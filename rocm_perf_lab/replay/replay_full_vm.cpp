#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstring>
#include <sys/mman.h>

static hsa_agent_t g_gpu_agent{};

static hsa_status_t find_gpu(hsa_agent_t agent, void*) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        g_gpu_agent = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

#include <unistd.h>

struct ReplayMetadata {
    std::string gpu_agent;
    std::string rocm_version;
    pid_t pid;
};

struct ReplayResult {
    ReplayMetadata metadata;
    size_t iterations;
    bool recopy;

    std::string kernel_name;
    std::string hsaco_path;

    std::vector<uint64_t> raw_ns;

    double avg_us;
    double min_us;
    double max_us;
};

static void print_json_output(const ReplayResult& r) {
    std::cout << "{\n";
    std::cout << "  \"kernel\": {\n";
    std::cout << "    \"name\": \"" << r.kernel_name << "\",\n";
    std::cout << "    \"hsaco_path\": \"" << r.hsaco_path << "\"\n";
    std::cout << "  },\n";
    std::cout << "  \"execution\": {\n";
    std::cout << "    \"iterations\": " << r.iterations << ",\n";
    std::cout << "    \"mode\": \"" << (r.recopy ? "stateless" : "stateful") << "\"\n";
    std::cout << "  },\n";
    std::cout << "  \"timing\": {\n";
    std::cout << "    \"unit\": \"microseconds\",\n";
    std::cout << "    \"average\": " << r.avg_us << ",\n";
    std::cout << "    \"min\": " << r.min_us << ",\n";
    std::cout << "    \"max\": " << r.max_us << "\n";
    std::cout << "  },\n";
    std::cout << "  \"environment\": {\n";
    std::cout << "    \"gpu_agent\": \"" << r.metadata.gpu_agent << "\",\n";
    std::cout << "    \"rocm_version\": \"" << r.metadata.rocm_version << "\",\n";
    std::cout << "    \"pid\": " << r.metadata.pid << "\n";
    std::cout << "  }\n";
    std::cout << "}\n";
}

static ReplayMetadata collect_metadata(hsa_agent_t agent) {
    ReplayMetadata meta{};

    // GPU agent name
    char name[64] = {};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name) == HSA_STATUS_SUCCESS) {
        meta.gpu_agent = name;
    } else {
        meta.gpu_agent = "unknown";
    }

    // ROCm runtime version
    uint16_t major = 0, minor = 0;
    if (hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major) == HSA_STATUS_SUCCESS &&
        hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &minor) == HSA_STATUS_SUCCESS) {
        meta.rocm_version = std::to_string(major) + "." + std::to_string(minor);
    } else {
        meta.rocm_version = "unknown";
    }

    // PID
    meta.pid = getpid();

    return meta;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: rocm_perf_replay_full_vm <capture_dir> "
                     "[--iterations N] [--no-recopy]\n";
        return 1;
    }

    std::string capture_dir = argv[1];

    size_t iterations = 1;
    bool recopy = true;
    bool json_output = false;
    std::string override_hsaco_path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoull(argv[++i]);
        } else if (arg == "--no-recopy") {
            recopy = false;
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg == "--hsaco" && i + 1 < argc) {
            override_hsaco_path = argv[++i];
        }
    }

    // ==========================================================
    // STAGE 0: PARSE REGION METADATA (BEFORE hsa_init)
    // ==========================================================

    std::ifstream meta(capture_dir + "/memory_regions.json");
    if (!meta) {
        std::cerr << "memory_regions.json not found\n";
        return 1;
    }

    std::string contents((std::istreambuf_iterator<char>(meta)),
                          std::istreambuf_iterator<char>());

    struct RegionMeta {
        uint64_t base;
        size_t size;
        uint64_t aligned_base;
        size_t aligned_size;
        size_t offset;
    };

    std::vector<RegionMeta> regions;

    size_t pos = 0;
    while ((pos = contents.find("\"base\":", pos)) != std::string::npos) {

        size_t start = contents.find_first_of("0123456789", pos);
        size_t end = contents.find_first_not_of("0123456789", start);
        uint64_t region_base = std::stoull(contents.substr(start, end - start));

        size_t size_pos = contents.find("\"size\":", end);
        size_t size_start = contents.find_first_of("0123456789", size_pos);
        size_t size_end = contents.find_first_not_of("0123456789", size_start);
        size_t size = std::stoull(contents.substr(size_start, size_end - size_start));

        const size_t page = 4096;
        uint64_t aligned_base = region_base & ~(page - 1);
        uint64_t end_addr = region_base + size;
        uint64_t aligned_end = (end_addr + page - 1) & ~(page - 1);
        size_t aligned_size = aligned_end - aligned_base;
        size_t offset = region_base - aligned_base;

        regions.push_back({region_base, size, aligned_base, aligned_size, offset});
        pos = size_end;
    }

    // ==========================================================
    // STAGE 0.5: PRE-MMAP TO STEER ROCr SVM APERTURE
    // ==========================================================

    struct PreMap { void* addr; size_t size; };
    std::vector<PreMap> premaps;

    for (const auto& r : regions) {
        void* addr = mmap(reinterpret_cast<void*>(r.aligned_base),
                          r.aligned_size,
                          PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                          -1,
                          0);
        if (addr != MAP_FAILED) {
            premaps.push_back({addr, r.aligned_size});
        }
    }

    // ==========================================================
    // STAGE 1: INIT HSA
    // ==========================================================

    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::cerr << "hsa_init failed\n";
        return 1;
    }

    for (auto& pm : premaps) {
        munmap(pm.addr, pm.size);
    }

    hsa_iterate_agents(find_gpu, nullptr);

    // Collect metadata and initialize result structure
    ReplayResult result{};
    result.metadata = collect_metadata(g_gpu_agent);
    result.iterations = iterations;
    result.recopy = recopy;

    // ==========================================================
    // STAGE 2: SELECT BACKING POOL
    // ==========================================================

    hsa_amd_memory_pool_t backing_pool{};
    bool found_pool = false;

    auto pool_cb = [](hsa_amd_memory_pool_t pool, void* data) {
        auto* ctx = reinterpret_cast<std::pair<hsa_amd_memory_pool_t*, bool*>*>(data);
        hsa_amd_segment_t segment;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
            &segment);
        bool alloc_allowed = false;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
            &alloc_allowed);

        if (segment == HSA_AMD_SEGMENT_GLOBAL && alloc_allowed) {
            *ctx->first = pool;
            *ctx->second = true;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    std::pair<hsa_amd_memory_pool_t*, bool*> pool_ctx{&backing_pool, &found_pool};
    hsa_amd_agent_iterate_memory_pools(g_gpu_agent, pool_cb, &pool_ctx);

    if (!found_pool) {
        std::cerr << "No suitable memory pool found\n";
        return 1;
    }

    // ==========================================================
    // STAGE 3: STRICT RESERVE + MAP (NO COPY YET)
    // ==========================================================

    struct RegionRuntime {
        void* reserved;
        size_t size;
        size_t offset;
        std::vector<char> blob;
    };

    std::vector<RegionRuntime> runtime_regions;

    for (const auto& r : regions) {

        void* reserved = nullptr;

        if (hsa_amd_vmem_address_reserve(&reserved,
                                         r.aligned_size,
                                         r.aligned_base,
                                         0) != HSA_STATUS_SUCCESS ||
            reinterpret_cast<uint64_t>(reserved) != r.aligned_base) {

            std::cerr << "Relocation detected or reserve failed at 0x"
                      << std::hex << r.base << "\n";
            return 1;
        }

        hsa_amd_vmem_alloc_handle_t handle{};
        if (hsa_amd_vmem_handle_create(backing_pool,
                                       r.aligned_size,
                                       (hsa_amd_memory_type_t)0,
                                       0,
                                       &handle) != HSA_STATUS_SUCCESS) {
            return 1;
        }

        if (hsa_amd_vmem_map(reserved,
                             r.aligned_size,
                             0,
                             handle,
                             0) != HSA_STATUS_SUCCESS) {
            return 1;
        }

        hsa_amd_memory_access_desc_t access{};
        access.agent_handle = g_gpu_agent;
        access.permissions = HSA_ACCESS_PERMISSION_RW;
        hsa_amd_vmem_set_access(reserved, r.aligned_size, &access, 1);

        std::stringstream fname;
        fname << capture_dir << "/memory/region_"
              << std::hex << r.base << ".bin";

        std::ifstream blobf(fname.str(), std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(blobf)),
                                std::istreambuf_iterator<char>());

        runtime_regions.push_back({
            reserved,
            r.size,
            r.offset,
            std::move(blob)
        });
    }

    // ==========================================================
    // STAGE 4: LOAD EXECUTABLE
    // ==========================================================

    std::string hsaco_path = override_hsaco_path.empty()
        ? (capture_dir + "/kernel.hsaco")
        : override_hsaco_path;

    result.hsaco_path = hsaco_path;

    std::ifstream hsaco_file(hsaco_path, std::ios::binary);
    if (!hsaco_file) {
        std::cerr << "Failed to open HSACO: " << hsaco_path << "\n";
        return 1;
    }

    std::vector<char> hsaco((std::istreambuf_iterator<char>(hsaco_file)),
                             std::istreambuf_iterator<char>());

    hsa_code_object_reader_t reader;
    hsa_code_object_reader_create_from_memory(
        hsaco.data(), hsaco.size(), &reader);

    hsa_executable_t executable;
    hsa_executable_create(HSA_PROFILE_FULL,
                          HSA_EXECUTABLE_STATE_UNFROZEN,
                          nullptr,
                          &executable);

    hsa_executable_load_agent_code_object(
        executable, g_gpu_agent, reader, nullptr, nullptr);

    hsa_executable_freeze(executable, nullptr);

    hsa_executable_symbol_t kernel_symbol{};
    hsa_executable_iterate_symbols(executable,
        [](hsa_executable_t,
           hsa_executable_symbol_t sym,
           void* data) -> hsa_status_t {
            uint32_t type;
            hsa_executable_symbol_get_info(sym,
                HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                &type);
            if (type == HSA_SYMBOL_KIND_KERNEL) {
                *reinterpret_cast<hsa_executable_symbol_t*>(data) = sym;
                return HSA_STATUS_INFO_BREAK;
            }
            return HSA_STATUS_SUCCESS;
        },
        &kernel_symbol);

    uint64_t kernel_object = 0;
    uint32_t kernarg_size = 0;
    uint32_t group_segment = 0;
    uint32_t private_segment = 0;

    hsa_executable_symbol_get_info(kernel_symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &kernel_object);
    hsa_executable_symbol_get_info(kernel_symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernarg_size);
    hsa_executable_symbol_get_info(kernel_symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &group_segment);
    hsa_executable_symbol_get_info(kernel_symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &private_segment);

    // Parse grid/block
    std::ifstream dfile(capture_dir + "/dispatch.json");
    std::string dcontents((std::istreambuf_iterator<char>(dfile)),
                           std::istreambuf_iterator<char>());

    auto extract_int = [&](const std::string& key) -> uint32_t {
        auto p = dcontents.find(key);
        auto s = dcontents.find_first_of("0123456789", p);
        auto e = dcontents.find_first_not_of("0123456789", s);
        return std::stoul(dcontents.substr(s, e - s));
    };

    auto extract_string = [&](const std::string& key) -> std::string {
        auto p = dcontents.find(key);
        if (p == std::string::npos) return "unknown";
        auto first_quote = dcontents.find('"', p + key.size());
        if (first_quote == std::string::npos) return "unknown";
        first_quote++;
        auto second_quote = dcontents.find('"', first_quote);
        if (second_quote == std::string::npos) return "unknown";
        return dcontents.substr(first_quote, second_quote - first_quote);
    };

    auto extract_int_array = [&](const std::string& key, uint32_t out[3]) {
        out[0] = out[1] = out[2] = 1;
        auto p = dcontents.find(key);
        if (p == std::string::npos) return;
        for (int i = 0; i < 3; i++) {
            auto s = dcontents.find_first_of("0123456789", p);
            if (s == std::string::npos) break;
            auto e = dcontents.find_first_not_of("0123456789", s);
            out[i] = std::stoul(dcontents.substr(s, e - s));
            p = e;
        }
    };

    uint32_t grid[3], block[3];
    extract_int_array("\"grid\":", grid);
    extract_int_array("\"block\":", block);

    std::string demangled = extract_string("\"demangled_name\":");
    std::string mangled   = extract_string("\"mangled_name\":");

    if (demangled != "unknown") {
        // Normalize: strip trailing clone / suffix annotations like " [clone .kd]"
        auto pos = demangled.find(" [");
        if (pos != std::string::npos) {
            result.kernel_name = demangled.substr(0, pos);
        } else {
            result.kernel_name = demangled;
        }
    } else if (mangled != "unknown") {
        result.kernel_name = mangled;
    } else {
        result.kernel_name = "unknown";
    }

    // ==========================================================
    // STAGE 5: MULTI-ITERATION DISPATCH WITH PROFILING
    // ==========================================================

    auto restore_memory = [&]() {
        for (auto& rr : runtime_regions) {
            void* dst = static_cast<void*>(
                static_cast<uint8_t*>(rr.reserved) + rr.offset);
            hsa_memory_copy(dst, rr.blob.data(), rr.size);
        }
    };

    void* kernarg = nullptr;
    hsa_amd_memory_pool_allocate(backing_pool,
                                 kernarg_size,
                                 0,
                                 &kernarg);

    std::ifstream kf(capture_dir + "/kernarg.bin", std::ios::binary);
    std::vector<char> kblob((std::istreambuf_iterator<char>(kf)),
                             std::istreambuf_iterator<char>());
    memcpy(kernarg, kblob.data(), kblob.size());

    hsa_queue_t* queue = nullptr;
    hsa_queue_create(g_gpu_agent,
                     128,
                     HSA_QUEUE_TYPE_MULTI,
                     nullptr,
                     nullptr,
                     private_segment,
                     group_segment,
                     &queue);

    hsa_status_t pst = hsa_amd_profiling_set_profiler_enabled(queue, 1);
    if (pst != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to enable profiling\n";
        return 1;
    }

    hsa_signal_t completion_signal;
    hsa_status_t ss = hsa_signal_create(1, 0, nullptr, &completion_signal);

    if (ss != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to create completion signal\n";
        return 1;
    }

    result.raw_ns.clear();
    result.raw_ns.reserve(iterations);

    restore_memory();

    for (size_t iter = 0; iter < iterations; ++iter) {

        if (iter > 0 && recopy) {
            restore_memory();
        }

        uint64_t index = hsa_queue_load_write_index_relaxed(queue);
        auto* packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
            queue->base_address) + (index % queue->size);

        memset(packet, 0, sizeof(*packet));

        uint32_t dims = 1;
        if (grid[2] > 1 || block[2] > 1) dims = 3;
        else if (grid[1] > 1 || block[1] > 1) dims = 2;

        packet->setup = dims << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        packet->workgroup_size_x = block[0];
        packet->workgroup_size_y = block[1];
        packet->workgroup_size_z = block[2];
        packet->grid_size_x = grid[0];
        packet->grid_size_y = grid[1];
        packet->grid_size_z = grid[2];
        packet->kernel_object = kernel_object;
        packet->kernarg_address = kernarg;
        packet->private_segment_size = private_segment;
        packet->group_segment_size = group_segment;
        packet->completion_signal = completion_signal;

        uint16_t header =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
            (1 << HSA_PACKET_HEADER_BARRIER);
        packet->header = header;

        hsa_queue_store_write_index_relaxed(queue, index + 1);
        hsa_signal_store_relaxed(queue->doorbell_signal, index);

        while (hsa_signal_wait_relaxed(
                   completion_signal,
                   HSA_SIGNAL_CONDITION_EQ,
                   0,
                   UINT64_MAX,
                   HSA_WAIT_STATE_ACTIVE) != 0) {}

        hsa_amd_profiling_dispatch_time_t time{};
        hsa_status_t dt = hsa_amd_profiling_get_dispatch_time(
            g_gpu_agent,
            completion_signal,
            &time);

        if (dt != HSA_STATUS_SUCCESS) {
            std::cerr << "Failed to get dispatch time\n";
            return 1;
        }

        result.raw_ns.push_back(time.end - time.start);

        hsa_signal_store_relaxed(completion_signal, 1);
    }

    if (!result.raw_ns.empty()) {
        uint64_t min = result.raw_ns[0];
        uint64_t max = result.raw_ns[0];
        uint64_t sum = 0;

        for (auto d : result.raw_ns) {
            if (d < min) min = d;
            if (d > max) max = d;
            sum += d;
        }

        double avg_ns = double(sum) / result.raw_ns.size();

        result.avg_us = avg_ns / 1000.0;
        result.min_us = min / 1000.0;
        result.max_us = max / 1000.0;

        if (json_output) {
            print_json_output(result);
        } else {
            std::cout << "Iterations: " << iterations << "\n";
            std::cout << "Mode: " << (recopy ? "stateless" : "stateful") << "\n";
            std::cout << "Average GPU time: " << result.avg_us << " us\n";
            std::cout << "Min: " << result.min_us << " us\n";
            std::cout << "Max: " << result.max_us << " us\n";
        }
    }

    hsa_shut_down();
    return 0;
}
