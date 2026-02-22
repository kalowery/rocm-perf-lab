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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: rocm_perf_replay_full_vm <capture_dir>\n";
        return 1;
    }

    std::string capture_dir = argv[1];

    // ---- Dump initial maps ----
    {
        std::ifstream maps("/proc/self/maps");
        std::cerr << "==== /proc/self/maps BEFORE hsa_init ====\n";
        std::cerr << maps.rdbuf();
        std::cerr << "=========================================\n";
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

    struct Region {
        uint64_t base;
        size_t size;
        uint64_t aligned_base;
        size_t aligned_size;
        size_t offset;
    };

    std::vector<Region> regions;

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
    //
    // ROCr (via libhsakmt) reserves large SVM aperture ranges during
    // hsa_init() using mmap(PROT_NONE). The aperture base is selected
    // heuristically based on the current process VA layout.
    //
    // If a captured VA region overlaps that aperture, strict
    // hsa_amd_vmem_address_reserve() will relocate and replay aborts.
    //
    // To make strict replay deterministic, we temporarily mmap the
    // captured VA ranges BEFORE hsa_init(). This forces ROCr to choose
    // alternate aperture locations that avoid those ranges.
    //
    // After hsa_init(), we munmap these placeholders and then perform
    // strict hsa_amd_vmem_address_reserve() at the original VAs.
    //
    // This does NOT relax strict replay semantics. It only shapes
    // the process VA topology so ROCr's internal aperture heuristic
    // cannot collide with captured regions.

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
    // STAGE 1: HSA INIT
    // ==========================================================

    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::cerr << "hsa_init failed\n";
        return 1;
    }

    {
        std::ifstream maps("/proc/self/maps");
        std::cerr << "==== /proc/self/maps AFTER hsa_init ====\n";
        std::cerr << maps.rdbuf();
        std::cerr << "=========================================\n";
    }

    // Remove placeholders
    for (const auto& pm : premaps) {
        munmap(pm.addr, pm.size);
    }
    premaps.clear();

    hsa_iterate_agents(find_gpu, nullptr);

    // ==========================================================
    // STAGE 2: SELECT BACKING POOL
    // ==========================================================

    hsa_amd_memory_pool_t backing_pool{};
    bool found_pool = false;

    auto pool_cb = [](hsa_amd_memory_pool_t pool, void* data) {
        auto* ctx = reinterpret_cast<std::pair<hsa_amd_memory_pool_t*, bool*>*>(data);
        hsa_amd_segment_t segment;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        bool alloc_allowed = false;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
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
        std::cerr << "No backing pool found\n";
        return 1;
    }

    // ==========================================================
    // STAGE 3: STRICT VM RESERVE + RESTORE
    // ==========================================================

    for (const auto& r : regions) {

        void* reserved = nullptr;
        hsa_status_t st =
            hsa_amd_vmem_address_reserve(&reserved,
                                         r.aligned_size,
                                         r.aligned_base,
                                         0);

        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "reserve failed at 0x"
                      << std::hex << r.base << "\n";
            return 1;
        }

        if (reinterpret_cast<uint64_t>(reserved) != r.aligned_base) {
            std::cerr << "Relocation detected for region 0x"
                      << std::hex << r.base
                      << " requested 0x" << r.aligned_base
                      << " got 0x" << reinterpret_cast<uint64_t>(reserved)
                      << "\n";
            return 1;
        }

        hsa_amd_vmem_alloc_handle_t handle{};
        st = hsa_amd_vmem_handle_create(
                backing_pool,
                r.aligned_size,
                (hsa_amd_memory_type_t)0,
                0,
                &handle);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "handle create failed\n";
            return 1;
        }

        st = hsa_amd_vmem_map(reserved, r.aligned_size, 0, handle, 0);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "vm map failed\n";
            return 1;
        }

        hsa_amd_memory_access_desc_t access{};
        access.agent_handle = g_gpu_agent;
        access.permissions = HSA_ACCESS_PERMISSION_RW;
        st = hsa_amd_vmem_set_access(reserved, r.aligned_size, &access, 1);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "set access failed\n";
            return 1;
        }

        std::stringstream fname;
        fname << capture_dir << "/memory/region_"
              << std::hex << r.base << ".bin";

        std::ifstream blobf(fname.str(), std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(blobf)),
                                std::istreambuf_iterator<char>());

        void* copy_dst =
            static_cast<void*>(
                static_cast<uint8_t*>(reserved) + r.offset);

        st = hsa_memory_copy(copy_dst, blob.data(), r.size);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "memory copy failed\n";
            return 1;
        }
    }

    std::cout << "Memory reconstructed.\n";

    // ==========================================================
    // STAGE 4: LOAD EXECUTABLE + DISPATCH KERNEL
    // ==========================================================

    std::ifstream hsaco_file(capture_dir + "/kernel.hsaco", std::ios::binary);
    if (!hsaco_file) {
        std::cerr << "kernel.hsaco not found\n";
        return 1;
    }

    std::vector<char> hsaco((std::istreambuf_iterator<char>(hsaco_file)),
                             std::istreambuf_iterator<char>());

    hsa_code_object_reader_t reader;
    if (hsa_code_object_reader_create_from_memory(hsaco.data(), hsaco.size(), &reader)
        != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to create code object reader\n";
        return 1;
    }

    hsa_executable_t executable;
    if (hsa_executable_create(HSA_PROFILE_FULL,
                              HSA_EXECUTABLE_STATE_UNFROZEN,
                              nullptr,
                              &executable) != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to create executable\n";
        return 1;
    }

    if (hsa_executable_load_agent_code_object(executable, g_gpu_agent, reader, nullptr, nullptr)
        != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to load code object\n";
        return 1;
    }

    if (hsa_executable_freeze(executable, nullptr) != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to freeze executable\n";
        return 1;
    }

    // Resolve first kernel symbol
    hsa_executable_symbol_t kernel_symbol{};
    bool found_symbol = false;

    auto sym_cb = [](hsa_executable_t,
                     hsa_executable_symbol_t sym,
                     void* data) -> hsa_status_t {
        auto* flag = reinterpret_cast<bool*>(data);
        uint32_t type;
        hsa_executable_symbol_get_info(sym,
            HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
            &type);
        if (type == HSA_SYMBOL_KIND_KERNEL) {
            *flag = true;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_executable_iterate_symbols(executable, sym_cb, &found_symbol);

    if (!found_symbol) {
        std::cerr << "No kernel symbol found\n";
        return 1;
    }

    // For simplicity, reload and capture first kernel symbol
    hsa_executable_iterate_symbols(executable,
        [](hsa_executable_t,
           hsa_executable_symbol_t sym,
           void* data) -> hsa_status_t {
            auto* out = reinterpret_cast<hsa_executable_symbol_t*>(data);
            uint32_t type;
            hsa_executable_symbol_get_info(sym,
                HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                &type);
            if (type == HSA_SYMBOL_KIND_KERNEL) {
                *out = sym;
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

    // Load dispatch.json for grid/block
    std::ifstream dfile(capture_dir + "/dispatch.json");
    std::string dcontents((std::istreambuf_iterator<char>(dfile)),
                           std::istreambuf_iterator<char>());

    auto extract_int = [&](const std::string& key) -> uint32_t {
        auto p = dcontents.find(key);
        if (p == std::string::npos) return 1;
        auto s = dcontents.find_first_of("0123456789", p);
        auto e = dcontents.find_first_not_of("0123456789", s);
        return static_cast<uint32_t>(std::stoul(dcontents.substr(s, e - s)));
    };

    uint32_t grid_x  = extract_int("\"grid\": [");
    uint32_t block_x = extract_int("\"block\": [");

    // Allocate kernarg
    void* kernarg = nullptr;
    if (hsa_amd_memory_pool_allocate(backing_pool, kernarg_size, 0, &kernarg)
        != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to allocate kernarg\n";
        return 1;
    }

    std::ifstream kf(capture_dir + "/kernarg.bin", std::ios::binary);
    std::vector<char> kblob((std::istreambuf_iterator<char>(kf)),
                             std::istreambuf_iterator<char>());
    memcpy(kernarg, kblob.data(), kblob.size());

    // Create queue
    hsa_queue_t* queue = nullptr;
    if (hsa_queue_create(g_gpu_agent,
                         128,
                         HSA_QUEUE_TYPE_MULTI,
                         nullptr,
                         nullptr,
                         private_segment,
                         group_segment,
                         &queue) != HSA_STATUS_SUCCESS) {
        std::cerr << "Queue creation failed\n";
        return 1;
    }

    hsa_signal_t completion_signal;
    hsa_signal_create(1, 0, nullptr, &completion_signal);

    uint64_t index = hsa_queue_load_write_index_relaxed(queue);

    auto* packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
        queue->base_address) + (index % queue->size);

    memset(packet, 0, sizeof(*packet));

    packet->setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    packet->workgroup_size_x = block_x;
    packet->workgroup_size_y = 1;
    packet->workgroup_size_z = 1;

    packet->grid_size_x = grid_x;
    packet->grid_size_y = 1;
    packet->grid_size_z = 1;

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
               HSA_WAIT_STATE_ACTIVE) != 0) {
    }

    std::cout << "Dispatch completed.\n";

    hsa_shut_down();
    return 0;
}
