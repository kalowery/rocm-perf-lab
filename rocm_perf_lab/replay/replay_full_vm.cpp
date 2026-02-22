#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstring>

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

    std::string base = argv[1];
    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::cerr << "hsa_init failed\n";
        return 1;
    }

    hsa_iterate_agents(find_gpu, nullptr);

    // ==========================================================
    // STAGE 1: RECONSTRUCT MEMORY BEFORE ANY EXECUTABLE LOAD
    // ==========================================================

    // ---- Select backing pool ----
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

    // ---- Load region metadata ----
    std::ifstream meta(base + "/memory_regions.json");
    if (!meta) {
        std::cerr << "memory_regions.json not found\n";
        return 1;
    }

    std::string contents((std::istreambuf_iterator<char>(meta)),
                          std::istreambuf_iterator<char>());

    size_t pos = 0;
    while ((pos = contents.find("\"base\":", pos)) != std::string::npos) {
        size_t start = contents.find_first_of("0123456789", pos);
        size_t end = contents.find_first_not_of("0123456789", start);
        uint64_t base = std::stoull(contents.substr(start, end - start));

        size_t size_pos = contents.find("\"size\":", end);
        size_t size_start = contents.find_first_of("0123456789", size_pos);
        size_t size_end = contents.find_first_not_of("0123456789", size_start);
        size_t size = std::stoull(contents.substr(size_start, size_end - size_start));

        // ---- Page-align reservation ----
        const size_t page = 4096;
        uint64_t aligned_base = base & ~(page - 1);
        uint64_t end_addr = base + size;
        uint64_t aligned_end = (end_addr + page - 1) & ~(page - 1);
        size_t aligned_size = aligned_end - aligned_base;
        size_t offset = base - aligned_base;

        void* reserved = nullptr;
        hsa_status_t st = hsa_amd_vmem_address_reserve(&reserved, aligned_size, aligned_base, 0);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "reserve failed at 0x" << std::hex << base << "\n";
            return 1;
        }

        hsa_amd_vmem_alloc_handle_t handle{};
        st = hsa_amd_vmem_handle_create(
                backing_pool,
                aligned_size,
                (hsa_amd_memory_type_t)0,
                0,
                &handle);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "handle create failed\n";
            return 1;
        }

        st = hsa_amd_vmem_map(reserved, aligned_size, 0, handle, 0);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "vm map failed\n";
            return 1;
        }

        hsa_amd_memory_access_desc_t access{};
        access.agent_handle = g_gpu_agent;
        access.permissions = HSA_ACCESS_PERMISSION_RW;
        st = hsa_amd_vmem_set_access(reserved, aligned_size, &access, 1);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "set access failed\n";
            return 1;
        }

        std::stringstream fname;
        fname << base << "/memory/region_"
              << std::hex << base << ".bin";

        std::ifstream blobf(fname.str(), std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(blobf)),
                                std::istreambuf_iterator<char>());

        void* copy_dst = static_cast<void*>(
            static_cast<uint8_t*>(reserved) + offset);

        st = hsa_memory_copy(copy_dst, blob.data(), size);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "memory copy failed\n";
            return 1;
        }

        pos = size_end;
    }

    std::cout << "Memory reconstructed.\n";

    // ==========================================================
    // STAGE 2: LOAD EXECUTABLE AFTER MEMORY RESTORE
    // ==========================================================

    std::ifstream hsaco_file(base + "/kernel.hsaco", std::ios::binary);
    if (!hsaco_file) {
        std::cerr << "kernel.hsaco not found\n";
        return 1;
    }

    std::vector<char> hsaco((std::istreambuf_iterator<char>(hsaco_file)),
                             std::istreambuf_iterator<char>());

    hsa_code_object_reader_t reader;
    hsa_code_object_reader_create_from_memory(hsaco.data(), hsaco.size(), &reader);

    hsa_executable_t executable;
    hsa_executable_create(HSA_PROFILE_FULL,
                          HSA_EXECUTABLE_STATE_UNFROZEN,
                          nullptr,
                          &executable);

    hsa_executable_load_agent_code_object(executable, g_gpu_agent, reader, nullptr, nullptr);
    hsa_executable_freeze(executable, nullptr);

    struct SymbolSearch { hsa_executable_symbol_t sym; bool found=false; } search;

    auto cb = [](hsa_executable_t, hsa_executable_symbol_t sym, void* data) {
        auto* s = reinterpret_cast<SymbolSearch*>(data);
        uint32_t type;
        hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &type);
        if (type == HSA_SYMBOL_KIND_KERNEL) {
            s->sym = sym;
            s->found = true;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_executable_iterate_symbols(executable, cb, &search);

    if (!search.found) {
        std::cerr << "No kernel symbol found\n";
        return 1;
    }

    uint64_t kernel_object = 0;
    uint32_t kernarg_size = 0;
    uint32_t private_segment_size = 0;
    uint32_t group_segment_size = 0;

    hsa_executable_symbol_get_info(search.sym,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &kernel_object);
    hsa_executable_symbol_get_info(search.sym,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernarg_size);
    hsa_executable_symbol_get_info(search.sym,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &private_segment_size);
    hsa_executable_symbol_get_info(search.sym,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &group_segment_size);

    // ---- Parse dispatch.json for dimensions ----
    std::ifstream dfile(base + "/dispatch.json");
    std::string dcontents((std::istreambuf_iterator<char>(dfile)),
                           std::istreambuf_iterator<char>());

    auto extract = [&](const std::string& key) -> uint32_t {
        auto p = dcontents.find(key);
        if (p == std::string::npos) return 1;
        auto s = dcontents.find_first_of("0123456789", p);
        auto e = dcontents.find_first_not_of("0123456789", s);
        return static_cast<uint32_t>(std::stoul(dcontents.substr(s, e - s)));
    };

    uint32_t grid_x  = extract("\"grid\": [");
    uint32_t block_x = extract("\"block\": [");

    // ---- Allocate kernarg ----
    void* kernarg = nullptr;
    hsa_amd_memory_pool_allocate(backing_pool, kernarg_size, 0, &kernarg);

    std::ifstream kf(base + "/kernarg.bin", std::ios::binary);
    std::vector<char> kblob((std::istreambuf_iterator<char>(kf)),
                             std::istreambuf_iterator<char>());
    memcpy(kernarg, kblob.data(), kblob.size());

    // ---- Create queue ----
    hsa_queue_t* queue = nullptr;
    hsa_status_t st = hsa_queue_create(
        g_gpu_agent,
        128,
        HSA_QUEUE_TYPE_MULTI,
        nullptr,
        nullptr,
        0,
        0,
        &queue);

    if (st != HSA_STATUS_SUCCESS) {
        std::cerr << "Queue creation failed\n";
        return 1;
    }

    // ---- Create completion signal ----
    hsa_signal_t completion_signal;
    hsa_signal_create(1, 0, nullptr, &completion_signal);

    // ---- Prepare dispatch packet ----
    uint64_t idx = hsa_queue_load_write_index_relaxed(queue);

    auto* pkt = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
        queue->base_address) + idx;

    memset(pkt, 0, sizeof(*pkt));

    pkt->kernel_object = kernel_object;
    pkt->kernarg_address = kernarg;

    pkt->grid_size_x = grid_x;
    pkt->grid_size_y = 1;
    pkt->grid_size_z = 1;

    pkt->workgroup_size_x = block_x;
    pkt->workgroup_size_y = 1;
    pkt->workgroup_size_z = 1;

    pkt->private_segment_size = private_segment_size;
    pkt->group_segment_size = group_segment_size;

    pkt->completion_signal = completion_signal;

    // ---- Setup packet header ----
    uint16_t header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER);

    pkt->header = header;
    pkt->setup =
        (1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);

    // ---- Publish packet ----
    hsa_queue_store_write_index_relaxed(queue, idx + 1);
    hsa_signal_store_relaxed(queue->doorbell_signal, idx);

    // ---- Wait for completion ----
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
