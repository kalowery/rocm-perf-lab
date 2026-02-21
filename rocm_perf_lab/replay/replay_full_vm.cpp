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

int main() {
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
    std::ifstream meta("../../isolate/tool/isolate_capture/memory_regions.json");
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

        void* reserved = nullptr;
        hsa_status_t st = hsa_amd_vmem_address_reserve(&reserved, size, base, 0);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "reserve failed at 0x" << std::hex << base << "\n";
            return 1;
        }

        hsa_amd_vmem_alloc_handle_t handle{};
        st = hsa_amd_vmem_handle_create(
                backing_pool,
                size,
                (hsa_amd_memory_type_t)0,
                0,
                &handle);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "handle create failed\n";
            return 1;
        }

        st = hsa_amd_vmem_map(reserved, size, 0, handle, 0);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "vm map failed\n";
            return 1;
        }

        hsa_amd_memory_access_desc_t access{};
        access.agent_handle = g_gpu_agent;
        access.permissions = HSA_ACCESS_PERMISSION_RW;
        st = hsa_amd_vmem_set_access(reserved, size, &access, 1);
        if (st != HSA_STATUS_SUCCESS) {
            std::cerr << "set access failed\n";
            return 1;
        }

        std::stringstream fname;
        fname << "../../isolate/tool/isolate_capture/memory/region_"
              << std::hex << base << ".bin";

        std::ifstream blobf(fname.str(), std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(blobf)),
                                std::istreambuf_iterator<char>());

        st = hsa_memory_copy(reserved, blob.data(), size);
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

    std::ifstream hsaco_file("../../isolate/tool/isolate_capture/kernel.hsaco", std::ios::binary);
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
    hsa_executable_symbol_get_info(search.sym,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &kernel_object);
    hsa_executable_symbol_get_info(search.sym,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernarg_size);

    // ---- Allocate kernarg ----
    void* kernarg = nullptr;
    hsa_amd_memory_pool_allocate(backing_pool, kernarg_size, 0, &kernarg);

    std::ifstream kf("../../isolate/tool/isolate_capture/kernarg.bin", std::ios::binary);
    std::vector<char> kblob((std::istreambuf_iterator<char>(kf)),
                             std::istreambuf_iterator<char>());
    memcpy(kernarg, kblob.data(), kblob.size());

    // ---- Create queue ----
    hsa_queue_t* queue = nullptr;
    hsa_queue_create(g_gpu_agent, 128, HSA_QUEUE_TYPE_MULTI,
                     nullptr, nullptr, 0, 0, &queue);

    uint64_t idx = hsa_queue_load_write_index_relaxed(queue);
    auto* pkt = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address) + idx;
    memset(pkt, 0, sizeof(*pkt));

    pkt->kernel_object = kernel_object;
    pkt->kernarg_address = kernarg;
    pkt->grid_size_x = 1;
    pkt->workgroup_size_x = 1;

    hsa_signal_store_relaxed(queue->doorbell_signal, idx);

    std::cout << "Dispatch submitted.\n";

    hsa_shut_down();
    return 0;
}
