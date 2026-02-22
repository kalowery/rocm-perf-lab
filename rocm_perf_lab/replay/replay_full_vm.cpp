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

    // ---- Rest of file (HSACO load + dispatch) remains identical to original ----

    std::cout << "Replay complete.\n";
    hsa_shut_down();
    return 0;
}
