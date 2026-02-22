#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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
        std::cerr << "Usage: vm_reserve_only <capture_dir>\n";
        return 1;
    }

    std::string capture_dir = argv[1];
    const uint32_t page_size = 4096;

    // ==============================
    // Parse regions BEFORE hsa_init
    // ==============================

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
    };

    std::vector<Region> regions;

    size_t pos = 0;
    while ((pos = contents.find("\"base\":", pos)) != std::string::npos) {
        size_t start = contents.find_first_of("0123456789", pos);
        size_t end = contents.find_first_not_of("0123456789", start);
        uint64_t base = std::stoull(contents.substr(start, end - start));

        size_t size_pos = contents.find("\"size\":", end);
        size_t size_start = contents.find_first_of("0123456789", size_pos);
        size_t size_end = contents.find_first_not_of("0123456789", size_start);
        size_t size = std::stoull(contents.substr(size_start, size_end - size_start));

        uint64_t aligned_base = base & ~(uint64_t(page_size) - 1);
        uint64_t end_addr = base + size;
        uint64_t aligned_end = (end_addr + page_size - 1) & ~(uint64_t(page_size) - 1);
        size_t aligned_size = aligned_end - aligned_base;

        regions.push_back({base, size, aligned_base, aligned_size});
        pos = size_end;
    }

    // ==============================
    // Pre-mmap to steer ROCr
    // ==============================

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

    // ==============================
    // hsa_init AFTER steering
    // ==============================

    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::cerr << "hsa_init failed\n";
        return 1;
    }

    for (auto& pm : premaps) {
        munmap(pm.addr, pm.size);
    }

    hsa_iterate_agents(find_gpu, nullptr);

    // ==============================
    // Strict reserve check
    // ==============================

    bool any_fail = false;
    bool any_reloc = false;
    size_t total_regions = 0;
    size_t total_bytes = 0;

    for (const auto& r : regions) {

        void* reserved = nullptr;
        hsa_status_t st = hsa_amd_vmem_address_reserve(
            &reserved,
            r.aligned_size,
            r.aligned_base,
            0);

        std::cout << "Region 0x" << std::hex << r.base
                  << " size " << std::dec << r.size << " -> ";

        if (st != HSA_STATUS_SUCCESS) {
            const char* status_str = nullptr;
            hsa_status_string(st, &status_str);
            std::cout << "FAIL (" << (status_str ? status_str : "unknown") << ")\n";
            any_fail = true;
        } else if (reserved != (void*)r.aligned_base) {
            std::cout << "RELOCATED (0x" << std::hex << reserved << ")\n";
            any_reloc = true;
        } else {
            std::cout << "OK\n";
        }

        total_regions++;
        total_bytes += r.size;
    }

    std::cout << "\nSummary:\n";
    std::cout << "  Regions: " << total_regions << "\n";
    std::cout << "  Total bytes: " << total_bytes << "\n";
    std::cout << "  Page size: " << page_size << "\n";

    hsa_shut_down();

    if (any_fail) return 1;
    if (any_reloc) return 2;
    return 0;
}
