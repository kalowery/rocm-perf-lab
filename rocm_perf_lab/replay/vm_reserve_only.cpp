#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <fstream>
#include <string>

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

    const uint32_t page_size = 4096;

    std::ifstream meta("../../isolate/tool/isolate_capture/memory_regions.json");
    if (!meta) {
        std::cerr << "memory_regions.json not found\n";
        return 1;
    }

    std::string contents((std::istreambuf_iterator<char>(meta)),
                          std::istreambuf_iterator<char>());

    size_t pos = 0;
    bool any_fail = false;
    bool any_reloc = false;
    size_t total_regions = 0;
    size_t total_bytes = 0;

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

        void* reserved = nullptr;
        hsa_status_t st = hsa_amd_vmem_address_reserve(
            &reserved, aligned_size, aligned_base, 0);

        std::cout << "Region 0x" << std::hex << base
                  << " size " << std::dec << size << " -> ";

        if (st != HSA_STATUS_SUCCESS) {
            const char* status_str = nullptr;
            hsa_status_string(st, &status_str);
            std::cout << "FAIL (" << (status_str ? status_str : "unknown") << ")\n";
            any_fail = true;
        } else if (reserved != (void*)aligned_base) {
            std::cout << "RELOCATED (0x" << std::hex << reserved << ")\n";
            any_reloc = true;
        } else {
            std::cout << "OK\n";
        }

        total_regions++;
        total_bytes += size;
        pos = size_end;
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
