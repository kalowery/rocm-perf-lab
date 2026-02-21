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

    std::ifstream meta("../../isolate/tool/isolate_capture/memory_regions.json");
    if (!meta) {
        std::cerr << "memory_regions.json not found\n";
        return 1;
    }

    std::string contents((std::istreambuf_iterator<char>(meta)),
                          std::istreambuf_iterator<char>());

    size_t pos = 0;
    bool all_ok = true;

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

        std::cout << "Reserve 0x" << std::hex << base
                  << " size " << std::dec << size << " -> ";

        if (st != HSA_STATUS_SUCCESS) {
            const char* status_str = nullptr;
            hsa_status_string(st, &status_str);
            std::cout << "FAIL (" << (status_str ? status_str : "unknown") << ")\n";
            all_ok = false;
        } else if (reserved != (void*)base) {
            std::cout << "MISMATCH (got 0x" << std::hex << reserved << ")\n";
            all_ok = false;
        } else {
            std::cout << "OK\n";
        }

        pos = size_end;
    }

    if (all_ok)
        std::cout << "All reservations succeeded.\n";

    hsa_shut_down();
    return all_ok ? 0 : 1;
}
