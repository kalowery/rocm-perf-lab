#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>

static hsa_agent_t g_gpu_agent{};
static std::string g_gpu_isa_name;

static hsa_status_t agent_callback(hsa_agent_t agent, void* data)
{
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);

    if (type == HSA_DEVICE_TYPE_GPU && g_gpu_agent.handle == 0) {
        g_gpu_agent = agent;

        hsa_isa_t isa;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);

        char isa_name[64] = {};
        hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);

        g_gpu_isa_name = isa_name;
    }

    return HSA_STATUS_SUCCESS;
}

int main()
{
    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to initialize HSA\n";
        return 1;
    }

    hsa_iterate_agents(agent_callback, nullptr);

    if (g_gpu_agent.handle == 0) {
        std::cerr << "No GPU agent found\n";
        return 1;
    }

    std::cout << "Detected GPU ISA: " << g_gpu_isa_name << "\n";

    std::ifstream meta("../../isolate/tool/isolate_capture/dispatch.json");
    if (!meta) {
        std::cerr << "dispatch.json not found\n";
        return 1;
    }

    std::string contents((std::istreambuf_iterator<char>(meta)),
                         std::istreambuf_iterator<char>());

    auto pos = contents.find("\"isa_name\"");
    if (pos == std::string::npos) {
        std::cerr << "ISA metadata missing\n";
        return 1;
    }

    auto start = contents.find("\"", pos + 10);
    auto end = contents.find("\"", start + 1);
    std::string captured_isa = contents.substr(start + 1, end - start - 1);

    std::cout << "Captured ISA: " << captured_isa << "\n";

    if (captured_isa != g_gpu_isa_name) {
        std::cerr << "ISA mismatch — replay unsafe\n";
        return 1;
    }

    std::cout << "ISA matches. Loading HSACO...\n";

    // Extract mangled_name
    auto mpos = contents.find("\"mangled_name\"");
    if (mpos == std::string::npos) {
        std::cerr << "mangled_name missing\n";
        return 1;
    }

    auto mstart = contents.find("\"", mpos + 14);
    auto mend = contents.find("\"", mstart + 1);
    std::string mangled_name = contents.substr(mstart + 1, mend - mstart - 1);

    std::cout << "Captured kernel: " << mangled_name << "\n";

    // Load HSACO blob
    std::ifstream hsaco_file("../../isolate/tool/isolate_capture/kernel.hsaco", std::ios::binary);
    if (!hsaco_file) {
        std::cerr << "kernel.hsaco not found\n";
        return 1;
    }

    std::vector<uint8_t> hsaco_blob((std::istreambuf_iterator<char>(hsaco_file)),
                                     std::istreambuf_iterator<char>());

    hsa_code_object_reader_t reader;
    if (hsa_code_object_reader_create_from_memory(
            hsaco_blob.data(), hsaco_blob.size(), &reader) != HSA_STATUS_SUCCESS) {
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

    if (hsa_executable_load_agent_code_object(
            executable, g_gpu_agent, reader, nullptr, nullptr) != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to load code object\n";
        return 1;
    }

    if (hsa_executable_freeze(executable, nullptr) != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to freeze executable\n";
        return 1;
    }

    std::cout << "Executable loaded. Resolving symbol...\n";

    struct SymbolSearch {
        std::string name;
        hsa_executable_symbol_t symbol{};
        bool found = false;
    } search{mangled_name};

    auto callback = [](hsa_executable_t exec,
                       hsa_executable_symbol_t symbol,
                       void* data) -> hsa_status_t {
        SymbolSearch* s = reinterpret_cast<SymbolSearch*>(data);

        char name[256] = {};
        hsa_executable_symbol_get_info(symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_NAME,
            name);

        if (s->name == name) {
            s->symbol = symbol;
            s->found = true;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    };

    hsa_executable_iterate_symbols(executable, callback, &search);

    if (!search.found) {
        std::cerr << "Kernel symbol not found in executable\n";
        return 1;
    }

    uint64_t kernel_object = 0;
    uint32_t kernarg_size = 0;
    uint32_t group_segment = 0;
    uint32_t private_segment = 0;

    hsa_executable_symbol_get_info(search.symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &kernel_object);

    hsa_executable_symbol_get_info(search.symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernarg_size);

    hsa_executable_symbol_get_info(search.symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &group_segment);

    hsa_executable_symbol_get_info(search.symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &private_segment);

    std::cout << "Resolved kernel object: " << kernel_object << "\n";
    std::cout << "Kernarg size: " << kernarg_size << "\n";
    std::cout << "Group segment: " << group_segment << "\n";
    std::cout << "Private segment: " << private_segment << "\n";

    std::cout << "Stage 2 complete.\n";

    hsa_shut_down();
    return 0;
}
