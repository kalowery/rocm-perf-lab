#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>

struct PoolCtx {
    hsa_amd_memory_pool_t* preferred_pool;
    bool* found_preferred;
    hsa_amd_memory_pool_t* fallback_pool;
    bool* found_fallback;
};

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

    std::cout << "Stage 2 complete. Testing VA reservations...\n";

    // ---- Load memory_regions.json ----
    std::ifstream mem_meta("../../isolate/tool/isolate_capture/memory_regions.json");
    if (!mem_meta) {
        std::cerr << "memory_regions.json not found\n";
        return 1;
    }

    std::string mem_contents(
        (std::istreambuf_iterator<char>(mem_meta)),
        std::istreambuf_iterator<char>());

    size_t mem_pos = 0;

    while ((pos = mem_contents.find("\"base\":", pos)) != std::string::npos) {

        size_t start = mem_contents.find_first_of("0123456789", pos);
        size_t end   = mem_contents.find_first_not_of("0123456789", start);
        uint64_t base = std::stoull(mem_contents.substr(start, end - start));

        size_t size_pos = mem_contents.find("\"size\":", end);
        size_t size_start = mem_contents.find_first_of("0123456789", size_pos);
        size_t size_end   = mem_contents.find_first_not_of("0123456789", size_start);
        size_t size = std::stoull(mem_contents.substr(size_start, size_end - size_start));

        std::cout << "Reserving VA @ 0x"
                  << std::hex << base
                  << " size " << std::dec << size << "... ";

        void* reserved = nullptr;

        hsa_status_t status = hsa_amd_vmem_address_reserve(
            &reserved,
            size,
            base,
            0);

        if (status != HSA_STATUS_SUCCESS || reserved != (void*)base) {
            std::cout << "FAILED\n";
            std::cerr << "VA reservation failed at 0x"
                      << std::hex << base << "\n";
            return 1;
        }

        std::cout << "OK\n";

        pos = size_end;
    }

    std::cout << "All VA reservations succeeded.\n";

    hsa_shut_down();
    return 0;

    // ---- Parse grid/block sizes (X dimension only for now) ----
    auto extract_int = [&](const std::string& key) -> uint32_t {
        auto p = contents.find(key);
        if (p == std::string::npos) return 0;
        auto start = contents.find_first_of("0123456789", p);
        auto end = contents.find_first_not_of("0123456789", start);
        return std::stoi(contents.substr(start, end - start));
    };

    uint32_t grid_x  = extract_int("\"grid\": [");
    uint32_t block_x = extract_int("\"block\": [");

    std::cout << "Grid X: " << grid_x << "  Block X: " << block_x << "\n";

    // ---- Load kernarg blob ----
    std::ifstream kfile("../../isolate/tool/isolate_capture/kernarg.bin", std::ios::binary);
    if (!kfile) {
        std::cerr << "kernarg.bin not found\n";
        return 1;
    }

    std::vector<uint8_t> kernarg_blob((std::istreambuf_iterator<char>(kfile)),
                                       std::istreambuf_iterator<char>());

    // ---- Find suitable kernarg pool (prefer fine-grained global, fallback to any allocatable global) ----
    hsa_amd_memory_pool_t kernarg_pool{};
    hsa_amd_memory_pool_t fallback_pool{};
    bool found_preferred = false;
    bool found_fallback = false;

    auto pool_cb = [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
        auto* ctx = reinterpret_cast<PoolCtx*>(data);

        auto& preferred_pool  = *ctx->preferred_pool;
        auto& found_preferred = *ctx->found_preferred;
        auto& fallback_pool   = *ctx->fallback_pool;
        auto& found_fallback  = *ctx->found_fallback;

        hsa_amd_segment_t segment;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
            &segment);

        uint32_t flags = 0;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
            &flags);

        bool alloc_allowed = false;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
            &alloc_allowed);

        if (segment != HSA_AMD_SEGMENT_GLOBAL || !alloc_allowed)
            return HSA_STATUS_SUCCESS;

        if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
            preferred_pool = pool;
            found_preferred = true;
            return HSA_STATUS_INFO_BREAK;
        }

        if (!found_fallback) {
            fallback_pool = pool;
            found_fallback = true;
        }

        return HSA_STATUS_SUCCESS;
    };

    PoolCtx ctx{
        &kernarg_pool,
        &found_preferred,
        &fallback_pool,
        &found_fallback
    };

    hsa_amd_agent_iterate_memory_pools(g_gpu_agent, pool_cb, &ctx);

    if (!found_preferred) {
        if (found_fallback) {
            kernarg_pool = fallback_pool;
        } else {
            std::cerr << "No suitable global allocatable pool found\n";
            return 1;
        }
    }

    // ---- Allocate kernarg buffer ----
    void* kernarg_device = nullptr;
    if (hsa_amd_memory_pool_allocate(kernarg_pool,
            kernarg_blob.size(),
            0,
            &kernarg_device) != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to allocate kernarg buffer\n";
        return 1;
    }

    memcpy(kernarg_device, kernarg_blob.data(), kernarg_blob.size());

    // ---- Create queue ----
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

    // ---- Create completion signal ----
    hsa_signal_t completion_signal;
    hsa_signal_create(1, 0, nullptr, &completion_signal);

    // ---- Build dispatch packet ----
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
    packet->kernarg_address = kernarg_device;
    packet->private_segment_size = private_segment;
    packet->group_segment_size = group_segment;
    packet->completion_signal = completion_signal;

    packet->header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER);

    hsa_queue_store_write_index_relaxed(queue, index + 1);
    hsa_signal_store_relaxed(queue->doorbell_signal, index);

    std::cout << "Kernel dispatched. Waiting...\n";

    hsa_signal_wait_acquire(
        completion_signal,
        HSA_SIGNAL_CONDITION_LT,
        1,
        UINT64_MAX,
        HSA_WAIT_STATE_ACTIVE);

    std::cout << "Kernel completed. Stage 3 complete.\n";

    hsa_shut_down();
    return 0;
}
