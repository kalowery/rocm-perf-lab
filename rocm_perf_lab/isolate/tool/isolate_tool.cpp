#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_api_trace.h>

#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <memory>
#include <cxxabi.h>

#define PUBLIC_API __attribute__((visibility("default")))

struct KernelInfo {
    std::string mangled_name;
    std::string demangled_name;
    uint32_t kernarg_size;
};

struct CapturedDispatch {
    uint64_t kernel_object;
    uint32_t grid[3];
    uint16_t block[3];
    uint32_t group_segment_size;
    uint32_t private_segment_size;
    std::vector<uint8_t> kernarg_copy;
};

static HsaApiTable* g_api_table = nullptr;

static decltype(hsa_executable_symbol_get_info)* real_symbol_get_info = nullptr;
static decltype(hsa_queue_create)* real_queue_create = nullptr;

static std::unordered_map<uint64_t, KernelInfo> g_kernel_cache;
static std::unordered_map<uint64_t, uint64_t> g_dispatch_counters;

static std::mutex g_kernel_mutex;
static std::mutex g_capture_mutex;

// Capture configuration
static std::string g_capture_kernel;
static std::unique_ptr<std::regex> g_capture_regex;
static long long g_capture_index = -1; // 0-based
static bool g_capture_enabled = false;
static bool g_capture_done = false;

/* ================================================================
   Code object tracking (HSACO capture groundwork)
   ================================================================ */

static std::unordered_map<uint64_t, std::vector<uint8_t>> g_pending_reader_blobs;
static std::unordered_map<uint64_t, std::vector<uint8_t>> g_executable_blobs;
static std::unordered_map<uint64_t, std::vector<uint8_t>> g_kernel_hsaco;

static uint64_t g_last_loaded_executable = 0;

static std::mutex g_code_object_mutex;

/* Original function pointers (to be wired later) */
static decltype(hsa_code_object_reader_create_from_memory)*
    real_reader_create_from_memory = nullptr;

static decltype(hsa_executable_load_agent_code_object)*
    real_load_agent_code_object = nullptr;

/* ================================================================
   Reader interception (memory-based only)
   ================================================================ */

static hsa_status_t intercepted_reader_create_from_memory(
    const void* code_object,
    size_t size,
    hsa_code_object_reader_t* reader)
{
    hsa_status_t status =
        real_reader_create_from_memory(code_object, size, reader);

    if (status == HSA_STATUS_SUCCESS && reader && size > 0) {
        std::vector<uint8_t> blob(size);
        memcpy(blob.data(), code_object, size);

        std::lock_guard<std::mutex> lock(g_code_object_mutex);
        g_pending_reader_blobs[reader->handle] = std::move(blob);
    }

    return status;
}

/* ================================================================
   Executable load interception
   ================================================================ */

static hsa_status_t intercepted_load_agent_code_object(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_code_object_reader_t reader,
    const char* options,
    hsa_loaded_code_object_t* loaded_code_object)
{
    hsa_status_t status =
        real_load_agent_code_object(
            executable,
            agent,
            reader,
            options,
            loaded_code_object);

    if (status == HSA_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_code_object_mutex);

        auto it = g_pending_reader_blobs.find(reader.handle);
        if (it != g_pending_reader_blobs.end()) {
            g_executable_blobs[executable.handle] = std::move(it->second);
            g_pending_reader_blobs.erase(it);
        }

        // Track most recently loaded executable
        g_last_loaded_executable = executable.handle;
    }

    return status;
}

static hsa_status_t intercepted_symbol_get_info(
    hsa_executable_symbol_t symbol,
    hsa_executable_symbol_info_t attribute,
    void* data)
{
    hsa_status_t status =
        real_symbol_get_info(symbol, attribute, data);

    if (status != HSA_STATUS_SUCCESS)
        return status;

    if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT)
    {
        uint64_t kernel_object = *reinterpret_cast<uint64_t*>(data);

        uint32_t kernarg_size = 0;
        real_symbol_get_info(
            symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            &kernarg_size);

        uint32_t name_len = 0;
        real_symbol_get_info(
            symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
            &name_len);

        std::string mangled;
        if (name_len > 0) {
            mangled.resize(name_len);
            real_symbol_get_info(
                symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_NAME,
                mangled.data());
        }

        // Attempt demangling
        std::string demangled;
        if (!mangled.empty()) {
            int status = 0;
            char* result = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
            if (status == 0 && result) {
                demangled = result;
                free(result);
            } else if (result) {
                free(result);
            }
        }

        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        g_kernel_cache[kernel_object] = { mangled, demangled, kernarg_size };
        
        // Associate kernel_object with last loaded executable HSACO if available
        {
            std::lock_guard<std::mutex> lock2(g_code_object_mutex);
            auto it = g_executable_blobs.find(g_last_loaded_executable);
            if (it != g_executable_blobs.end()) {
                g_kernel_hsaco[kernel_object] = it->second;
            }
        }
    }

    return status;
}

static void OnSubmitPackets(
    const void* in_packets,
    uint64_t count,
    uint64_t user_queue_index,
    void* data,
    hsa_amd_queue_intercept_packet_writer writer)
{
    const hsa_kernel_dispatch_packet_t* packets =
        reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(in_packets);

    for (uint64_t i = 0; i < count; ++i)
    {
        const auto& pkt = packets[i];

        uint16_t type = pkt.header & 0xFF;
        if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
            continue;

        KernelInfo info{};
        {
            std::lock_guard<std::mutex> lock(g_kernel_mutex);
            auto it = g_kernel_cache.find(pkt.kernel_object);
            if (it != g_kernel_cache.end())
                info = it->second;
            else
                continue;
        }

        if (info.kernarg_size == 0)
            continue;

        bool should_capture = false;
        uint64_t dispatch_index = 0;

        {
            std::lock_guard<std::mutex> lock(g_capture_mutex);

            dispatch_index = g_dispatch_counters[pkt.kernel_object]++;

            if (g_capture_enabled && !g_capture_done) {
                const std::string& match_name =
                    !info.demangled_name.empty() ? info.demangled_name : info.mangled_name;

                if (g_capture_regex &&
                    std::regex_search(match_name, *g_capture_regex) &&
                    dispatch_index == (uint64_t)g_capture_index) {
                    should_capture = true;
                    g_capture_done = true;
                }
            }
        }

        if (!should_capture)
            continue;

        // Perform capture outside lock
        std::vector<uint8_t> kernarg_copy(info.kernarg_size);
        memcpy(kernarg_copy.data(),
               reinterpret_cast<const void*>(pkt.kernarg_address),
               info.kernarg_size);

        // Persist to filesystem
        system("mkdir -p isolate_capture");

        std::ofstream meta("isolate_capture/dispatch.json");
        meta << "{\n";
        meta << "  \"mangled_name\": \"" << info.mangled_name << "\",\n";
        meta << "  \"demangled_name\": \"" << info.demangled_name << "\",\n";
        meta << "  \"kernel_object\": " << pkt.kernel_object << ",\n";
        meta << "  \"grid\": [" << pkt.grid_size_x << ", " << pkt.grid_size_y << ", " << pkt.grid_size_z << "],\n";
        meta << "  \"block\": [" << pkt.workgroup_size_x << ", " << pkt.workgroup_size_y << ", " << pkt.workgroup_size_z << "],\n";
        meta << "  \"group_segment_size\": " << pkt.group_segment_size << ",\n";
        meta << "  \"private_segment_size\": " << pkt.private_segment_size << ",\n";
        meta << "  \"kernarg_size\": " << info.kernarg_size << ",\n";
        meta << "  \"dispatch_index\": " << dispatch_index << "\n";
        meta << "}\n";
        meta.close();

        std::ofstream bin("isolate_capture/kernarg.bin", std::ios::binary);
        bin.write(reinterpret_cast<const char*>(kernarg_copy.data()), kernarg_copy.size());
        bin.close();
    }

    writer(in_packets, count);
}

static hsa_status_t intercepted_queue_create(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue)
{
    hsa_status_t status =
        g_api_table->amd_ext_->hsa_amd_queue_intercept_create_fn(
            agent,
            size,
            type,
            callback,
            data,
            private_segment_size,
            group_segment_size,
            queue);

    if (status == HSA_STATUS_SUCCESS)
    {
        g_api_table->amd_ext_->hsa_amd_queue_intercept_register_fn(
            *queue,
            OnSubmitPackets,
            nullptr);
    }

    return status;
}

extern "C"
PUBLIC_API bool OnLoad(
    HsaApiTable* table,
    uint64_t runtime_version,
    uint64_t failed_tool_count,
    const char* const* failed_tool_names)
{
    g_api_table = table;

    // Parse environment variables for selective capture
    const char* kernel_env = getenv("ISOLATE_KERNEL");
    const char* index_env  = getenv("ISOLATE_DISPATCH_INDEX");

    if (kernel_env && index_env) {
        g_capture_kernel = kernel_env;
        g_capture_index = atoll(index_env);
        try {
            g_capture_regex = std::make_unique<std::regex>(g_capture_kernel);
            g_capture_enabled = true;
        } catch (...) {
            g_capture_enabled = false;
        }
    }

    real_symbol_get_info =
        table->core_->hsa_executable_symbol_get_info_fn;

    real_queue_create =
        table->core_->hsa_queue_create_fn;

    real_reader_create_from_memory =
        table->core_->hsa_code_object_reader_create_from_memory_fn;

    real_load_agent_code_object =
        table->core_->hsa_executable_load_agent_code_object_fn;

    table->core_->hsa_executable_symbol_get_info_fn =
        intercepted_symbol_get_info;

    table->core_->hsa_queue_create_fn =
        intercepted_queue_create;

    table->core_->hsa_code_object_reader_create_from_memory_fn =
        intercepted_reader_create_from_memory;

    table->core_->hsa_executable_load_agent_code_object_fn =
        intercepted_load_agent_code_object;

    return true;
}

extern "C"
PUBLIC_API void OnUnload()
{
    // No-op: capture persists immediately when selected
}
