#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_api_trace.h>

#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <iostream>

#define PUBLIC_API __attribute__((visibility("default")))

struct KernelInfo {
    std::string name;
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
static std::vector<CapturedDispatch> g_dispatches;

static std::mutex g_kernel_mutex;
static std::mutex g_dispatch_mutex;

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

        std::string name;
        if (name_len > 0) {
            name.resize(name_len);
            real_symbol_get_info(
                symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_NAME,
                name.data());
        }

        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        g_kernel_cache[kernel_object] = { name, kernarg_size };
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

        CapturedDispatch cd{};
        cd.kernel_object = pkt.kernel_object;

        cd.grid[0] = pkt.grid_size_x;
        cd.grid[1] = pkt.grid_size_y;
        cd.grid[2] = pkt.grid_size_z;

        cd.block[0] = pkt.workgroup_size_x;
        cd.block[1] = pkt.workgroup_size_y;
        cd.block[2] = pkt.workgroup_size_z;

        cd.group_segment_size = pkt.group_segment_size;
        cd.private_segment_size = pkt.private_segment_size;

        cd.kernarg_copy.resize(info.kernarg_size);

        memcpy(cd.kernarg_copy.data(),
               reinterpret_cast<const void*>(pkt.kernarg_address),
               info.kernarg_size);

        {
            std::lock_guard<std::mutex> lock(g_dispatch_mutex);
            g_dispatches.push_back(std::move(cd));
        }
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

    real_symbol_get_info =
        table->core_->hsa_executable_symbol_get_info_fn;

    real_queue_create =
        table->core_->hsa_queue_create_fn;

    table->core_->hsa_executable_symbol_get_info_fn =
        intercepted_symbol_get_info;

    table->core_->hsa_queue_create_fn =
        intercepted_queue_create;

    return true;
}

extern "C"
PUBLIC_API void OnUnload()
{
    std::lock_guard<std::mutex> lock(g_dispatch_mutex);

    std::ofstream out("isolate_output.json");
    out << "{ \"dispatch_count\": "
        << g_dispatches.size()
        << " }\n";
}
