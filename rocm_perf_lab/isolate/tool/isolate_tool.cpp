#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_api_trace.h>

#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
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

static std::unordered_map<hsa_queue_t*, hsa_agent_t> g_queue_agents;
static std::mutex g_queue_agent_mutex;

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

static std::mutex g_code_object_mutex;

/* ================================================================
   Device virtual memory tracking (Phase 1 groundwork)
   ================================================================ */

struct DeviceRegion {
    uint64_t base;
    size_t size;

    bool is_pool_alloc;
    hsa_agent_t agent;
    hsa_amd_memory_pool_t pool;

    bool is_vmem;
    uint64_t handle;

    uint32_t access_mask;
};

static std::vector<DeviceRegion> g_device_regions;
static std::mutex g_region_mutex;

/* ================================================================
   Symbol iteration callback for HSACO association
   ================================================================ */

static hsa_status_t hsaco_symbol_callback(
    hsa_executable_t executable,
    hsa_executable_symbol_t symbol,
    void* data)
{
    auto* blob = reinterpret_cast<std::vector<uint8_t>*>(data);

    hsa_symbol_kind_t kind;
    real_symbol_get_info(
        symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
        &kind);

    if (kind != HSA_SYMBOL_KIND_KERNEL)
        return HSA_STATUS_SUCCESS;

    uint64_t kernel_object = 0;
    real_symbol_get_info(
        symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &kernel_object);

    if (kernel_object == 0)
        return HSA_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lock(g_code_object_mutex);
    g_kernel_hsaco[kernel_object] = *blob;

    return HSA_STATUS_SUCCESS;
}

/* Original function pointers (to be wired later) */
static decltype(hsa_code_object_reader_create_from_memory)*
    real_reader_create_from_memory = nullptr;

static decltype(hsa_executable_load_agent_code_object)*
    real_load_agent_code_object = nullptr;

static decltype(hsa_amd_memory_pool_allocate)*
    real_memory_pool_allocate = nullptr;

static decltype(hsa_amd_memory_pool_free)*
    real_memory_pool_free = nullptr;

static decltype(hsa_amd_vmem_address_reserve)*
    real_vmem_address_reserve = nullptr;

static decltype(hsa_amd_vmem_address_free)*
    real_vmem_address_free = nullptr;

static decltype(hsa_amd_vmem_handle_create)*
    real_vmem_handle_create = nullptr;

static decltype(hsa_amd_vmem_handle_release)*
    real_vmem_handle_release = nullptr;

static decltype(hsa_amd_vmem_map)*
    real_vmem_map = nullptr;

static decltype(hsa_amd_vmem_unmap)*
    real_vmem_unmap = nullptr;

static decltype(hsa_amd_vmem_set_access)*
    real_vmem_set_access = nullptr;

/* ================================================================
   Reader interception (memory-based only)
   ================================================================ */

/* ================================================================
   Memory pool allocation interception
   ================================================================ */

static hsa_status_t intercepted_memory_pool_allocate(
    hsa_amd_memory_pool_t pool,
    size_t size,
    uint32_t flags,
    void** ptr)
{
    hsa_status_t status = real_memory_pool_allocate(pool, size, flags, ptr);

    if (status == HSA_STATUS_SUCCESS && ptr && *ptr) {
        DeviceRegion region{};
        region.base = reinterpret_cast<uint64_t>(*ptr);
        region.size = size;
        region.is_pool_alloc = true;
        region.pool = pool;
        region.is_vmem = false;
        region.handle = 0;
        region.access_mask = 0;

        std::lock_guard<std::mutex> lock(g_region_mutex);
        if (g_device_regions.size() < g_device_regions.capacity()) {
            g_device_regions.push_back(region);
        }
    }

    return status;
}

static hsa_status_t intercepted_memory_pool_free(void* ptr)
{
    hsa_status_t status = real_memory_pool_free(ptr);

    if (status == HSA_STATUS_SUCCESS && ptr) {
        uint64_t base = reinterpret_cast<uint64_t>(ptr);
        std::lock_guard<std::mutex> lock(g_region_mutex);
        for (size_t i = 0; i < g_device_regions.size(); ++i) {
            if (g_device_regions[i].base == base) {
                g_device_regions[i] = g_device_regions.back();
                g_device_regions.pop_back();
                break;
            }
        }
    }

    return status;
}

/* ================================================================
   Reader interception (memory-based only)
   ================================================================ */

/* ================================================================
   VMEM interception (forward-only for Phase 1)
   ================================================================ */

static hsa_status_t intercepted_vmem_address_reserve(
    void** va,
    size_t size,
    uint64_t address,
    uint64_t flags)
{
    hsa_status_t status = real_vmem_address_reserve(va, size, address, flags);

    if (status == HSA_STATUS_SUCCESS && va && *va) {
        DeviceRegion region{};
        region.base = reinterpret_cast<uint64_t>(*va);
        region.size = size;
        region.is_pool_alloc = false;
        region.is_vmem = true;
        region.handle = 0;
        region.access_mask = 0;

        std::lock_guard<std::mutex> lock(g_region_mutex);
        if (g_device_regions.size() < g_device_regions.capacity()) {
            g_device_regions.push_back(region);
        }
    }

    return status;
}

static hsa_status_t intercepted_vmem_address_free(
    void* va,
    size_t size)
{
    return real_vmem_address_free(va, size);
}

static hsa_status_t intercepted_vmem_handle_create(
    hsa_amd_memory_pool_t pool,
    size_t size,
    hsa_amd_memory_type_t type,
    uint64_t flags,
    hsa_amd_vmem_alloc_handle_t* memory_handle)
{
    return real_vmem_handle_create(pool, size, type, flags, memory_handle);
}

static hsa_status_t intercepted_vmem_handle_release(
    hsa_amd_vmem_alloc_handle_t memory_handle)
{
    return real_vmem_handle_release(memory_handle);
}

static hsa_status_t intercepted_vmem_map(
    void* va,
    size_t size,
    size_t in_offset,
    hsa_amd_vmem_alloc_handle_t memory_handle,
    uint64_t flags)
{
    hsa_status_t status =
        real_vmem_map(va, size, in_offset, memory_handle, flags);

    if (status == HSA_STATUS_SUCCESS && va) {
        uint64_t base = reinterpret_cast<uint64_t>(va);

        std::lock_guard<std::mutex> lock(g_region_mutex);
        for (size_t i = 0; i < g_device_regions.size(); ++i) {
            if (g_device_regions[i].base == base &&
                g_device_regions[i].is_vmem) {
                g_device_regions[i].handle = memory_handle.handle;
                g_device_regions[i].size = size;
                break;
            }
        }
    }

    return status;
}

static hsa_status_t intercepted_vmem_unmap(
    void* va,
    size_t size)
{
    hsa_status_t status = real_vmem_unmap(va, size);

    if (status == HSA_STATUS_SUCCESS && va) {
        uint64_t base = reinterpret_cast<uint64_t>(va);

        std::lock_guard<std::mutex> lock(g_region_mutex);
        for (size_t i = 0; i < g_device_regions.size(); ++i) {
            if (g_device_regions[i].base == base &&
                g_device_regions[i].is_vmem) {
                g_device_regions[i] = g_device_regions.back();
                g_device_regions.pop_back();
                break;
            }
        }
    }

    return status;
}

static hsa_status_t intercepted_vmem_set_access(
    void* va,
    size_t size,
    const hsa_amd_memory_access_desc_t* desc,
    size_t desc_cnt)
{
    hsa_status_t status =
        real_vmem_set_access(va, size, desc, desc_cnt);

    if (status == HSA_STATUS_SUCCESS && va && desc && desc_cnt > 0) {
        uint64_t base = reinterpret_cast<uint64_t>(va);

        std::lock_guard<std::mutex> lock(g_region_mutex);
        for (size_t i = 0; i < g_device_regions.size(); ++i) {
            if (g_device_regions[i].base == base &&
                g_device_regions[i].is_vmem) {
                for (size_t j = 0; j < desc_cnt; ++j) {
                    g_device_regions[i].access_mask |=
                        static_cast<uint32_t>(desc[j].permissions);
                }
                break;
            }
        }
    }

    return status;
}

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
        std::vector<uint8_t>* blob_ptr = nullptr;

        {
            std::lock_guard<std::mutex> lock(g_code_object_mutex);

            auto it = g_pending_reader_blobs.find(reader.handle);
            if (it != g_pending_reader_blobs.end()) {
                g_executable_blobs[executable.handle] = std::move(it->second);
                g_pending_reader_blobs.erase(it);
            }

            auto exec_it = g_executable_blobs.find(executable.handle);
            if (exec_it != g_executable_blobs.end()) {
                blob_ptr = &exec_it->second;
            }
        }

        if (blob_ptr) {
            g_api_table->core_->hsa_executable_iterate_symbols_fn(
                executable,
                hsaco_symbol_callback,
                blob_ptr);
        }
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

        std::string demangled;
        if (!mangled.empty()) {
            int status_dm = 0;
            char* result = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status_dm);
            if (status_dm == 0 && result) {
                demangled = result;
                free(result);
            } else if (result) {
                free(result);
            }
        }

        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        g_kernel_cache[kernel_object] = { mangled, demangled, kernarg_size };
    }

    return status;
}

static void snapshot_device_memory()
{
    FILE* sentinel = fopen("snapshot_called.txt", "w");
    if (sentinel) {
        fprintf(sentinel, "snapshot invoked\n");
        fclose(sentinel);
    }

    std::vector<DeviceRegion> regions_copy;
    {
        std::lock_guard<std::mutex> lock(g_region_mutex);
        regions_copy = g_device_regions;
    }

    mkdir("isolate_capture/memory", 0755);

    FILE* meta = fopen("isolate_capture/memory_regions.json", "w");
    if (!meta) return;

    fprintf(meta, "{\n  \"regions\": [\n");

    for (size_t i = 0; i < regions_copy.size(); ++i) {
        const DeviceRegion& r = regions_copy[i];

        void* host_buf = malloc(r.size);
        if (!host_buf) continue;

        hsa_status_t status =
            hsa_memory_copy(host_buf,
                            reinterpret_cast<void*>(r.base),
                            r.size);

        if (status == HSA_STATUS_SUCCESS) {
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "isolate_capture/memory/region_%lx.bin",
                     r.base);

            FILE* f = fopen(filename, "wb");
            if (f) {
                fwrite(host_buf, 1, r.size, f);
                fclose(f);
            }

            fprintf(meta,
                "    {\"base\": %lu, \"size\": %zu, "
                "\"is_pool\": %s, \"is_vmem\": %s, "
                "\"handle\": %lu, \"access\": %u}%s\n",
                r.base,
                r.size,
                r.is_pool_alloc ? "true" : "false",
                r.is_vmem ? "true" : "false",
                r.handle,
                r.access_mask,
                (i + 1 < regions_copy.size()) ? "," : "");
        }

        free(host_buf);
    }

    fprintf(meta, "  ]\n}\n");
    fclose(meta);
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

        // Persist HSACO blob if available
        std::vector<uint8_t> hsaco_copy;
        {
            std::lock_guard<std::mutex> lock(g_code_object_mutex);
            auto it = g_kernel_hsaco.find(pkt.kernel_object);
            if (it != g_kernel_hsaco.end()) {
                hsaco_copy = it->second;
            }
        }

        if (!hsaco_copy.empty()) {
            std::ofstream hsaco("isolate_capture/kernel.hsaco", std::ios::binary);
            hsaco.write(reinterpret_cast<const char*>(hsaco_copy.data()), hsaco_copy.size());
            hsaco.close();
        }

        hsa_queue_t* queue = reinterpret_cast<hsa_queue_t*>(data);

        hsa_agent_t agent = {};
        {
            std::lock_guard<std::mutex> lock(g_queue_agent_mutex);
            auto it = g_queue_agents.find(queue);
            if (it != g_queue_agents.end())
                agent = it->second;
        }

        char agent_name[64] = {};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);

        hsa_isa_t isa;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);

        char isa_name[64] = {};
        hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);

        uint32_t wavefront_size = 0;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size);

        std::ofstream meta("isolate_capture/dispatch.json");
        meta << "{\n";
        meta << "  \"mangled_name\": \"" << info.mangled_name << "\",\n";
        meta << "  \"demangled_name\": \"" << info.demangled_name << "\",\n";
        meta << "  \"agent_name\": \"" << agent_name << "\",\n";
        meta << "  \"isa_name\": \"" << isa_name << "\",\n";
        meta << "  \"wavefront_size\": " << wavefront_size << ",\n";
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

        snapshot_device_memory();
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
        {
            std::lock_guard<std::mutex> lock(g_queue_agent_mutex);
            g_queue_agents[*queue] = agent;
        }

        g_api_table->amd_ext_->hsa_amd_queue_intercept_register_fn(
            *queue,
            OnSubmitPackets,
            *queue);
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

    // Pre-reserve region tracking capacity to avoid allocations in hooks
    g_device_regions.reserve(256);

    real_symbol_get_info =
        table->core_->hsa_executable_symbol_get_info_fn;

    real_queue_create =
        table->core_->hsa_queue_create_fn;

    real_reader_create_from_memory =
        table->core_->hsa_code_object_reader_create_from_memory_fn;

    real_load_agent_code_object =
        table->core_->hsa_executable_load_agent_code_object_fn;

    real_memory_pool_allocate =
        table->amd_ext_->hsa_amd_memory_pool_allocate_fn;

    real_memory_pool_free =
        table->amd_ext_->hsa_amd_memory_pool_free_fn;

    real_vmem_address_reserve =
        table->amd_ext_->hsa_amd_vmem_address_reserve_fn;

    real_vmem_address_free =
        table->amd_ext_->hsa_amd_vmem_address_free_fn;

    real_vmem_handle_create =
        table->amd_ext_->hsa_amd_vmem_handle_create_fn;

    real_vmem_handle_release =
        table->amd_ext_->hsa_amd_vmem_handle_release_fn;

    real_vmem_map =
        table->amd_ext_->hsa_amd_vmem_map_fn;

    real_vmem_unmap =
        table->amd_ext_->hsa_amd_vmem_unmap_fn;

    real_vmem_set_access =
        table->amd_ext_->hsa_amd_vmem_set_access_fn;

    if (!real_memory_pool_allocate) {
        fprintf(stderr, "[DEBUG] hsa_amd_memory_pool_allocate_fn is NULL at OnLoad\n");
    }
    if (!real_memory_pool_free) {
        fprintf(stderr, "[DEBUG] hsa_amd_memory_pool_free_fn is NULL at OnLoad\n");
    }

    table->core_->hsa_executable_symbol_get_info_fn =
        intercepted_symbol_get_info;

    table->core_->hsa_queue_create_fn =
        intercepted_queue_create;

    table->core_->hsa_code_object_reader_create_from_memory_fn =
        intercepted_reader_create_from_memory;

    table->core_->hsa_executable_load_agent_code_object_fn =
        intercepted_load_agent_code_object;

    table->amd_ext_->hsa_amd_memory_pool_allocate_fn =
        intercepted_memory_pool_allocate;

    table->amd_ext_->hsa_amd_memory_pool_free_fn =
        intercepted_memory_pool_free;

    table->amd_ext_->hsa_amd_vmem_address_reserve_fn =
        intercepted_vmem_address_reserve;

    table->amd_ext_->hsa_amd_vmem_address_free_fn =
        intercepted_vmem_address_free;

    table->amd_ext_->hsa_amd_vmem_handle_create_fn =
        intercepted_vmem_handle_create;

    table->amd_ext_->hsa_amd_vmem_handle_release_fn =
        intercepted_vmem_handle_release;

    table->amd_ext_->hsa_amd_vmem_map_fn =
        intercepted_vmem_map;

    table->amd_ext_->hsa_amd_vmem_unmap_fn =
        intercepted_vmem_unmap;

    table->amd_ext_->hsa_amd_vmem_set_access_fn =
        intercepted_vmem_set_access;

    return true;
}

extern "C"
PUBLIC_API void OnUnload()
{
    // No-op: capture persists immediately when selected
}
