/*
 * rocprofiler_backend.cpp — rocprofiler-sdk based GPU kernel tracing backend
 *
 * Alternative to hsa_intercept.cpp: uses rocprofiler-sdk's callback tracing
 * API instead of HSA_TOOLS_LIB/OnLoad. Works even when RTLD_GLOBAL pre-loads
 * (e.g. vLLM v0.31's env_override.py) break HSA tool interception.
 *
 * Entry point: rocprofiler_configure() — discovered by rocprofiler-register
 * via global symbol scan.
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 */

#include "trace_db.h"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>

namespace {

// ---- Global state ----

rocprofiler_client_finalize_t g_fini_func = nullptr;
rocprofiler_context_id_t     g_ctx;

// agent_id.handle → device index (GPU-only, 0-based)
std::unordered_map<uint64_t, int> g_agent_device_map;
std::mutex                        g_agent_mutex;

// kernel_id → demangled name
std::unordered_map<uint64_t, std::string> g_kernel_names;
std::mutex                                g_kernel_mutex;

std::atomic<bool> g_shutdown_done{false};

// Track the PID that called tool_init — forked children detect PID change
pid_t g_init_pid = 0;
std::string g_db_pattern;  // RTL_OUTPUT pattern with %p
std::atomic<bool> g_fork_db_initialized{false};

void ensure_db_for_this_process() {
    if (g_init_pid == getpid()) return;
    bool expected = false;
    if (!g_fork_db_initialized.compare_exchange_strong(expected, true)) return;

    std::string db_path = g_db_pattern;
    auto pos = db_path.find("%p");
    if (pos != std::string::npos) {
        db_path.replace(pos, 2, std::to_string(getpid()));
    }
    if (trace_db::get_trace_db().open(db_path)) {
        fprintf(stderr, "rtl: forked worker DB opened: %s (pid=%d)\n",
                db_path.c_str(), getpid());
    }
}

// ---- Kernel name demangling (matches hsa_intercept.cpp logic) ----

std::string demangle_kernel_name(const char* raw) {
    if (!raw || !*raw) return "<unknown>";
    std::string name(raw);

    if (name.size() > 3 && name.compare(name.size() - 3, 3, ".kd") == 0) {
        name.resize(name.size() - 3);
    }

    if (name.size() >= 2 && name[0] == '_' && name[1] == 'Z') {
        int status = 0;
        char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
        if (status == 0 && demangled) {
            std::string full(demangled);
            free(demangled);

            size_t start = 0;
            auto first_colon = full.find("::");
            auto first_space = full.find(' ');
            if (first_space != std::string::npos && first_colon != std::string::npos
                && first_space < first_colon) {
                start = first_space + 1;
            }

            static constexpr const char* ANON_NS = "(anonymous namespace)";
            static constexpr size_t ANON_NS_LEN = 21;
            size_t end = std::string::npos;
            size_t pos = start;
            while (pos < full.size()) {
                pos = full.find('(', pos);
                if (pos == std::string::npos) break;
                if (full.compare(pos, ANON_NS_LEN, ANON_NS) == 0) {
                    pos += ANON_NS_LEN;
                    continue;
                }
                end = pos;
                break;
            }
            if (end == std::string::npos) end = full.size();
            name = full.substr(start, end - start);
        }
    }

    return name;
}

const std::string& resolve_kernel_name(uint64_t kernel_id) {
    static const std::string unknown = "<unknown>";
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    auto it = g_kernel_names.find(kernel_id);
    return (it != g_kernel_names.end()) ? it->second : unknown;
}

int resolve_device_id(uint64_t agent_handle) {
    std::lock_guard<std::mutex> lock(g_agent_mutex);
    auto it = g_agent_device_map.find(agent_handle);
    return (it != g_agent_device_map.end()) ? it->second : -1;
}

// ---- Callbacks ----

void kernel_dispatch_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*,
                              void*) {
    if (record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH)
        return;

    if (record.operation == ROCPROFILER_KERNEL_DISPATCH_COMPLETE &&
        record.phase == ROCPROFILER_CALLBACK_PHASE_NONE) {
        ensure_db_for_this_process();
        auto* data = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(
            record.payload);
        if (!data) return;

        const auto& info = data->dispatch_info;
        int device_id = resolve_device_id(info.agent_id.handle);
        const std::string& kname = resolve_kernel_name(info.kernel_id);

        char dispatch_info[128];
        snprintf(dispatch_info, sizeof(dispatch_info),
                 "wg=%u,%u,%u grid=%u,%u,%u",
                 info.workgroup_size.x, info.workgroup_size.y, info.workgroup_size.z,
                 info.grid_size.x, info.grid_size.y, info.grid_size.z);

        trace_db::get_trace_db().record_kernel(
            kname.c_str(),
            device_id,
            info.queue_id.handle,
            data->start_timestamp,
            data->end_timestamp,
            record.correlation_id.internal,
            dispatch_info,
            trace_db::current_roctx_id());
    }
}

void code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*,
                          void*) {
    if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
        return;

    if (record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER &&
        record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
        auto* data = static_cast<
            rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
            record.payload);
        if (!data || !data->kernel_name) return;

        std::string demangled = demangle_kernel_name(data->kernel_name);
        std::lock_guard<std::mutex> lock(g_kernel_mutex);
        g_kernel_names[data->kernel_id] = std::move(demangled);
    }
}

// ---- Agent enumeration ----

rocprofiler_status_t agent_enumerate_cb(rocprofiler_agent_version_t,
                                        const void** agents,
                                        size_t       num_agents,
                                        void*) {
    int gpu_idx = 0;
    std::lock_guard<std::mutex> lock(g_agent_mutex);
    for (size_t i = 0; i < num_agents; ++i) {
        auto* agent = static_cast<const rocprofiler_agent_t*>(agents[i]);
        if (agent->type == ROCPROFILER_AGENT_TYPE_GPU) {
            g_agent_device_map[agent->id.handle] = gpu_idx++;
        }
    }
    fprintf(stderr, "rtl: found %d GPU agent(s) [rocprofiler-sdk]\n", gpu_idx);
    return ROCPROFILER_STATUS_SUCCESS;
}

// ---- Shutdown ----

void shutdown() {
    bool expected = false;
    if (!g_shutdown_done.compare_exchange_strong(expected, true))
        return;

    auto& db = trace_db::get_trace_db();
    db.flush();
    db.close();

    fprintf(stderr, "rtl: rocprofiler-sdk backend shutdown complete\n");
}

// ---- Tool init / fini ----

int rtl_tool_init(rocprofiler_client_finalize_t fini_func, void*) {
    g_fini_func = fini_func;

    const char* output_env = std::getenv("RTL_OUTPUT");
    g_db_pattern = output_env ? output_env : "rtl_trace_%p.db";
    g_init_pid = getpid();

    std::string db_path = g_db_pattern;
    auto pos = db_path.find("%p");
    if (pos != std::string::npos) {
        db_path.replace(pos, 2, std::to_string(getpid()));
    }

    if (!trace_db::get_trace_db().open(db_path)) {
        fprintf(stderr, "rtl: ERROR: failed to open trace DB: %s\n", db_path.c_str());
        return -1;
    }
    fprintf(stderr, "rtl: trace DB opened: %s\n", db_path.c_str());

    rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, agent_enumerate_cb,
        sizeof(rocprofiler_agent_t), nullptr);

    auto status = rocprofiler_create_context(&g_ctx);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: rocprofiler_create_context failed (%d)\n", status);
        return -1;
    }

    rocprofiler_tracing_operation_t kernel_ops[] = {
        ROCPROFILER_KERNEL_DISPATCH_COMPLETE
    };
    status = rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
        kernel_ops, 1, kernel_dispatch_callback, nullptr);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: configure kernel dispatch tracing failed (%d)\n", status);
        return -1;
    }

    rocprofiler_tracing_operation_t code_ops[] = {
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER
    };
    status = rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
        code_ops, 1, code_object_callback, nullptr);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: configure code object tracing failed (%d)\n", status);
        return -1;
    }

    status = rocprofiler_start_context(g_ctx);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: rocprofiler_start_context failed (%d)\n", status);
        return -1;
    }

    std::atexit(shutdown);
    fprintf(stderr, "rtl: rocprofiler-sdk backend initialized (kernel dispatch + code object tracing)\n");
    return 0;
}

void rtl_tool_fini(void*) {
    shutdown();
}

} // anonymous namespace

// ---- rocprofiler entry point ----

extern "C" {

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id) {
    if (std::getenv("RTL_ROCP_DISABLE"))
        return nullptr;

    client_id->name = "rocm-trace-lite";

    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;
    fprintf(stderr, "rtl: rocprofiler-sdk v%u.%u.%u [%s] (priority=%u)\n",
            major, minor, patch, runtime_version ? runtime_version : "", priority);

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        rtl_tool_init,
        rtl_tool_fini,
        nullptr
    };

    return &cfg;
}

} // extern "C"

// ---- Provide rtl_trigger_shutdown for trace_db API ----

namespace trace_db {
void rtl_trigger_shutdown() {
    shutdown();
}
} // namespace trace_db
