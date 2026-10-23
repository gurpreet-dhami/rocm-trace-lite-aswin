/*
 * rocprofiler_backend.cpp — Hybrid rocprofiler-sdk + HSA queue intercept backend
 *
 * Entry point: rocprofiler_configure() — discovered by rocprofiler-register.
 * Uses rocprofiler_at_intercept_table_registration to obtain the HSA API table,
 * then installs queue intercept + signal injection for per-kernel GPU timing
 * (identical mechanism to hsa_intercept.cpp's OnLoad path).
 *
 * This backend works even when RTLD_GLOBAL pre-loads (vLLM v0.31+) break
 * HSA_TOOLS_LIB/OnLoad, and captures CUDA graph replay kernels.
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 */

#include "trace_db.h"
#include "hip_api_intercept.h"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <hsa/amd_hsa_signal.h>

#include <cxxabi.h>
#include <inttypes.h>
#include <unistd.h>

using namespace trace_db;

namespace rocp_backend {

// ---- Original HSA API tables ----

static CoreApiTable g_orig_core;
static AmdExtTable g_orig_ext;

// ---- Kernel symbol map ----

static std::mutex g_symbol_mutex;
static std::unordered_map<uint64_t, std::string> g_symbols;

// ---- Agent tracking ----

static std::vector<hsa_agent_t> g_gpu_agents;
static std::mutex g_agent_mutex;

// ---- Queue info ----

struct QueueInfo {
    int device_id;
    uint64_t queue_handle;
    uint64_t hw_queue_addr;
};
static std::mutex g_queue_mutex;
static std::unordered_map<uint64_t, QueueInfo> g_queue_map;

static std::atomic<uint64_t> g_dispatch_id{0};
static bool g_intercept_available = false;

// ---- RTL_MODE ----

enum class RtlMode { STANDARD = 0, LITE = 1, FULL = 2 };
static RtlMode g_rtl_mode = RtlMode::LITE;

// ---- Lock-free signal pool (Vyukov MPMC ring) ----

static constexpr size_t CENTRAL_CAPACITY = 4096;
static constexpr size_t CENTRAL_MASK = CENTRAL_CAPACITY - 1;

struct alignas(64) CentralSlot {
    std::atomic<uint64_t> sequence;
    hsa_signal_t signal;
};

static CentralSlot g_central_slots[CENTRAL_CAPACITY];
static std::atomic<uint64_t> g_central_head alignas(64) {0};
static std::atomic<uint64_t> g_central_tail alignas(64) {0};

static bool central_try_pop(hsa_signal_t& out) {
    uint64_t pos = g_central_head.load(std::memory_order_relaxed);
    while (true) {
        CentralSlot& slot = g_central_slots[pos & CENTRAL_MASK];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        int64_t diff = (int64_t)seq - (int64_t)(pos + 1);
        if (diff == 0) {
            if (g_central_head.compare_exchange_weak(pos, pos + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                out = slot.signal;
                slot.sequence.store(pos + CENTRAL_CAPACITY, std::memory_order_release);
                return true;
            }
        } else if (diff < 0) {
            return false;
        } else {
            pos = g_central_head.load(std::memory_order_relaxed);
        }
    }
}

static bool central_try_push(hsa_signal_t sig) {
    uint64_t pos = g_central_tail.load(std::memory_order_relaxed);
    while (true) {
        CentralSlot& slot = g_central_slots[pos & CENTRAL_MASK];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        int64_t diff = (int64_t)seq - (int64_t)pos;
        if (diff == 0) {
            if (g_central_tail.compare_exchange_weak(pos, pos + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                slot.signal = sig;
                slot.sequence.store(pos + 1, std::memory_order_release);
                return true;
            }
        } else if (diff < 0) {
            return false;
        } else {
            pos = g_central_tail.load(std::memory_order_relaxed);
        }
    }
}

static hsa_signal_t acquire_signal() {
    hsa_signal_t sig;
    if (central_try_pop(sig)) {
        g_orig_core.hsa_signal_store_relaxed_fn(sig, 1);
        return sig;
    }
    hsa_status_t st = g_orig_core.hsa_signal_create_fn(1, 0, nullptr, &sig);
    if (st != HSA_STATUS_SUCCESS) sig.handle = 0;
    return sig;
}

static void release_signal(hsa_signal_t sig) {
    if (sig.handle == 0) return;
    if (central_try_push(sig)) return;
    g_orig_core.hsa_signal_destroy_fn(sig);
}

// ---- Diagnostic counters ----

static std::atomic<uint64_t> g_total_intercepts{0};
static std::atomic<uint64_t> g_drop_shutdown{0};
static std::atomic<uint64_t> g_drop_not_kernel{0};
static std::atomic<uint64_t> g_drop_batch_skip{0};
static std::atomic<uint64_t> g_drop_no_qi{0};
static std::atomic<uint64_t> g_drop_sig_fail{0};
static std::atomic<uint64_t> g_drop_ts_fail{0};
static std::atomic<uint64_t> g_drop_ts_invalid{0};
static std::atomic<uint64_t> g_injected{0};
static std::atomic<uint64_t> g_injected_fallback{0};
static std::atomic<uint64_t> g_recorded_ok{0};

// ---- Completion worker ----

static std::string lookup_kernel_name(uint64_t kernel_object);

struct DispatchData {
    uint64_t dispatch_id;
    uint64_t correlation_id;
    uint64_t kernel_object;
    int device_id;
    uint64_t queue_id;
    hsa_signal_t profiling_signal;
    hsa_signal_t original_signal;
    uint64_t hw_queue_addr;
    uint16_t wg_x, wg_y, wg_z;
    uint32_t grid_x, grid_y, grid_z;
    uint64_t roctx_id;
};

static std::mutex g_work_mutex;
static std::condition_variable g_work_cv;
static std::deque<DispatchData*> g_work_queue;
static std::atomic<bool> g_shutdown{false};
static std::thread g_worker;

// Track init PID for fork detection
static pid_t g_init_pid = 0;
static std::string g_db_pattern;
static std::atomic<bool> g_fork_db_initialized{false};

static void ensure_db_for_this_process() {
    if (g_init_pid == getpid()) return;
    bool expected = false;
    if (!g_fork_db_initialized.compare_exchange_strong(expected, true)) return;

    std::string db_path = g_db_pattern;
    auto pos = db_path.find("%p");
    if (pos != std::string::npos)
        db_path.replace(pos, 2, std::to_string(getpid()));
    if (trace_db::get_trace_db().open(db_path))
        fprintf(stderr, "rtl: forked worker DB opened: %s (pid=%d)\n",
                db_path.c_str(), getpid());
}

static void completion_worker() {
    while (true) {
        DispatchData* dd = nullptr;
        {
            std::unique_lock<std::mutex> lock(g_work_mutex);
            g_work_cv.wait(lock, [] {
                return !g_work_queue.empty() || g_shutdown.load(std::memory_order_acquire);
            });
            if (g_work_queue.empty()) {
                if (g_shutdown.load(std::memory_order_acquire)) break;
                continue;
            }
            dd = g_work_queue.front();
            g_work_queue.pop_front();
        }

        static constexpr uint64_t WAIT_TIMEOUT_NS = 100000000ULL;
        bool abandoned = false;
        while (true) {
            hsa_signal_value_t wait_val = g_orig_core.hsa_signal_wait_scacquire_fn(
                dd->profiling_signal, HSA_SIGNAL_CONDITION_LT, 1,
                WAIT_TIMEOUT_NS, HSA_WAIT_STATE_BLOCKED);
            if (wait_val < 1) break;
            if (g_shutdown.load(std::memory_order_acquire)) { abandoned = true; break; }
        }

        if (abandoned) {
            if (dd->original_signal.handle != 0)
                g_orig_core.hsa_signal_subtract_screlease_fn(dd->original_signal, 1);
            release_signal(dd->profiling_signal);
            delete dd;
            continue;
        }

        hsa_amd_profiling_dispatch_time_t time{};
        if (dd->device_id < 0 || (size_t)dd->device_id >= g_gpu_agents.size()) {
            g_drop_ts_fail.fetch_add(1, std::memory_order_relaxed);
        } else {
            hsa_status_t status = g_orig_ext.hsa_amd_profiling_get_dispatch_time_fn(
                g_gpu_agents[dd->device_id], dd->profiling_signal, &time);

            if (status != HSA_STATUS_SUCCESS) {
                g_drop_ts_fail.fetch_add(1, std::memory_order_relaxed);
            } else if (time.end <= time.start) {
                g_drop_ts_invalid.fetch_add(1, std::memory_order_relaxed);
            } else {
                ensure_db_for_this_process();
                std::string name = lookup_kernel_name(dd->kernel_object);
                auto kernel_cb = trace_db::get_kernel_event_callback();
                if (kernel_cb) {
                    trace_db::KernelEventRecord rec;
                    rec.name = name.c_str();
                    rec.device_id = dd->device_id;
                    rec.queue_id = dd->queue_id;
                    rec.start_ns = time.start;
                    rec.end_ns = time.end;
                    rec.correlation_id = dd->correlation_id;
                    rec.wg_x = dd->wg_x; rec.wg_y = dd->wg_y; rec.wg_z = dd->wg_z;
                    rec.grid_x = dd->grid_x; rec.grid_y = dd->grid_y; rec.grid_z = dd->grid_z;
                    try {
                        kernel_cb(rec, trace_db::get_kernel_event_callback_data());
                    } catch (...) {}
                } else {
                    char dispatch_info[128];
                    snprintf(dispatch_info, sizeof(dispatch_info),
                             "hwq=0x%" PRIx64 " wg=%u,%u,%u grid=%u,%u,%u",
                             dd->hw_queue_addr,
                             (unsigned)dd->wg_x, (unsigned)dd->wg_y, (unsigned)dd->wg_z,
                             dd->grid_x, dd->grid_y, dd->grid_z);
                    get_trace_db().record_kernel(name.c_str(), dd->device_id, dd->queue_id,
                                                  time.start, time.end, dd->correlation_id,
                                                  dispatch_info, dd->roctx_id);
                }
                g_recorded_ok.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (dd->original_signal.handle != 0)
            g_orig_core.hsa_signal_subtract_screlease_fn(dd->original_signal, 1);
        release_signal(dd->profiling_signal);
        delete dd;
    }
}

// ---- Kernel name lookup ----

static std::string lookup_kernel_name(uint64_t kernel_object) {
    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    auto it = g_symbols.find(kernel_object);
    if (it != g_symbols.end()) return it->second;
    char buf[32];
    snprintf(buf, sizeof(buf), "kernel_0x%lx", kernel_object);
    return buf;
}

static hsa_status_t symbol_iterate_cb(hsa_executable_t,
                                       hsa_executable_symbol_t symbol, void*) {
    hsa_symbol_kind_t kind;
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (kind != HSA_SYMBOL_KIND_KERNEL) return HSA_STATUS_SUCCESS;

    uint64_t handle;
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &handle) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    uint32_t name_len;
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_len) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (name_len == 0) return HSA_STATUS_SUCCESS;

    std::string name(name_len, '\0');
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &name[0]) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    if (name.size() > 3 && name.compare(name.size() - 3, 3, ".kd") == 0)
        name.resize(name.size() - 3);

    if (name.size() >= 2 && name[0] == '_' && name[1] == 'Z') {
        int demangle_status = 0;
        char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &demangle_status);
        if (demangle_status == 0 && demangled) {
            std::string full(demangled);
            free(demangled);
            size_t start = 0;
            auto first_colon = full.find("::");
            auto first_space = full.find(' ');
            if (first_space != std::string::npos && first_colon != std::string::npos
                && first_space < first_colon)
                start = first_space + 1;
            static constexpr const char* ANON_NS = "(anonymous namespace)";
            static constexpr size_t ANON_NS_LEN = 21;
            size_t end = std::string::npos;
            size_t pos = start;
            while (pos < full.size()) {
                pos = full.find('(', pos);
                if (pos == std::string::npos) break;
                if (full.compare(pos, ANON_NS_LEN, ANON_NS) == 0) { pos += ANON_NS_LEN; continue; }
                end = pos;
                break;
            }
            if (end == std::string::npos) end = full.size();
            name = full.substr(start, end - start);
        }
    }

    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    g_symbols[handle] = name;
    return HSA_STATUS_SUCCESS;
}

// ---- Queue intercept callback ----

static void queue_intercept_cb(const void* in_packets, uint64_t count,
                                uint64_t, void* data,
                                hsa_amd_queue_intercept_packet_writer writer) {
    g_total_intercepts.fetch_add(1, std::memory_order_relaxed);

    if (g_shutdown.load(std::memory_order_acquire)) {
        g_drop_shutdown.fetch_add(count, std::memory_order_relaxed);
        writer(in_packets, count);
        return;
    }

    QueueInfo* qi = static_cast<QueueInfo*>(data);
    if (!qi) {
        g_drop_no_qi.fetch_add(count, std::memory_order_relaxed);
        writer(in_packets, count);
        return;
    }

    const size_t pkt_size = sizeof(hsa_kernel_dispatch_packet_t);
    const size_t buf_bytes = count * pkt_size;
    char stack_buf[64 * 64];
    std::vector<char> heap_buf;
    char* mod_buf;
    if (buf_bytes <= sizeof(stack_buf)) {
        mod_buf = stack_buf;
    } else {
        heap_buf.resize(buf_bytes);
        mod_buf = heap_buf.data();
    }
    memcpy(mod_buf, in_packets, buf_bytes);

    DispatchData* pending_stack[64];
    std::vector<DispatchData*> pending_vec;
    DispatchData** dd_list;
    if (count <= 64) {
        dd_list = pending_stack;
    } else {
        pending_vec.resize(count);
        dd_list = pending_vec.data();
    }
    uint64_t dd_count = 0;

    const bool batch_mode = (count > 1);
    if (batch_mode && g_rtl_mode != RtlMode::FULL) {
        g_drop_batch_skip.fetch_add(count, std::memory_order_relaxed);
        writer(in_packets, count);
        return;
    }

    for (uint64_t i = 0; i < count; i++) {
        hsa_kernel_dispatch_packet_t* pkt =
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(mod_buf + i * pkt_size);

        uint8_t type = ((pkt->header >> HSA_PACKET_HEADER_TYPE) & 0xFF);
        bool is_kernel = (type == HSA_PACKET_TYPE_KERNEL_DISPATCH);

        if (!is_kernel && pkt->kernel_object != 0
            && pkt->workgroup_size_x > 0 && pkt->grid_size_x > 0) {
            is_kernel = true;
            g_injected_fallback.fetch_add(1, std::memory_order_relaxed);
        }

        if (!is_kernel) {
            g_drop_not_kernel.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        if (g_rtl_mode == RtlMode::LITE && pkt->completion_signal.handle != 0) {
            g_drop_not_kernel.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        hsa_signal_t prof_sig = acquire_signal();
        if (prof_sig.handle == 0) {
            g_drop_sig_fail.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        hsa_signal_t orig_sig = pkt->completion_signal;
        pkt->completion_signal = prof_sig;

        auto* dd = new DispatchData;
        dd->dispatch_id = g_dispatch_id.fetch_add(1, std::memory_order_relaxed);
        dd->correlation_id = next_correlation_id();
        dd->roctx_id = trace_db::current_roctx_id();
        dd->kernel_object = pkt->kernel_object;
        dd->device_id = qi->device_id;
        dd->queue_id = qi->queue_handle;
        dd->profiling_signal = prof_sig;
        dd->original_signal = orig_sig;
        dd->hw_queue_addr = qi->hw_queue_addr;
        dd->wg_x = pkt->workgroup_size_x;
        dd->wg_y = pkt->workgroup_size_y;
        dd->wg_z = pkt->workgroup_size_z;
        dd->grid_x = pkt->grid_size_x;
        dd->grid_y = pkt->grid_size_y;
        dd->grid_z = pkt->grid_size_z;

        dd_list[dd_count++] = dd;
        g_injected.fetch_add(1, std::memory_order_relaxed);
    }

    writer(mod_buf, count);

    if (dd_count > 0) {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        for (uint64_t i = 0; i < dd_count; i++)
            g_work_queue.push_back(dd_list[i]);
        g_work_cv.notify_one();
    }
}

// ---- HSA API wrappers (installed via intercept table) ----

static hsa_status_t my_hsa_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    void* data, uint32_t private_segment_size,
    uint32_t group_segment_size, hsa_queue_t** queue) {

    if (!g_intercept_available) {
        hsa_status_t status = g_orig_core.hsa_queue_create_fn(
            agent, size, type, callback, data,
            private_segment_size, group_segment_size, queue);
        if (status == HSA_STATUS_SUCCESS && g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn)
            g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn(*queue, true);
        return status;
    }

    uint32_t safe_priv = (private_segment_size == UINT32_MAX) ? 0 : private_segment_size;
    uint32_t safe_grp = (group_segment_size == UINT32_MAX) ? 0 : group_segment_size;
    hsa_status_t status = g_orig_ext.hsa_amd_queue_intercept_create_fn(
        agent, size, type, callback, data, safe_priv, safe_grp, queue);

    if (status != HSA_STATUS_SUCCESS) {
        return g_orig_core.hsa_queue_create_fn(agent, size, type, callback, data,
                                                private_segment_size, group_segment_size, queue);
    }

    g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn(*queue, true);

    auto* qi = new QueueInfo;
    qi->device_id = 0;
    qi->queue_handle = (uint64_t)(*queue);
    qi->hw_queue_addr = (*queue) ? (uint64_t)(*queue)->base_address : 0;
    {
        std::lock_guard<std::mutex> lock(g_agent_mutex);
        for (size_t i = 0; i < g_gpu_agents.size(); i++) {
            if (g_gpu_agents[i].handle == agent.handle) {
                qi->device_id = (int)i;
                break;
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_queue_map[(uint64_t)(*queue)] = *qi;
    }

    g_orig_ext.hsa_amd_queue_intercept_register_fn(*queue, queue_intercept_cb, qi);
    return HSA_STATUS_SUCCESS;
}

#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
static decltype(AmdExtTable::hsa_amd_queue_create_fn) g_orig_amd_queue_create = nullptr;

static hsa_status_t my_hsa_amd_queue_create(
    hsa_agent_t agent, hsa_amd_queue_create_desc_t* descs, uint32_t num_descs) {

    if (!descs || num_descs == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    hsa_status_t first_error = HSA_STATUS_SUCCESS;

    for (uint32_t i = 0; i < num_descs; i++) {
        auto& d = descs[i];
        d.queue = nullptr;
        if (d.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE || !g_intercept_available) {
            hsa_status_t st = g_orig_amd_queue_create(agent, &d, 1);
            if (st != HSA_STATUS_SUCCESS && first_error == HSA_STATUS_SUCCESS) first_error = st;
            continue;
        }
        uint32_t queue_size = d.queue_size_bytes / sizeof(hsa_kernel_dispatch_packet_t);
        if (queue_size == 0) queue_size = 64;
        uint32_t priv_seg = d.engine.compute.private_segment_size;
        uint32_t safe_priv = (priv_seg == UINT32_MAX) ? 0 : priv_seg;
        hsa_queue_t* queue = nullptr;
        hsa_status_t st = g_orig_ext.hsa_amd_queue_intercept_create_fn(
            agent, queue_size, d.engine.compute.type, d.callback, d.callback_data,
            safe_priv, 0, &queue);
        if (st != HSA_STATUS_SUCCESS) {
            st = g_orig_amd_queue_create(agent, &d, 1);
            if (st != HSA_STATUS_SUCCESS && first_error == HSA_STATUS_SUCCESS) first_error = st;
            continue;
        }
        d.queue = queue;
        g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn(queue, true);
        if (d.priority != HSA_AMD_QUEUE_PRIORITY_NORMAL && g_orig_ext.hsa_amd_queue_set_priority_fn)
            g_orig_ext.hsa_amd_queue_set_priority_fn(queue, d.priority);
        auto* qi = new QueueInfo;
        qi->device_id = 0;
        qi->queue_handle = (uint64_t)queue;
        qi->hw_queue_addr = queue ? (uint64_t)queue->base_address : 0;
        {
            std::lock_guard<std::mutex> lock(g_agent_mutex);
            for (size_t j = 0; j < g_gpu_agents.size(); j++) {
                if (g_gpu_agents[j].handle == agent.handle) { qi->device_id = (int)j; break; }
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            g_queue_map[(uint64_t)queue] = *qi;
        }
        g_orig_ext.hsa_amd_queue_intercept_register_fn(queue, queue_intercept_cb, qi);
    }
    return first_error;
}
#endif

static hsa_status_t my_hsa_executable_freeze(hsa_executable_t executable, const char* options) {
    hsa_status_t status = g_orig_core.hsa_executable_freeze_fn(executable, options);
    if (status == HSA_STATUS_SUCCESS)
        hsa_executable_iterate_symbols(executable, symbol_iterate_cb, nullptr);
    return status;
}

static hsa_status_t agent_iterate_cb(hsa_agent_t agent, void*) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        std::lock_guard<std::mutex> lock(g_agent_mutex);
        g_gpu_agents.push_back(agent);
        g_orig_ext.hsa_amd_profiling_async_copy_enable_fn(true);
    }
    return HSA_STATUS_SUCCESS;
}

// ---- Shutdown ----

static void shutdown() {
    static std::atomic<bool> shutdown_done{false};
    if (shutdown_done.exchange(true)) return;

    fprintf(stderr, "\n=== rtl diagnostic (PID %d, rocprofiler-sdk backend) ===\n", getpid());
    fprintf(stderr, "  intercept calls:     %" PRIu64 "\n", g_total_intercepts.load());
    fprintf(stderr, "  signals injected:    %" PRIu64 "\n", g_injected.load());
    if (g_injected_fallback.load() > 0)
        fprintf(stderr, "  fallback detect:     %" PRIu64 "\n", g_injected_fallback.load());
    fprintf(stderr, "  drop (shutdown):     %" PRIu64 "\n", g_drop_shutdown.load());
    fprintf(stderr, "  drop (not kernel):   %" PRIu64 "\n", g_drop_not_kernel.load());
    fprintf(stderr, "  drop (batch skip):   %" PRIu64 "\n", g_drop_batch_skip.load());
    fprintf(stderr, "  drop (no qi):        %" PRIu64 "\n", g_drop_no_qi.load());
    fprintf(stderr, "  drop (sig pool):     %" PRIu64 "\n", g_drop_sig_fail.load());
    fprintf(stderr, "  drop (ts fail):      %" PRIu64 "\n", g_drop_ts_fail.load());
    fprintf(stderr, "  drop (ts invalid):   %" PRIu64 "\n", g_drop_ts_invalid.load());
    fprintf(stderr, "  recorded OK:         %" PRIu64 "\n", g_recorded_ok.load());
    fprintf(stderr, "====================================\n\n");

    g_shutdown.store(true, std::memory_order_release);
    g_work_cv.notify_all();

    if (g_worker.joinable()) g_worker.join();

    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        while (!g_work_queue.empty()) {
            auto* dd = g_work_queue.front();
            g_work_queue.pop_front();
            if (dd->original_signal.handle != 0)
                g_orig_core.hsa_signal_subtract_screlease_fn(dd->original_signal, 1);
            release_signal(dd->profiling_signal);
            delete dd;
        }
    }

    if (trace_db::is_trace_ready()) {
        trace_db::get_trace_db().flush();
        trace_db::get_trace_db().close();
    }

    {
        uint64_t head = g_central_head.load(std::memory_order_relaxed);
        uint64_t tail = g_central_tail.load(std::memory_order_relaxed);
        for (uint64_t i = head; i < tail; i++) {
            CentralSlot& slot = g_central_slots[i & CENTRAL_MASK];
            if (slot.signal.handle != 0) g_orig_core.hsa_signal_destroy_fn(slot.signal);
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_queue_map.clear();
    }
}

// ---- HSA intercept table callback (replaces OnLoad) ----

static std::atomic<bool> g_table_setup_done{false};

static void hsa_table_callback(rocprofiler_intercept_table_t type,
                                uint64_t, uint64_t lib_instance,
                                void** tables, uint64_t num_tables, void*) {
    if (type != ROCPROFILER_HSA_TABLE || num_tables < 1) return;
    if (lib_instance != 0) return;

    // In forked workers, this callback fires again but the parent already set
    // up the API table wrappers. The child inherits them — re-running setup
    // would overwrite g_orig_core/g_orig_ext with rocprofiler-sdk's wrappers
    // (not the real HSA functions), breaking the call chain.
    // Only re-initialize the worker thread and signal pool for the new process.
    if (g_table_setup_done.load(std::memory_order_acquire)) {
        // Forked child: reinit worker infrastructure only
        g_shutdown.store(false, std::memory_order_release);
        g_central_head.store(0, std::memory_order_relaxed);
        g_central_tail.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < CENTRAL_CAPACITY; i++)
            g_central_slots[i].sequence.store(i, std::memory_order_relaxed);
        size_t filled = 0;
        for (size_t i = 0; i < 64; i++) {
            hsa_signal_t sig;
            if (g_orig_core.hsa_signal_create_fn(1, 0, nullptr, &sig) == HSA_STATUS_SUCCESS) {
                if (central_try_push(sig)) filled++;
                else { g_orig_core.hsa_signal_destroy_fn(sig); break; }
            }
        }
        g_worker = std::thread(completion_worker);
        fprintf(stderr, "rtl: forked worker reinit (pid=%d, %zu signals)\n", getpid(), filled);
        return;
    }
    g_table_setup_done.store(true, std::memory_order_release);

    auto* table = static_cast<HsaApiTable*>(tables[0]);

    // Save original API tables (real HSA functions, before any wrapping)
    g_orig_core = *table->core_;
    g_orig_ext = *table->amd_ext_;

    // Replace queue creation and executable freeze
    table->core_->hsa_queue_create_fn = my_hsa_queue_create;
    table->core_->hsa_executable_freeze_fn = my_hsa_executable_freeze;

#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
    if (table->amd_ext_->hsa_amd_queue_create_fn != nullptr) {
        g_orig_amd_queue_create = g_orig_ext.hsa_amd_queue_create_fn;
        table->amd_ext_->hsa_amd_queue_create_fn = my_hsa_amd_queue_create;
        fprintf(stderr, "rtl: hsa_amd_queue_create intercepted (ROCm 10+ path)\n");
    }
#endif

    // Discover GPU agents
    hsa_iterate_agents(agent_iterate_cb, nullptr);
    fprintf(stderr, "rtl: found %zu GPU agent(s) [rocprofiler-sdk + intercept table]\n",
            g_gpu_agents.size());

    // Parse RTL_MODE
    const char* mode_env = getenv("RTL_MODE");
    if (mode_env) {
        if (strcmp(mode_env, "lite") == 0) g_rtl_mode = RtlMode::LITE;
        else if (strcmp(mode_env, "standard") == 0 || strcmp(mode_env, "default") == 0)
            g_rtl_mode = RtlMode::STANDARD;
        else if (strcmp(mode_env, "full") == 0) g_rtl_mode = RtlMode::FULL;
        else if (strcmp(mode_env, "hip") == 0) {
            g_rtl_mode = RtlMode::STANDARD;
            hip_api::g_enabled.store(true);
        }
    }
    static const char* mode_names[] = {"standard", "lite", "full"};
    fprintf(stderr, "rtl: mode=%s%s\n", mode_names[(int)g_rtl_mode],
            hip_api::g_enabled.load() ? "+hip" : "");

    // Probe queue intercept
    if (!g_gpu_agents.empty()) {
        hsa_queue_t* probe_q = nullptr;
        hsa_status_t probe_st = g_orig_ext.hsa_amd_queue_intercept_create_fn(
            g_gpu_agents[0], 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &probe_q);
        if (probe_st == HSA_STATUS_SUCCESS && probe_q) {
            g_intercept_available = true;
            hsa_queue_destroy(probe_q);
            fprintf(stderr, "rtl: queue intercept available (signal injection enabled)\n");
        } else {
            g_intercept_available = false;
            fprintf(stderr, "rtl: queue intercept NOT available (0x%x)\n", (unsigned)probe_st);
        }
    }

    // Initialize signal pool
    g_central_head.store(0, std::memory_order_relaxed);
    g_central_tail.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < CENTRAL_CAPACITY; i++)
        g_central_slots[i].sequence.store(i, std::memory_order_relaxed);

    size_t filled = 0;
    for (size_t i = 0; i < 64; i++) {
        hsa_signal_t sig;
        if (g_orig_core.hsa_signal_create_fn(1, 0, nullptr, &sig) == HSA_STATUS_SUCCESS) {
            if (central_try_push(sig)) filled++;
            else { g_orig_core.hsa_signal_destroy_fn(sig); break; }
        }
    }
    fprintf(stderr, "rtl: signal pool initialized (%zu signals)\n", filled);

    // Start completion worker
    g_worker = std::thread(completion_worker);
    fprintf(stderr, "rtl: completion worker started\n");

    std::atexit(shutdown);
}

// ---- Tool init / fini ----

static int rtl_tool_init(rocprofiler_client_finalize_t, void*) {
    const char* output_env = std::getenv("RTL_OUTPUT");
    g_db_pattern = output_env ? output_env : "rtl_trace_%p.db";
    g_init_pid = getpid();

    std::string db_path = g_db_pattern;
    auto pos = db_path.find("%p");
    if (pos != std::string::npos)
        db_path.replace(pos, 2, std::to_string(getpid()));

    if (!trace_db::get_kernel_event_callback() && !trace_db::get_api_event_callback()) {
        if (!trace_db::get_trace_db().open(db_path)) {
            fprintf(stderr, "rtl: ERROR: failed to open trace DB: %s\n", db_path.c_str());
            return -1;
        }
        fprintf(stderr, "rtl: trace DB opened: %s\n", db_path.c_str());
    }

    return 0;
}

static void rtl_tool_fini(void*) {
    shutdown();
}

} // namespace rocp_backend

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

    // Register for HSA API table interception
    auto status = rocprofiler_at_intercept_table_registration(
        rocp_backend::hsa_table_callback, ROCPROFILER_HSA_TABLE, nullptr);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: rocprofiler_at_intercept_table_registration failed (%d)\n",
                status);
        return nullptr;
    }
    fprintf(stderr, "rtl: registered for HSA intercept table\n");

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        rocp_backend::rtl_tool_init,
        rocp_backend::rtl_tool_fini,
        nullptr
    };
    return &cfg;
}

} // extern "C"

namespace trace_db {
void rtl_trigger_shutdown() {
    rocp_backend::shutdown();
}
} // namespace trace_db
