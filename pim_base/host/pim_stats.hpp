#pragma once
// Per-batch measurement for simulator runs.
//
// Simulator wall-clock time says nothing about real PIM hardware, so each batch records what a cost
// model needs instead: DPU cycles of every launch (per DPU), bytes moved between host and DPUs (useful
// and padded, per DPU), and host wall time with the simulator's own time (DPU execution and transfers)
// taken out. Rows are appended to <prefix>_batches.csv and <prefix>_dpus.csv when --stats-csv is set.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

extern "C" {
#include <dpu.h>
}

namespace pim_stats {

using clock_type = std::chrono::steady_clock;

struct Batch {
    std::string op;
    int64_t batch_idx = 0;
    int64_t ops = 0;
    int launches = 0;
    uint64_t pim_cycles_bsp = 0;           // sum over launches of the slowest DPU's cycles
    std::vector<uint64_t> dpu_cycles;      // per DPU, summed over launches
    std::vector<uint64_t> dpu_send_bytes;  // per DPU, useful bytes host -> DPU
    std::vector<uint64_t> dpu_recv_bytes;  // per DPU, useful bytes DPU -> host
    uint64_t send_bytes_padded = 0;        // bytes actually transferred (every DPU gets the largest size)
    uint64_t recv_bytes_padded = 0;
    uint64_t send_bytes_max_dpu = 0;       // sum over launches of the largest per-DPU transfer
    uint64_t recv_bytes_max_dpu = 0;
    double sim_seconds = 0;                // wall time spent in simulated DPU runs and transfers
    clock_type::time_point start;
};

inline std::string csv_prefix;  // empty: recording off
inline std::string run_tag;     // free-form label written to every row (e.g. the data distribution)
inline int nr_dpus_recorded = 0;
inline bool active = false;
inline Batch cur;

inline void configure(const std::string& prefix, const std::string& tag, int nr_dpus) {
    csv_prefix = prefix;
    run_tag = tag;
    nr_dpus_recorded = nr_dpus;
    if (csv_prefix.empty()) return;
    // Write headers once per file
    auto header = [](const std::string& path, const char* line) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (f != nullptr) { std::fclose(f); return; }
        f = std::fopen(path.c_str(), "w");
        if (f == nullptr) { std::perror(path.c_str()); return; }
        std::fputs(line, f);
        std::fclose(f);
    };
    header(csv_prefix + "_batches.csv",
           "tag,nr_dpus,op,batch_idx,ops,launches,pim_cycles_bsp,pim_cycles_mean,dpu_cycles_max_total,"
           "send_bytes,recv_bytes,send_bytes_padded,recv_bytes_padded,send_bytes_max_dpu,recv_bytes_max_dpu,"
           "wall_s,sim_s,cpu_s\n");
    header(csv_prefix + "_dpus.csv", "tag,nr_dpus,op,batch_idx,dpu,cycles,send_bytes,recv_bytes\n");
}

inline void begin(const std::string& op, int64_t batch_idx, int64_t ops) {
    if (csv_prefix.empty()) return;
    cur = Batch();
    cur.op = op;
    cur.batch_idx = batch_idx;
    cur.ops = ops;
    cur.dpu_cycles.assign(nr_dpus_recorded, 0);
    cur.dpu_send_bytes.assign(nr_dpus_recorded, 0);
    cur.dpu_recv_bytes.assign(nr_dpus_recorded, 0);
    cur.start = clock_type::now();
    active = true;
}

// Called after every successful launch: reads launch_cycles from all DPUs
inline void record_launch(dpu_set_t set) {
    if (!active) return;
    std::vector<uint64_t> cycles(nr_dpus_recorded, 0);
    dpu_set_t d;
    uint32_t i;
    DPU_FOREACH(set, d, i) { DPU_ASSERT(dpu_prepare_xfer(d, &cycles[i])); }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "launch_cycles", 0, sizeof(uint64_t), DPU_XFER_DEFAULT));
    uint64_t mx = 0;
    for (int j = 0; j < nr_dpus_recorded; j++) {
        cur.dpu_cycles[j] += cycles[j];
        mx = std::max(mx, cycles[j]);
    }
    cur.pim_cycles_bsp += mx;
    cur.launches++;
}

inline void record_send(int dpu, uint64_t useful_bytes) {
    if (active) cur.dpu_send_bytes[dpu] += useful_bytes;
}
inline void record_recv(int dpu, uint64_t useful_bytes) {
    if (active) cur.dpu_recv_bytes[dpu] += useful_bytes;
}
// One parallel transfer: every DPU receives/sends per_dpu_bytes (n_dpus = 1 for a single-DPU copy)
inline void record_send_transfer(uint64_t per_dpu_bytes, int n_dpus) {
    if (!active) return;
    cur.send_bytes_padded += per_dpu_bytes * n_dpus;
    cur.send_bytes_max_dpu += per_dpu_bytes;
}
inline void record_recv_transfer(uint64_t per_dpu_bytes, int n_dpus) {
    if (!active) return;
    cur.recv_bytes_padded += per_dpu_bytes * n_dpus;
    cur.recv_bytes_max_dpu += per_dpu_bytes;
}
inline void add_sim_time(double seconds) {
    if (active) cur.sim_seconds += seconds;
}

// Times a simulator-side section (DPU run or transfer) so it can be subtracted from host time
struct SimTimer {
    clock_type::time_point t0 = clock_type::now();
    ~SimTimer() { add_sim_time(std::chrono::duration<double>(clock_type::now() - t0).count()); }
};

inline void end() {
    if (!active) return;
    active = false;
    double wall = std::chrono::duration<double>(clock_type::now() - cur.start).count();
    uint64_t sum_cycles = 0, max_total = 0, send = 0, recv = 0;
    for (int j = 0; j < nr_dpus_recorded; j++) {
        sum_cycles += cur.dpu_cycles[j];
        max_total = std::max(max_total, cur.dpu_cycles[j]);
        send += cur.dpu_send_bytes[j];
        recv += cur.dpu_recv_bytes[j];
    }
    FILE* f = std::fopen((csv_prefix + "_batches.csv").c_str(), "a");
    if (f != nullptr) {
        std::fprintf(f, "%s,%d,%s,%lld,%lld,%d,%llu,%.1f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.6f,%.6f,%.6f\n",
                     run_tag.c_str(), nr_dpus_recorded, cur.op.c_str(), (long long)cur.batch_idx,
                     (long long)cur.ops, cur.launches, (unsigned long long)cur.pim_cycles_bsp,
                     (double)sum_cycles / nr_dpus_recorded, (unsigned long long)max_total,
                     (unsigned long long)send, (unsigned long long)recv,
                     (unsigned long long)cur.send_bytes_padded, (unsigned long long)cur.recv_bytes_padded,
                     (unsigned long long)cur.send_bytes_max_dpu, (unsigned long long)cur.recv_bytes_max_dpu,
                     wall, cur.sim_seconds, std::max(0.0, wall - cur.sim_seconds));
        std::fclose(f);
    }
    f = std::fopen((csv_prefix + "_dpus.csv").c_str(), "a");
    if (f != nullptr) {
        for (int j = 0; j < nr_dpus_recorded; j++) {
            std::fprintf(f, "%s,%d,%s,%lld,%d,%llu,%llu,%llu\n", run_tag.c_str(), nr_dpus_recorded,
                         cur.op.c_str(), (long long)cur.batch_idx, j, (unsigned long long)cur.dpu_cycles[j],
                         (unsigned long long)cur.dpu_send_bytes[j], (unsigned long long)cur.dpu_recv_bytes[j]);
        }
        std::fclose(f);
    }
}

}  // namespace pim_stats
