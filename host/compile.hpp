#pragma once

#include "dpu_ctrl.hpp"
#include "macro_common.h"
#include "macro.hpp"
#include "timer.hpp"
#include <mutex>
#include <shared_mutex>

#define NULL_pt(type) ((type)-1)

// Directory holding the DPU binaries; the Makefile sets it to BUILDDIR.
#ifndef DPU_BINARY_DIR
#define DPU_BINARY_DIR "build"
#endif

const string dpu_insert_binary = DPU_BINARY_DIR "/zd_tree_dpu_insert";
const string dpu_box_fetch_binary = DPU_BINARY_DIR "/zd_tree_dpu_box_fetch";
const string dpu_box_count_binary = DPU_BINARY_DIR "/zd_tree_dpu_box_count";
const string dpu_knn_binary = DPU_BINARY_DIR "/zd_tree_dpu_knn";
const string dpu_misc_binary = DPU_BINARY_DIR "/zd_tree_dpu_misc";

int32_t wram_save_pos[NR_DPUS];

inline void init_wram_save_pos() {
    parfor_wrap(0, NR_DPUS, [&](size_t i) {wram_save_pos[i] = NULL_pt(int32_t);});
}

void dpu_heap_load() {
    time_nested("DPU WRAM recovery", [&]() {
        uint64_t consensus_no = (uint64_t)CPU_DPU_CONSENSUS_NO;
        DPU_FOREACH(dpu_set, dpu, each_dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &wram_save_pos[each_dpu]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
            "wram_heap_save_addr", 0, sizeof(int32_t), DPU_XFER_ASYNC));
        DPU_ASSERT(dpu_broadcast_to(dpu_set, "wram_load_consensus_location",
            0, &consensus_no, sizeof(uint64_t), SEND_RECEIVE_ASYNC_STATE));
    });
}

void dpu_heap_save() {
    time_nested("DPU WRAM backup", [&]() {
        DPU_FOREACH(dpu_set, dpu, each_dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &wram_save_pos[each_dpu]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "wram_heap_save_addr", 0, sizeof(int32_t), DPU_XFER_DEFAULT));
    });
}

enum dpu_binary {
    empty,
    insert_binary,
    box_fetch_binary,
    box_count_binary,
    knn_binary,
    misc_binary
};
dpu_binary current_dpu_binary = dpu_binary::empty;

inline void dpu_binary_switch_core(dpu_binary target) {
    switch (target) {
        case dpu_binary::insert_binary: {
            dpu_control::load(dpu_insert_binary);
            break;
        }
        case dpu_binary::box_fetch_binary: {
            dpu_control::load(dpu_box_fetch_binary);
            break;
        }
        case dpu_binary::box_count_binary: {
            dpu_control::load(dpu_box_count_binary);
            break;
        }
        case dpu_binary::knn_binary: {
            dpu_control::load(dpu_knn_binary);
            break;
        }
        case dpu_binary::misc_binary: {
            dpu_control::load(dpu_misc_binary);
            break;
        }
        default: {
            ASSERT(false);
            break;
        }
    }
}

mutex switch_mutex;

inline void dpu_binary_switch_to(dpu_binary target) {
    unique_lock wLock(switch_mutex);
    time_nested("switchto" + std::to_string(target), [&]() {
        if (current_dpu_binary != target) {
            cpu_coverage_timer->end();
            time_nested("lock", [&]() {
                dpu_control::dpu_mutex.lock();
            });
            cpu_coverage_timer->start();
            if (current_dpu_binary != dpu_binary::empty) {
                dpu_heap_save();
                dpu_binary_switch_core(target);
                current_dpu_binary = target;
                dpu_heap_load();
            } else {
                dpu_binary_switch_core(target);
                current_dpu_binary = target;
            }
            time_nested("unlock", [&]() {
                dpu_control::dpu_mutex.unlock();
            });
        }
    });
}