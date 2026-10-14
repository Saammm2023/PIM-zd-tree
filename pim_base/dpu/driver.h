#pragma once
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <stdint.h>
#include <stdio.h>
#include "macro.h"
#include "task_utils.h"
#include "task_framework_dpu.h"

BARRIER_INIT(main_loop_barrier, NR_TASKLETS);
BARRIER_INIT(init_barrier, NR_TASKLETS);

void execute(int lft, int rt);
void init();

#ifdef DPU_ENERGY
extern uint64_t op_count;
extern uint64_t db_size_count;
extern uint64_t cycle_count;
#endif

void wram_heap_save();
void wram_heap_load();

// Cycles of the last launch on this DPU (whole run(), all tasklets), read by the host after each launch
__host uint64_t launch_cycles;
BARRIER_INIT(cycle_barrier, NR_TASKLETS);

void run() {
#ifdef DPU_ENERGY
    perfcounter_t initial_time = perfcounter_config(COUNT_CYCLES, false);
#endif
    uint32_t tid = me();
    if(tid == 0) perfcounter_config(COUNT_CYCLES, true);
    if(tid == 0) wram_heap_load();

    init_io_manager();
    barrier_wait(&init_barrier);

    for (int T = 0; T < recv_block_cnt; T++) {
        if (tid == 0) {
            mem_reset();
            init_block_header(T);
            init();
        }
        barrier_wait(&main_loop_barrier);
        if (recv_block_task_cnt == 0) {
            init_block_type(tid, FIXED_LENGTH, 0, 0);
            finish_reply(0, tid);
        } else {
            uint32_t lft = recv_block_task_cnt * tid / NR_TASKLETS;
            uint32_t rt = recv_block_task_cnt * (tid + 1) / NR_TASKLETS;
            execute(lft, rt);
        }
        barrier_wait(&main_loop_barrier);
    }

    finish_io_manager(tid);
    barrier_wait(&cycle_barrier);
    if(tid == 0) launch_cycles = perfcounter_get();
#ifdef DPU_ENERGY
    cycle_count += perfcounter_get() - initial_time;
#endif
    if(tid == 0) wram_heap_save();
}
