#pragma once
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <mutex.h>
#include <stdint.h>
#include <stdio.h>

#include "common.h"
#include "macro.h"
#include "task_utils.h"
#include "configs_dpu.h"
#include "task_framework_dpu.h"
#include "node_dpu.h"
#include "geometry.h"

// DPU ID
int64_t DPU_ID = -1;

uint64_t range_per_dpu;
uint64_t local_range_start;
uint64_t local_range_end;

static inline void dpu_init_func(int32_t dpu_id, int32_t nr_of_dpus) {
    DPU_ID = dpu_id;
    range_per_dpu = UINT64_MAX / nr_of_dpus + 1;
    local_range_start = local_range_end * dpu_id;
    local_range_end = range_per_dpu + local_range_start - 1;
}


/* Memory Management */

__mram_noinit Bnode b_buffer_tmp[B_BUFFER_SIZE / sizeof(Bnode)];
__mram_noinit Pnode p_buffer_tmp[P_BUFFER_SIZE / sizeof(Pnode)];

MUTEX_INIT(b_lock);
MUTEX_INIT(p_lock);

uint64_t bcnt;
mBptr b_buffer;
// mBptr b_buffer_start, b_buffer_end;
uint64_t pcnt;
mPptr p_buffer;
// mPptr p_buffer_start, p_buffer_end;

__mram_noinit int64_t mrambuffer[MRAM_BUFFER_SIZE / sizeof(int64_t)];

mBptr root;

__mram_noinit int64_t send_varlen_offset_tmp[NR_TASKLETS][MAX_TASK_COUNT_PER_TASKLET_PER_BLOCK];
__mram_noinit uint8_t send_varlen_buffer_tmp[NR_TASKLETS][MAX_TASK_BUFFER_SIZE_PER_TASKLET];

extern mpint64_t send_varlen_offset[];
extern mpuint8_t send_varlen_buffer[];

static inline void storage_init() {
    bcnt = 1; pcnt = 0;
    b_buffer = b_buffer_tmp;
    p_buffer = p_buffer_tmp;
}

/* B Nodes */

static inline void bnode_init() {
    root = b_buffer;
    root->height = 0;
    root->key = 0;
    root->parent = store_node_parent(INVALID_MBPTR);
    root->subtree_size = 0;
    vectorT v;
    vector_ones(&v, INT64_MIN); root->box_min = v;
    vector_ones(&v, INT64_MAX); root->box_max = v;
    for(int8_t i = 0; i < DB_SIZE; i++) root->children[i] = null_pptr;
}

static inline mBptr alloc_new_bnode() {
    mBptr addr;
    mutex_lock(b_lock);
    addr = &(b_buffer[bcnt]);
    bcnt++;
    mutex_unlock(b_lock);
    ALWAYS_IN_DPU_ASSERT(addr < b_buffer_tmp + B_BUFFER_SIZE / sizeof(Bnode),
                         "internal-node storage full (%d nodes); store fewer points per DPU\n",
                         (int)(B_BUFFER_SIZE / sizeof(Bnode)));
    addr->subtree_size = 0;
    return addr;
}

/* P Nodes */

static inline mPptr alloc_new_pnode() {
    mPptr addr;
    mutex_lock(p_lock);
    addr = &(p_buffer[pcnt]);
    pcnt++;
    mutex_unlock(p_lock);
    ALWAYS_IN_DPU_ASSERT(addr < p_buffer_tmp + P_BUFFER_SIZE / sizeof(Pnode),
                         "leaf storage full (%d leaves); store fewer points per DPU\n",
                         (int)(P_BUFFER_SIZE / sizeof(Pnode)));
    addr->num = 0;
    return addr;
}


/* Used for WRAM heap stroage for DPU program reloading */

typedef struct WRAMHeap {

    int64_t DPU_ID;
    uint64_t range_per_dpu;

    uint64_t bcnt;
    mBptr root;
    mBptr bbuffer;

    uint64_t pcnt;
    mPptr pbuffer;

    mpint64_t send_varlen_offset[NR_TASKLETS];
    mpuint8_t send_varlen_buffer[NR_TASKLETS];

#ifdef DPU_ENERGY
    uint64_t op_cnt;
    uint64_t db_size_cnt;
    uint64_t cycle_cnt;
#endif

} WRAMHeap __attribute__((aligned (8)));

__host mpuint8_t wram_heap_save_addr = NULL_pt(mpuint8_t);  // IRAM friendly
__host uint64_t wram_load_consensus_location = (uint64_t)CPU_DPU_CONSENSUS_NO;
__mram_noinit uint8_t wram_heap_save_addr_tmp[sizeof(WRAMHeap) << 1];

void wram_heap_save() {
    mpuint8_t saveAddr = wram_heap_save_addr;
    WRAMHeap heapInfo;
    heapInfo.DPU_ID = DPU_ID;
    heapInfo.range_per_dpu = range_per_dpu;
    heapInfo.root = root;
    heapInfo.bcnt = bcnt;
    heapInfo.bbuffer = b_buffer;
    heapInfo.pcnt = pcnt;
    heapInfo.pbuffer = p_buffer;
#ifdef DPU_ENERGY
    heapInfo.op_cnt = op_count;
    heapInfo.db_size_cnt = db_size_count;
    heapInfo.cycle_cnt = cycle_count;
#endif
    for(int i = 0; i < NR_TASKLETS; i++){
        heapInfo.send_varlen_offset[i] = send_varlen_offset[i];
        heapInfo.send_varlen_buffer[i] = send_varlen_buffer[i];
    }
    if(saveAddr == NULL_pt(mpuint8_t)) saveAddr = wram_heap_save_addr_tmp;
    mram_write(&heapInfo, saveAddr, sizeof(WRAMHeap));
    wram_heap_save_addr = saveAddr;
}

void wram_heap_init() {
    storage_init();
    wram_heap_save_addr = NULL_pt(mpuint8_t);
    for(int i = 0; i < NR_TASKLETS; i++) {
        send_varlen_offset[i] = send_varlen_offset_tmp[i];
        send_varlen_buffer[i] = send_varlen_buffer_tmp[i];
    }
#ifdef DPU_ENERGY
    op_count = 0;
    db_size_count = 0;
    cycle_count = 0;
#endif
}

void wram_heap_load() {
    if(wram_load_consensus_location == CPU_DPU_CONSENSUS_NO) {
        wram_load_consensus_location = 0;
        mpuint8_t saveAddr = wram_heap_save_addr;
        if(saveAddr == NULL_pt(mpuint8_t)) wram_heap_init();
        else {
            WRAMHeap heapInfo;
            mram_read((mpuint8_t)saveAddr, &heapInfo, sizeof(WRAMHeap));

            DPU_ID = heapInfo.DPU_ID;
            range_per_dpu = heapInfo.range_per_dpu;
            local_range_start = local_range_end * DPU_ID;
            local_range_end = range_per_dpu + local_range_start - 1;

            root = heapInfo.root;
            bcnt = heapInfo.bcnt;
            b_buffer = heapInfo.bbuffer;
            pcnt = heapInfo.pcnt;
            p_buffer = heapInfo.pbuffer;

            for(int i = 0; i < NR_TASKLETS; i++) {
                send_varlen_offset[i] = heapInfo.send_varlen_offset[i];
                send_varlen_buffer[i] = heapInfo.send_varlen_buffer[i];
            }
#ifdef DPU_ENERGY
            op_count = heapInfo.op_cnt;
            db_size_count = heapInfo.db_size_cnt;
            cycle_count = heapInfo.cycle_cnt;
#endif
        }
    }
}