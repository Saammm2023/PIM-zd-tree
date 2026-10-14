#pragma once

#include <mutex.h>
#include <seqread.h>
#include <barrier.h>
#include "macro.h"
#include "debug.h"
#include "task_framework_common.h"

BARRIER_INIT(compress_variable_barrier, NR_TASKLETS);
BARRIER_INIT(task_dpu_barrier, NR_TASKLETS);

/* -------------- Task Framework -------------- */

__host mpuint8_t recv_buffer = (mpuint8_t)DPU_MRAM_HEAP_POINTER + DPU_MRAM_HEAP_START_SAFE_BUFFER;

// recv: EPOCH_NUM(8) + BLOCK_CNT(8) + TOTAL_SIZE(8) + Blocks{TASK_TYPE(8) + TASK_CNT(8) + TOTAL_SIZE(8)} + Offsets
__host volatile int64_t recv_epoch_number;
__host volatile int64_t recv_block_cnt;
__host volatile int64_t recv_total_size;
__host mpint64_t recv_block_offsets;

// recv : EPOCH_NUM(8) + TASK_COUNT(8) + TASK_SIZE(8)

// fixed length : + TASKS
// __host __mram_ptr uint8_t* recv_task_start = DPU_MRAM_HEAP_POINTER + CPU_DPU_PREFIX_LEN;
// variable length : + TASKS + OFFSETS
// __host __mram_ptr int64_t* recv_offsets;
__host mpuint8_t recv_block;
__host volatile int64_t recv_block_task_type;
__host volatile int64_t recv_block_task_cnt;
__host volatile int64_t recv_block_task_size;
__host mpuint8_t recv_block_tasks;
__host mpint64_t recv_block_task_offsets;
#define FIXED_LENGTH 0
#define VARIABLE_LENGTH 1
__host int recv_block_content_type;
__host int recv_block_fixlen;


// recv: BUFFER_STATE(8) + BLOCK_CNT(8) + TOTAL_SIZE(8) + Blocks{TASK_TYPE(8) + TASK_CNT(8) + TOTAL_SIZE(8)} + Offsets
__host __mram_ptr uint8_t* send_buffer = (mpuint8_t)DPU_MRAM_HEAP_POINTER + DPU_SEND_BUFFER_OFFSET + DPU_MRAM_HEAP_START_SAFE_BUFFER;
__host int64_t send_buffer_state;
__host int64_t send_block_cnt;
__host int64_t send_total_size;
__host int64_t send_block_offsets[MAX_IO_BLOCKS];

__host mpuint8_t send_block;
__host int64_t send_block_task_type;
__host int64_t send_block_task_cnt;
__host int64_t send_block_task_size;
__host mpuint8_t send_block_tasks;
__host mpint64_t send_block_task_offsets;
#define FIXED_LENGTH 0
#define VARIABLE_LENGTH 1
__host int send_block_content_type;
__host int send_block_fixlen;

__host int64_t send_varlen_task_cnt[NR_TASKLETS];
__host int64_t send_varlen_task_size[NR_TASKLETS];
mpint64_t send_varlen_offset[NR_TASKLETS];
mpuint8_t send_varlen_buffer[NR_TASKLETS];

static inline void print_io_buffer(mpuint8_t buffer) {
    mpint64_t buf = (mpint64_t)buffer;
    TASK_IN_DPU_ASSERT((buf[2] % sizeof(int64_t)) == 0, "print io buffer: invalid length\n");
    int size = buf[2] / sizeof(int64_t);
    printf("***\n");
    for (int i = 0; i < size; i ++) {
        printf("%d * %llx\n", i, buf[i]);
    }
    printf("\n***\n");
}

/* ---------------------------- IO Manager Init ---------------------------- */
static inline void init_io_manager() {
    mpint64_t buf = (mpint64_t)recv_buffer;
    recv_epoch_number = buf[0];
    recv_block_cnt = buf[1];
    recv_total_size = buf[2];
    TASK_IN_DPU_ASSERT_EXEC(recv_total_size <= MAX_TASK_BUFFER_SIZE_PER_DPU, {
        printf("io manager overflow: %lld\n", recv_total_size);
    });
    recv_block_offsets =
        (mpint64_t)(recv_buffer + recv_total_size - S64(recv_block_cnt));

    send_block = send_buffer + DPU_CPU_HEADER;
    send_buffer_state = DPU_BUFFER_SUCCEED;
    send_block_cnt = 0;
}

static void init_block_header(int i) {
    recv_block = recv_buffer + recv_block_offsets[i];
    recv_block_tasks = recv_block + CPU_DPU_BLOCK_HEADER;
    mpint64_t buf = (mpint64_t)recv_block;
    recv_block_task_type = buf[0];
    recv_block_task_cnt = buf[1];
    recv_block_task_size = buf[2];
}

static void init_block_offset(bool fixed) {
    if (fixed) {
        recv_block_content_type = FIXED_LENGTH;
    } else {
        recv_block_content_type = VARIABLE_LENGTH;
        recv_block_task_offsets =
            (mpint64_t)(recv_block + recv_block_task_size -
                        recv_block_task_cnt * sizeof(int64_t));
    }
}

static void init_block_type(int tasklet_id, int type, int recvlen,
                            int sendlen) {
    send_varlen_task_cnt[tasklet_id] = 0;
    send_varlen_task_size[tasklet_id] = 0;
    if (tasklet_id == 0) {
        send_block_content_type = type;
        send_block_fixlen = sendlen;
        recv_block_fixlen = recvlen;

        send_block_tasks = send_block + DPU_CPU_BLOCK_HEADER;
        // Fixed-length replies fill MRAM up to its end (64 MB above 0x08000000); fail clearly instead of
        // writing past it, which shows up on the host only as "dpu is in fault".
        ALWAYS_IN_DPU_ASSERT(
            type != FIXED_LENGTH ||
                (uint64_t)(uint32_t)send_block_tasks +
                        (uint64_t)recv_block_task_cnt * sendlen <=
                    0x08000000ull + (64ull << 20),
            "reply area overflow: %lld replies of %d B; lower the batch size per DPU\n",
            recv_block_task_cnt, sendlen);
        int i = send_block_cnt++;
        send_block_offsets[i] = send_block - send_buffer;
    }
    IN_DPU_ASSERT(recvlen >= 0 || recv_block_content_type == VARIABLE_LENGTH,
                  "isb! inv\n");
    barrier_wait(&task_dpu_barrier);
}

#define init_block_with_type(tasktype, replytype)                             \
    {                                                                         \
        init_block_offset(task_fixed(tasktype));                              \
        int type = (task_fixed(replytype)) ? FIXED_LENGTH : VARIABLE_LENGTH;  \
        init_block_type(me(), type, task_len(tasktype), task_len(replytype)); \
    }

/* ---------------------------- MRAM - WRAM Boost ----------------------------
 */
seqreader_buffer_t task_reader_local_cache[NR_TASKLETS];
seqreader_t sr[NR_TASKLETS];
int curpos[NR_TASKLETS];
void* curaddr[NR_TASKLETS];

static void* init_task_seqreader(int tasklet_id, int l) {
    mpuint8_t maddr = recv_block_tasks + l * recv_block_fixlen;
    task_reader_local_cache[tasklet_id] = seqread_alloc();
    return seqread_init(task_reader_local_cache[tasklet_id], maddr,
                        &sr[tasklet_id]);
}

static inline void* nxt_task(int tasklet_id, void* ptr) {
    return seqread_get(ptr, recv_block_fixlen, &sr[tasklet_id]);
}

static void init_task_reader(int l) {
    int tasklet_id = me();
    curpos[tasklet_id] = l;
    curaddr[tasklet_id] = init_task_seqreader(tasklet_id, l);
}

// static void move_task_reader(int tasklet_id, int l) {
//     curpos[tasklet_id] = l;
//     curaddr[tasklet_id] =
//         seqread_seek(recv_block_tasks + l * recv_block_fixlen,
//         &sr[tasklet_id]);
// }

static inline void* get_task_cached(int pos) {
    int tasklet_id = me();
    int cp = curpos[tasklet_id];
    TASK_IN_DPU_ASSERT_EXEC((pos == cp) || (pos == cp + 1), {
        printf("get_task error! cp=%d pos=%d\n", cp, pos);
    });
    if (pos == cp + 1) {
        curaddr[tasklet_id] = nxt_task(tasklet_id, curaddr[tasklet_id]);
        curpos[tasklet_id] = pos;
    }
    return curaddr[tasklet_id];
}

static inline __mram_ptr uint8_t* get_task(int i) {
    if (recv_block_content_type == FIXED_LENGTH) {
        TASK_IN_DPU_ASSERT(i >= 0 && recv_block_fixlen > 0,
                           "get task: length error\n");
        return recv_block_tasks + recv_block_fixlen * i;
    } else if (recv_block_content_type == VARIABLE_LENGTH) {
        TASK_IN_DPU_ASSERT_EXEC(i >= 0 && i < recv_block_task_cnt, {
            printf("get task: length error i=%d taskcnt=%lld\n", i,
                   recv_block_task_cnt);
        });
        return recv_block + recv_block_task_offsets[i];
    } else {
        print_io_buffer(recv_buffer);
        TASK_IN_DPU_ASSERT(false, "get task: invalid recv buffer type\n");
        return NULL;
    }
}

/* ---------------------------- Push reply & Finish ----------------------------
 */
static inline mpuint8_t push_fixed_reply_zero_copy(int i) {
    return send_block_tasks + i * send_block_fixlen;
}

static inline void push_fixed_reply(int i, void* buffer) {
    mram_write(buffer, push_fixed_reply_zero_copy(i), send_block_fixlen);
}

static inline mpuint8_t push_variable_reply_zero_copy(int tasklet_id,
                                                      size_t length) {
    TASK_IN_DPU_ASSERT(tasklet_id < NR_TASKLETS,
                       "push variable reply: wrong tasklet id");
    int64_t cnt = send_varlen_task_cnt[tasklet_id]++;
    int64_t size = send_varlen_task_size[tasklet_id];
    send_varlen_offset[tasklet_id][cnt] = size;
    send_varlen_task_size[tasklet_id] += length;
    TASK_IN_DPU_ASSERT(
        send_varlen_task_size[tasklet_id] <= MAX_TASK_BUFFER_SIZE_PER_TASKLET,
        "send task size overflow\n");
    TASK_IN_DPU_ASSERT(send_varlen_task_cnt[tasklet_id] <=
                           MAX_TASK_COUNT_PER_TASKLET_PER_BLOCK,
                       "send task count overflow\n");
    return send_varlen_buffer[tasklet_id] + size;
}

static inline void push_variable_reply(int tasklet_id, void* buffer,
                                       size_t length) {
    mram_write(buffer, push_variable_reply_zero_copy(tasklet_id, length),
               length);
}

static inline mpuint8_t push_variable_reply_head(int tasklet_id) {
    return send_varlen_buffer[tasklet_id] + send_varlen_task_size[tasklet_id];
}

static inline void push_variable_reply_commit(int tasklet_id, int length) {
    push_variable_reply_zero_copy(tasklet_id, length);
}

static inline void finish_fixed_reply(int length, int tasklet_id) {
    TASK_IN_DPU_ASSERT(send_block_content_type == DPU_BLOCK_FIXLEN,
                       "finish fixed reply: wrong type\n");
    barrier_wait(&task_dpu_barrier);
    if (tasklet_id == 0) {
        mpint64_t buf = (mpint64_t)send_block;
        buf[0] = DPU_BLOCK_FIXLEN;
        buf[1] = length;
        buf[2] = DPU_CPU_BLOCK_HEADER + send_block_fixlen * length;
        send_block += buf[2];
    }
}

static inline void finish_variable_reply(int tasklet_id) {
    TASK_IN_DPU_ASSERT(send_block_content_type == DPU_BLOCK_VARLEN,
                       "finish variable reply: wrong type\n");
    barrier_wait(&task_dpu_barrier);
    int64_t prefix_size = 0, total_cnt = 0, prefix_cnt = 0, total_size = 0;
    for (int i = 0; i < NR_TASKLETS; i++) {
        if (i < tasklet_id) {
            prefix_size += send_varlen_task_size[i];
            prefix_cnt += send_varlen_task_cnt[i];
        }
        total_size += send_varlen_task_size[i];
        total_cnt += send_varlen_task_cnt[i];
    }

    int task_start = DPU_CPU_BLOCK_HEADER + prefix_size;
    int offset_start =
        DPU_CPU_BLOCK_HEADER + total_size + sizeof(int64_t) * prefix_cnt;

    int64_t cnt = send_varlen_task_cnt[tasklet_id];
    int64_t size = send_varlen_task_size[tasklet_id];

    TASK_IN_DPU_ASSERT_EXEC(
        cnt == 0 || send_varlen_offset[tasklet_id][0] == 0, {
            printf("%d %lld compress variable reply: incorrect start offset\n",
                   tasklet_id, send_varlen_offset[tasklet_id][0]);
        });

    for (int i = 0; i < cnt; i++) {
        send_varlen_offset[tasklet_id][i] += task_start;
    }

    mram_to_mram(send_block + task_start, send_varlen_buffer[tasklet_id], size);

    mram_to_mram(send_block + offset_start, send_varlen_offset[tasklet_id],
                 cnt * sizeof(int64_t));

    mpint64_t buf = (mpint64_t)send_block;
    buf[0] = DPU_BLOCK_VARLEN;
    buf[1] = total_cnt;
    buf[2] = DPU_CPU_BLOCK_HEADER + total_cnt * sizeof(int64_t) + total_size;
    barrier_wait(&task_dpu_barrier);
    if (tasklet_id == 0) {
        send_block += buf[2];
    }
}

static inline void finish_reply(int length, int tasklet_id) {
    if (send_block_content_type == FIXED_LENGTH) {
        finish_fixed_reply(length, tasklet_id);
    } else {
        finish_variable_reply(tasklet_id);
    }
}

// void print_io_buffer() {
//     mpint64_t buf = (mpint64_t)send_buffer;
//     TASK_IN_DPU_ASSERT((buf[2] % sizeof(int64_t)) == 0,
//                   "finish io manager: error\n");
//     for (int i = 0; i < buf[2] / sizeof(int64_t); i++) {
//         printf("%llx\t", buf[i]);
//     }
//     printf("\n");
// }

static inline void finish_io_manager(int tasklet_id) {
    if (tasklet_id == 0) {
        mpint64_t buf = (mpint64_t)send_buffer;
        buf[0] = DPU_BUFFER_SUCCEED;
        buf[1] = send_block_cnt;
        buf[2] = send_block - send_buffer + sizeof(int64_t) * send_block_cnt;
        TASK_IN_DPU_ASSERT(send_block_cnt <= MAX_IO_BLOCKS,
                           "finish io manager: too much io blocks");
        TASK_IN_DPU_ASSERT(buf[2] <= MAX_TASK_BUFFER_SIZE_PER_DPU,
                           "finish io manager: buffer overflow\n");
        mram_write(send_block_offsets, send_block,
                   sizeof(int64_t) * send_block_cnt);
        printf("finish cnt=%lld size=%lld\n", buf[1], buf[2]);
    }
}
