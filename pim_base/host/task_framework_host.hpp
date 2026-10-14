#pragma once

#include "timer.hpp"
#include "debug.hpp"
#include "dpu_ctrl.hpp"
#include "macro.hpp"
#include "sort.hpp"
#include "task_framework_common.h"

#include "pim_interface_header.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/internal/integer_sort.h>
#include <parlay/papi/papi_util_impl.h>
#include <mutex>

#define SEND_RECEIVE_ASYNC_STATE (DPU_XFER_DEFAULT)

// IRAM friendly
using namespace std;

atomic<uint64_t> total_communication = 0;
atomic<uint64_t> total_actual_communication = 0;

struct count_size {
    int32_t cnt;
    int32_t size;
};

static inline count_size inc_cs(atomic<count_size>* target, int c, int s,
                                bool atomic) {
    count_size ret, nxt;
    if (atomic) {
        while (true) {
            ret = target->load();
            nxt.cnt = ret.cnt + c;
            nxt.size = ret.size + s;
            if (atomic_compare_exchange_weak(target, &ret, nxt)) {
                break;
            }
        }
    } else {
        ret = *target;
        nxt.cnt = ret.cnt + c;
        nxt.size = ret.size + s;
        target->store(nxt);
    }
    return ret;
}

enum Block_IO_Type { invalid, send, receive };

enum Block_Content_Type { fixed_length, variable_length };

enum State {
    idle,
    pre_init,
    loading_tasks,
    loading_finished,
    waiting_for_sync,
    supplying_responces
};

class IO_Task_Block {
   public:
    int target;
    State state;
    Block_Content_Type content_type;
    uint8_t* base;
    int64_t* base64;
    int64_t* offsets;
    int task_length;
    atomic<count_size> cs;

    void init(Block_Content_Type ct, int task_type, uint8_t* _base,
              int64_t* offset_buf, int length, int target) {
        this->target = target;
        this->content_type = ct;
        this->base = _base;
        this->base64 = (int64_t*)_base;
        if (content_type == fixed_length) {
            task_length = length;
        } else {
            this->offsets = offset_buf;
        }
        base64[0] = task_type;
        cs = (count_size){.cnt = 0, .size = CPU_DPU_BLOCK_HEADER};
        this->state = loading_tasks;
    }

    void* push_task_zero_copy(int length, bool atomic, int* cnt) {
        ASSERT(state == loading_tasks);
        if (content_type == fixed_length) {
            ASSERT(length == -1);
            length = task_length;
        }
        count_size send_cs = inc_cs(&cs, 1, length, atomic);
        CHECK_BOUND(send_cs.cnt < MAX_TASK_COUNT_PER_DPU_PER_BLOCK,
                    "DPU %d: %d tasks in one block (max %d)", target, send_cs.cnt, MAX_TASK_COUNT_PER_DPU_PER_BLOCK);
        CHECK_BOUND(send_cs.size < MAX_TASK_BUFFER_SIZE_PER_DPU,
                    "DPU %d: %d task bytes (max %d)", target, send_cs.size, MAX_TASK_BUFFER_SIZE_PER_DPU);
        if (cnt != NULL) {
            *cnt = send_cs.cnt;
        }
        if (content_type == variable_length) {
            offsets[send_cs.cnt] = send_cs.size;
        }
        return base + send_cs.size;
    }

    void* push_multiple_tasks_zero_copy(int length, int count, bool atomic, int* cnt) {
        ASSERT(state == loading_tasks);
        ASSERT(content_type == fixed_length);
        length = task_length;
        count_size send_cs = inc_cs(&cs, count, count * length, atomic);
        CHECK_BOUND(send_cs.cnt < MAX_TASK_COUNT_PER_DPU_PER_BLOCK,
                    "DPU %d: %d tasks in one block (max %d)", target, send_cs.cnt, MAX_TASK_COUNT_PER_DPU_PER_BLOCK);
        CHECK_BOUND(send_cs.size < MAX_TASK_BUFFER_SIZE_PER_DPU,
                    "DPU %d: %d task bytes (max %d)", target, send_cs.size, MAX_TASK_BUFFER_SIZE_PER_DPU);
        if (cnt != NULL) {
            *cnt = send_cs.cnt;
        }
        return base + send_cs.size;
    }

    int finish() {
        ASSERT(state == loading_tasks);
        count_size finish_cs = cs.load();
        int total_size = finish_cs.size;
        if (content_type == variable_length) {
            total_size += sizeof(int64_t) * finish_cs.cnt;
            memcpy(base + finish_cs.size, offsets, S64(finish_cs.cnt));
        }
        base64[1] = finish_cs.cnt;
        base64[2] = total_size;
        state = loading_finished;
        return total_size;
    }

    int count() { return cs.load().cnt; }

    int size() { return cs.load().size; }

    int expected_reply_length(int reply_length, Block_Content_Type ct) {
        ASSERT(state == loading_finished);
        count_size rep_cs = cs.load();
        if (ct == fixed_length) {
            return reply_length * rep_cs.cnt;
        } else {
            int ret = (reply_length + sizeof(int64_t)) * rep_cs.cnt;
            return ret;
        }
    }

    void switch_to_reply(uint8_t* _base, int length, Block_Content_Type _ct) {
        ASSERT(state == loading_finished);
        this->state = supplying_responces;
        this->base = _base;
        this->content_type = _ct;
        this->base64 = (int64_t*)_base;
        int _cnt = (int)(this->base64[1]);
        int _size = (int)(this->base64[2]);
        count_size send_cs = cs.load();
#ifdef ZHAOYW_CPU_DEBUG
        if (!(_cnt == send_cs.cnt)) {
            printf("wrong %d %d\n", _cnt, send_cs.cnt);
            fflush(stdout);
        }
#endif
        ASSERT(_cnt == send_cs.cnt);
        cs = (count_size){.cnt = _cnt, .size = _size};
        if (_ct == fixed_length) {
            this->task_length = length;
        } else {
            this->offsets = (int64_t*)(base + _size - _cnt * sizeof(int64_t));
        }
    }

    void* ith(int i) {
        ASSERT(i < cs.load().cnt);
        uint8_t* ret;
        if (content_type == fixed_length) {
            i = DPU_CPU_HEADER + i * this->task_length;
            ASSERT_EXEC(i < cs.load().size, {
                printf("ith! i=%d cnt=%d tasklen=%d size=%d\n", i,
                       cs.load().cnt, this->task_length, cs.load().size);
            });
            ret = base + i;
        } else {
            ret = base + offsets[i];
        }
        return ret;
    }
} __attribute__((aligned(64)));

enum Batch_Transmit_Type { broadcast, direct };

struct task_idx {
    uint16_t id;
    int16_t size;
    uint32_t offset;
};

class IO_Task_Batch {
   public:
    Batch_Transmit_Type btt;
    State state;
    Block_Content_Type ct;
    int task_length;
    IO_Task_Block tbs[NR_DPUS];

    void init(Batch_Transmit_Type _btt, Block_Content_Type _ct, int task_type,
              uint8_t* _bases[],
              int64_t offset_bufs[][MAX_TASK_COUNT_PER_DPU_PER_BLOCK],
              int length) {
        state = loading_tasks;
        btt = _btt;
        ct = _ct;
        task_length = length;
#ifdef ZHAOYW_CPU_DEBUG
        if (btt == broadcast) {
            ASSERT(ct == fixed_length);
        }
        if (ct == fixed_length) {
            ASSERT(offset_bufs == NULL);
        } else {
            ASSERT(length == -1);
        }
#endif
        if (btt == broadcast) {
            tbs[0].init(ct, task_type, _bases[0], offset_bufs[0], length, 0);
        } else {
            parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
                tbs[i].init(ct, task_type, _bases[i], offset_bufs[i], length, i);
            });
        }
    }

    template <bool id_from_func, typename TaskFunc, typename Id>  // g(task)
    void push_task_from_array_by_isort(int n, TaskFunc taskf, Id id,
                                       slice<int*, int*> location) {
        using TaskType = decltype(taskf(0));
        auto In = parlay::delayed_seq<TaskType>(n, taskf);

        parlay::sequence<uint32_t> offset;
        int num_buckets = nr_of_dpus;

        auto buffers = parlay::tabulate(nr_of_dpus, [&](size_t i) -> TaskType* {
            return (TaskType*)(tbs[i].base64 + CPU_DPU_BLOCK_HEADER_I64);
        });

        auto counts = parlay::sequence<int>(nr_of_dpus);

        sort_task_direct<id_from_func>(make_slice(In), id, make_slice(buffers),
                                       location, make_slice(counts));

        parlay::parallel_for(0, num_buckets, [&](size_t i) {
            ASSERT(counts[i] < MAX_TASK_COUNT_PER_DPU_PER_BLOCK);
            tbs[i].cs = (count_size){
                .cnt = (int)counts[i],
                .size = (int)(CPU_DPU_BLOCK_HEADER + counts[i] * sizeof(TaskType))};
        });
        return;
    }

    template <typename TaskFunc, typename IdFunc>  // g(index)
    void push_task_sorted(int n, int num_buckets, TaskFunc taskf, IdFunc g,
                          slice<int*, int*> location) {
        using TaskType = decltype(taskf(0));

        auto starts = parlay::sequence<int>(num_buckets, 0);
        auto ends = parlay::sequence<int>(num_buckets, 0);
        ends[g(n - 1)] = n;
        parlay::parallel_for(0, n - 1, [&](size_t i) {
            if (g(i) != g(i + 1)) {
                ends[g(i)] = i + 1;
                starts[g(i + 1)] = i + 1;
            }
        });

        auto buffers = parlay::tabulate(nr_of_dpus, [&](size_t i) -> TaskType* {
            return (TaskType*)(tbs[i].base64 + CPU_DPU_BLOCK_HEADER_I64);
        });

        auto counts = parlay::tabulate(
            nr_of_dpus, [&](size_t i) { return ends[i] - starts[i]; });

        parlay::parallel_for(0, n, [&](size_t i) {
            int id = g(i);
            int offset = i - starts[id];
            buffers[id][offset] = taskf(i);
            location[i] = offset;
        });

        parlay::parallel_for(0, num_buckets, [&](size_t i) {
            ASSERT_EXEC(counts[i] < MAX_TASK_COUNT_PER_DPU_PER_BLOCK, {
                cout << "i=" << i << " count=" << counts[i] << endl;
            });
            tbs[i].cs = (count_size){
                .cnt = (int)counts[i],
                .size = (int)(CPU_DPU_BLOCK_HEADER + counts[i] * sizeof(TaskType))};
        });
        return;
    }

    template <typename F, typename G>
    void push_task_from_array(int length, F idx_generator, G task_generator) {
        if (length > 100000) {
            assert(false);
        } else {
            parfor_wrap(0, length, [&](size_t i) {
                task_idx idx = idx_generator(i);
                int task_id = idx.offset;
                int offset;
                auto ptr = push_task_zero_copy(idx.id, idx.size, true, &offset);
                task_generator(task_id, offset, ptr);
            });
        }
    }

    void* push_task_zero_copy(int send_id, int length, bool atomic,
                              int* cnt = nullptr) {
        ASSERT(state == loading_tasks);
#ifdef ZHAOYW_CPU_DEBUG
        if (btt == broadcast) {
            ASSERT(send_id == -1);
        } else {
            ASSERT_EXEC((send_id >= 0 && send_id < nr_of_dpus),
                        { printf("send id = %d\n", send_id); });
        }
#endif
        if (btt == broadcast) {
            send_id = 0;
        }
        return tbs[send_id].push_task_zero_copy(length, atomic, cnt);
    }

    void* push_multiple_tasks_zero_copy(int send_id, int length, int count,
                                        bool atomic, int* cnt = nullptr) {
        ASSERT(state == loading_tasks);
        // ASSERT(ct == fixed_length);
#ifdef ZHAOYW_CPU_DEBUG
        if (btt == broadcast) {
            ASSERT(send_id == -1);
        } else {
            ASSERT_EXEC((send_id >= 0 && send_id < nr_of_dpus),
                        { printf("send id = %d\n", send_id); });
        }
#endif
        if (btt == broadcast) {
            send_id = 0;
        }
        return tbs[send_id].push_multiple_tasks_zero_copy(length, count, atomic, cnt);
    }

    bool finish(uint8_t** starts) {
        bool empty = true;
        auto tsk = [this, &starts, &empty](int i) {
            int len = tbs[i].finish();
            starts[i] += len;
            if (len > CPU_DPU_BLOCK_HEADER) {
                empty = false;
            }
        };
        if (btt == broadcast) {
            tsk(0);
        } else {
            parfor_wrap(0, nr_of_dpus, [&](size_t i) { tsk(i); });
        }
        return !empty;
    }

    void expected_reply_length(int* add_into, int reply_length,
                               Block_Content_Type _ct) {
        ASSERT(state == loading_finished);
        int r = (btt == broadcast) ? 1 : nr_of_dpus;
        parlay::parallel_for(0, r, [&](size_t i) {
            add_into[i] += tbs[i].expected_reply_length(reply_length, _ct);
        });
    }

    void supply_responce(uint8_t** _bases, int length, Block_Content_Type _ct) {
        ASSERT(state == loading_finished);
        state = supplying_responces;
        int r = (btt == broadcast) ? 1 : nr_of_dpus;
        parlay::parallel_for(0, r, [&](size_t i) {
            tbs[i].switch_to_reply(_bases[i], length, _ct);
        });
    }

    void* ith(int receive_id, int offset) {
#ifdef ZHAOYW_CPU_DEBUG
        if (btt == broadcast) {
            ASSERT(receive_id == -1);
        } else {
            ASSERT(receive_id >= 0 && receive_id < nr_of_dpus);
        }
#endif
        if (btt == broadcast) {
            receive_id = 0;
        }
        return tbs[receive_id].ith(offset);
    }

    void* get_reply(int offset, int receive_id) {  // obsolete api
        return ith(receive_id, offset);
    }
};

class IO_Manager {
   private:
    int64_t offsets[MAX_IO_BLOCKS][NR_DPUS];
    int reply_length[MAX_IO_BLOCKS];
    Block_Content_Type reply_ct[MAX_IO_BLOCKS];
    int cnt, size, broadcast_cnt, direct_cnt;

    // memory buffers
    int64_t direct_offsets[NR_DPUS][MAX_TASK_COUNT_PER_DPU_PER_BLOCK];
    uint8_t (*direct_buffer)[MAX_TASK_BUFFER_SIZE_PER_DPU];
    uint8_t* direct_buffer_addr[NR_DPUS];
    uint8_t* direct_buffer_heads[NR_DPUS];
    // uint8_t* direct_buffer_tails[NR_DPUS];
    int64_t direct_batch_offsets[NR_DPUS][MAX_IO_BLOCKS];
    int direct_receive_length[NR_DPUS];  // expected receive length

    int64_t broadcast_offsets[1][MAX_TASK_COUNT_PER_DPU_PER_BLOCK];
    uint8_t broadcast_buffer[1][MAX_TASK_BUFFER_SIZE_PER_DPU];
    uint8_t* broadcast_buffer_head[1];
    uint8_t* broadcast_buffer_tail[1];  // used to detect overflow. not used yet
                                        // !!! ???
    int64_t broadcast_batch_offsets[1][MAX_IO_BLOCKS];
    int broadcast_receive_length[1];

    static inline void deactivate_scheduling_based_on_interface() {
        parlay::deactivate_scheduling(IO_Manager::using_upmem_interface);
    }

   public:
    size_t tid; // the worker id of the controlling thread
    int id; // the id of this io manager
    inline static mutex alloc_io_manager_mutex;
    inline static atomic<IO_Manager*> working_manager;
    State io_manager_state;
    IO_Task_Batch tbs[MAX_IO_BLOCKS];

    static bool using_upmem_interface;

    IO_Manager() {
        direct_buffer = new uint8_t[NR_DPUS][MAX_TASK_BUFFER_SIZE_PER_DPU];
        uint8_t* buf = (uint8_t*)direct_buffer;
        size_t size = 1ull * NR_DPUS * MAX_TASK_BUFFER_SIZE_PER_DPU;
        parlay::parallel_for(0, size, [&](size_t i) {
            buf[i] = 0;
        });
        // memset(direct_buffer, 0, sizeof(uint8_t) * NR_DPUS * MAX_TASK_BUFFER_SIZE_PER_DPU);
    }

    void reset() {
        unique_lock wLock(alloc_io_manager_mutex);
        ASSERT(tid == worker_id());
        tid = (size_t)-1;
        io_manager_state = idle;
    }

    void init() {
        ASSERT(io_manager_state == pre_init);
        ASSERT(tid == worker_id());
        cnt = 0;
        broadcast_cnt = direct_cnt = 0;
        broadcast_buffer_head[0] = broadcast_buffer[0] + CPU_DPU_HEADER;
        broadcast_receive_length[0] = 0;
        broadcast_batch_offsets[0][0] = CPU_DPU_HEADER;
        parlay::parallel_for(0, nr_of_dpus, [&](int i) {
            direct_receive_length[i] = 0;
            direct_buffer_addr[i] = direct_buffer[i];
            direct_buffer_heads[i] = direct_buffer[i] + CPU_DPU_HEADER;
            direct_batch_offsets[i][0] = CPU_DPU_HEADER;
        });
        io_manager_state = loading_finished;
    }

    IO_Task_Batch* alloc_task_batch(Batch_Transmit_Type btt,
                                    Block_Content_Type send_ct,
                                    Block_Content_Type receive_ct,
                                    int task_type, int len, int reply_len) {
        ASSERT(tid == worker_id());
        ASSERT(io_manager_state == loading_finished);
        ASSERT(!(receive_ct == variable_length && btt == broadcast));

        int i = cnt++;
        IO_Task_Batch& tb = tbs[i];
        reply_length[i] = reply_len;
        reply_ct[i] = receive_ct;
        if (btt == broadcast) {
            ASSERT(send_ct == fixed_length);
            broadcast_cnt++;
            if (send_ct == fixed_length) {
                tb.init(btt, send_ct, task_type, broadcast_buffer_head, NULL, len);
            }
        } else {
            direct_cnt++;
            if (send_ct == fixed_length) {
                tb.init(btt, send_ct, task_type, direct_buffer_heads, NULL, len);
            } else {
                ASSERT(len == -1);
                tb.init(btt, send_ct, task_type, direct_buffer_heads, direct_offsets, -1);
            }
        }
        io_manager_state = loading_tasks;
        tb.state = loading_tasks;
        return &tb;
    }

    template <typename Task, typename Reply>
    IO_Task_Batch* alloc(Batch_Transmit_Type btt) {
        ASSERT(tid == worker_id());
        Block_Content_Type send_ct = Task::fixed ? fixed_length : variable_length;
        Block_Content_Type receive_ct = Reply::fixed ? fixed_length : variable_length;
        int task_type = Task::id;
        int len = Task::task_len;
        int reply_len = Reply::task_len;
        return alloc_task_batch(btt, send_ct, receive_ct, task_type, len, reply_len);
    }

    void finish_task_batch() {
        ASSERT(io_manager_state == loading_tasks);
        int i = cnt - 1;
        ASSERT(i < MAX_IO_BLOCKS);
        ASSERT(tbs[i].state == loading_tasks);
        if (tbs[i].btt == broadcast) {
            tbs[i].finish(broadcast_buffer_head);
            broadcast_batch_offsets[0][broadcast_cnt] = broadcast_buffer_head[0] - broadcast_buffer[0];
            ASSERT((broadcast_buffer_head[0] - broadcast_buffer[0]) < MAX_TASK_BUFFER_SIZE_PER_DPU);
        } else {
            tbs[i].finish(direct_buffer_heads);
            for (int j = 0; j < nr_of_dpus; j++) {
                direct_batch_offsets[j][direct_cnt] = direct_buffer_heads[j] - direct_buffer[j];
                CHECK_BOUND((direct_buffer_heads[j] - direct_buffer[j]) < MAX_TASK_BUFFER_SIZE_PER_DPU,
                            "DPU %d: %ld bytes queued (max %d)", j, (long)(direct_buffer_heads[j] - direct_buffer[j]),
                            MAX_TASK_BUFFER_SIZE_PER_DPU);
            }
        }
        io_manager_state = loading_finished;
        tbs[i].state = loading_finished;
    }

    void print_all_buffer(bool x16 = false) {
        bool sending = (io_manager_state == loading_tasks || io_manager_state == loading_finished);
        ASSERT(tid == worker_id());
        if (sending) {
            if (direct_cnt > 0) {
                printf("\nDirect:\n");
                for (int i = 0; i < nr_of_dpus; i++) {
                    int64_t* buf = (int64_t*)direct_buffer[i];
                    ASSERT((buf[2] % sizeof(int64_t)) == 0);
                    int64_t size = buf[2];
                    if (broadcast_cnt > 0) {
                        int64_t* buff = (int64_t*)broadcast_buffer[0];
                        size -= buff[2] - CPU_DPU_HEADER;
                    }
                    size /= sizeof(int64_t);
                    printf("i=%d\t", i);
                    for (int t = 0; t < size; t++) {
                        if (x16) {
                            printf("%lx\t", buf[t]);
                        } else {
                            printf("%ld\t", buf[t]);
                        }
                    }
                    printf("\n");
                }
            }

            if (broadcast_cnt > 0) {
                int64_t* buf = (int64_t*)broadcast_buffer[0];
                int64_t size = (broadcast_buffer_head[0] - broadcast_buffer[0]);
                ASSERT((size % sizeof(int64_t)) == 0);
                size /= sizeof(int64_t);
                printf("\nBroadcast: %ld\n", size);
                for (int t = 0; t < size; t++) {
                    if (x16) {
                        printf("%lx\t", buf[t]);
                    } else {
                        printf("%ld\t", buf[t]);
                    }
                }
                printf("\n");
            }
        } else {
            int r = (direct_cnt == 0) ? 1 : nr_of_dpus;
            for (int i = 0; i < r; i++) {
                int64_t* buf = (int64_t*)direct_buffer[i];
                ASSERT((buf[2] % sizeof(int64_t)) == 0);
                int64_t size = buf[2];
                size /= sizeof(int64_t);
                printf("i=%d\t", i);
                for (int t = 0; t < size; t++) {
                    if (x16) {
                        printf("%lx\t", buf[t]);
                    } else {
                        printf("%ld\t", buf[t]);
                    }
                }
                printf("\n");
            }
        }
    }

    bool send_task() {
        ASSERT(tid == worker_id());
        ASSERT(cnt > 0 && tbs[cnt - 1].state == loading_finished);
        ASSERT((direct_cnt > 0) || (broadcast_cnt > 0));

        time_start("pre send");

        // calculate reply length
        broadcast_receive_length[0] = 0;
        memset(direct_receive_length, 0, sizeof(direct_receive_length));
        {
            for (int i = 0; i < cnt; i++) {
                if (tbs[i].btt == broadcast) {
                    tbs[i].expected_reply_length(broadcast_receive_length, reply_length[i], fixed_length);
                } else {
                    tbs[i].expected_reply_length(direct_receive_length, reply_length[i], reply_ct[i]);
                }
            }
        }

        int broadcast_length = broadcast_buffer_head[0] - broadcast_buffer[0] - CPU_DPU_HEADER;
        int cnt_length = sizeof(int64_t) * cnt;

        auto get_direct_size = [&]() -> int {
            if (direct_cnt == 0) {
                return 0;
            }
            int ret = 0;
            for (int i = 0; i < nr_of_dpus; i++) {
                int64_t* start = (int64_t*)direct_buffer[i];
                int task_size = direct_buffer_heads[i] - direct_buffer[i] - CPU_DPU_HEADER;
                if(task_size > ret) ret = task_size;
                task_size += CPU_DPU_HEADER + broadcast_length + cnt_length;
                start[0] = epoch_number;
                start[1] = cnt;
                start[2] = task_size;
                CHECK_BOUND(task_size <= MAX_TASK_BUFFER_SIZE_PER_DPU, "DPU %d: %d bytes to send (max %d)", i,
                            task_size, MAX_TASK_BUFFER_SIZE_PER_DPU);
                if (broadcast_cnt > 0) {
                    memcpy(direct_buffer_heads[i], broadcast_batch_offsets[0],
                           sizeof(int64_t) * broadcast_cnt);
                    for (int j = 0; j < direct_cnt; j++) {
                        direct_batch_offsets[i][j] += broadcast_length;
                    }
                }
                memcpy(direct_buffer_heads[i] + sizeof(int64_t) * broadcast_cnt,
                       direct_batch_offsets[i], sizeof(int64_t) * direct_cnt);
            }
            return ret;
        };

        int direct_length = get_direct_size();

        time_end("pre send");

#ifdef INFO_IO_BALANCE
        {
            int size_sum = 0, size_max = 0;
            auto get_size_sum = [&](int& size_sum, int& size_max) -> void {
                if (direct_cnt == 0) {
                    size_sum = (CPU_DPU_HEADER + broadcast_length + cnt_length) * nr_of_dpus;
                    size_max = CPU_DPU_HEADER + broadcast_length + cnt_length;
                } else {
                    for (int i = 0; i < nr_of_dpus; i++) {
                        int64_t* start = (int64_t*)direct_buffer[i];
                        size_sum += start[2];
                        size_max = (size_max > start[2]) ? size_max : start[2];
                    }
                }
            };
            get_size_sum(size_sum, size_max);

            total_communication += size_sum;
            total_actual_communication += (uint64_t)size_max * (uint64_t)nr_of_dpus;

#ifdef PRINT_IO
            printf(
                "send %d : dircnt=%d broadcnt=%d sum=%d max=%d"
                "ratio=%lf\n",
                epoch_number, direct_cnt, broadcast_cnt, size_sum, size_max,
                ((double)size_sum / size_max) / nr_of_dpus);
#endif
        }
#endif

        // parlay::deactivate_scheduling(true);

        time_nested("trigger", [&]() {
            if (direct_cnt == 0) {
                parlay::deactivate_scheduling(true);
                int64_t* start = (int64_t*)broadcast_buffer[0];
                int size = CPU_DPU_HEADER + broadcast_length + cnt_length;
                start[0] = epoch_number;
                start[1] = cnt;
                start[2] = size;
                ASSERT(size <= MAX_TASK_BUFFER_SIZE_PER_DPU);
                memcpy(broadcast_buffer_head[0], broadcast_batch_offsets[0],
                       sizeof(int64_t) * cnt);
#ifdef IRAM_FRIENDLY
                DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME,
                                            DPU_MRAM_HEAP_START_SAFE_BUFFER,
                                            broadcast_buffer[0], size,
                                            SEND_RECEIVE_ASYNC_STATE));
#else
                DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                        broadcast_buffer[0], size,
                                        SEND_RECEIVE_ASYNC_STATE));
#endif
            } else if (broadcast_cnt == 0) {
                ASSERT(broadcast_length == 0);
                int size = CPU_DPU_HEADER + broadcast_length + direct_length + cnt_length;
                // DPU_FOREACH(dpu_set, dpu, each_dpu) {
                //     DPU_ASSERT(dpu_prepare_xfer(dpu, direct_buffer[each_dpu]));
                // }
#ifdef IRAM_FRIENDLY
                // DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                //                          DPU_MRAM_HEAP_POINTER_NAME,
                //                          DPU_MRAM_HEAP_START_SAFE_BUFFER, size,
                //                          SEND_RECEIVE_ASYNC_STATE));
                IO_Manager::deactivate_scheduling_based_on_interface();
                namespace_pim_interface::SendToPIM(
                    (uint8_t**)direct_buffer_addr, 0, DPU_MRAM_HEAP_POINTER_NAME, DPU_MRAM_HEAP_START_SAFE_BUFFER,
                    size, (SEND_RECEIVE_ASYNC_STATE == DPU_XFER_ASYNC)
                );
#else
                // DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                //                      DPU_MRAM_HEAP_POINTER_NAME, 0, size,
                //                      SEND_RECEIVE_ASYNC_STATE));
                namespace_pim_interface::SendToPIM(
                    (uint8_t**)direct_buffer_addr, 0, DPU_MRAM_HEAP_POINTER_NAME, 0,
                    size, (SEND_RECEIVE_ASYNC_STATE == DPU_XFER_ASYNC)
                );
#endif
            } else {  // both
                // header
                // DPU_FOREACH(dpu_set, dpu, each_dpu) {
                //     DPU_ASSERT(dpu_prepare_xfer(dpu, direct_buffer[each_dpu]));
                // }
                // DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                //                          DPU_MRAM_HEAP_POINTER_NAME,
                //                          DPU_MRAM_HEAP_START_SAFE_BUFFER,
                //                          CPU_DPU_HEADER, DPU_XFER_ASYNC));
                IO_Manager::deactivate_scheduling_based_on_interface();
                namespace_pim_interface::SendToPIM(
                    (uint8_t**)direct_buffer_addr, 0, DPU_MRAM_HEAP_POINTER_NAME,
                    DPU_MRAM_HEAP_START_SAFE_BUFFER,
                    CPU_DPU_HEADER, (SEND_RECEIVE_ASYNC_STATE == DPU_XFER_ASYNC)
                );
                // broadcast
                parlay::deactivate_scheduling(true);
                DPU_ASSERT(dpu_broadcast_to(
                    dpu_set, DPU_MRAM_HEAP_POINTER_NAME,
                    CPU_DPU_HEADER + DPU_MRAM_HEAP_START_SAFE_BUFFER,
                    broadcast_buffer[0] + CPU_DPU_HEADER, broadcast_length,
                    SEND_RECEIVE_ASYNC_STATE));

                // direct
                // DPU_FOREACH(dpu_set, dpu, each_dpu) {
                //     DPU_ASSERT(dpu_prepare_xfer(
                //         dpu, direct_buffer[each_dpu] + CPU_DPU_HEADER));
                // }
                // DPU_ASSERT(dpu_push_xfer(
                //     dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                //     CPU_DPU_HEADER + broadcast_length +
                //         DPU_MRAM_HEAP_START_SAFE_BUFFER,
                //     direct_length + cnt_length, SEND_RECEIVE_ASYNC_STATE));
                IO_Manager::deactivate_scheduling_based_on_interface();
                namespace_pim_interface::SendToPIM(
                    (uint8_t**)direct_buffer_addr, CPU_DPU_HEADER, DPU_MRAM_HEAP_POINTER_NAME,
                    CPU_DPU_HEADER + broadcast_length + DPU_MRAM_HEAP_START_SAFE_BUFFER,
                    direct_length + cnt_length, false
                );
            }
        });

        parlay::deactivate_scheduling(false);

        io_manager_state = loading_finished;
        return true;
    }

    auto direct_receive_maxlen() {
        auto lengths = parlay::make_slice(direct_receive_length, direct_receive_length + nr_of_dpus);
        auto maxele = parlay::max_element(lengths);
        return *maxele;
    }

    void receive_from_direct(int offset, int length) {
        // DPU_FOREACH(dpu_set, dpu, each_dpu) {
        //     DPU_ASSERT(dpu_prepare_xfer(dpu, direct_buffer[each_dpu] + offset));
        // }
#ifdef IRAM_FRIENDLY
        // DPU_ASSERT(dpu_push_xfer(
        //     dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
        //     DPU_SEND_BUFFER_OFFSET + offset + DPU_MRAM_HEAP_START_SAFE_BUFFER,
        //     length, SEND_RECEIVE_ASYNC_STATE));
        namespace_pim_interface::ReceiveFromPIM(
            (uint8_t**)direct_buffer_addr, offset, DPU_MRAM_HEAP_POINTER_NAME,
            DPU_SEND_BUFFER_OFFSET + offset + DPU_MRAM_HEAP_START_SAFE_BUFFER,
            length, (SEND_RECEIVE_ASYNC_STATE == DPU_XFER_ASYNC)
        );
#else
        // DPU_ASSERT(dpu_push_xfer(
        //     dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
        //     DPU_SEND_BUFFER_OFFSET + offset, length, SEND_RECEIVE_ASYNC_STATE));
        namespace_pim_interface::ReceiveFromPIM(
            (uint8_t**)direct_buffer_addr, offset, DPU_MRAM_HEAP_POINTER_NAME,
            DPU_SEND_BUFFER_OFFSET + offset,
            length, (SEND_RECEIVE_ASYNC_STATE == DPU_XFER_ASYNC)
        );
#endif
    }

    void receive_from_broadcast(int offset, int length) {
        DPU_FOREACH(dpu_set, dpu, each_dpu) {
            if (each_dpu == 0) {  // !!! ???
#ifdef IRAM_FRIENDLY
                DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                                         DPU_SEND_BUFFER_OFFSET + offset +
                                             DPU_MRAM_HEAP_START_SAFE_BUFFER,
                                         direct_buffer[0] + offset, length));
#else
                DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                                         DPU_SEND_BUFFER_OFFSET + offset,
                                         direct_buffer[0] + offset, length));
#endif
            }
        }
    }

    bool receive_task() {
        time_start("pre_working");
        ASSERT(tid == worker_id());
        ASSERT(io_manager_state == loading_finished);

        int cnt_length = (sizeof(int64_t) + DPU_CPU_BLOCK_HEADER) * cnt;

        int broadcast_length = broadcast_receive_length[0];
        int direct_length = direct_receive_maxlen();
        int receive_length =
            DPU_CPU_HEADER + broadcast_length + direct_length + cnt_length;

        ASSERT(broadcast_cnt != 0 || broadcast_length == 0);
        ASSERT(direct_cnt != 0 || direct_length == 0);
        time_end("pre_working");

#ifndef ZHAOYW_CPU_DEBUG
        if ((broadcast_length + direct_length) == 0) {
            io_manager_state = waiting_for_sync;
            return false;
        }
#endif

        // parlay::deactivate_scheduling(true);
        time_nested("trigger", [&]() {
            if (direct_cnt == 0) {  // only broadcast, one DPU_CPU_HEADER
                parlay::deactivate_scheduling(true);
                receive_from_broadcast(0, receive_length);
            } else if (broadcast_cnt == 0) {
                IO_Manager::deactivate_scheduling_based_on_interface();
                receive_from_direct(0, receive_length);
            } else {  // receive all DPU_CPU_HEADERS independently
                IO_Manager::deactivate_scheduling_based_on_interface();
                receive_from_direct(0, receive_length);
            }
        });
        io_manager_state = waiting_for_sync;
        return true;
    }

    bool sync() {
        ASSERT(io_manager_state == waiting_for_sync);

        auto quit = [&]() {
            return false;
        };

        auto check_error = [&]() {
            int r = nr_of_dpus;
            bool fail = false;
            for (int i = 0; i < r; i++) {
                int64_t* buf = (int64_t*)direct_buffer[i];
                if (buf[0] == DPU_BUFFER_ERROR) {  // error
                    dpu_control::print_log([&](auto x) -> bool { return x == i; });
                    cout << "Quit From DPU!" << endl;
                    fail = true;
                    break;
                }
            }
            if (fail) {
                dpu_control::free_the_dpus();
                exit(-1);
            }
        };

        auto more_fetching = [&](int64_t* lengths, int receive_length) {
            int64_t exact_length = 0;
            for (int j = 0; j < nr_of_dpus; j++) {
                if (direct_cnt == 0 && j > 0) {
                    break;
                }
                int64_t* buf = (int64_t*)direct_buffer[j];
                lengths[j] = buf[2];
                ASSERT(buf[2] > 0 && buf[2] < MAX_TASK_BUFFER_SIZE_PER_DPU);
                exact_length = max(exact_length, lengths[j]);
            }
            ASSERT(exact_length < MAX_TASK_BUFFER_SIZE_PER_DPU);
            if (exact_length > receive_length) {
                receive_from_direct(receive_length, exact_length - receive_length);
                // DPU_ASSERT(dpu_sync(dpu_set));
            } else {
            }
        };

        // time_nested("wait", [&]() { DPU_ASSERT(dpu_sync(dpu_set)); });
        parlay::deactivate_scheduling(false);

        int receive_length = DPU_CPU_HEADER + (sizeof(int64_t) + DPU_CPU_BLOCK_HEADER) * cnt;

        int broadcast_length = broadcast_receive_length[0];
        int direct_length = direct_receive_maxlen();
        receive_length += broadcast_length + direct_length;

#ifdef ZHAOYW_CPU_DEBUG
        check_error();
        if ((broadcast_length + direct_length) == 0) {
            return quit();
        }
#else
        if ((broadcast_length + direct_length) == 0) {
            return quit();
        }
#endif

#ifdef INFO_IO_BALANCE
        time_nested("io count", [&](){
            int size_sum = 0, size_max = 0;
            auto get_size_sum = [&](int& size_sum, int& size_max) -> void {
                if (direct_cnt == 0) {
                    int64_t* start = (int64_t*)direct_buffer[0];
                    size_sum = start[2] * nr_of_dpus;
                    size_max = start[2];
                } else {
                    for (int i = 0; i < nr_of_dpus; i++) {
                        int64_t* start = (int64_t*)direct_buffer[i];
                        size_sum += start[2];
                        size_max = (size_max > start[2]) ? size_max : start[2];
                    }
                }
            };
            get_size_sum(size_sum, size_max);
            total_communication += size_sum;
            total_actual_communication += (uint64_t)size_max * (uint64_t)nr_of_dpus;
#ifdef PRINT_IO
            printf("receive %d : sum=%d max=%d ratio=%lf\n", epoch_number,
                   size_sum, size_max,
                   ((double)size_sum / size_max) / nr_of_dpus);
#endif
        });
#endif

        int64_t lengths[NR_DPUS];
        time_nested("more fetching", [&]() {
            more_fetching(lengths, receive_length);
        });

        time_nested("post receiving", [&]() {
            int64_t receive_batch_offsets[NR_DPUS][MAX_IO_BLOCKS];
            parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
                if (direct_cnt == 0 && i > 0) {
                    return;
                }
                int64_t* buf = (int64_t*)direct_buffer[i];
                int delta = lengths[i] / sizeof(int64_t) - cnt;
                for (int j = 0; j < cnt; j++) {
                    receive_batch_offsets[i][j] = buf[delta + j];
                    ASSERT(buf[delta + j] <= MAX_TASK_BUFFER_SIZE_PER_DPU &&
                           buf[delta + j] > 0);
                }
            });

            for (int i = 0; i < broadcast_cnt; i++) {
                ASSERT(tbs[i].ct == fixed_length);
                uint8_t* bases[1];
                bases[0] = direct_buffer[0] + receive_batch_offsets[0][i];
                tbs[i].supply_responce(bases, reply_length[i], reply_ct[i]);
            }

            for (int i = broadcast_cnt; i < cnt; i++) {
                ASSERT(tbs[i].btt == direct);
                uint8_t* bases[NR_DPUS];
                for (int j = 0; j < nr_of_dpus; j++) {
                    bases[j] = direct_buffer[j] + receive_batch_offsets[j][i];
                }
                tbs[i].supply_responce(bases, reply_length[i], reply_ct[i]);
            }
        });
        io_manager_state = supplying_responces;
        return true;
    }

    bool successful_send;

    bool exec() {
        ASSERT(tid == worker_id());
        cpu_coverage_timer->end();
        time_nested(string("lock"), [&]() {
            dpu_control::dpu_mutex.lock();
        });
        cpu_coverage_timer->start();

        epoch_number++;

        ASSERT(working_manager.load() == nullptr);
        working_manager = this;

        bool successful_send = false;
        time_nested("send", [&]() {
            successful_send = send_task();
        });

        bool ret = false;
        if (successful_send) {
            cpu_coverage_timer->end();
            pim_coverage_timer->start();
            time_nested("dpu", [&]() {
                DPU_ASSERT(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
                while (!dpu_control::ready()) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                time_nested("wait", [&]() {
                    dpu_error_t err = dpu_sync(dpu_set);
                    if (err != DPU_OK) {
                        // Capacity checks on the DPU print their reason to the DPU log before faulting
                        std::fprintf(stderr, "DPU error: %s; DPU logs follow\n", dpu_error_to_string(err));
                        dpu_control::print_all_log();
                        std::exit(1);
                    }
                });
            });
            pim_coverage_timer->end();
            cpu_coverage_timer->start();

            time_nested("receive", [&]() {
                receive_task();
                // always use these two together, sync receive is SYNCHRONOUS
                ret = sync();
            });
            working_manager = nullptr;
        }
        working_manager = nullptr;
        time_nested(string("unlock"), [&]() {
            dpu_control::dpu_mutex.unlock();
        });
        return ret;
    }
};

bool IO_Manager::using_upmem_interface = true;

const int NUM_IO_MANAGERS = 5;
IO_Manager** io_managers;

inline void init_io_managers() {
    io_managers = new IO_Manager*[NUM_IO_MANAGERS];
    for (int i = 0; i < NUM_IO_MANAGERS; i++) {
        io_managers[i] = new IO_Manager();
        io_managers[i]->tid = worker_id();
        io_managers[i]->id = i;
        io_managers[i]->reset();
    }
}

inline auto alloc_io_manager() {
    unique_lock wLock(IO_Manager::alloc_io_manager_mutex);
    for (int i = 0; i < NUM_IO_MANAGERS; i++) {
        if (io_managers[i]->io_manager_state == idle) {
            ASSERT(io_managers[i]->tid == (size_t)-1);
            io_managers[i]->io_manager_state = pre_init;
            io_managers[i]->tid = worker_id();
            return io_managers[i];
        }
    }
    assert(false);
    return io_managers[0];
}