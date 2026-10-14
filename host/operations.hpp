#pragma once
#include <stdint.h>
#include <utility>
#include <cstring>

#include <parlay/primitives.h>
#include "debug.hpp"
#include "task_utils.hpp"
#include "task_framework_host.hpp"
#include "dpu_ctrl.hpp"
#include "timer.hpp"
#include "geometry.hpp"
#include "utils.hpp"
#include "heap.hpp"

#if LX_NORM == 2
#include <cmath>
#endif

using namespace std;

class pim_zd_tree {
private:
    /* Auxiliary Structures and Functions */

    struct box_dpu_id {
        int litmin;
        int litmax;
        int bigmin;
        int bigmax;
        box_dpu_id(): litmax(-1), bigmin(INT_MAX) {}
        inline bool same_dpu() {return litmin == bigmax;}
        inline int size() {
            if(litmax < bigmin) return (bigmax + litmax - litmin - bigmin + 2);
            else return (bigmax - litmin + 1);
        }
        inline void set_litmin_bigmax(int litmin_, int bigmax_) {litmin = litmin_; bigmax = bigmax_;}
        inline void set_litmax_bigmin(int litmax_, int bigmin_) {litmax = litmax_; bigmin = bigmin_;}
    };

    uint64_t range_size_each_dpu;
    uint64_t epoch_num;

    pptr *op_addrs;
    int32_t *op_taskpos;
    int *target_dpu;

public:
    /* Main Operation Functions */
    static atomic<int64_t> nr_points;  // Total number of points stored in the tree

    int64_t length;  // Batch size

    vectorT *vector_input;
    vectorT *vector_output;
    int64_t *i64_io;

    int8_t key_to_dpu_id_mode;
    uint64_t *partition_borders;

    pim_zd_tree() {
        this->vector_input = new vectorT[BATCH_SIZE];
        this->vector_output = new vectorT[BATCH_SIZE];
        this->i64_io = new int64_t[BATCH_SIZE];
        this->op_addrs = new pptr[BATCH_SIZE];
        this->op_taskpos = new int32_t[BATCH_SIZE];
        this->target_dpu = new int[BATCH_SIZE];
        this->range_size_each_dpu = UINT64_MAX / nr_of_dpus + 1;
        this->epoch_num = 0;
        this->key_to_dpu_id_mode = 0;
        this->partition_borders = new uint64_t[nr_of_dpus + 1];
    }

    ~pim_zd_tree() {
        delete [] this->vector_input;
        delete [] this->vector_output;
        delete [] this->i64_io;
        delete [] this->op_addrs;
        delete [] this->op_taskpos;
        delete [] this->target_dpu;
        delete [] this->partition_borders;
    }

    uint16_t key_to_dpu_id(uint64_t key) {
        if(key_to_dpu_id_mode == 0) return (uint16_t)(key / this->range_size_each_dpu);
        else if(key_to_dpu_id_mode == 1) {
            // Binary search
            uint16_t ll = 0, rr = nr_of_dpus, mid;
            while(ll + 16 < rr) {
                mid = (ll + rr) / 2;
                if(key < this->partition_borders[mid]) {
                    rr = mid;
                }
                else if(key >= this->partition_borders[mid + 1]) {
                    ll = mid + 1;
                }
                else {
                    return mid;
                }
            }
            for(mid = ll; mid < nr_of_dpus; mid++) {
                if(key >= this->partition_borders[mid] && key < this->partition_borders[mid + 1]) {
                    break;
                }
            }
            return mid;
        }
        else if(key_to_dpu_id_mode == 2) {
            // Interpolation search
            uint16_t ret = (uint16_t)(key / this->range_size_each_dpu);
            while(this->partition_borders[ret + 1] <= key && ret < nr_of_dpus - 1) {
                ret++;
            }
            while(this->partition_borders[ret] > key && ret > 0) {
                ret--;
            }
            return ret;
        }
        else return -1;
    }
    void reset_epoch_num() { this->epoch_num = 0; }
    void print_current_epoch() { printf("Current epoch: %llu\n", this->epoch_num); }


/* Interfaces for database operations */

public:
    void init_range() {
        print_current_epoch();
        cpu_coverage_timer->start();
        printf("\n********** INIT RANGES **********\n");
#ifdef DPU_INIT_ON
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        IO_Task_Batch* batch = io->alloc<dpu_init_range_task, empty_task_reply>(direct);
        parfor_wrap(0, nr_of_dpus, [&](size_t i) {
            auto it = (dpu_init_range_task*)batch->push_task_zero_copy(i, -1, false);
            it->range_start = this->partition_borders[i];
            it->range_end = this->partition_borders[i + 1];
        });
        io->finish_task_batch();
        ASSERT(io->exec());
        io->reset();
#endif
        printf("\n****** INIT RANGES Finished ******\n");
        cpu_coverage_timer->end();
    }

    void insert(vectorT *vec_input = nullptr, bool debug_print = false) {
#ifdef INSERT_NODE_ON
        print_current_epoch();
        cpu_coverage_timer->start();
        time_start("insert");

        if(vec_input == nullptr) vec_input = this->vector_input;
        // One SINGLE_INSERT task carries every point bound for one leaf, and the DPU builds the new
        // subtree in a fixed per-tasklet MRAM scratch area (MAX_INSERT_TASK_LEN points). A leaf that
        // receives more points than that keeps a strided sample in this round; the other points are
        // re-searched and inserted in a further round. The same happens to whole leaf groups beyond
        // MAX_INSERT_TASKS_PER_DPU on one DPU (the IO framework holds at most
        // MAX_TASK_COUNT_PER_DPU_PER_BLOCK tasks per DPU per block). Rounds are only added in those cases
        // (e.g. the first batch into an empty tree, where all points of a DPU hit one or two root
        // children, or a later batch with more than about 12.8k points per DPU).
        const int64_t batch_length = this->length;
        parlay::sequence<vectorT> round_input, deferred;
        while(true) {
            time_start("init");
            IO_Manager *io;
            IO_Task_Batch *single_search_batch, *single_insert_batch;
            auto key_wrap_seq = parlay::tabulate(this->length, [&](int32_t i) {
                return std::make_pair(coord_to_key(&(vec_input[i])), i);
            });
            parlay::integer_sort_inplace(key_wrap_seq, [&](std::pair<uint64_t, int32_t> kw) {return kw.first;});
            uint64_t *key_seq = (uint64_t*)this->i64_io;
            auto key_idx_seq = parlay::tabulate(this->length, [&](int32_t i) {
                key_seq[i] = key_wrap_seq[i].first;
                return key_wrap_seq[i].second;
            });
            time_end("init");

            time_nested("search", [&]() {
                time_nested("taskgen", [&]() {
                    parfor_wrap(0, this->length, [&](size_t i) {
                        this->target_dpu[i] = key_to_dpu_id(key_seq[i]);
                    });
                    io = alloc_io_manager();
                    io->init();
                    single_search_batch = io->alloc<Single_search_task, Single_search_reply>(direct);
                    single_search_batch->push_task_sorted(
                        this->length, nr_of_dpus,
                        [&](size_t i) { return (Single_search_task){.key = key_seq[i]}; },
                        [&](size_t i) { return this->target_dpu[i]; },
                        parlay::make_slice(this->op_taskpos, this->op_taskpos + this->length)
                    );
                    io->finish_task_batch();
                });
                time_nested("exec", [&](){ASSERT(io->exec());});
                time_nested("get result", [&]() {
                    parfor_wrap(0, this->length, [&](size_t i) {
                        Single_search_reply *rep = (Single_search_reply*)single_search_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                        this->op_addrs[i] = rep->addr;
                    });
                    io->reset();
                });
            });

            time_nested("insert_vector", [&]() {
                time_nested("group by target", [&]() {
                    // Points are in key order, but points bound for the same target are not always
                    // contiguous: new points on both sides of an existing subtree C both target the edge
                    // above C (same parent and slot). Two tasks for one edge make the second edge split
                    // build on top of the first one's node and disconnect C. Stable-sort by target so each
                    // target gets exactly one task, with its points still in key order.
                    auto perm = parlay::tabulate(this->length, [](size_t i) { return (uint32_t)i; });
                    parlay::sort_inplace(perm, [&](uint32_t a, uint32_t b) {
                        const pptr &x = this->op_addrs[a], &y = this->op_addrs[b];
                        if(x.id != y.id) return x.id < y.id;
                        if(x.addr != y.addr) return x.addr < y.addr;
                        if(x.data_type != y.data_type) return x.data_type < y.data_type;
                        if(x.info != y.info) return x.info < y.info;
                        return a < b;
                    });
                    auto sorted_addrs = parlay::tabulate(this->length, [&](size_t i) { return this->op_addrs[perm[i]]; });
                    key_idx_seq = parlay::tabulate(this->length, [&](size_t i) { return key_idx_seq[perm[i]]; });
                    parfor_wrap(0, this->length, [&](size_t i) { this->op_addrs[i] = sorted_addrs[i]; });
                });
                time_nested("taskgen", [&]() {
                    io = alloc_io_manager();
                    io->init();
                    single_insert_batch = io->alloc_task_batch(direct, variable_length, fixed_length, SINGLE_INSERT_TSK, -1, 0);

                    auto pptr_diff_seq = parlay::delayed_tabulate(this->length, [&](size_t i)->bool {
                        return (i == 0) || !equal_pptr_strong(this->op_addrs[i], this->op_addrs[i - 1]);
                    });
                    int pptr_diff_num = parlay::count(pptr_diff_seq, true);
                    auto pptr_diff_idx = parlay::pack_index<uint32_t>(pptr_diff_seq);
                    auto group_len = [&](int i)->int {
                        return ((i == pptr_diff_num - 1) ? this->length : pptr_diff_idx[i + 1]) - pptr_diff_idx[i];
                    };
                    int max_group_len = parlay::reduce(parlay::delayed_tabulate(pptr_diff_num, [&](size_t i) { return group_len(i); }), parlay::maxm<int>());
                    // Rank of each leaf group among the groups bound for the same DPU
                    parlay::sequence<int> group_rank_in_dpu(pptr_diff_num);
                    int max_groups_per_dpu = 0;
                    {
                        std::vector<int> groups_per_dpu(nr_of_dpus, 0);
                        for(int i = 0; i < pptr_diff_num; i++) {
                            group_rank_in_dpu[i] = groups_per_dpu[this->op_addrs[pptr_diff_idx[i]].id]++;
                        }
                        for(int c : groups_per_dpu) max_groups_per_dpu = std::max(max_groups_per_dpu, c);
                    }
                    if(max_group_len > MAX_INSERT_TASK_LEN || max_groups_per_dpu > MAX_INSERT_TASKS_PER_DPU) {
                        // Keep every stride-th point of an oversized group and only the first
                        // MAX_INSERT_TASKS_PER_DPU groups of each DPU; defer the rest
                        parlay::sequence<bool> keep(this->length, true);
                        parfor_wrap(0, pptr_diff_num, [&](size_t i) {
                            int len = group_len(i);
                            if(group_rank_in_dpu[i] >= MAX_INSERT_TASKS_PER_DPU) {
                                for(int j = 0, k = pptr_diff_idx[i]; j < len; j++, k++) keep[k] = false;
                            }
                            else if(len > MAX_INSERT_TASK_LEN) {
                                int stride = (len + MAX_INSERT_TASK_LEN - 1) / MAX_INSERT_TASK_LEN;
                                for(int j = 0, k = pptr_diff_idx[i]; j < len; j++, k++) keep[k] = (j % stride == 0);
                            }
                        });
                        auto kept_idx = parlay::pack_index<uint32_t>(keep);
                        auto deferred_idx = parlay::pack_index<uint32_t>(parlay::delayed_map(keep, [](bool b) { return !b; }));
                        deferred = parlay::tabulate(deferred_idx.size(), [&](size_t i) {
                            return vec_input[key_idx_seq[deferred_idx[i]]];
                        });
                        int kept_num = kept_idx.size();
                        auto kept_diff_idx = parlay::pack_index<uint32_t>(parlay::delayed_tabulate(kept_num, [&](size_t i)->bool {
                            return (i == 0) || !equal_pptr_strong(this->op_addrs[kept_idx[i]], this->op_addrs[kept_idx[i - 1]]);
                        }));
                        int kept_diff_num = kept_diff_idx.size();
                        parfor_wrap(0, kept_diff_num, [&](int i) {
                            int len = (
                                (i == kept_diff_num - 1) ?
                                (kept_num - kept_diff_idx[i]) :
                                (kept_diff_idx[i + 1] - kept_diff_idx[i])
                            );
                            pptr addr = this->op_addrs[kept_idx[kept_diff_idx[i]]];
                            Single_insert_task *tsk = (Single_insert_task*)(single_insert_batch->push_task_zero_copy(
                                addr.id,
                                SINGLE_INSERT_TSK_SIZE(len),
                                true
                            ));
                            tsk->addr = addr;
                            tsk->len = len;
                            for(int j = 0, k = kept_diff_idx[i]; j < len; j++, k++) {
                                tsk->v[j] = vec_input[key_idx_seq[kept_idx[k]]];
                            }
                        });
                    }
                    else if(pptr_diff_num >= this->length / 5) {
                        parfor_wrap(0, this->length, [&](int i) {
                            if(pptr_diff_seq[i]) {
                                int len = 1, j;
                                for(j = i + 1; j < this->length; j++, len++) {
                                    if(!equal_pptr_strong(this->op_addrs[j], this->op_addrs[j - 1])) {
                                        break;
                                    }
                                }
                                pptr addr = this->op_addrs[i];
                                Single_insert_task *tsk = (Single_insert_task*)(single_insert_batch->push_task_zero_copy(
                                    addr.id,
                                    SINGLE_INSERT_TSK_SIZE(len),
                                    true
                                ));
                                tsk->addr = addr;
                                tsk->len = len;
                                for(j = 0; j < len; j++, i++) {
                                    tsk->v[j] = vec_input[key_idx_seq[i]];
                                }
                            }
                        });
                    }
                    else {
                        parfor_wrap(0, pptr_diff_num, [&](int i) {
                            int len = (
                                (i == pptr_diff_num - 1) ?
                                (this->length - pptr_diff_idx[i]) :
                                (pptr_diff_idx[i + 1] - pptr_diff_idx[i])
                            );
                            pptr addr = this->op_addrs[pptr_diff_idx[i]];
                            Single_insert_task *tsk = (Single_insert_task*)(single_insert_batch->push_task_zero_copy(
                                addr.id,
                                SINGLE_INSERT_TSK_SIZE(len),
                                true
                            ));
                            tsk->addr = addr;
                            tsk->len = len;
                            for(int j = 0, k = pptr_diff_idx[i]; j < len; j++, k++) {
                                tsk->v[j] = vec_input[key_idx_seq[k]];
                            }
                        });
                    }

                    io->finish_task_batch();
                });
                time_nested("exec", [&](){ASSERT(io->exec());});
                time_nested("get result", [&]() {io->reset();});
            });

            // nr_points += this->length;
            std::atomic_fetch_add(&(this->nr_points), this->length - (int64_t)deferred.size());

            if(deferred.empty()) break;
            round_input = std::move(deferred);
            deferred = parlay::sequence<vectorT>();
            vec_input = round_input.data();
            this->length = round_input.size();
        }
        this->length = batch_length;

        time_end("insert");
        cpu_coverage_timer->end();
        this->epoch_num++;
#endif
    }

    /* 
        Box range queries. Return the number of existing points in the queried box, or fetch them.
        count_or_fetch = true, return the counted numbers; false, fetch the points.
        Set pim_zd_tree::length to be the number of boxes.
        Put a vector pair in pim_zd_tree::vector_input as box boundaries.
    */
    void box_range(bool count_or_fetch = true, int expected_length = 100, vectorT *vec_input = nullptr) {
#if (defined BOX_RANGE_FETCH_ON) || (defined BOX_RANGE_COUNT_ON)
        print_current_epoch();
        cpu_coverage_timer->start();
        time_start("box");

        if(vec_input == nullptr) vec_input = this->vector_input;
        parlay::sequence<int> box_dpu_num;
        int total_query_num;
        IO_Manager *io;
        IO_Task_Batch *box_batch;
        
        time_nested("taskgen", [&]() {
            parlay::sequence<box_dpu_id> box_idx(this->length);
            box_dpu_num = parlay::tabulate(this->length, [&](size_t i) {
                box_boundary_swap(vec_input[i << 1], vec_input[(i << 1) + 1]);
                uint64_t key1 = coord_to_key(&(vec_input[i << 1]));
                uint64_t key2 = coord_to_key(&(vec_input[(i << 1) + 1]));
                box_idx[i].set_litmin_bigmax(
                    key_to_dpu_id(key1),
                    key_to_dpu_id(key2)
                );
                if(box_idx[i].same_dpu()) return (int)1;
                else {
                    auto box_split_res = box_split(key1, key2);
                    box_idx[i].set_litmax_bigmin(
                        key_to_dpu_id(box_split_res.first),
                        key_to_dpu_id(box_split_res.second)
                    );
                    return box_idx[i].size();
                }
            });
            total_query_num = parlay::scan_inplace(box_dpu_num);

            io = alloc_io_manager();
            io->init();
            if(count_or_fetch) {
                box_batch = io->alloc<Box_count_task, Box_count_reply>(direct);
            } else {
                box_batch = io->alloc_task_batch(direct, fixed_length, variable_length, BOX_FETCH_TSK, 
                                                 sizeof(Box_fetch_task), BOX_FETCH_REP_SIZE(expected_length));
            }
            auto tsk_seq = parlay::sequence<std::pair<vectorT, vectorT>>(total_query_num);

            parfor_wrap(0, this->length, [&](size_t i) {
                int start_idx = box_dpu_num[i];
                int end_idx = (i == this->length - 1 ? total_query_num : box_dpu_num[i + 1]);
                auto tsk = std::make_pair(vec_input[i << 1], vec_input[(i << 1) + 1]);
                if(end_idx - start_idx <= 1) {
                    this->target_dpu[start_idx] = box_idx[i].litmin;
                    tsk_seq[start_idx] = tsk;
                } else {
                    int j;
                    if(box_idx[i].litmax < box_idx[i].bigmin) {
                        for(j = box_idx[i].litmin; j <= box_idx[i].litmax; j++, start_idx++) {
                            this->target_dpu[start_idx] = j;
                            tsk_seq[start_idx] = tsk;
                        }
                        for(j = box_idx[i].bigmin; j <= box_idx[i].bigmax; j++, start_idx++) {
                            this->target_dpu[start_idx] = j;
                            tsk_seq[start_idx] = tsk;
                        }
                    }
                    else {
                        for(j = box_idx[i].litmin; j <= box_idx[i].bigmax; j++, start_idx++) {
                            this->target_dpu[start_idx] = j;
                            tsk_seq[start_idx] = tsk;
                        }
                    }
                }
            });
            if(count_or_fetch) {
                box_batch->push_task_from_array_by_isort<false>(
                    total_query_num,
                    [&](size_t i) {
                        Box_count_task tsk;
                        tsk.vec_min = tsk_seq[i].first;
                        tsk.vec_max = tsk_seq[i].second;
                        return tsk;
                    },
                    parlay::make_slice(this->target_dpu, this->target_dpu + total_query_num),
                    parlay::make_slice(this->op_taskpos, this->op_taskpos + total_query_num)
                );
            } else {
                box_batch->push_task_from_array_by_isort<false>(
                    total_query_num,
                    [&](size_t i) {
                        Box_fetch_task tsk;
                        tsk.vec_min = tsk_seq[i].first;
                        tsk.vec_max = tsk_seq[i].second;
                        return tsk;
                    },
                    parlay::make_slice(this->target_dpu, this->target_dpu + total_query_num),
                    parlay::make_slice(this->op_taskpos, this->op_taskpos + total_query_num)
                );
            }
            io->finish_task_batch();
        });

        time_nested("exec", [&](){ASSERT(io->exec());});
        time_nested("get result", [&]() {
            if(count_or_fetch) {
                parfor_wrap(0, this->length, [&](size_t i) {
                    this->i64_io[i] = 0;
                    int end_idx = (i == this->length - 1 ? total_query_num : box_dpu_num[i + 1]);
                    for(int j = box_dpu_num[i]; j < end_idx; j++) {
                        this->i64_io[i] += ((Box_count_reply*)box_batch->ith(this->target_dpu[j], this->op_taskpos[j]))->count;
                    }
                });
            } else {
                parlay::sequence<int> return_size_seq = parlay::tabulate(total_query_num, [&](size_t i) {
                    return (int)(((Box_fetch_reply*)box_batch->ith(this->target_dpu[i], this->op_taskpos[i]))->len);
                });
                int total_return_num = parlay::scan_inplace(return_size_seq);
                parfor_wrap(0, this->length, [&](size_t i) {
                    this->i64_io[i] = return_size_seq[box_dpu_num[i]];
                });
                this->i64_io[this->length] = total_return_num;
                ASSERT(total_return_num <= BATCH_SIZE);
                if(total_return_num > BATCH_SIZE) return;
                parfor_wrap(0, total_query_num, [&](size_t i) {
                    int len = (i == total_query_num - 1 ? total_return_num : return_size_seq[i + 1]) - return_size_seq[i];
                    Box_fetch_reply *rep = (Box_fetch_reply*)box_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                    memcpy(this->vector_output + return_size_seq[i], rep->v, S64(MULTIPLY_NR_DIMENSION(len)));
                });
            }
            io->reset();
        });

        time_end("box");
        cpu_coverage_timer->end();
        this->epoch_num++;
#endif
    }

    void knn(int knn_k = 10, vectorT *vec_input = nullptr) {
#ifdef KNN_ON
        print_current_epoch();
        cpu_coverage_timer->start();
        time_start("knn");

        if(vec_input == nullptr) vec_input = this->vector_input;
        IO_Manager *io;
        IO_Task_Batch *knn_batch;

#if (LX_NORM == 2) && !defined(LX_NORM_ON_DPU)
        parlay::sequence<uint32_t> needs_further_processing_idx;
        time_nested("first round", [&]() {
            time_nested("taskgen", [&]() {
                parfor_wrap(0, this->length, [&](size_t i) {
                    this->target_dpu[i] = key_to_dpu_id(coord_to_key(&(vec_input[i])));
                });
                io = alloc_io_manager();
                io->init();
                knn_batch = io->alloc_task_batch(direct, fixed_length, variable_length, KNN_TSK, sizeof(knn_task), KNN_REP_SIZE(knn_k));
                knn_batch->push_task_from_array_by_isort<false>(
                    this->length,
                    [&](size_t i) {
                        knn_task tsk;
                        tsk.k = knn_k;
                        tsk.center = vec_input[i];
                        return tsk;
                    },
                    parlay::make_slice(this->target_dpu, this->target_dpu + this->length),
                    parlay::make_slice(this->op_taskpos, this->op_taskpos + this->length)
                );
                io->finish_task_batch();
            });
            time_nested("exec", [&](){ASSERT(io->exec());});
            time_nested("get result", [&]() {
                parfor_wrap(0, this->length, [&](size_t i) {
                    knn_reply *rep = (knn_reply*)knn_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                    heap_host heap(knn_k);
                    vectorT vec;
                    int64_t distance;
                    int j, k;
                    for(j = 0; j < rep->len; j++) {
                        vec = vector_sub(vec_input + i, rep->v + j);
                        distance = vector_norm(&vec);
                        heap.enqueue(distance, rep->v + j);
                    }
                    memcpy(this->vector_output + i * knn_k, heap.vector_storage, S64(MULTIPLY_NR_DIMENSION(knn_k)));
                });
                io->reset();
            });
        });

        time_nested("second round", [&]() {
            parlay::sequence<int> box_dpu_num;
            int total_return_num;

            time_nested("taskgen", [&]() {
                parlay::sequence<box_dpu_id> box_idx(this->length);
                box_dpu_num = parlay::tabulate(this->length, [&](size_t i) {
                    vectorT vec;
                    uint64_t key1, key2;
                    vec = vector_sub(this->vector_output + i * knn_k, vec_input + i);
                    int64_t radius = sqrt(vector_norm(&vec));
                    vector_ones(&vec, radius);
                    vec = vector_sub_zero_bounded(vec_input + i, &vec);
                    key1 = coord_to_key(&vec);
                    vector_ones(&vec, radius);
                    vec = vector_add(vec_input + i, &vec);
                    key2 = coord_to_key(&vec);
                    box_idx[i].set_litmin_bigmax(key_to_dpu_id(key1),  key_to_dpu_id(key2));
                    auto box_split_res = box_split(key1, key2);
                    box_idx[i].set_litmax_bigmin(
                        key_to_dpu_id(box_split_res.first),
                        key_to_dpu_id(box_split_res.second)
                    );
                    int ret = box_idx[i].size() - 1;
                    this->i64_io[i] = (ret <= 0 ? -1 : radius * radius);
                    return ret;
                });
                total_return_num = parlay::scan_inplace(box_dpu_num);

                io = alloc_io_manager();
                io->init();
                knn_batch = io->alloc_task_batch(direct, fixed_length, variable_length, KNN_BOUNDED_TSK, sizeof(knn_bounded_task), KNN_REP_SIZE(knn_k));
                parfor_wrap(0, this->length, [&](size_t i) {
                    if(this->i64_io[i] < 0) return;
                    vectorT *vec = vec_input + i;
                    int this_dpu_idx = key_to_dpu_id(coord_to_key(vec));
                    int start_idx = box_dpu_num[i];
                    int j;
                    knn_bounded_task *tsk;
                    if(box_idx[i].litmax < box_idx[i].bigmin) {
                        for(j = box_idx[i].litmin; j <= box_idx[i].litmax; j++) {
                            if(j == this_dpu_idx) {
                                continue;
                            }
                            this->target_dpu[start_idx] = j;
                            tsk = (knn_bounded_task*)knn_batch->push_task_zero_copy(
                                j, sizeof(knn_bounded_task), true, this->op_taskpos + start_idx
                            );
                            tsk->k = knn_k;
                            tsk->center = *vec;
                            tsk->radius = this->i64_io[i];
                            start_idx++;
                        }
                        for(j = box_idx[i].bigmin; j <= box_idx[i].bigmax; j++) {
                            if(j == this_dpu_idx) {
                                continue;
                            }
                            this->target_dpu[start_idx] = j;
                            tsk = (knn_bounded_task*)knn_batch->push_task_zero_copy(
                                j, sizeof(knn_bounded_task), true, this->op_taskpos + start_idx
                            );
                            tsk->k = knn_k;
                            tsk->center = *vec;
                            tsk->radius = this->i64_io[i];
                            start_idx++;
                        }
                    }
                    else {
                        for(j = box_idx[i].litmin; j <= box_idx[i].bigmax; j++) {
                            if(j == this_dpu_idx) {
                                continue;
                            }
                            this->target_dpu[start_idx] = j;
                            tsk = (knn_bounded_task*)knn_batch->push_task_zero_copy(
                                j, sizeof(knn_bounded_task), true, this->op_taskpos + start_idx
                            );
                            tsk->k = knn_k;
                            tsk->center = *vec;
                            tsk->radius = this->i64_io[i];
                            start_idx++;
                        }
                    }
                });
                io->finish_task_batch();
            });
            time_nested("exec", [&](){ASSERT(io->exec());});
            time_nested("get result", [&]() {
                parfor_wrap(0, this->length, [&](size_t i) {
                    if(this->i64_io[i] < 0) return;
                    heap_host heap(knn_k);
                    vectorT *center = vec_input + i;
                    vectorT vec, *vec_pt;
                    int64_t distance;
                    int j;
                    for(vec_pt = this->vector_output + knn_k * i, j = 0; j < knn_k; j++, vec_pt++) {
                        vec = vector_sub(center, vec_pt);
                        distance = vector_norm(&vec);
                        heap.enqueue(distance, vec_pt);
                    }
                    int end_idx = (i == this->length - 1 ? total_return_num : box_dpu_num[i + 1]);
                    knn_reply *rep;
                    for(j = box_dpu_num[i]; j < end_idx; j++) {
                        rep = (knn_reply*)knn_batch->ith(this->target_dpu[j], this->op_taskpos[j]);
                        for(int k = 0; k < rep->len; k++) {
                            vec_pt = rep->v + k;
                            vec = vector_sub(center, vec_pt);
                            distance = vector_norm(&vec);
                            heap.enqueue(distance, vec_pt);
                        }
                    }
                    memcpy(this->vector_output + i * knn_k, heap.vector_storage, S64(MULTIPLY_NR_DIMENSION(knn_k)));
                });
                io->reset();
            });
        });
#else
        parlay::sequence<uint32_t> needs_further_processing_idx;
        time_nested("first round", [&]() {
            time_nested("taskgen", [&]() {
                parfor_wrap(0, this->length, [&](size_t i) {
                    this->target_dpu[i] = key_to_dpu_id(coord_to_key(&(vec_input[i])));
                });
                io = alloc_io_manager();
                io->init();
                knn_batch = io->alloc_task_batch(direct, fixed_length, variable_length, KNN_TSK, sizeof(knn_task), KNN_REP_SIZE(knn_k));
                knn_batch->push_task_from_array_by_isort<false>(
                    this->length,
                    [&](size_t i) {
                        knn_task tsk;
                        tsk.k = knn_k;
                        tsk.center = vec_input[i];
                        return tsk;
                    },
                    parlay::make_slice(this->target_dpu, this->target_dpu + this->length),
                    parlay::make_slice(this->op_taskpos, this->op_taskpos + this->length)
                );
                io->finish_task_batch();
            });
            time_nested("exec", [&](){ASSERT(io->exec());});
            time_nested("get result", [&]() {
                parfor_wrap(0, this->length, [&](size_t i) {
                    knn_reply *rep = (knn_reply*)knn_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                    this->i64_io[i] = rep->len;
                    memcpy(this->vector_output + i * knn_k, rep->v, S64(MULTIPLY_NR_DIMENSION(knn_k)));
                });
                io->reset();
                needs_further_processing_idx = parlay::pack_index<uint32_t>(
                    parlay::delayed_tabulate(this->length, [&](size_t i)->bool {
                        return (this->i64_io[i] >= 0);
                    })
                );
            });
        });

        int needs_further_processing_knn_num = needs_further_processing_idx.size();
        if(needs_further_processing_knn_num > 0) {
            time_nested("second round", [&]() {
                parlay::sequence<int> box_dpu_num;
                int total_return_num;

                time_nested("taskgen", [&]() {
                    parlay::sequence<box_dpu_id> box_idx(needs_further_processing_knn_num);
                    box_dpu_num = parlay::tabulate(needs_further_processing_knn_num, [&](size_t i) {
                        vectorT vec;
                        uint64_t key1, key2;
                        int64_t radius = this->i64_io[needs_further_processing_idx[i]];
#if LX_NORM == 2
                        radius = (int64_t)sqrt(radius);
#endif
                        vector_ones(&vec, radius);
                        vec = vector_sub_zero_bounded(&(vec_input[needs_further_processing_idx[i]]), &vec);
                        key1 = coord_to_key(&vec);
                        vector_ones(&vec, radius);
                        vec = vector_add(&(vec_input[needs_further_processing_idx[i]]), &vec);
                        key2 = coord_to_key(&vec);
                        box_idx[i].set_litmin_bigmax(key_to_dpu_id(key1), key_to_dpu_id(key2));
                        auto box_split_res = box_split(key1, key2);
                        box_idx[i].set_litmax_bigmin(
                            key_to_dpu_id(box_split_res.first),
                            key_to_dpu_id(box_split_res.second)
                        );
                        return box_idx[i].size() - 1;
                    });
                    total_return_num = parlay::scan_inplace(box_dpu_num);

                    io = alloc_io_manager();
                    io->init();
                    knn_batch = io->alloc_task_batch(direct, fixed_length, variable_length, KNN_BOUNDED_TSK, sizeof(knn_bounded_task), KNN_REP_SIZE(knn_k));
                    parfor_wrap(0, needs_further_processing_knn_num, [&](size_t i) {
                        vectorT vec = vec_input[needs_further_processing_idx[i]];
                        int this_dpu_idx = key_to_dpu_id(coord_to_key(&vec));
                        int start_idx = box_dpu_num[i];
                        int j;
                        knn_bounded_task *tsk;
                        if(box_idx[i].litmax < box_idx[i].bigmin) {
                            for(j = box_idx[i].litmin; j <= box_idx[i].litmax; j++) {
                                if(j == this_dpu_idx) {
                                    continue;
                                }
                                this->target_dpu[start_idx] = j;
                                tsk = (knn_bounded_task*)knn_batch->push_task_zero_copy(
                                    j, sizeof(knn_bounded_task), true, this->op_taskpos + start_idx
                                );
                                tsk->k = knn_k;
                                tsk->center = vec;
                                tsk->radius = this->i64_io[needs_further_processing_idx[i]];
                                start_idx++;
                            }
                            for(j = box_idx[i].bigmin; j <= box_idx[i].bigmax; j++) {
                                if(j == this_dpu_idx) {
                                    continue;
                                }
                                this->target_dpu[start_idx] = j;
                                tsk = (knn_bounded_task*)knn_batch->push_task_zero_copy(
                                    j, sizeof(knn_bounded_task), true, this->op_taskpos + start_idx
                                );
                                tsk->k = knn_k;
                                tsk->center = vec;
                                tsk->radius = this->i64_io[needs_further_processing_idx[i]];
                                start_idx++;
                            }
                        }
                        else {
                            for(j = box_idx[i].litmin; j <= box_idx[i].bigmax; j++) {
                                if(j == this_dpu_idx) {
                                    continue;
                                }
                                this->target_dpu[start_idx] = j;
                                tsk = (knn_bounded_task*)knn_batch->push_task_zero_copy(
                                    j, sizeof(knn_bounded_task), true, this->op_taskpos + start_idx
                                );
                                tsk->k = knn_k;
                                tsk->center = vec;
                                tsk->radius = this->i64_io[needs_further_processing_idx[i]];
                                start_idx++;
                            }
                        }
                    });
                    io->finish_task_batch();
                });
                time_nested("exec", [&](){ASSERT(io->exec());});
                time_nested("get result", [&]() {
                    parfor_wrap(0, needs_further_processing_knn_num, [&](size_t i) {
                        heap_host heap(knn_k);
                        vectorT center = vec_input[needs_further_processing_idx[i]];
                        vectorT vec, *vec_pt;
                        int64_t distance;
                        int j;
                        for(j = 0; j < knn_k; j++) {
                            vec_pt = &(this->vector_output[j + knn_k * needs_further_processing_idx[i]]);
                            vec = vector_sub(&center, vec_pt);
                            distance = vector_norm(&vec);
                            heap.enqueue(distance, vec_pt);
                        }
                        int end_idx = (i == needs_further_processing_knn_num - 1 ? total_return_num : box_dpu_num[i + 1]);
                        knn_reply *rep;
                        for(j = box_dpu_num[i]; j < end_idx; j++) {
                            rep = (knn_reply*)knn_batch->ith(this->target_dpu[j], this->op_taskpos[j]);
                            for(int k = 0; k < rep->len; k++) {
                                vec_pt = &(rep->v[k]);
                                vec = vector_sub(&center, vec_pt);
                                distance = vector_norm(&vec);
                                heap.enqueue(distance, vec_pt);
                            }
                        }
                        memcpy(this->vector_output + knn_k * needs_further_processing_idx[i], heap.vector_storage, S64(MULTIPLY_NR_DIMENSION(knn_k)));
                    });
                    io->reset();
                });
            });
        }
#endif
        time_end("knn");
        cpu_coverage_timer->end();
        this->epoch_num++;
#endif
    }

    void search_maximum_match(bool print_res = false, bool debug_fetch = false, bool debug_print = false, uint64_t default_key = 0) {
#ifdef SEARCH_TEST_ON
        print_current_epoch();
        cpu_coverage_timer->start();
        time_start("search max match");

        parlay::sequence<uint64_t> key_seq, return_seq;
        IO_Manager *io;
        IO_Task_Batch *single_key_search_batch;

        time_nested("taskgen", [&]() {
            key_seq = parlay::tabulate(this->length, [&](size_t i){ return coord_to_key(&(this->vector_input[i])); });
            parfor_wrap(0, this->length, [&](size_t i){ this->target_dpu[i] = key_to_dpu_id(key_seq[i]); });
            io = alloc_io_manager();
            io->init();
            single_key_search_batch = io->alloc<Single_key_search_task, Single_key_search_reply>(direct);
            single_key_search_batch->push_task_from_array_by_isort<false>(
                this->length,
                [&](size_t i) {
                    return (Single_key_search_task){.key = key_seq[i]};
                },
                parlay::make_slice(this->target_dpu, this->target_dpu + this->length),
                parlay::make_slice(this->op_taskpos, this->op_taskpos + this->length)
            );
            io->finish_task_batch();
        });
        time_nested("exec", [&](){ASSERT(io->exec());});
        time_nested("get result", [&]() {
            return_seq = parlay::tabulate(this->length, [&](size_t i) {
                Single_key_search_reply *rep = (Single_key_search_reply*)single_key_search_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                return rep->key;
            });
            io->reset();
        });

        time_end("search max match");

        if(print_res) {
            cout<<"------------ Search Results --------------"<<endl;
            int match_num = 0;
            for(int i = 0; i < this->length; i++) {
                if(key_seq[i] == return_seq[i]) match_num++;
                else cout<<hex<<key_seq[i]<<" "<<return_seq[i]<<endl;
            }
            cout<<"Match Num: "<<dec<<match_num<<"; Unmatch Num: "<<(this->length - match_num)<<endl;
        }

#if (defined FETCH_NODE_ON) && (defined DPU_STORAGE_STAT_ON)
        if(debug_fetch) {
            time_start("fetch node");
            auto idx_seq = parlay::pack_index<uint32_t>(parlay::delayed_tabulate(this->length, [&](size_t i) {
                return key_seq[i] != return_seq[i] && key_seq[i] != default_key;
            }));
            int tmp_length = idx_seq.size();
            auto target_dpu_seq = parlay::tabulate(tmp_length, [&](size_t i){
                return (size_t)key_to_dpu_id(key_seq[idx_seq[i]]);
            });
            if(debug_print) for(int i = 0; i < tmp_length; i++) cout<<key_seq[idx_seq[i]]<<" "<<target_dpu_seq[i]<<endl;
            parlay::integer_sort_inplace(target_dpu_seq);
            idx_seq = parlay::pack_index<uint32_t>(parlay::delayed_tabulate(target_dpu_seq.size(), [&](size_t i){
                return (i == 0) || (target_dpu_seq[i - 1] != target_dpu_seq[i]);
            }));
            tmp_length = idx_seq.size();
            IO_Task_Batch *dpu_stat_batch;
            parlay::sequence<dpu_storage_stat_reply> dpu_stats_return_seq;

            time_nested("DPU stats", [&]() {
                time_nested("taskgen", [&]() {
                    parfor_wrap(0, tmp_length, [&](size_t i){ this->target_dpu[i] = target_dpu_seq[idx_seq[i]]; });
                    io = alloc_io_manager();
                    io->init();
                    dpu_stat_batch = io->alloc<dpu_storage_stat_task, dpu_storage_stat_reply>(direct);
                    dpu_stat_batch->push_task_from_array_by_isort<false>(
                        tmp_length,
                        [&](size_t i) {
                            return (dpu_storage_stat_task){.dpu_id = this->target_dpu[i]};
                        },
                        parlay::make_slice(this->target_dpu, this->target_dpu + tmp_length),
                        parlay::make_slice(this->op_taskpos, this->op_taskpos + tmp_length)
                    );
                    io->finish_task_batch();
                });
                time_nested("exec", [&](){ASSERT(io->exec());});
                time_nested("get result", [&]() {
                    dpu_stats_return_seq = parlay::tabulate(tmp_length, [&](size_t i) {
                        dpu_storage_stat_reply *rep = (dpu_storage_stat_reply*)dpu_stat_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                        return *rep;
                    });
                    io->reset();
                });
            });
            for(int i = 0; i < tmp_length; i++) {
                cout<<"DPU: "<<this->target_dpu[i]<<"; Bcnt: "<<dpu_stats_return_seq[i].bcnt<<"; Pcnt: "<<dpu_stats_return_seq[i].pcnt;
                cout<<endl;
            }
            
            int total_node_to_fetch;
            IO_Task_Batch *node_fetch_batch;
            parlay::sequence<pptr> pptr_seq_tmp;
            parlay::sequence<Fetch_node_reply> fetch_return_seq;

            time_nested("fetch dpu nodes", [&]() {
                time_nested("taskgen", [&]() {
                    auto int_seq = parlay::tabulate(tmp_length, [&](size_t i){return (int)(dpu_stats_return_seq[i].bcnt + dpu_stats_return_seq[i].pcnt);});
                    total_node_to_fetch = parlay::scan_inplace(int_seq);
                    pptr_seq_tmp = parlay::sequence<pptr>(total_node_to_fetch);
                    parfor_wrap(0, tmp_length, [&](size_t i) {
                        int idx_tmp, dpu_id = target_dpu_seq[idx_seq[i]];
                        for(int j = 0; j < dpu_stats_return_seq[i].bcnt; j++) {
                            idx_tmp = int_seq[i] + j;
                            this->target_dpu[idx_tmp] = dpu_id;
                            pptr_seq_tmp[idx_tmp] = (pptr) {
                                .data_type = B_NODE_DATA_TYPE,
                                .info = 0,
                                .id = dpu_id,
                                .addr = j,
                            };
                        }
                        for(int j = 0; j < dpu_stats_return_seq[i].pcnt; j++) {
                            idx_tmp = int_seq[i] + j + dpu_stats_return_seq[i].bcnt;
                            this->target_dpu[idx_tmp] = dpu_id;
                            pptr_seq_tmp[idx_tmp] = (pptr) {
                                .data_type = P_NODE_DATA_TYPE,
                                .info = 0,
                                .id = dpu_id,
                                .addr = j,
                            };
                        }
                    });
                    io = alloc_io_manager();
                    io->init();
                    node_fetch_batch = io->alloc<Fetch_node_w_pptr_task, Fetch_node_reply>(direct);
                    node_fetch_batch->push_task_from_array_by_isort<false>(
                        total_node_to_fetch,
                        [&](size_t i) {
                            return (Fetch_node_w_pptr_task){.addr = pptr_seq_tmp[i]};
                        },
                        parlay::make_slice(this->target_dpu, this->target_dpu + total_node_to_fetch),
                        parlay::make_slice(this->op_taskpos, this->op_taskpos + total_node_to_fetch)
                    );
                    io->finish_task_batch();
                });
                time_nested("exec", [&](){ASSERT(io->exec());});
                time_nested("get result", [&]() {
                    fetch_return_seq = parlay::tabulate(total_node_to_fetch, [&](size_t i) {
                        Fetch_node_reply *rep = (Fetch_node_reply*)node_fetch_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                        return *rep;
                    });
                    io->reset();
                });
            });

            if(print_res) {
                cout<<"------------ Fetch Results --------------"<<endl;
                for(int i = 0; i < total_node_to_fetch; i++) {
                    cout<<"*********************"<<endl;
                    cout<<dec<<"Target Addr: "; cout_pptr(pptr_seq_tmp[i]);
                    cout<<"Node key: "<<hex<<fetch_return_seq[i].key<<"; Height: "<<dec<<fetch_return_seq[i].height<<endl;
                    cout<<dec<<"Parent: "; cout_pptr(fetch_return_seq[i].parent);
                    if(pptr_seq_tmp[i].data_type == P_NODE_DATA_TYPE) {
                        cout<<"len: "<<dec<<fetch_return_seq[i].len<<endl;
                        for(int j = 0; j <= fetch_return_seq[i].len && j < LEAF_SIZE; j++)
                            cout<<hex<<fetch_return_seq[i].keys[j]<<endl;
                    }
                    else if(pptr_seq_tmp[i].data_type == B_NODE_DATA_TYPE) {
                        for(int j = 0; j < DB_SIZE; j++) {
                            pptr tmp_addr = I64_TO_PPTR(fetch_return_seq[i].keys[j]);
                            if(valid_pptr(tmp_addr)) {
                                cout<<dec<<j<<": ";
                                cout_pptr(tmp_addr);
                            }
                        }
                    }
                }
            }
            
            time_end("fetch node");
        }
#endif
        cpu_coverage_timer->end();
        this->epoch_num++;
#endif
    }

    void fetch_all_nodes(bool print_res = false, bool debug_print = false) {
#if (defined FETCH_NODE_ON) && (defined DPU_STORAGE_STAT_ON)
        print_current_epoch();
        cpu_coverage_timer->start();
        IO_Manager *io;
        IO_Task_Batch *single_key_search_batch;

            time_start("fetch node");
            
            IO_Task_Batch *dpu_stat_batch;
            parlay::sequence<dpu_storage_stat_reply> dpu_stats_return_seq;
            int tmp_length = this->length;

            time_nested("DPU stats", [&]() {
                time_nested("taskgen", [&]() {
                    parfor_wrap(0, tmp_length, [&](size_t i) {
                        uint64_t key = coord_to_key(&(this->vector_input[i]));
                        this->target_dpu[i] = key_to_dpu_id(key);
                    });
                    io = alloc_io_manager();
                    io->init();
                    dpu_stat_batch = io->alloc<dpu_storage_stat_task, dpu_storage_stat_reply>(direct);
                    dpu_stat_batch->push_task_from_array_by_isort<false>(
                        tmp_length,
                        [&](size_t i) {
                            return (dpu_storage_stat_task){.dpu_id = this->target_dpu[i]};
                        },
                        parlay::make_slice(this->target_dpu, this->target_dpu + tmp_length),
                        parlay::make_slice(this->op_taskpos, this->op_taskpos + tmp_length)
                    );
                    io->finish_task_batch();
                });
                time_nested("exec", [&](){ASSERT(io->exec());});
                time_nested("get result", [&]() {
                    dpu_stats_return_seq = parlay::tabulate(tmp_length, [&](size_t i) {
                        dpu_storage_stat_reply *rep = (dpu_storage_stat_reply*)dpu_stat_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                        return *rep;
                    });
                    io->reset();
                });
            });
            for(int i = 0; i < tmp_length; i++) {
                cout<<"DPU: "<<this->target_dpu[i]<<"; Bcnt: "<<dpu_stats_return_seq[i].bcnt<<"; Pcnt: "<<dpu_stats_return_seq[i].pcnt;
                cout<<endl;
            }
            
            int total_node_to_fetch;
            IO_Task_Batch *node_fetch_batch;
            parlay::sequence<pptr> pptr_seq_tmp;
            parlay::sequence<Fetch_node_reply> fetch_return_seq;

            time_nested("fetch dpu nodes", [&]() {
                time_nested("taskgen", [&]() {
                    auto int_seq = parlay::tabulate(tmp_length, [&](size_t i){return (int)(dpu_stats_return_seq[i].bcnt + dpu_stats_return_seq[i].pcnt);});
                    total_node_to_fetch = parlay::scan_inplace(int_seq);
                    pptr_seq_tmp = parlay::sequence<pptr>(total_node_to_fetch);
                    auto target_dpu_seq = parlay::tabulate(tmp_length, [&](size_t i) { return this->target_dpu[i]; });
                    parfor_wrap(0, tmp_length, [&](size_t i) {
                        int idx_tmp, dpu_id = target_dpu_seq[i];
                        for(int j = 0; j < dpu_stats_return_seq[i].bcnt; j++) {
                            idx_tmp = int_seq[i] + j;
                            this->target_dpu[idx_tmp] = dpu_id;
                            pptr_seq_tmp[idx_tmp] = (pptr) {
                                .data_type = B_NODE_DATA_TYPE,
                                .info = 0,
                                .id = dpu_id,
                                .addr = j,
                            };
                        }
                        for(int j = 0; j < dpu_stats_return_seq[i].pcnt; j++) {
                            idx_tmp = int_seq[i] + j + dpu_stats_return_seq[i].bcnt;
                            this->target_dpu[idx_tmp] = dpu_id;
                            pptr_seq_tmp[idx_tmp] = (pptr) {
                                .data_type = P_NODE_DATA_TYPE,
                                .info = 0,
                                .id = dpu_id,
                                .addr = j,
                            };
                        }
                    });
                    io = alloc_io_manager();
                    io->init();
                    node_fetch_batch = io->alloc<Fetch_node_w_pptr_task, Fetch_node_reply>(direct);
                    node_fetch_batch->push_task_from_array_by_isort<false>(
                        total_node_to_fetch,
                        [&](size_t i) {
                            return (Fetch_node_w_pptr_task){.addr = pptr_seq_tmp[i]};
                        },
                        parlay::make_slice(this->target_dpu, this->target_dpu + total_node_to_fetch),
                        parlay::make_slice(this->op_taskpos, this->op_taskpos + total_node_to_fetch)
                    );
                    io->finish_task_batch();
                });
                time_nested("exec", [&](){ASSERT(io->exec());});
                time_nested("get result", [&]() {
                    fetch_return_seq = parlay::tabulate(total_node_to_fetch, [&](size_t i) {
                        Fetch_node_reply *rep = (Fetch_node_reply*)node_fetch_batch->ith(this->target_dpu[i], this->op_taskpos[i]);
                        return *rep;
                    });
                    io->reset();
                });
            });

            if(print_res) {
                cout<<"------------ Fetch Results --------------"<<endl;
                for(int i = 0; i < total_node_to_fetch; i++) {
                    cout<<"*********************"<<endl;
                    cout<<dec<<"Target Addr: "; cout_pptr(pptr_seq_tmp[i]);
                    cout<<"Node key: "<<hex<<fetch_return_seq[i].key<<"; Height: "<<dec<<fetch_return_seq[i].height<<endl;
                    cout<<dec<<"Parent: "; cout_pptr(fetch_return_seq[i].parent);
                    if(pptr_seq_tmp[i].data_type == P_NODE_DATA_TYPE) {
                        cout<<"len: "<<dec<<fetch_return_seq[i].len<<endl;
                        for(int j = 0; j <= fetch_return_seq[i].len && j < LEAF_SIZE; j++)
                            cout<<hex<<fetch_return_seq[i].keys[j]<<endl;
                    }
                    else if(pptr_seq_tmp[i].data_type == B_NODE_DATA_TYPE) {
                        for(int j = 0; j < DB_SIZE; j++) {
                            pptr tmp_addr = I64_TO_PPTR(fetch_return_seq[i].keys[j]);
                            if(valid_pptr(tmp_addr)) {
                                cout<<dec<<j<<": ";
                                cout_pptr(tmp_addr);
                            }
                        }
                    }
                }
            }
            
            time_end("fetch node");
        cpu_coverage_timer->end();
        this->epoch_num++;
#endif
    }

};

atomic<int64_t> pim_zd_tree::nr_points = atomic<int64_t>(0);