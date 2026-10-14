#pragma once
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>
#include "macro.h"
#include "task_utils.h"
#include "task_framework_dpu.h"
#include "node_dpu.h"
#include "storage.h"
#include "utils_dpu.h"

void vector_min(vectorT *src, vectorT *dst);
void vector_max(vectorT *src, vectorT *dst);

/* ----------------- Search Leaf Node -------------------- */

typedef struct Bnode_metadata_for_search {
    int16_t height;
    int16_t subtree_size;
    int32_t parent;
    uint64_t key;
} Bnode_metadata_for_search;
#define BNODE_METADATA_FOR_SEARCH_SIZE (16)

static inline pptr b_search(uint64_t key, bool mismatch_return_parent) {
    mBptr tmp = root;
    int idx = -1;
    bool continue_sign = true;
    pptr addr;
    Bnode_metadata_for_search bnode;
    while(continue_sign) {
        continue_sign = false;
        m_read(tmp, &bnode, BNODE_METADATA_FOR_SEARCH_SIZE);
        if(check_match_height(key, bnode.key, bnode.height)) {
            idx = lookup_next_bit_chunk(key, bnode.height);
            addr = tmp->children[idx];
            if(valid_pptr(addr)) {
                if(addr.data_type == B_NODE_DATA_TYPE) {
                    tmp = pptr_to_mbptr(addr);
                    continue_sign = true;
                }
                // Else: Search head into Pnode
            }
            else {
                // This direction is empty
                addr = mbptr_to_pptr(tmp);
            }
        }
        else {
            // Exist a mismatch in the key, indicating the search ends in the middle of an edge
            if(mismatch_return_parent) addr = mbptr_to_pptr(load_node_parent(bnode.parent));
            else addr = mbptr_to_pptr(tmp);
        }
    };
    addr.info = (int8_t)idx;
    return addr;
}

#ifdef SEARCH_TEST_ON
static inline uint64_t p_search(mPptr addr, uint64_t key) {
    uint64_t tmp_key;
#ifdef DPU_KEYS_STORED_IN_PNODE
    uint64_t res = addr->keys[0];
#else
    vectorT vec = addr->v[0];
    uint64_t res = coord_to_key(&vec);
#endif
    int height = maximum_match_height(res, key), num = addr->num, tmp_height;
    for(int i = 1; i < num; i++) {
#ifdef DPU_KEYS_STORED_IN_PNODE
        tmp_key = addr->keys[i];
#else
        vec = addr->v[i];
        tmp_key = coord_to_key(&vec);
#endif
        tmp_height = maximum_match_height(key, tmp_key);
        if(tmp_height > height) {
            res = tmp_key;
            height = tmp_height;
        }
    }
    return res;
}
#endif


/* ----------------- Insert New Vectors ----------------- */

#ifdef INSERT_NODE_ON

#define INSERT_WRAM_KEY_BUF_SIZE (32)

/* Auxiliary functions */

static inline void p_insert_naive(mPptr addr, int8_t num, mpvector vec) {
    bool insert_mode = num <= (LEAF_SIZE >> 1);
    int8_t i = 0;
    int height;
    Pnode pnode;
    if(insert_mode) m_read(addr, &pnode, PNODE_METADATA_SIZE);
    else m_read(addr, &pnode, sizeof(Pnode));
    vectorT *vec_pt = pnode.v + pnode.num;
    uint64_t key_tmp;
#ifdef DPU_KEYS_STORED_IN_PNODE
    uint64_t *key_addr = pnode.keys + pnode.num;
#endif
    m_read(vec, vec_pt, S64(MULTIPLY_NR_DIMENSION(num)));
    if(pnode.num == 0) {
        i = 1;
        vec_pt++;
        pnode.height = 64;
        pnode.key = coord_to_key(pnode.v);
        pnode.box_min = pnode.v[0];
        pnode.box_max = pnode.v[0];
#ifdef DPU_KEYS_STORED_IN_PNODE
        *key_addr = pnode.key;
        key_addr++;
#endif
    }
    for(; i < num; i++, vec_pt++) {
        key_tmp = coord_to_key(vec_pt);
#ifdef DPU_KEYS_STORED_IN_PNODE
        *key_addr = key_tmp;
        key_addr++;
#endif
        height = maximum_match_height(key_tmp, pnode.key);
        if(height < pnode.height) pnode.height = height;
        vector_min(vec_pt, &pnode.box_min);
        vector_max(vec_pt, &pnode.box_max);
    }
    pnode.key = prune_tail_bits(pnode.key, pnode.height);
    if(insert_mode) {
        m_write(pnode.v + pnode.num, addr->v + pnode.num, S64(MULTIPLY_NR_DIMENSION(num)));
#ifdef DPU_KEYS_STORED_IN_PNODE
        m_write(pnode.keys + pnode.num, addr->keys + pnode.num, S64(num));
#endif
        pnode.num += num;
        m_write(&pnode, addr, PNODE_METADATA_SIZE);
    }
    else {
        pnode.num += num;
        for(i = pnode.num; i < LEAF_SIZE; i++) {
#ifdef DPU_KEYS_STORED_IN_PNODE
            pnode.keys[i] = 0;
#endif
            vector_ones(pnode.v + i, 0);
        }
        m_write(&pnode, addr, sizeof(Pnode));
    }
}

typedef struct int64_pair {
    int64_t begin;
    int64_t end;
    mBptr parent;
    int64_t idx;
} int64_pair __attribute__((aligned (8)));


/* Maintain ancestor counters: Max subtree size pruned by MAX_RANGE_QUERY_SIZE to avoid locks */
static inline void maintain_ancestor_counter(mBptr addr, int num) {
    int lock_idx;
    int64_t subtree_size_1 = 0, subtree_size_2 = 0;
    mBptr parent;
    bool continue_signal = true;
    while(continue_signal && addr != INVALID_MBPTR) {
        lock_idx = bnode_mutex_hash(addr);
        mutex_pool_lock(&bnode_lock_pool, lock_idx);
        parent = load_node_parent(addr->parent);
        subtree_size_1 = addr->subtree_size;
        subtree_size_1 += num;
        addr->subtree_size = subtree_size_1;
        mutex_pool_unlock(&bnode_lock_pool, lock_idx);
        subtree_size_2 = (parent == INVALID_MBPTR ? 0 : parent->subtree_size);
        continue_signal = (subtree_size_1 - num < MAX_RANGE_QUERY_SIZE) || (subtree_size_2 < MAX_RANGE_QUERY_SIZE);
        addr = parent;
    }
}

/* Main function for insert vectors */
static void p_insert(mPptr addr, int idx, int num, mpvector vec, mpvoid buf, int buf_size, uint64_t *key_buf_wram) {
    int addr_num = addr->num;
    if(num + addr_num <= LEAF_SIZE) {
        p_insert_naive(addr, num, vec);
    }
    else {
        // Vector and key buffer takes 75% of the buf space
        mpvector vec_buf = (mpvector)buf;
        mpuint64_t key_buf = (mpuint64_t)(buf + buf_size * 3 * NR_DIMENSION / 4 / (NR_DIMENSION + 1));
        __mram_ptr struct int64_pair *stack_buf = (__mram_ptr struct int64_pair*)(buf + buf_size * 3 / 4);
        // The pair queue owns the last quarter of buf; it used to run on into the next tasklet's scratch
        __mram_ptr struct int64_pair *stack_buf_end = stack_buf + (buf_size / 4) / sizeof(struct int64_pair);
        int total_num = addr_num + num;
        IN_DPU_ASSERT(total_num <= INSERT_SCRATCH_CAPACITY, "p_insert: task exceeds MRAM scratch\n");
        int i, j, k;
        bool use_wram_key_buf = total_num <= INSERT_WRAM_KEY_BUF_SIZE;

        // Assume input vectors from the tasks are already sorted on CPU
        // Sort input vectors and current vectors
        {
            int16_t key_idx[LEAF_SIZE];
            for(int i = 0; i < addr_num; i++) key_idx[i] = i;
            uint64_t key1, key2;
            vectorT tmp_vec;

            for(i = 0; i < addr_num - 1; i++) {
#ifdef DPU_KEYS_STORED_IN_PNODE
                key1 = addr->keys[key_idx[i]];
#else
                tmp_vec = addr->v[key_idx[i]];
                key1 = coord_to_key(&tmp_vec);
#endif
                for(j = i + 1; j < addr_num; j++) {
#ifdef DPU_KEYS_STORED_IN_PNODE
                    key2 = addr->keys[key_idx[j]];
#else
                    tmp_vec = addr->v[key_idx[j]];
                    key2 = coord_to_key(&tmp_vec);
#endif
                    if(key1 > key2) {
                        // Swap
                        key1 = key2;
                        key2 = key_idx[j]; key_idx[j] = key_idx[i]; key_idx[i] = key2;
                    }
                }
            }

            uint64_t key_idx_j;
#ifndef DPU_KEYS_STORED_IN_PNODE
            vectorT tmp_vec_1;
#endif
            for(i = 0, j = 0, k = 0; i < num; i++, k++) {
                tmp_vec = vec[i];
                key1 = coord_to_key(&tmp_vec);
#ifdef DPU_KEYS_STORED_IN_PNODE
                while(j < addr_num) {
                    key_idx_j = addr->keys[key_idx[j]];
                    if(key_idx_j > key1) break;
                    if(use_wram_key_buf) key_buf_wram[k] = key_idx_j;
                    else key_buf[k] = key_idx_j;
                    vec_buf[k] = addr->v[key_idx[j]];
                    j++; k++;
                }
#else
                while(j < addr_num) {
                    tmp_vec_1 = addr->v[key_idx[j]];
                    key_idx_j = coord_to_key(&tmp_vec_1);
                    if(key_idx_j > key1) break;
                    if(use_wram_key_buf) key_buf_wram[k] = key_idx_j;
                    else key_buf[k] = key_idx_j;
                    vec_buf[k] = tmp_vec_1;
                    j++; k++;
                }
#endif
                vec_buf[k] = tmp_vec;
                if(use_wram_key_buf) key_buf_wram[k] = key1;
                else key_buf[k] = key1;
            }
            while(j < addr_num) {
#ifdef DPU_KEYS_STORED_IN_PNODE
                vec_buf[k] = addr->v[key_idx[j]];
                if(use_wram_key_buf) key_buf_wram[k] = addr->keys[key_idx[j]];
                else key_buf[k] = addr->keys[key_idx[j]];
#else
                tmp_vec_1 = addr->v[key_idx[j]];
                vec_buf[k] = tmp_vec_1;
                if(use_wram_key_buf) key_buf_wram[k] = coord_to_key(&tmp_vec_1);
                else key_buf[k] = coord_to_key(&tmp_vec_1);
#endif
                j++; k++;
            }
        }

        // Recursively build the new trie

        int32_t child_start[DB_SIZE], child_end[DB_SIZE];
        int vec_start, vec_end, tmp_int, step;
        int8_t height, tmp_idx;
        uint64_t key;
        mPptr p_addr = addr;
        mBptr b_addr = INVALID_MBPTR;
        __dma_aligned Bnode bnode;
        __dma_aligned Pnode pnode;
        __mram_ptr struct int64_pair *pair_start = stack_buf, *pair_end = stack_buf + 1;
        struct int64_pair pair_content;
        pair_content = (struct int64_pair) {
            .begin = 0,
            .end = total_num - 1,
            .parent = load_node_parent(addr->parent),
            .idx = idx,
        };
        *pair_start = pair_content;
        for(i = 0; i < DB_SIZE; i++) child_end[i] = -1;
        for(i = 1; i < LEAF_SIZE; i++) {
#ifdef DPU_KEYS_STORED_IN_PNODE
            pnode.keys[i] = 0;
#endif
            vector_ones(pnode.v + i, 0);
        }
        vectorT tmp_vec;

        while(pair_start != pair_end) {
            pair_content = *pair_start;
            vec_start = pair_content.begin;
            vec_end = pair_content.end;
            for(i = 0; i < DB_SIZE; i++) child_start[i] = INT32_MAX >> 1;
            key = (use_wram_key_buf ? key_buf_wram[vec_start] : key_buf[vec_start]);
            height = maximum_match_height(key, (use_wram_key_buf ? key_buf_wram[vec_end] : key_buf[vec_end]));
            // Allocate new B node
            b_addr = alloc_new_bnode();
            bnode.parent = store_node_parent(pair_content.parent);
            pair_content.parent->children[pair_content.idx] = mbptr_to_pptr(b_addr);
            bnode.height = height;
            key = prune_tail_bits(key, height);
            bnode.key = key;
            bnode.subtree_size = vec_end - vec_start + 1;
            bnode.box_min = key_to_coord(key, false);
            key |= ((((uint64_t)1) << (64 - height)) - (uint64_t)1);
            bnode.box_max = key_to_coord(key, true);

            // Count children point num
            step = vec_end - vec_start + 1;
            if(step < DB_SIZE) {
                for(i = vec_start; i <= vec_end; i++) {
                    tmp_idx = lookup_next_bit_chunk((use_wram_key_buf ? key_buf_wram[i] : key_buf[i]), height);
                    if(child_start[tmp_idx] > i) child_start[tmp_idx] = i;
                    child_end[tmp_idx] = i;
                }
            }
            else {
                step = step >> (DB_SIZE_LOG - 1);
                i = vec_start;
                tmp_idx = lookup_next_bit_chunk((use_wram_key_buf ? key_buf_wram[i] : key_buf[i]), height);
                if(child_start[tmp_idx] > i) child_start[tmp_idx] = i;
                child_end[tmp_idx] = i;
                for(i += step; i <= vec_end; i += step) {
                    tmp_idx = lookup_next_bit_chunk((use_wram_key_buf ? key_buf_wram[i] : key_buf[i]), height);
                    child_end[tmp_idx] = i;
                    if(child_start[tmp_idx] > i) {
                        for(j = i - step + 1; j <= i; j++) {
                            tmp_idx = lookup_next_bit_chunk((use_wram_key_buf ? key_buf_wram[j] : key_buf[j]), height);
                            if(child_start[tmp_idx] > j) child_start[tmp_idx] = j;
                            child_end[tmp_idx] = j;
                        }
                    }
                }
                for(i = i - step + 1; i <= vec_end; i++) {
                    tmp_idx = lookup_next_bit_chunk((use_wram_key_buf ? key_buf_wram[i] : key_buf[i]), height);
                    if(child_start[tmp_idx] > i) child_start[tmp_idx] = i;
                    child_end[tmp_idx] = i;
                }
            }
            // Construct
            for(i = 0; i < DB_SIZE; i++) {
                k = child_start[i];
                tmp_int = child_end[i] - k + 1;
                if(tmp_int > LEAF_SIZE) {
                    // Need further split
                    pair_content = (struct int64_pair) {
                        .begin = k,
                        .end = child_end[i],
                        .parent = b_addr,
                        .idx = i,
                    };
                    *pair_end = pair_content;
                    pair_end++; if(pair_end >= stack_buf_end) pair_end = stack_buf;
                    bnode.children[i] = null_pptr;
                }
                else if(tmp_int > 0) {
                    // Allocate new P node
                    if(p_addr == INVALID_MPPTR) p_addr = alloc_new_pnode();
                    pnode.num = tmp_int;
                    height = maximum_match_height(
                        (use_wram_key_buf ? key_buf_wram[k] : key_buf[k]),
                        (use_wram_key_buf ? key_buf_wram[child_end[i]] : key_buf[child_end[i]])
                    );
                    pnode.height = height;
                    pnode.key = prune_tail_bits((use_wram_key_buf ? key_buf_wram[k] : key_buf[k]), height);
                    pnode.parent = store_node_parent(b_addr);
                    bnode.children[i] = mpptr_to_pptr(p_addr);
                    vector_ones(&pnode.box_min, INT64_MAX);
                    vector_ones(&pnode.box_max, INT64_MIN);
                    for(j = 0; j < tmp_int; j++) {
#ifdef DPU_KEYS_STORED_IN_PNODE
                        pnode.keys[j] = (use_wram_key_buf ? key_buf_wram[j + k] : key_buf[j + k]);
#endif
                        tmp_vec = vec_buf[j + k];
                        pnode.v[j] = tmp_vec;
                        vector_min(&tmp_vec, &pnode.box_min);
                        vector_max(&tmp_vec, &pnode.box_max);
                    }
                    m_write(&pnode, p_addr, sizeof(Pnode));
                    p_addr = INVALID_MPPTR;
                }
                else bnode.children[i] = null_pptr;
            }
            m_write(&bnode, b_addr, sizeof(Bnode));
            pair_start++; if(pair_start >= stack_buf_end) pair_start = stack_buf;
        }
    }
}

static inline mBptr b_insert(mBptr addr, int idx, int num, mpvector vec, mpvoid buf, int buf_size, uint64_t *key_buf_wram) {
    // Assume input vectors from the tasks are already sorted on CPU
    mPptr p_addr;
    mBptr parent;
    int lock_idx = bnode_mutex_hash(addr);
    mutex_pool_lock(&bnode_lock_pool, lock_idx);
    addr->subtree_size += num;
    parent = load_node_parent(addr->parent);
    mutex_pool_unlock(&bnode_lock_pool, lock_idx);
    if(!get_mbptr_child_exist(addr, idx)) {
        p_addr = alloc_new_pnode();
        addr->children[idx] = mpptr_to_pptr(p_addr);
        p_addr->parent = store_node_parent(addr);
        p_insert(p_addr, idx, num, vec, buf, buf_size, key_buf_wram);
    }
    else {
        // Allocate new B node
        mBptr b_addr;
        vectorT tmp_vec = vec[0];
        int8_t height_min, height_tmp, idx_tmp, idx2;
        mBptr original_b;
        pptr original_b_addr;
        uint64_t key_original_b, key1;
        int32_t ll = 0, rr = num - 1, i, j, tmp_int, range_num;
        {
            key1 = coord_to_key(&tmp_vec);
            tmp_vec = vec[rr];
            uint64_t key2 = coord_to_key(&tmp_vec);
            original_b_addr = addr->children[idx];
            original_b = pptr_to_mbptr(original_b_addr);
            key_original_b = original_b->key;
            height_min = maximum_match_height(key1, key_original_b);
            height_tmp = maximum_match_height(key2, key_original_b);
            if(height_min > height_tmp) height_min = height_tmp;
            else height_tmp = height_min;
        }
        while(rr >= ll) {
            height_tmp = lookup_next_bit_chunk(key_original_b, height_min);
            b_addr = alloc_new_bnode();
            for(int8_t i = 0; i < DB_SIZE; i++) b_addr->children[i] = null_pptr;
            b_addr->height = height_min;
            key1 = prune_tail_bits(key_original_b, height_min);
            b_addr->key = key1;
            b_addr->parent = store_node_parent(addr);
            addr->children[idx] = mbptr_to_pptr(b_addr);
            b_addr->children[height_tmp] = original_b_addr;
            b_addr->box_min = key_to_coord(key1, false);
            key1 |= ((((uint64_t)1) << (64 - height_min)) - (uint64_t)1);
            b_addr->box_max = key_to_coord(key1, true);
            lock_idx = bnode_mutex_hash(original_b);
            mutex_pool_lock(&bnode_lock_pool, lock_idx);
            original_b->parent = store_node_parent(b_addr);
            b_addr->subtree_size = original_b->subtree_size + rr - ll + 1;
            mutex_pool_unlock(&bnode_lock_pool, lock_idx);

            // Build children
            j = -1;
            tmp_int = ll;
            tmp_vec = vec[ll];
            key1 = coord_to_key(&tmp_vec);
            idx_tmp = lookup_next_bit_chunk(key1, height_min);
            range_num = 1;
            if(idx_tmp == height_tmp) j = ll;
            else if(idx_tmp < height_tmp) ll++;
            for(i = tmp_int + 1; i <= rr; i++) {
                tmp_vec = vec[i];
                key1 = coord_to_key(&tmp_vec);
                idx2 = lookup_next_bit_chunk(key1, height_min);
                if(idx2 == height_tmp) j = i;
                else if(idx2 < height_tmp) ll++;
                if(idx2 == idx_tmp) range_num++;
                else {
                    if(idx_tmp != height_tmp) {
                        p_addr = alloc_new_pnode();
                        b_addr->children[idx_tmp] = mpptr_to_pptr(p_addr);
                        p_addr->parent = store_node_parent(b_addr);
                        p_insert(p_addr, idx_tmp, range_num, vec + tmp_int, buf, buf_size, key_buf_wram);
                    }
                    tmp_int = i;
                    idx_tmp = idx2;
                    range_num = 1;
                }
            }
            if(idx_tmp != height_tmp) {
                p_addr = alloc_new_pnode();
                b_addr->children[idx_tmp] = mpptr_to_pptr(p_addr);
                p_addr->parent = store_node_parent(b_addr);
                p_insert(p_addr, idx_tmp, range_num, vec + tmp_int, buf, buf_size, key_buf_wram);
            }
            rr = j;
            idx = height_tmp;
            addr = b_addr;
            height_min += DB_SIZE_LOG;
        }
    }
    return parent;
}

#endif


/* ----------------- Fetch Entire Node ----------------- */

#ifdef FETCH_NODE_ON
static inline void fetch_single_node(pptr addr, int i) {
    Fetch_node_reply tsr;
    tsr.addr = addr;
    tsr.len = 0;
    for(int j = 0; j < LEAF_SIZE; j++) tsr.keys[j] = INVALID_KEY;
    if(addr.data_type == P_NODE_DATA_TYPE) {
        mPptr p_addr = pptr_to_mpptr(addr);
        tsr.parent = mbptr_to_pptr(load_node_parent(p_addr->parent));
        tsr.key = p_addr->key;
        tsr.height = p_addr->height;
        tsr.len = p_addr->num;
#ifdef DPU_KEYS_STORED_IN_PNODE
        for(int j = 0; j < tsr.len; j++) tsr.keys[j] = p_addr->keys[j];
#else
        vectorT vec;
        for(int j = 0; j < tsr.len; j++) {
            vec = p_addr->v[j];
            tsr.keys[j] = coord_to_key(&vec);
        }
#endif
        for(int j = tsr.len; j < LEAF_SIZE; j++) tsr.keys[j] = PPTR_TO_U64(null_pptr);
    }
    else if(addr.data_type == B_NODE_DATA_TYPE) {
        mBptr b_addr = pptr_to_mbptr(addr);
        tsr.parent = mbptr_to_pptr(load_node_parent(b_addr->parent));
        tsr.key = b_addr->key;
        tsr.height = b_addr->height;
        tsr.len = DB_SIZE;
        pptr tmp_addr;
        for(int j = 0; j < DB_SIZE; j++) {
            tmp_addr = b_addr->children[j];
            tsr.keys[j] = PPTR_TO_U64(tmp_addr);
        }
    }
    push_fixed_reply(i, &tsr);
}
#endif
