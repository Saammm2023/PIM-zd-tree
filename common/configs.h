#pragma once

#define NR_DIMENSION (3)

/* Size of the batch size */
#define BATCH_SIZE (2100000)

/* Maximum sizes of box range and kNN queries */
#define MAX_RANGE_QUERY_SIZE (1000)
#define MAX_KNN_SIZE (125)

/* Size threshold of a leaf node in the tree */
#define LEAF_SIZE (16)

/* Block size of internal nodes */
#define DB_SIZE (16)  // size of the data block
#define DB_SIZE_LOG (4)  // DB_SIZE = 2 ^ DB_SIZE_LOG
#define DB_SIZE_LOG_LOG (2)  // DB_SIZE_LOG = 2 ^ DB_SIZE_LOG_LOG
#define DB_SIZE_PLUS_ONE (17)  // DB_SIZE_PLUS_ONE = DB_SIZE + 1

/* Coord <-----> Key */
// 64-bit int
// #define COORD_MAX (INT64_MAX)
// #define KEY_START_POS (2)  // The first position where in coord start to convert into key [1, 64]
// #define KEY_START_POS_MINUS_ONE (1)  // KEY_START_POS_MINUS_ONE = KEY_START_POS - 1
// // 32-bit int
#define COORD_MAX ((int64_t)INT32_MAX)
#define KEY_START_POS (34)  // The first position where in coord start to convert into key [1, 64]
#define KEY_START_POS_MINUS_ONE (33)  // KEY_START_POS_MINUS_ONE = KEY_START_POS - 1

#define LX_NORM (1)

/* MRAM scratch shared by the tasklets, MRAM_BUFFER_SIZE / NR_TASKLETS each */
#define MRAM_BUFFER_SIZE (3 << 19) // 1.5 MB

/* p_insert keeps the vectors and keys of one insert task plus the leaf's existing points
   (at most LEAF_SIZE) in 3/4 of the tasklet's scratch: 3072 points with 12 tasklets and 3 dimensions.
   The host never sends an insert task longer than MAX_INSERT_TASK_LEN. */
#define INSERT_SCRATCH_CAPACITY (3 * (MRAM_BUFFER_SIZE / NR_TASKLETS) / (32 * (NR_DIMENSION + 1)))
#define MAX_INSERT_TASK_LEN (INSERT_SCRATCH_CAPACITY - LEAF_SIZE)

/* MRAM budget (64 MB per DPU). Static data ends at about 52.4 MB + MAX_TASK_BUFFER_SIZE_PER_DPU, because
   send_varlen_buffer_tmp is itself MAX_TASK_BUFFER_SIZE_PER_DPU bytes. The heap after it holds the receive area
   (MAX_TASK_BUFFER_SIZE_PER_DPU) and then the reply area, which gets whatever is left:
   about 14.05 MB - 2 * MAX_TASK_BUFFER_SIZE_PER_DPU.
   Lookups and the search phase of inserts reply 8 B per operation, and inserts send 24 B per point, so
   5.75 MB balances the two at about 245k operations per DPU per batch. The original 6.4 MB left only
   0.93 MB for replies (about 119k operations), which made larger batches fault. */
#define MAX_TASK_BUFFER_SIZE_PER_DPU (5888 << 10) // 5.75 MB
#define MAX_TASK_COUNT_PER_DPU_PER_BLOCK ((100 << 10) >> 3) // 100 KB = 12.5 K
/* insert() sends one task per target leaf; leaf groups beyond this on one DPU go to another round */
#define MAX_INSERT_TASKS_PER_DPU (MAX_TASK_COUNT_PER_DPU_PER_BLOCK - 1)
