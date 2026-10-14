#pragma once

#include <stdlib.h>
#include "debug_settings.h"
#include "task_framework_common.h"

// Always-on capacity check: prints the reason to the DPU log and faults the DPU. The host prints the
// DPU logs when a launch fails (see exec() in task_framework_host.hpp).
#define ALWAYS_IN_DPU_ASSERT(x, ...)          \
    {                                         \
        if (!(x)) {                           \
            printf("tasklet-%d: ", me());     \
            printf(__VA_ARGS__);              \
            abort();                          \
        }                                     \
    }

#ifdef ZHAOYW_DEBUG_TASK
#define TASK_IN_DPU_ASSERT(x, y) IN_DPU_ASSERT(x, y)
#define TASK_IN_DPU_ASSERT_EXEC(x, y) IN_DPU_ASSERT_EXEC(x, y)
#else
#define TASK_IN_DPU_ASSERT(x, y) {}
#define TASK_IN_DPU_ASSERT_EXEC(x, y) {}
#endif

#ifdef ZHAOYW_DEBUG_SPACE
#define SPACE_IN_DPU_ASSERT(x, y) IN_DPU_ASSERT(x, y)
#else
#define SPACE_IN_DPU_ASSERT(x, y) {}
#endif

extern __mram_ptr uint8_t* send_buffer;

#ifdef ZHAOYW_DEBUG_CORE
#define EXIT() {printf("tasklet-%d: EXIT()!", me()); (*(__mram_ptr int64_t*)send_buffer) = DPU_BUFFER_ERROR; exit(0); }
#else
#define EXIT() {}
#endif

#ifdef ZHAOYW_DEBUG
#define IN_DPU_ASSERT(x, y) {if(!(x)){printf("tasklet-%d: %s", me(), (y));(*(__mram_ptr int64_t*)send_buffer) = DPU_BUFFER_ERROR;exit(0);}}
#define IN_DPU_ASSERT_EXEC(x, y) {if(!(x)){y;(*(__mram_ptr int64_t*)send_buffer) = DPU_BUFFER_ERROR;exit(0);}}
#define DEBUG(x) {x;}
#else
#define IN_DPU_ASSERT(x, y) {}
#define IN_DPU_ASSERT_EXEC(x, y) {}
#define DEBUG(x) {}
#endif