#pragma once
#include "debug_settings.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

// Always-on bound check for the IO buffers: an overflow there corrupts host memory silently otherwise.
#define CHECK_BOUND(x, ...)                                        \
    if (!(x)) {                                                    \
        std::fprintf(stderr, "%s:%d: buffer bound exceeded: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__);                         \
        std::fprintf(stderr, "\n");                               \
        std::abort();                                              \
    }

#ifdef ZHAOYW_CPU_DEBUG
#define ASSERT(x) assert(x)
#define ASSERT_EXEC(x, y) \
    if (!(x)) {           \
        y;                \
        assert(false);    \
    }
#else
#define ASSERT(x) {(void)(x);}
#define ASSERT_EXEC(x, y) {(void)(x);}
#endif
