#pragma once
// PIM-zd-tree side of the data generator: converts datagen_core points to vectorT and measures the
// dataset's skew. See datagen_core.hpp for the distributions.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "common.h"
#include "utils.h"
#include "datagen_core.hpp"

static_assert(NR_DIMENSION == datagen::kDim, "datagen_core generates 3-D points");

namespace datagen {

inline vectorT to_vector(const coords& c) {
    vectorT v;
    v.x = (COORD)c[0];
    v.y = (COORD)c[1];
    v.z = (COORD)c[2];
    return v;
}

inline vectorT point(uint64_t stream, uint64_t index) { return to_vector(point_coords(stream, index)); }

// Gini coefficient of point counts over nr_bins equal ranges of the z-order key space (paper Sec. 7.2
// uses 2048 bins). 0 = perfectly even, close to 1 = all points in a few bins.
inline double key_range_gini(const vectorT* pts, size_t n, int nr_bins) {
    if (n == 0 || nr_bins <= 1) return 0;
    std::vector<uint64_t> counts(nr_bins, 0);
    const unsigned __int128 span = (unsigned __int128)1 << 64;
    for (size_t i = 0; i < n; i++) {
        uint64_t key = coord_to_key(const_cast<vectorT*>(&pts[i]));
        int bin = (int)(((unsigned __int128)key * nr_bins) / span);
        counts[bin]++;
    }
    std::sort(counts.begin(), counts.end());
    long double weighted = 0, total = 0;
    for (int i = 0; i < nr_bins; i++) {
        weighted += (long double)(i + 1) * counts[i];
        total += counts[i];
    }
    return (double)(2 * weighted / (nr_bins * total) - (long double)(nr_bins + 1) / nr_bins);
}

}  // namespace datagen
