#pragma once
// Data generators for the experiments: uniform points, or skewed points drawn from Zipf-weighted clusters.
//
// Every point is a pure function of (seed, stream, index), so a run is reproducible for a given --seed
// and independent of thread scheduling. Streams keep warm-up and timed batches from repeating points.
//
// zipf: nr_clusters cluster centres are placed uniformly. Cluster c (rank c + 1) is chosen with probability
// proportional to 1 / (c + 1)^s, and the point is placed around its centre with a triangular distribution of
// half-width 2^cluster_width_log2 per dimension, wrapping around the coordinate space. Keys use the top 21 bits of each 31-bit coordinate
// (coord >> 10), so cluster_width_log2 should stay well above 10, or points start to share keys.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/utilities.h>

#include "common.h"
#include "utils.h"

namespace datagen {

enum class Dist { uniform, zipf };

struct Config {
    Dist dist = Dist::uniform;
    double zipf_s = 1.0;
    int nr_clusters = 1024;
    int cluster_width_log2 = 22;
    uint64_t seed = 1;
};

inline Config cfg;
inline std::vector<double> cluster_cdf;      // cumulative Zipf weights, normalised to 1
inline std::vector<vectorT> cluster_centres;

inline const char* dist_name() { return cfg.dist == Dist::zipf ? "zipf" : "uniform"; }

inline uint64_t rnd(uint64_t stream, uint64_t index, uint64_t salt) {
    return parlay::hash64(cfg.seed * 0x9E3779B97F4A7C15ull + (stream << 48) + (salt << 40) + index);
}
inline COORD rnd_coord(uint64_t stream, uint64_t index, uint64_t salt) {
    return (COORD)(rnd(stream, index, salt) & COORD_MAX);
}

inline bool init(const std::string& dist, double zipf_s, int nr_clusters, int cluster_width_log2,
                 uint64_t seed) {
    if (dist == "uniform") cfg.dist = Dist::uniform;
    else if (dist == "zipf") cfg.dist = Dist::zipf;
    else return false;
    cfg.zipf_s = zipf_s;
    cfg.nr_clusters = std::max(1, nr_clusters);
    cfg.cluster_width_log2 = cluster_width_log2;
    cfg.seed = seed;
    if (cfg.dist == Dist::zipf) {
        cluster_cdf.resize(cfg.nr_clusters);
        cluster_centres.resize(cfg.nr_clusters);
        double sum = 0;
        for (int c = 0; c < cfg.nr_clusters; c++) {
            sum += 1.0 / std::pow((double)(c + 1), cfg.zipf_s);
            cluster_cdf[c] = sum;
        }
        for (int c = 0; c < cfg.nr_clusters; c++) {
            cluster_cdf[c] /= sum;
#if NR_DIMENSION == 2
            cluster_centres[c].x = rnd_coord(0, c, 1);
            cluster_centres[c].y = rnd_coord(0, c, 2);
#elif NR_DIMENSION == 3
            cluster_centres[c].x = rnd_coord(0, c, 1);
            cluster_centres[c].y = rnd_coord(0, c, 2);
            cluster_centres[c].z = rnd_coord(0, c, 3);
#else
            for (int k = 0; k < NR_DIMENSION; k++) cluster_centres[c].x[k] = rnd_coord(0, c, 1 + k);
#endif
        }
    }
    return true;
}

// Centre plus a triangular offset in [-w, w]
inline COORD around(COORD centre, uint64_t stream, uint64_t index, uint64_t salt) {
    const int64_t w = (int64_t)1 << cfg.cluster_width_log2;
    uint64_t r = rnd(stream, index, salt);
    int64_t a = (int64_t)(r & 0xffffffffull) % (w + 1);
    int64_t b = (int64_t)(r >> 32) % (w + 1);
    // Wrap around instead of clamping: clamping piles points onto the faces of the space, where many
    // share coordinates and keys
    int64_t v = ((int64_t)centre + a - b) & COORD_MAX;
    return (COORD)v;
}

inline vectorT point(uint64_t stream, uint64_t index) {
    vectorT v;
    if (cfg.dist == Dist::uniform) {
#if NR_DIMENSION == 2
        v.x = rnd_coord(stream, index, 1);
        v.y = rnd_coord(stream, index, 2);
#elif NR_DIMENSION == 3
        v.x = rnd_coord(stream, index, 1);
        v.y = rnd_coord(stream, index, 2);
        v.z = rnd_coord(stream, index, 3);
#else
        for (int k = 0; k < NR_DIMENSION; k++) v.x[k] = rnd_coord(stream, index, 1 + k);
#endif
        return v;
    }
    double u = (double)(rnd(stream, index, 7) >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
    int c = (int)(std::upper_bound(cluster_cdf.begin(), cluster_cdf.end(), u) - cluster_cdf.begin());
    c = std::min(c, cfg.nr_clusters - 1);
    const vectorT& ctr = cluster_centres[c];
#if NR_DIMENSION == 2
    v.x = around(ctr.x, stream, index, 1);
    v.y = around(ctr.y, stream, index, 2);
#elif NR_DIMENSION == 3
    v.x = around(ctr.x, stream, index, 1);
    v.y = around(ctr.y, stream, index, 2);
    v.z = around(ctr.z, stream, index, 3);
#else
    for (int k = 0; k < NR_DIMENSION; k++) v.x[k] = around(ctr.x[k], stream, index, 1 + k);
#endif
    return v;
}

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
