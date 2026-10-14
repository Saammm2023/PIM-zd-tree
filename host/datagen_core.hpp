#pragma once
// Data generator core, with no PIM-zd-tree dependencies: the PIM host (host/datagen.hpp) and the CPU
// baseline (cpu_baseline/) both include it, so both see exactly the same points.
//
// Uniform points, or skewed points drawn from Zipf-weighted clusters. Every point is a pure function of
// (seed, stream, index), so a run is reproducible for a given seed and independent of thread scheduling.
// Streams keep warm-up and timed batches from repeating points: stream 1 = warm-up inserts, stream 2 = timed
// inserts.
//
// zipf: nr_clusters cluster centres are placed uniformly. Cluster c (rank c + 1) is chosen with probability
// proportional to 1 / (c + 1)^s, and the point is placed around its centre with a triangular distribution of
// half-width 2^cluster_width_log2 per dimension, wrapping around the coordinate space. PIM-zd-tree keys use
// the top 21 bits of each 31-bit coordinate (coord >> 10), so cluster_width_log2 should stay well above 10,
// or points start to share keys.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace datagen {

constexpr int kDim = 3;
constexpr int64_t kCoordMax = 0x7fffffff;  // coordinates are in [0, 2^31)
using coords = std::array<int64_t, kDim>;

enum class Dist { uniform, zipf };

struct Config {
    Dist dist = Dist::uniform;
    double zipf_s = 1.0;
    int nr_clusters = 1024;
    int cluster_width_log2 = 22;
    uint64_t seed = 1;
};

inline Config cfg;
inline std::vector<double> cluster_cdf;  // cumulative Zipf weights, normalised to 1
inline std::vector<coords> cluster_centres;

inline const char* dist_name() { return cfg.dist == Dist::zipf ? "zipf" : "uniform"; }

// Same function as parlay::hash64, copied so the result does not depend on the parlaylib version
inline uint64_t hash64(uint64_t u) {
    uint64_t v = u * 3935559000370003845ull + 2691343689449507681ull;
    v ^= v >> 21;
    v ^= v << 37;
    v ^= v >> 4;
    v *= 4768777513237032717ull;
    v ^= v << 20;
    v ^= v >> 41;
    v ^= v << 5;
    return v;
}

inline uint64_t rnd(uint64_t stream, uint64_t index, uint64_t salt) {
    return hash64(cfg.seed * 0x9E3779B97F4A7C15ull + (stream << 48) + (salt << 40) + index);
}
inline int64_t rnd_coord(uint64_t stream, uint64_t index, uint64_t salt) {
    return (int64_t)(rnd(stream, index, salt) & kCoordMax);
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
            for (int k = 0; k < kDim; k++) cluster_centres[c][k] = rnd_coord(0, c, 1 + k);
        }
    }
    return true;
}

// Centre plus a triangular offset in [-w, w]. Wraps around instead of clamping: clamping piles points onto
// the faces of the space, where many share coordinates and keys.
inline int64_t around(int64_t centre, uint64_t stream, uint64_t index, uint64_t salt) {
    const int64_t w = (int64_t)1 << cfg.cluster_width_log2;
    uint64_t r = rnd(stream, index, salt);
    int64_t a = (int64_t)(r & 0xffffffffull) % (w + 1);
    int64_t b = (int64_t)(r >> 32) % (w + 1);
    return (centre + a - b) & kCoordMax;
}

inline coords point_coords(uint64_t stream, uint64_t index) {
    coords v;
    if (cfg.dist == Dist::uniform) {
        for (int k = 0; k < kDim; k++) v[k] = rnd_coord(stream, index, 1 + k);
        return v;
    }
    double u = (double)(rnd(stream, index, 7) >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
    int c = (int)(std::upper_bound(cluster_cdf.begin(), cluster_cdf.end(), u) - cluster_cdf.begin());
    c = std::min(c, cfg.nr_clusters - 1);
    for (int k = 0; k < kDim; k++) v[k] = around(cluster_centres[c][k], stream, index, 1 + k);
    return v;
}

}  // namespace datagen
