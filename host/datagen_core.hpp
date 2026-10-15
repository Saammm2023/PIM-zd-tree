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
#include <cstdio>
#include <string>
#include <vector>

namespace datagen {

constexpr int kDim = 3;
constexpr int64_t kCoordMax = 0x7fffffff;  // coordinates are in [0, 2^31)
using coords = std::array<int64_t, kDim>;

enum class Dist { uniform, zipf, file };

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

// Points read with --input (scripts/prepare_points.py writes the format). Stream 1 (warm-up) is the start of the
// file, stream 2 (timed inserts) the points after the first file_warm.
inline std::vector<coords> file_points;
inline uint64_t file_warm = 0;

inline const char* dist_name() {
    return cfg.dist == Dist::zipf ? "zipf" : cfg.dist == Dist::file ? "file" : "uniform";
}

// Binary point file: uint64 n, uint64 dim (3), then n * 3 int64 coordinates in [0, 2^31)
inline bool load_point_file(const std::string& path, std::vector<coords>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { std::perror(path.c_str()); return false; }
    uint64_t hdr[2];
    bool ok = std::fread(hdr, sizeof(uint64_t), 2, f) == 2 && hdr[1] == (uint64_t)kDim;
    if (ok) {
        out.resize(hdr[0]);
        ok = std::fread(out.data(), sizeof(coords), hdr[0], f) == hdr[0];
    }
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "%s: not a 3-D point file\n", path.c_str());
    for (const coords& c : out)
        for (int64_t v : c)
            if (ok && (v < 0 || v > kCoordMax)) {
                std::fprintf(stderr, "%s: coordinate %lld outside [0, 2^31)\n", path.c_str(), (long long)v);
                ok = false;
            }
    return ok;
}

// Use a point file instead of generated data; needs at least warm + timed points
inline bool init_file(const std::string& path, uint64_t warm, uint64_t timed) {
    if (!load_point_file(path, file_points)) return false;
    if (file_points.size() < warm + timed) {
        std::fprintf(stderr, "%s has %zu points, the run needs %llu\n", path.c_str(), file_points.size(),
                     (unsigned long long)(warm + timed));
        return false;
    }
    cfg.dist = Dist::file;
    file_warm = warm;
    return true;
}

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
    if (cfg.dist == Dist::file) return file_points[stream == 1 ? index : file_warm + index];
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

// ---- Lookup queries over the warm-up points (stream 1, indices 0..n-1) ----
// uniform: every inserted point equally likely. zipf: point of popularity rank r (1-based) is asked with
// probability proportional to 1 / r^s, so a few hot keys get many of the queries; ranks are mapped to points by
// a fixed permutation so hot points are scattered over the data set (and over the DPUs).

enum class QueryDist { uniform, zipf, mix };

struct QueryConfig {
    QueryDist dist = QueryDist::uniform;
    double zipf_s = 1.0;
    uint64_t n = 0;     // number of points queries are drawn from
    uint64_t mult = 1;  // rank -> point permutation: (rank * mult + 12345) mod n, mult coprime with n
    double mix_fraction = 0;  // mix: share of queries taken from query_file_points (the rest uniform)
};

inline QueryConfig qcfg;
inline std::vector<double> query_cdf;
inline std::vector<coords> query_file_points;  // e.g. Varden points (paper Fig. 9), not stored in the tree

inline uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) { uint64_t t = a % b; a = b; b = t; }
    return a;
}

inline bool init_queries(const std::string& dist, double zipf_s, uint64_t n) {
    if (dist == "uniform") qcfg.dist = QueryDist::uniform;
    else if (dist == "zipf") qcfg.dist = QueryDist::zipf;
    else if (dist == "mix") qcfg.dist = QueryDist::mix;
    else return false;
    qcfg.zipf_s = zipf_s;
    qcfg.n = n;
    qcfg.mult = 1000003;
    while (gcd_u64(qcfg.mult, n) != 1) qcfg.mult += 2;
    if (qcfg.dist == QueryDist::zipf) {
        query_cdf.resize(n);
        double sum = 0;
        for (uint64_t r = 0; r < n; r++) {
            sum += 1.0 / std::pow((double)(r + 1), zipf_s);
            query_cdf[r] = sum;
        }
        for (auto& v : query_cdf) v /= sum;
    }
    return true;
}

// mix: a fraction of the queries come from a separate point file (spatial hot spots, generally absent from the tree)
inline bool init_query_mix(const std::string& path, double fraction) {
    if (!load_point_file(path, query_file_points) || query_file_points.empty()) return false;
    qcfg.mix_fraction = fraction;
    return true;
}

// Index (into warm-up stream 1) of query q of lookup round r
inline uint64_t query_point_index(uint64_t round, uint64_t q) {
    uint64_t h = rnd(4 + round, q, 9);
    if (qcfg.dist != QueryDist::zipf) return h % qcfg.n;
    double u = (double)(h >> 11) * (1.0 / 9007199254740992.0);
    uint64_t rank = (uint64_t)(std::upper_bound(query_cdf.begin(), query_cdf.end(), u) - query_cdf.begin());
    if (rank >= qcfg.n) rank = qcfg.n - 1;
    return (uint64_t)(((unsigned __int128)rank * qcfg.mult + 12345) % qcfg.n);
}

// Coordinates of query q of lookup round r
inline coords query_point(uint64_t round, uint64_t q) {
    if (qcfg.dist == QueryDist::mix) {
        double u = (double)(rnd(4 + round, q, 11) >> 11) * (1.0 / 9007199254740992.0);
        if (u < qcfg.mix_fraction)
            return query_file_points[rnd(4 + round, q, 12) % query_file_points.size()];
    }
    return point_coords(1, query_point_index(round, q));
}

}  // namespace datagen
