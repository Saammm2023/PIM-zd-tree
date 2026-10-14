// CPU shared-memory baselines for the PIM-zd-tree experiments: the Pkd-tree and the zd-tree from
// SpaceTreeLib (PSI), run on the same points and the same workload as build*/zd_tree_host.
//
// Workload (same flags and meaning as the PIM host):
//   warm-up: --insert-round (-I) batches of --insert-batch-size (-i) points, stream 1 (first batch: build)
//   timed inserts: --test-round (-r) batches of --test-batch-size (-b) points, stream 2 (with -t 1)
//   lookups (-s 1): --search-batch-size (-S) points sampled from the warm-up batches (the first S / I of
//   each), plus one random point that is not in the data set
//
// Each batch appends a row to <prefix>_cpu.csv: tree,tag,threads,op,batch_idx,ops,seconds,found.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "psi/dependence/splitter.h"
#include "psi/kd_tree.h"
#include "baselines/zdtree_3d/zdtree_3d.hpp"

#include "../host/datagen_core.hpp"

namespace {

struct Options {
    std::string tree = "pkd";
    long insert_batch_size = 50000, insert_round = 10;
    int test_type = 0;
    long test_batch_size = 10000, test_round = 2;
    int search_type = 0;
    long search_batch_size = 20000;
    std::string dist = "uniform";
    double zipf_s = 1.0;
    int clusters = 1024, cluster_width_log2 = 22, seed = 1;
    std::string csv, tag;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--tree") o.tree = next();
        else if (a == "-i" || a == "--insert-batch-size") o.insert_batch_size = std::atol(next());
        else if (a == "-I" || a == "--insert-round") o.insert_round = std::atol(next());
        else if (a == "-t" || a == "--test-type") o.test_type = std::atoi(next());
        else if (a == "-b" || a == "--test-batch-size") o.test_batch_size = std::atol(next());
        else if (a == "-r" || a == "--test-round") o.test_round = std::atol(next());
        else if (a == "-s" || a == "--search-type") o.search_type = std::atoi(next());
        else if (a == "-S" || a == "--search-batch-size") o.search_batch_size = std::atol(next());
        else if (a == "--dist") o.dist = next();
        else if (a == "--zipf-s") o.zipf_s = std::atof(next());
        else if (a == "--clusters") o.clusters = std::atoi(next());
        else if (a == "--cluster-width-log2") o.cluster_width_log2 = std::atoi(next());
        else if (a == "--seed") o.seed = std::atoi(next());
        else if (a == "--stats-csv") o.csv = next();
        else if (a == "--stats-tag") o.tag = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); std::exit(1); }
    }
    return o;
}

using clock_type = std::chrono::steady_clock;

// Point lookup: is this exact point in the tree?
template <class Tree, class Point>
bool lookup(Tree& tree, const Point& p) {
    if constexpr (std::is_same_v<Point, ZD3D::geobase::Point>) {
        // 1-NN search: the point is present when its nearest neighbour is at distance 0 (the wrapper's
        // range_count is not const, so the library's contains() does not compile for it)
        size_t k = 1;
        auto nn = tree.tree.knn_report(k, p);
        return !nn.empty() && nn.top().second == 0;
    } else {
        return tree.contains(p);
    }
}

template <class Tree, class Point>
void run(const Options& o, const char* tree_name) {
    auto make_point = [](const datagen::coords& c) {
        Point p;
        // The bundled zd-tree interleaves only the low 21 bits of each coordinate into its Morton code, so it
        // gets coord >> 10: the same 21 bits per dimension that PIM-zd-tree's keys use
        constexpr int shift = std::is_same_v<Point, ZD3D::geobase::Point> ? 10 : 0;
        for (int k = 0; k < datagen::kDim; k++) p[k] = c[k] >> shift;
        return p;
    };
    auto batch = [&](uint64_t stream, uint64_t first, long n) {
        return parlay::tabulate(n, [&](size_t i) { return make_point(datagen::point_coords(stream, first + i)); });
    };

    FILE* csv = nullptr;
    if (!o.csv.empty()) {
        std::string path = o.csv + "_cpu.csv";
        bool fresh = (std::fopen(path.c_str(), "r") == nullptr);
        csv = std::fopen(path.c_str(), "a");
        if (csv != nullptr && fresh) std::fputs("tree,tag,threads,op,batch_idx,ops,seconds,found\n", csv);
    }
    auto record = [&](const char* op, long idx, long ops, double sec, long found) {
        std::printf("%s %s batch %ld: %ld ops in %.6f s (%.2f Mop/s)%s\n", tree_name, op, idx, ops, sec,
                    ops / sec / 1e6, found >= 0 ? (" found " + std::to_string(found)).c_str() : "");
        if (csv != nullptr)
            std::fprintf(csv, "%s,%s,%zu,%s,%ld,%ld,%.6f,%ld\n", tree_name, o.tag.c_str(),
                         (size_t)parlay::num_workers(), op, idx, ops, sec, found);
    };
    auto seconds_since = [](clock_type::time_point t0) {
        return std::chrono::duration<double>(clock_type::now() - t0).count();
    };

    Tree tree;
    for (long j = 0; j < o.insert_round; j++) {
        auto pts = batch(1, (uint64_t)j * o.insert_batch_size, o.insert_batch_size);
        auto t0 = clock_type::now();
        if (j == 0) tree.build(parlay::make_slice(pts));
        else tree.batch_insert(parlay::make_slice(pts));
        record("warmup_insert", j, o.insert_batch_size, seconds_since(t0), -1);
    }

    if (o.test_type == 1) {
        for (long j = 0; j < o.test_round; j++) {
            auto pts = batch(2, (uint64_t)j * o.test_batch_size, o.test_batch_size);
            auto t0 = clock_type::now();
            tree.batch_insert(parlay::make_slice(pts));
            record("insert", j, o.test_batch_size, seconds_since(t0), -1);
        }
    }

    if (o.search_type == 1) {
        // Same queries as the PIM host: the first S / I points of each warm-up batch, then random points
        long per_batch = o.search_batch_size / o.insert_round;
        long sampled = per_batch * o.insert_round;
        long n = o.search_batch_size + 1;
        auto queries = parlay::tabulate(n, [&](size_t q) {
            if ((long)q < sampled) {
                long j = q / per_batch, i = q % per_batch;
                return make_point(datagen::point_coords(1, (uint64_t)j * o.insert_batch_size + i));
            }
            datagen::coords c;
            for (int k = 0; k < datagen::kDim; k++) c[k] = datagen::rnd_coord(3, q, 1 + k);
            return make_point(c);
        });
        std::atomic<long> found{0};
        auto t0 = clock_type::now();
        parlay::parallel_for(0, n, [&](size_t q) {
            if (lookup(tree, queries[q])) found.fetch_add(1, std::memory_order_relaxed);
        });
        double sec = seconds_since(t0);
        record("lookup", 0, n, sec, found.load());
        std::printf("Lookup: %ld of %ld queries found (%ld were sampled from the data)\n", found.load(), n, sampled);
    }
    if (csv != nullptr) std::fclose(csv);
    tree.delete_tree();
}

}  // namespace

int main(int argc, char** argv) {
    Options o = parse(argc, argv);
    if (!datagen::init(o.dist, o.zipf_s, o.clusters, o.cluster_width_log2, (uint64_t)o.seed)) {
        std::fprintf(stderr, "unknown --dist %s (use uniform or zipf)\n", o.dist.c_str());
        return 1;
    }
    std::printf("CPU baseline: tree=%s threads=%zu dist=%s\n", o.tree.c_str(), (size_t)parlay::num_workers(),
                datagen::dist_name());
    if (o.tree == "pkd") {
        using point = psi::basic_point<long, datagen::kDim>;
        using split_rule = psi::orthogonal_split_rule<psi::max_stretch_dim<point>, psi::object_median<point>>;
        run<psi::kd_tree<psi::tree_traits<point, split_rule>>, point>(o, "pkd-tree");
    } else if (o.tree == "zd") {
        using point = ZD3D::geobase::Point;
        using split_rule = psi::orthogonal_split_rule<psi::rotate_dim<point>, psi::spatial_median<point>>;
        run<ZD3D::zdtree<point, split_rule>, point>(o, "zd-tree");
    } else {
        std::fprintf(stderr, "unknown --tree %s (use pkd or zd)\n", o.tree.c_str());
        return 1;
    }
    return 0;
}
