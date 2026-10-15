#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <iostream>
#include <cstdint>
#include <string>
#include <cmath>
#include <chrono>
#include <argparse/argparse.hpp>
#include <climits>

#ifdef USE_PAPI
#include "parlay/papi/papi_util_impl.h"
#endif

#include "host.hpp"
#include "operations.hpp"
#include "datagen.hpp"

using namespace std;

const int maxTopLevelThreads = 3;

/* Argv for main function */
int64_t insert_batch_size;
int insert_round;
int test_batch_size;
int test_round;
int search_batch_size;
int search_type; /* 1: Point search; 2: Box range count; 3: Box fetch; 4: kNN */
int expected_box_size;
bool print_timer;
bool debug_print;
int test_type;
int top_level_threads;
std::string interface_type;
std::string stats_csv_prefix;
std::string stats_tag;
std::string data_dist;
double zipf_s;
int nr_clusters;
int cluster_width_log2;
int data_seed;
std::string placement;
bool verify_each_batch;
std::string query_dist;
std::string input_file;
std::string query_file;
double query_mix;
double query_zipf_s;
int search_round;
bool lookup_dedup;
bool lookup_adaptive;
double adaptive_threshold;
double adaptive_repeat_threshold;
int hot_cache_size;
COORD input_coord_max[NR_DIMENSION];

void host_parse_arguments(int argc, char *argv[]) {
    argparse::ArgumentParser parser("PIM-zd-tree");

    // Insert options
    parser.add_argument("-i", "--insert-batch-size")
        .help("Insert batch size")
        .default_value(50000)
        .scan<'i', int>();
    parser.add_argument("-I", "--insert-round")
        .help("Insert rounds")
        .default_value(10)
        .scan<'i', int>();

    // Test options
    parser.add_argument("-t", "--test-type")
        .help("Test type")
        .default_value(0)
        .scan<'i', int>();
    parser.add_argument("-b", "--test-batch-size")
        .help("Test batch size")
        .default_value(10000)
        .scan<'i', int>();
    parser.add_argument("-r", "--test-round")
        .help("Test rounds")
        .default_value(2)
        .scan<'i', int>();
    parser.add_argument("-e", "--expected-box-size")
        .help("Expected box size")
        .default_value(100)
        .scan<'i', int>();

    // Search options
    parser.add_argument("-s", "--search-type")
        .help("Search type")
        .default_value(0)
        .scan<'i', int>();
    parser.add_argument("-S", "--search-batch-size")
        .help("Search batch size")
        .default_value(20000)
        .scan<'i', int>();

    // Interface and runtime options
    parser.add_argument("--interface")
        .help("Backend interface")
        .default_value(std::string("upmem"));
    parser.add_argument("--debug")
        .help("Enable debug output")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--print-timer")
        .help("Print timing information")
        .default_value(true)
        .implicit_value(true);
    parser.add_argument("--top-level-threads")
        .help("Number of top-level threads")
        .default_value(1)
        .scan<'i', int>();
    parser.add_argument("--stats-csv")
        .help("Append per-batch DPU cycles and bytes to <prefix>_batches.csv and <prefix>_dpus.csv")
        .default_value(std::string(""));
    parser.add_argument("--stats-tag")
        .help("Label written to every stats row (e.g. the data distribution)")
        .default_value(std::string(""));
    parser.add_argument("--dist")
        .help("Point distribution for inserted data: uniform or zipf (Zipf-weighted clusters)")
        .default_value(std::string("uniform"));
    parser.add_argument("--zipf-s")
        .help("Zipf exponent over clusters (larger = more skew)")
        .default_value(1.0)
        .scan<'g', double>();
    parser.add_argument("--clusters")
        .help("Number of clusters for --dist zipf")
        .default_value(1024)
        .scan<'i', int>();
    parser.add_argument("--cluster-width-log2")
        .help("Cluster half-width per dimension is 2^this (coordinates are 31-bit)")
        .default_value(22)
        .scan<'i', int>();
    parser.add_argument("--placement")
        .help("Key ranges per DPU: quantile (equal points per DPU, from the warm-up data) or equal (equal-width ranges)")
        .default_value(std::string("quantile"));
    parser.add_argument("--verify-each-batch")
        .help("After every warm-up batch, look up all points inserted so far (slow; correctness check)")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--query-dist")
        .help("Lookup queries: sample (first S/I points of each warm-up batch, plus one absent point), "
              "uniform (any warm-up point) or zipf (hot keys, see --query-zipf-s)")
        .default_value(std::string("sample"));
    parser.add_argument("--input")
        .help("Read points from this file (scripts/prepare_points.py) instead of generating them; "
              "warm-up = its first -i x -I points, timed inserts = the points after them")
        .default_value(std::string(""));
    parser.add_argument("--query-file")
        .help("Point file for --query-dist mix (e.g. Varden points, paper Fig. 9)")
        .default_value(std::string(""));
    parser.add_argument("--query-mix")
        .help("--query-dist mix: fraction of lookups taken from --query-file, the rest uniform over stored points")
        .default_value(0.0)
        .scan<'g', double>();
    parser.add_argument("--query-zipf-s")
        .help("Zipf exponent of key popularity for --query-dist zipf")
        .default_value(1.0)
        .scan<'g', double>();
    parser.add_argument("--search-round")
        .help("Number of lookup batches of --search-batch-size queries (uniform/zipf query-dist)")
        .default_value(1)
        .scan<'i', int>();
    parser.add_argument("--lookup-dedup")
        .help("Send each distinct key once per lookup batch (contribution)")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--lookup-adaptive")
        .help("Use dedup/cache only for batches whose busiest DPU would get > --adaptive-threshold x the mean lookups (contribution; implies --lookup-dedup)")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--adaptive-threshold")
        .help("--lookup-adaptive uses dedup/cache when busiest DPU's lookups > this x mean")
        .default_value(1.2)
        .scan<'g', double>();
    parser.add_argument("--adaptive-repeat-threshold")
        .help("--lookup-adaptive skips dedup/cache while the last batch that used them saved less than this share "
              "of its lookups (re-checked every 8 batches)")
        .default_value(0.05)
        .scan<'g', double>();
    parser.add_argument("--hot-cache-size")
        .help("Keys in the host-side hot-key cache, 0 = off (contribution; implies --lookup-dedup)")
        .default_value(0)
        .scan<'i', int>();
    parser.add_argument("--seed")
        .help("Seed for the data generator")
        .default_value(1)
        .scan<'i', int>();

    parser.parse_args(argc, argv);

    // Assign parsed values to globals
    insert_batch_size  = parser.get<int>("--insert-batch-size");
    insert_round       = parser.get<int>("--insert-round");
    test_type          = parser.get<int>("--test-type");
    test_batch_size    = parser.get<int>("--test-batch-size");
    test_round         = parser.get<int>("--test-round");
    expected_box_size  = parser.get<int>("--expected-box-size");
    search_type        = parser.get<int>("--search-type");
    search_batch_size  = parser.get<int>("--search-batch-size");
    interface_type     = parser.get<std::string>("--interface");
    debug_print        = parser.get<bool>("--debug");
    print_timer        = parser.get<bool>("--print-timer");
    top_level_threads  = parser.get<int>("--top-level-threads");
    stats_csv_prefix   = parser.get<std::string>("--stats-csv");
    stats_tag          = parser.get<std::string>("--stats-tag");
    data_dist          = parser.get<std::string>("--dist");
    zipf_s             = parser.get<double>("--zipf-s");
    nr_clusters        = parser.get<int>("--clusters");
    cluster_width_log2 = parser.get<int>("--cluster-width-log2");
    data_seed          = parser.get<int>("--seed");
    placement          = parser.get<std::string>("--placement");
    verify_each_batch  = parser.get<bool>("--verify-each-batch");
    query_dist         = parser.get<std::string>("--query-dist");
    input_file         = parser.get<std::string>("--input");
    query_file         = parser.get<std::string>("--query-file");
    query_mix          = parser.get<double>("--query-mix");
    query_zipf_s       = parser.get<double>("--query-zipf-s");
    search_round       = parser.get<int>("--search-round");
    lookup_dedup       = parser.get<bool>("--lookup-dedup");
    hot_cache_size     = parser.get<int>("--hot-cache-size");
    lookup_adaptive    = parser.get<bool>("--lookup-adaptive");
    adaptive_threshold = parser.get<double>("--adaptive-threshold");
    adaptive_repeat_threshold = parser.get<double>("--adaptive-repeat-threshold");
    if(hot_cache_size > 0 || lookup_adaptive) lookup_dedup = true;
    if(query_dist != "sample" && query_dist != "uniform" && query_dist != "zipf" && query_dist != "mix") {
        fprintf(stderr, "unknown --query-dist %s (use sample, uniform, zipf or mix)\n", query_dist.c_str());
        exit(1);
    }
    if(placement != "quantile" && placement != "equal") {
        fprintf(stderr, "unknown --placement %s (use quantile or equal)\n", placement.c_str());
        exit(1);
    }

    for (int i = 0; i < NR_DIMENSION; ++i)
        input_coord_max[i] = INT32_MAX;
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char *argv[]) {
    printf("------------------- Start ---------------------\n");
    host_parse_arguments(argc, argv);
    host_init(interface_type);
    pim_stats::configure(stats_csv_prefix, stats_tag, nr_of_dpus);
    if(!datagen::init(data_dist, zipf_s, nr_clusters, cluster_width_log2, (uint64_t)data_seed)) {
        fprintf(stderr, "unknown --dist %s (use uniform or zipf)\n", data_dist.c_str());
        return 1;
    }
    if(!input_file.empty()) {
        uint64_t warm = (uint64_t)insert_batch_size * insert_round;
        uint64_t timed = (test_type == 1) ? (uint64_t)test_batch_size * test_round : 0;
        if(!datagen::init_file(input_file, warm, timed)) return 1;
    }
    
    pim_zd_tree zd_tree;
    pim_zd_tree *zd_forest[maxTopLevelThreads];
    zd_forest[0] = &zd_tree;
    for(int i = 1; i < maxTopLevelThreads; i++) zd_forest[i] = nullptr;
    bool needs_pipeline = (top_level_threads > 1) && (test_batch_size > 0) && (test_round > 0) && (test_type >= 1) && (test_type <= 4);
    if(needs_pipeline) for(int i = 1; i < top_level_threads; i++) zd_forest[i] = new pim_zd_tree;

    printf("------------- Data Structure Init ------------\n");

    int search_per_batch = search_batch_size / insert_round;
    int sampled_search_num = search_per_batch * insert_round;
    int64_t total_insert_size = insert_batch_size * insert_round;
    vectorT *vecs, *vec_dataset;
    bool need_to_search = search_batch_size > 0 && (search_type == 1 || search_type == 2 || search_type == 3 || search_type == 4);
    if(need_to_search) {
        if(search_type == 2 || search_type == 3 || search_type == 1) {
            vecs = new vectorT[search_batch_size + 1];
            // Queries past sampled_search_num (at least the extra (search_batch_size + 1)-th) are random
            // points, not taken from the dataset. The upper bound was search_per_batch + 1, which left them
            // uninitialised whenever insert_round > 1.
            parfor_wrap(sampled_search_num, search_batch_size + 1, [&](size_t i) {
#if NR_DIMENSION == 2
                vecs[i].x = abs(rn_gen::parallel_rand()) & COORD_MAX;
                vecs[i].y = abs(rn_gen::parallel_rand()) & COORD_MAX;
#elif NR_DIMENSION == 3
                vecs[i].x = abs(rn_gen::parallel_rand()) & COORD_MAX;
                vecs[i].y = abs(rn_gen::parallel_rand()) & COORD_MAX;
                vecs[i].z = abs(rn_gen::parallel_rand()) & COORD_MAX;
#else
                for(int j = 0; j < NR_DIMENSION; j++) vecs[i].x[j] = abs(rn_gen::parallel_rand()) & COORD_MAX;
#endif
            });
        }
        if(search_type == 2 || search_type == 3 || search_type == 4)
            vec_dataset = new vectorT[total_insert_size];
    }

    cpu_coverage_timer->start();
    dpu_binary_switch_to(dpu_binary::insert_binary);
    cpu_coverage_timer->end();
    cpu_coverage_timer->reset();
    pim_coverage_timer->reset();
    parlay::sequence<vectorT> vectors_from_file(1);
    parlay::sequence<uint64_t> sorted_idx_from_file;
    parlay::sequence<vectorT> vectors_from_varden(1);
    size_t varden_counter = 0;
    zd_tree.key_to_dpu_id_mode = 0;
    zd_tree.partition_borders[0] = 0;
    zd_tree.partition_borders[nr_of_dpus] = UINT64_MAX;
    parfor_wrap(1, nr_of_dpus, [&](size_t i) {
        zd_tree.partition_borders[i] = UINT64_MAX / nr_of_dpus * i;
    });
    if(placement == "quantile") {
        // Each DPU owns a key range holding the same number of warm-up points: the throughput-optimized
        // PIM-zd-tree splits L0 by subtree size (theta_L0 = n/P), so every subtree below L0 has about n/P
        // points. The borders are fixed from the warm-up data rather than maintained dynamically.
        auto keys = parlay::tabulate(total_insert_size, [&](size_t i) {
            vectorT v = datagen::point(1, i);
            return coord_to_key(&v);
        });
        parlay::sort_inplace(keys);
        for(int i = 1; i < nr_of_dpus; i++) {
            uint64_t b = keys[(size_t)((__int128)total_insert_size * i / nr_of_dpus)];
            zd_tree.partition_borders[i] = std::max(b, zd_tree.partition_borders[i - 1] + 1);
        }
        zd_tree.key_to_dpu_id_mode = 1;
    }
    printf("Placement: %s\n", placement.c_str());
    COORD tmp_coord_max[NR_DIMENSION];
    for(int i = 0; i < NR_DIMENSION; i++) {
        tmp_coord_max[i] = COORD_MAX;
    }
    zd_tree.init_range();

    for(int j = 0; j < insert_round; j++) {
        zd_tree.length = insert_batch_size;
        parfor_wrap(0, zd_tree.length, [&](size_t i) {
            zd_tree.vector_input[i] = datagen::point(1, (uint64_t)j * insert_batch_size + i);
            if(need_to_search && search_type != 1) vec_dataset[j * insert_batch_size + i] = zd_tree.vector_input[i];
            if(need_to_search && i < search_per_batch && search_type < 4 && search_type > 0)
                vecs[j * search_per_batch + i] = zd_tree.vector_input[i];
        });
        pim_stats::begin("warmup_insert", j, insert_batch_size);
        zd_tree.insert(zd_tree.vector_input, debug_print);
        pim_stats::end();
        if(verify_each_batch) {
            // Look up every point inserted so far; each should match exactly (Unmatch Num: 0)
            cpu_coverage_timer->start();
            dpu_binary_switch_to(dpu_binary::misc_binary);
            cpu_coverage_timer->end();
            zd_tree.length = (int64_t)(j + 1) * insert_batch_size;
            parfor_wrap(0, zd_tree.length, [&](size_t i) { zd_tree.vector_input[i] = datagen::point(1, i); });
            printf("Verify after warm-up batch %d: ", j);
            zd_tree.search_maximum_match(true, false, false);
            cpu_coverage_timer->start();
            dpu_binary_switch_to(dpu_binary::insert_binary);
            cpu_coverage_timer->end();
        }
    }

    {
        // Skew of the warm-up data set, as in paper Sec. 7.2: Gini of point counts over equal key ranges
        auto dataset = parlay::tabulate(total_insert_size, [&](size_t i) { return datagen::point(1, i); });
        double gini_2048 = datagen::key_range_gini(dataset.data(), dataset.size(), 2048);
        double gini_dpus = datagen::key_range_gini(dataset.data(), dataset.size(), nr_of_dpus);
        printf("Dataset: dist=%s zipf_s=%g clusters=%d width_log2=%d points=%lld gini_2048=%.4f gini_%d_bins=%.4f\n",
               datagen::dist_name(), zipf_s, nr_clusters, cluster_width_log2, (long long)total_insert_size,
               gini_2048, nr_of_dpus, gini_dpus);
        if(!stats_csv_prefix.empty()) {
            std::string path = stats_csv_prefix + "_dataset.csv";
            FILE* f = fopen(path.c_str(), "r");
            bool new_file = (f == nullptr);
            if(f != nullptr) fclose(f);
            f = fopen(path.c_str(), "a");
            if(f != nullptr) {
                if(new_file) fputs("tag,nr_dpus,dist,zipf_s,clusters,width_log2,seed,points,gini_2048,gini_dpu_bins\n", f);
                fprintf(f, "%s,%d,%s,%g,%d,%d,%d,%lld,%.4f,%.4f\n", stats_tag.c_str(), nr_of_dpus,
                        datagen::dist_name(), zipf_s, nr_clusters, cluster_width_log2, data_seed,
                        (long long)total_insert_size, gini_2048, gini_dpus);
                fclose(f);
            }
        }
    }

    if(print_timer) {
        cout<<"Total dataset size: "<<pim_zd_tree::nr_points.load()<<endl;
        cout<<dec<<"------------- Init timers -------------"<<endl;
        print_all_timers(print_type::pt_full);
        print_all_timers(print_type::pt_name);
        print_all_timers(print_type::pt_time);
        
    }
    reset_all_timers();
    zd_tree.reset_epoch_num();

    auto timer_program_start = std::chrono::high_resolution_clock::now();
    auto timer_program_stop = std::chrono::high_resolution_clock::now();
    auto program_duration = std::chrono::duration_cast<microseconds>(timer_program_stop - timer_program_start);
    int64_t total_test_time = 0;
#ifdef USE_PAPI
    papi_init_program(parlay::num_workers());
#endif
    
    total_communication = 0;
    total_actual_communication = 0;

    if(test_type == 1) {
        cpu_coverage_timer->start();
        dpu_binary_switch_to(dpu_binary::insert_binary);
        cpu_coverage_timer->end();
        cpu_coverage_timer->reset();
        pim_coverage_timer->reset();
        
        zd_tree.length = test_batch_size;
        vectorT *vec_to_search = new vectorT[test_round * test_batch_size];
        parfor_wrap(0, test_round * test_batch_size, [&](size_t i) {
            vec_to_search[i] = datagen::point(2, i);
        });
        
        timer_program_start = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
        papi_reset_counters();
        papi_turn_counters(true);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(true, parlay::num_workers());
#endif
        for(int j = 0; j < test_round; j++) {
            pim_stats::begin("insert", j, test_batch_size);
            zd_tree.insert(vec_to_search + j * test_batch_size, debug_print);
            pim_stats::end();
        }
#ifdef USE_PAPI
        papi_turn_counters(false);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(false, parlay::num_workers());
#endif
        timer_program_stop = std::chrono::high_resolution_clock::now();
        delete [] vec_to_search;
        program_duration = std::chrono::duration_cast<microseconds>(timer_program_stop - timer_program_start);
        total_test_time = program_duration.count();
    }
    else if(test_type == 2 || test_type == 3) {
        cpu_coverage_timer->start();
        if(test_type == 2) dpu_binary_switch_to(dpu_binary::box_count_binary);
        else dpu_binary_switch_to(dpu_binary::box_fetch_binary);
        cpu_coverage_timer->end();
        cpu_coverage_timer->reset();
        pim_coverage_timer->reset();

        int64_t box_edge_size = COORD_MAX / pow(pim_zd_tree::nr_points.load() / expected_box_size, 1.0 / NR_DIMENSION) / 2.0;
        vectorT boxes;
#if NR_DIMENSION == 2
        boxes = (vectorT){.x = box_edge_size, .y = box_edge_size};
#elif NR_DIMENSION == 3
        boxes = (vectorT){.x = box_edge_size, .y = box_edge_size, .z = box_edge_size};
#else
        for(int j = 0; j < NR_DIMENSION; j++) boxes.x[j] = box_edge_size;
#endif
        int actual_test_round = test_round / top_level_threads;
        vectorT *vec_to_search = new vectorT[test_round * test_batch_size * 2];
        parfor_wrap(0, test_round * test_batch_size, [&](size_t i) {
            vectorT vec;
#if NR_DIMENSION == 2
            vec.x = abs(rn_gen::parallel_rand()) & COORD_MAX;
            vec.y = abs(rn_gen::parallel_rand()) & COORD_MAX;
#elif NR_DIMENSION == 3
            vec.x = abs(rn_gen::parallel_rand()) & COORD_MAX;
            vec.y = abs(rn_gen::parallel_rand()) & COORD_MAX;
            vec.z = abs(rn_gen::parallel_rand()) & COORD_MAX;
#else
            for(int j = 0; j < NR_DIMENSION; j++) vec.x[j] = abs(rn_gen::parallel_rand()) & COORD_MAX;
#endif
            vec_to_search[i << 1] = vector_sub_zero_bounded(&vec, &boxes);
            vec_to_search[(i << 1) + 1] = vector_add(&vec, &boxes);
        });

        timer_program_start = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
        papi_reset_counters();
        papi_turn_counters(true);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(true, parlay::num_workers());
#endif
        time_nested("Box Operation", [&]() {
            parfor_wrap(0, top_level_threads, [&](size_t tid) {
                time_nested("thread " + std::to_string(tid), [&]() {
                    zd_forest[tid]->length = test_batch_size;
                    for(int j = 0; j < actual_test_round; j++) {
                        zd_forest[tid]->box_range(test_type == 2, expected_box_size, vec_to_search + tid * actual_test_round * test_batch_size * 2 + j * test_batch_size * 2);
                    }
                });
            }, true, 1);
        });
#ifdef USE_PAPI
        papi_turn_counters(false);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(false, parlay::num_workers());
#endif
        timer_program_stop = std::chrono::high_resolution_clock::now();
        delete [] vec_to_search;
        program_duration = std::chrono::duration_cast<microseconds>(timer_program_stop - timer_program_start);
        total_test_time = program_duration.count();
    }
    else if(test_type == 4) {
        cpu_coverage_timer->start();
        dpu_binary_switch_to(dpu_binary::knn_binary);
        cpu_coverage_timer->end();
        cpu_coverage_timer->reset();
        pim_coverage_timer->reset();

        int actual_test_round = test_round / top_level_threads;
        vectorT *vec_to_search = new vectorT[test_round * test_batch_size];
        parfor_wrap(0, test_round * test_batch_size, [&](size_t i) {
#if NR_DIMENSION == 2
            vec_to_search[i].x = abs(rn_gen::parallel_rand()) & COORD_MAX;
            vec_to_search[i].y = abs(rn_gen::parallel_rand()) & COORD_MAX;
#elif NR_DIMENSION == 3
            vec_to_search[i].x = abs(rn_gen::parallel_rand()) & COORD_MAX;
            vec_to_search[i].y = abs(rn_gen::parallel_rand()) & COORD_MAX;
            vec_to_search[i].z = abs(rn_gen::parallel_rand()) & COORD_MAX;
#else
            for(int j = 0; j < NR_DIMENSION; j++) vec_to_search[i].x[j] = abs(rn_gen::parallel_rand()) & COORD_MAX;
#endif
        });

        timer_program_start = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
        papi_reset_counters();
        papi_turn_counters(true);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(true, parlay::num_workers());
#endif
        time_nested("kNN Operation", [&]() {
            parfor_wrap(0, top_level_threads, [&](size_t tid) {
                time_nested("thread " + std::to_string(tid), [&]() {
                    zd_forest[tid]->length = test_batch_size;
                    if(expected_box_size > 0 && expected_box_size <= MAX_KNN_SIZE) {
                        for(int j = 0; j < actual_test_round; j++) {
                            zd_forest[tid]->knn(expected_box_size, vec_to_search + tid * actual_test_round * test_batch_size + j * test_batch_size);
                        }
                    }
                });
            }, true, 1);
        });
#ifdef USE_PAPI
        papi_turn_counters(false);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(false, parlay::num_workers());
#endif
        timer_program_stop = std::chrono::high_resolution_clock::now();
        delete [] vec_to_search;
        program_duration = std::chrono::duration_cast<microseconds>(timer_program_stop - timer_program_start);
        total_test_time = program_duration.count();
    }

    if(print_timer) {
        cout<<dec<<"------------- Test timers -------------"<<endl;
        print_all_timers(print_type::pt_full);
        print_all_timers(print_type::pt_name);
        print_all_timers(print_type::pt_time);
        cout<<dec<<"------------- Test counters -----------"<<endl;
        cout<<"Total time in test (us): "<<dec<<total_test_time<<endl;
        cout<<"Total communication: "<<total_communication<<endl;
        cout<<"Total actual communication: "<<total_actual_communication<<endl;
#ifdef USE_PAPI
        papi_print_counters(1);
#endif
    }
    reset_all_timers();
    zd_tree.reset_epoch_num();

    if(need_to_search) {
        if(search_type == 1) {
            cpu_coverage_timer->start();
            dpu_binary_switch_to(dpu_binary::misc_binary);
            cpu_coverage_timer->end();
            cpu_coverage_timer->reset();
            pim_coverage_timer->reset();
            zd_tree.lookup_dedup = lookup_dedup;
            zd_tree.lookup_adaptive = lookup_adaptive;
            zd_tree.adaptive_threshold = adaptive_threshold;
            zd_tree.adaptive_repeat_threshold = adaptive_repeat_threshold;
            if(lookup_dedup) zd_tree.dedup_table.reserve(search_batch_size);  // setup, outside the measured batches
            zd_tree.hot_cache.capacity = hot_cache_size;
            if(query_dist == "sample") {
                zd_tree.length = search_batch_size + 1;
                parfor_wrap(0, zd_tree.length, [&](size_t i) {
                    zd_tree.vector_input[i] = vecs[i];
                });
                pim_stats::begin("lookup", 0, zd_tree.length);
                zd_tree.search_maximum_match(true, false, debug_print);
                pim_stats::end();
            }
            else {
                // Queries over the warm-up points: every one should be found (Unmatch Num: 0), except the
                // --query-file points of a mix, which are generally not stored
                datagen::init_queries(query_dist, query_zipf_s, (uint64_t)total_insert_size);
                if(query_dist == "mix" && !datagen::init_query_mix(query_file, query_mix)) return 1;
                for(int r = 0; r < search_round; r++) {
                    zd_tree.length = search_batch_size;
                    parfor_wrap(0, zd_tree.length, [&](size_t i) {
                        zd_tree.vector_input[i] = datagen::to_vector(datagen::query_point(r, i));
                    });
                    pim_stats::begin("lookup", r, zd_tree.length);
                    zd_tree.search_maximum_match(true, false, debug_print);
                    pim_stats::end();
                }
                if(zd_tree.lookup_adaptive)
                    printf("Adaptive lookups: dedup skipped in %llu of %llu batches (%llu for lack of repeats)\n",
                           (unsigned long long)zd_tree.adaptive_skipped, (unsigned long long)zd_tree.adaptive_batches,
                           (unsigned long long)zd_tree.adaptive_norepeat_skipped);
                if(zd_tree.hot_cache.capacity > 0)
                    printf("Hot-key cache: %zu keys, %llu hits of %llu lookups\n", zd_tree.hot_cache.keys.size(),
                           (unsigned long long)zd_tree.hot_cache.hits, (unsigned long long)zd_tree.hot_cache.lookups);
            }
        }
        else if(search_type == 2 || search_type == 3) {
            // Box ranges
            cpu_coverage_timer->start();
            if(search_type == 2) dpu_binary_switch_to(dpu_binary::box_count_binary);
            else dpu_binary_switch_to(dpu_binary::box_fetch_binary);
            cpu_coverage_timer->end();
            cpu_coverage_timer->reset();
            pim_coverage_timer->reset();

            int acutal_batch_num = search_batch_size / expected_box_size;
            int64_t box_edge_size = COORD_MAX / pow(pim_zd_tree::nr_points.load() / expected_box_size, 1.0 / NR_DIMENSION) / 2.0;
            vectorT boxes;
#if NR_DIMENSION == 2
            boxes = (vectorT){.x = box_edge_size, .y = box_edge_size};
#elif NR_DIMENSION == 3
            boxes = (vectorT){.x = box_edge_size, .y = box_edge_size, .z = box_edge_size};
#else
            for(int j = 0; j < NR_DIMENSION; j++) boxes.x[j] = box_edge_size;
#endif
            zd_tree.length = acutal_batch_num;
            int *counts = new int[acutal_batch_num];
            int err_num = 0;
            parfor_wrap(0, acutal_batch_num, [&](size_t i) {
                zd_tree.vector_input[i << 1] = vector_sub_zero_bounded(&vecs[i], &boxes);
                zd_tree.vector_input[(i << 1) + 1] = vector_add(&vecs[i], &boxes);
                counts[i] = 0;
                int dpu_id = -1;
                for(int j = 0; j < total_insert_size; j++) {
                    if(vector_in_box(&vec_dataset[j], &zd_tree.vector_input[i << 1], &zd_tree.vector_input[(i << 1) + 1])) {
                        counts[i]++;
                    }
                }
            });
            zd_tree.box_range(search_type == 2, expected_box_size);
            bool correct_in_check, printed;
            for(int i = 0; i < acutal_batch_num; i++) {
                uint64_t key1 = coord_to_key(&zd_tree.vector_input[i << 1]);
                uint64_t key2 = coord_to_key(&zd_tree.vector_input[(i << 1) + 1]);
                if(search_type == 2 && counts[i] != zd_tree.i64_io[i]) {
                    err_num++;
                    printf("Query %d: %d %lld\n", i, counts[i], zd_tree.i64_io[i]);
                }
                else if(search_type == 3) {
                    correct_in_check = true;
                    printed = false;
                    int sub_err_num = 0;
                    if(counts[i] != zd_tree.i64_io[i + 1] - zd_tree.i64_io[i]) correct_in_check = false;
                    else {
                        for(int j = zd_tree.i64_io[i]; j < zd_tree.i64_io[i + 1]; j++) {
                            if(!vector_in_box(&zd_tree.vector_output[j], &zd_tree.vector_input[i << 1], &zd_tree.vector_input[(i << 1) + 1])) {
                                correct_in_check = false;
                                sub_err_num++;
#if NR_DIMENSION == 3
                                if(!printed) {
                                    printed = true;
                                    printf("Vec: %lld %lld %lld\nMin: %lld %lld %lld\nMax: %lld %lld %lld\n",
                                        zd_tree.vector_output[j].x, zd_tree.vector_output[j].y, zd_tree.vector_output[j].z,
                                        zd_tree.vector_input[i << 1].x, zd_tree.vector_input[i << 1].y, zd_tree.vector_input[i << 1].z,
                                        zd_tree.vector_input[(i << 1) + 1].x, zd_tree.vector_input[(i << 1) + 1].y, zd_tree.vector_input[(i << 1) + 1].z
                                    );
                                }
#endif
                            }
                        }
                    }
                    if(!correct_in_check) {
                        err_num++;
                        printf("Query %d: %d %lld; %d\n", i, counts[i], zd_tree.i64_io[i + 1] - zd_tree.i64_io[i], sub_err_num);
                    }
                }
            }
            printf("Total err num: %d\n", err_num);
            delete [] counts;
        }
        else if(search_type == 4) {
            // kNN
            cpu_coverage_timer->start();
            dpu_binary_switch_to(dpu_binary::knn_binary);
            cpu_coverage_timer->end();
            cpu_coverage_timer->reset();
            pim_coverage_timer->reset();

            int acutal_batch_num = search_batch_size / expected_box_size;
            zd_tree.length = acutal_batch_num;
            parfor_wrap(0, acutal_batch_num, [&](size_t i) {
#if NR_DIMENSION == 2
                zd_tree.vector_input[i].x = abs(rn_gen::parallel_rand()) & COORD_MAX;
                zd_tree.vector_input[i].y = abs(rn_gen::parallel_rand()) & COORD_MAX;
#elif NR_DIMENSION == 3
                zd_tree.vector_input[i].x = abs(rn_gen::parallel_rand()) & COORD_MAX;
                zd_tree.vector_input[i].y = abs(rn_gen::parallel_rand()) & COORD_MAX;
                zd_tree.vector_input[i].z = abs(rn_gen::parallel_rand()) & COORD_MAX;
#else
                for(int j = 0; j < NR_DIMENSION; j++) zd_tree.vector_input[i].x[j] = abs(rn_gen::parallel_rand()) & COORD_MAX;
#endif
            });
            zd_tree.knn(expected_box_size);

            int64_t distance1, distance2, tmp;
            vectorT vec;
            heap_host heap(expected_box_size);
            int err_num = 0;
            for(int i = 0; i < acutal_batch_num; i++) {
                distance1 = 0;
                for(int j = 0; j < expected_box_size; j++) {
                    vec = vector_sub(
                        &zd_tree.vector_input[i],
                        &zd_tree.vector_output[i * expected_box_size + j]
                    );
                    tmp = vector_norm(&vec);
                    if(tmp > distance1) distance1 = tmp;
                }
                heap.num = 0;
                for(int j = 0; j < total_insert_size; j++) {
                    vec = vector_sub(&zd_tree.vector_input[i], &vec_dataset[j]);
                    tmp = vector_norm(&vec);
                    heap.enqueue(tmp, &vec_dataset[j]);
                }
                distance2 = heap.distance_storage[0];
                if(distance1 != distance2) {
                    printf("Query %d: %lld %lld\n", i, distance1, distance2);
                    printf("Second round radius: %lld\n", zd_tree.i64_io[i]);
                    printf("PIM\n");
                    for(int j = 0; j < expected_box_size; j++) {
                        vec = vector_sub(
                            &zd_tree.vector_input[i],
                            &zd_tree.vector_output[i * expected_box_size + j]
                        );
                        tmp = vector_norm(&vec);
                        printf("%lld ", tmp);
                        tmp = vector_norm_dpu(&vec);
                        printf("%lld\n", tmp);
                    }
                    printf("Baseline\n");
                    for(int j = 0; j < heap.num; j++) {
                        vec = vector_sub(
                            zd_tree.vector_input + i,
                            heap.vector_storage + j
                        );
                        tmp = vector_norm_dpu(&vec);
                        printf("%lld %lld\n", heap.distance_storage[j], tmp);
                    }
                    err_num++;
                }
            }
            printf("Total err: %d\n", err_num);
        }
    }
    zd_tree.reset_epoch_num();

    host_end();
    if(need_to_search) {
        if(search_type == 2 || search_type == 3 || search_type == 1) delete [] vecs;
        if(search_type == 2 || search_type == 3 || search_type == 4) delete [] vec_dataset;
    }
    if(needs_pipeline) for(int i = 1; i < maxTopLevelThreads; i++) {
        if(zd_forest[i] != nullptr) delete zd_forest[i];
    }
    return 0;
}