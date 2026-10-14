#!/usr/bin/env bash
# Experiment sweep for the report. Writes CSVs under results/raw/ and one log per run under results/logs/.
#   scripts/run_sweep.sh [scaling|batch|skewcmp|contrib|cpu|all]
# Needs: source ~/upmem-sdk/env-sim.sh, and cpu_baseline/cpu_bench built (make -C cpu_baseline).
set -u
cd "$(dirname "$0")/.."
RAW=results/raw
LOGS=results/logs
mkdir -p "$RAW" "$LOGS"

declare -A DIST=(
    [uniform]="--dist uniform"
    [moderate]="--dist zipf --zipf-s 0.6 --cluster-width-log2 28"
    [strong]="--dist zipf --zipf-s 1.5 --cluster-width-log2 22"
)
DATASETS="uniform moderate strong"
# 800k warm-up points, 3 timed insert batches, one lookup batch. 8 DPUs then hold ~119k points each,
# under the ~130k leaf-storage limit.
WARMUP="-i 80000 -I 10"

build() {
    for d in "$@"; do
        make NR_DPUS=$d BUILDDIR=build$d -j6 > "$LOGS/build$d.log" 2>&1 || { echo "build $d failed"; exit 1; }
    done
}

run_pim() {  # run_pim <name> <dpus> <csv-prefix> <tag> <args...>
    local name=$1 d=$2 prefix=$3 tag=$4; shift 4
    local log="$LOGS/$name.log"
    if timeout 3600 ./build$d/zd_tree_host "$@" --stats-csv "$prefix" --stats-tag "$tag" > "$log" 2>&1; then
        echo "ok    $name  $(grep -o 'Match Num.*' "$log" | tail -1)"
    else
        echo "FAIL  $name  $(grep -oE '(leaf|internal-node) storage full|DPU error[^;]*|buffer bound exceeded' "$log" | head -1)"
        echo "$tag,$d,FAIL,$(grep -oE '(leaf|internal-node) storage full|buffer bound exceeded' "$log" | head -1)" >> "${prefix}_failures.csv"
    fi
}

scaling() {
    build 8 16 32 64
    for d in 8 16 32 64; do
        for ds in $DATASETS; do
            for pl in quantile equal; do
                run_pim "scaling_${d}_${ds}_${pl}" $d "$RAW/scaling" "${ds}_${pl}" \
                    $WARMUP -t 1 -b 50000 -r 3 -s 1 -S 50000 ${DIST[$ds]} --placement $pl
            done
        done
    done
}

batch() {
    build 64
    for ds in uniform strong; do
        for b in 5000 10000 20000 50000 100000 200000; do
            run_pim "batch_64_${ds}_${b}" 64 "$RAW/batch" "${ds}_quantile" \
                $WARMUP -t 1 -b $b -r 3 -s 1 -S $b ${DIST[$ds]} --placement quantile
        done
    done
}

skewcmp() {
    # Strong skew at a size the range-partitioned design can still hold (its hottest DPU overflows leaf storage
    # at 800k points), so both placements can be compared on the same data
    build 8 16 32 64
    for d in 8 16 32 64; do
        for pl in quantile equal; do
            run_pim "skewcmp_${d}_strong_${pl}" $d "$RAW/skewcmp" "strong_${pl}" \
                -i 20000 -I 10 -t 1 -b 20000 -r 3 -s 1 -S 20000 ${DIST[strong]} --placement $pl
        done
    done
}

contrib() {
    # Our addition: lookups with hot keys. 10 lookup batches of 50k queries after 800k warm-up points.
    build 16 64
    declare -A VARIANT=(
        [base]=""
        [dedup]="--lookup-dedup"
        [cache]="--hot-cache-size 256"
        [adaptive]="--lookup-adaptive --hot-cache-size 256"
    )
    local LOOK="-s 1 -S 50000 --search-round 10"
    for d in 16 64; do
        for q in uniform 0.6 0.8 1.0 1.2 1.5; do
            local qarg="--query-dist zipf --query-zipf-s $q"
            [ "$q" = uniform ] && qarg="--query-dist uniform"
            for v in base dedup cache adaptive; do
                run_pim "contrib_${d}_q${q}_${v}" $d "$RAW/contrib" "uniform_quantile_q${q}_${v}" \
                    $WARMUP $LOOK $qarg ${VARIANT[$v]} ${DIST[uniform]}
            done
        done
    done
    # Cache size, 64 DPUs, s = 1.2 (size 0 = dedup only, 256 = the main runs above)
    for c in 64 1024 4096 16384; do
        run_pim "contrib_64_q1.2_cache$c" 64 "$RAW/contrib" "uniform_quantile_q1.2_cache$c" \
            $WARMUP $LOOK --query-dist zipf --query-zipf-s 1.2 --hot-cache-size $c ${DIST[uniform]}
    done
    # Hot keys on strongly skewed data
    for v in base dedup cache adaptive; do
        run_pim "contrib_64_strongdata_q1.0_${v}" 64 "$RAW/contrib" "strong_quantile_q1.0_${v}" \
            $WARMUP $LOOK --query-dist zipf --query-zipf-s 1.0 ${VARIANT[$v]} ${DIST[strong]}
    done
    # CPU reference on the same query streams (20 threads, 3 repetitions)
    rm -f "$RAW/contrib_cpu.csv"
    for rep in 1 2 3; do
        for tree in pkd zd; do
            for q in uniform 0.6 0.8 1.0 1.2 1.5; do
                local qarg="--query-dist zipf --query-zipf-s $q"
                [ "$q" = uniform ] && qarg="--query-dist uniform"
                PARLAY_NUM_THREADS=20 cpu_baseline/cpu_bench --tree $tree $WARMUP $LOOK $qarg ${DIST[uniform]} \
                    --stats-csv "$RAW/contrib" --stats-tag "uniform_q$q" --rep $rep > "$LOGS/cpu_contrib_${tree}_q${q}_$rep.log" 2>&1 \
                    && echo "ok    cpu contrib $tree q$q rep $rep" || echo "FAIL  cpu contrib $tree q$q rep $rep"
            done
        done
    done
}

cpu() {
    # Wall-clock CPU timings vary run to run; each configuration runs REPS times and the analysis takes the median
    local REPS=3
    rm -f "$RAW/scaling_cpu.csv" "$RAW/batch_cpu.csv"
    for rep in $(seq 1 $REPS); do
        for ds in $DATASETS; do
            for tree in pkd zd; do
                for th in 1 4 20; do
                    PARLAY_NUM_THREADS=$th cpu_baseline/cpu_bench --tree $tree $WARMUP -t 1 -b 50000 -r 3 -s 1 -S 50000 \
                        ${DIST[$ds]} --stats-csv "$RAW/scaling" --stats-tag "$ds" --rep $rep \
                        > "$LOGS/cpu_${tree}_${ds}_${th}_$rep.log" 2>&1 \
                        && echo "ok    cpu $tree $ds $th threads rep $rep" || echo "FAIL  cpu $tree $ds $th threads rep $rep"
                done
            done
        done
        for ds in uniform strong; do
            for tree in pkd zd; do
                for b in 5000 10000 20000 50000 100000 200000; do
                    PARLAY_NUM_THREADS=20 cpu_baseline/cpu_bench --tree $tree $WARMUP -t 1 -b $b -r 3 -s 1 -S $b \
                        ${DIST[$ds]} --stats-csv "$RAW/batch" --stats-tag "$ds" --rep $rep \
                        > "$LOGS/cpu_batch_${tree}_${ds}_${b}_$rep.log" 2>&1 \
                        && echo "ok    cpu batch $tree $ds $b rep $rep" || echo "FAIL  cpu batch $tree $ds $b rep $rep"
                done
            done
        done
    done
}

case "${1:-all}" in
    scaling) scaling ;;
    batch) batch ;;
    skewcmp) skewcmp ;;
    contrib) contrib ;;
    cpu) cpu ;;
    all) cpu; scaling; batch; skewcmp; contrib ;;
    *) echo "usage: $0 [scaling|batch|skewcmp|contrib|cpu|all]"; exit 1 ;;
esac
