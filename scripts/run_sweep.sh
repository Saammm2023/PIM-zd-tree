#!/usr/bin/env bash
# Experiment sweep for the report. Writes CSVs under results/raw/ and one log per run under results/logs/.
#   scripts/run_sweep.sh [scaling|batch|skewcmp|contrib|varden|varden_mix|adaptive_rerun|cpu|all]
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

varden() {
    # Varden benchmark (SpaceTreeLib's generator; see SETUP_SIMULATOR.md "Varden"). Needs
    # $VARDEN/varden_data.pts and $VARDEN/varden_query.pts.
    local V=${VARDEN:-../varden}
    [ -f "$V/varden_data.pts" ] && [ -f "$V/varden_query.pts" ] || { echo "Varden files not found in $V"; return 1; }
    build 8 16 32 64
    rm -f "$RAW"/varden_*
    local LOOK="-s 1 -S 50000"
    # 1. Varden as data: 800k points, PIM-zd-tree (equal ranges overflow), and 200k points for both placements
    for d in 8 16 32 64; do
        run_pim "varden_scaling_${d}" $d "$RAW/varden_scaling" "varden_quantile" \
            $WARMUP -t 1 -b 50000 -r 3 $LOOK --input "$V/varden_data.pts" --placement quantile
        for pl in quantile equal; do
            run_pim "varden_small_${d}_${pl}" $d "$RAW/varden_small" "varden_${pl}" \
                -i 20000 -I 10 -t 1 -b 20000 -r 3 -s 1 -S 20000 --input "$V/varden_data.pts" --placement $pl
        done
    done
    # 2. Hot keys over Varden data, 64 DPUs: baseline vs our addition
    for q in uniform 1.0 1.2; do
        local qarg="--query-dist zipf --query-zipf-s $q"
        [ "$q" = uniform ] && qarg="--query-dist uniform"
        for v in base cache adaptive; do
            local varg=""
            [ $v = cache ] && varg="--hot-cache-size 256"
            [ $v = adaptive ] && varg="--lookup-adaptive --hot-cache-size 256"
            run_pim "varden_hot_64_q${q}_${v}" 64 "$RAW/varden_hot" "varden_quantile_q${q}_${v}" \
                $WARMUP $LOOK --search-round 10 $qarg $varg --input "$V/varden_data.pts"
        done
    done
    varden_mix
    varden_cpu
}

varden_mix() {
    # Paper Fig. 9 for point lookups: uniform data, a fraction of lookups from Varden points, 64 DPUs
    local V=${VARDEN:-../varden}
    local LOOK="-s 1 -S 50000"
    build 64
    rm -f "$RAW"/varden_mix_batches.csv "$RAW"/varden_mix_dpus.csv "$RAW"/varden_mix_dataset.csv "$RAW"/varden_mix_failures.csv
    for f in 0 0.0001 0.001 0.01 0.02 0.05 0.1; do
        for v in base adaptive; do
            local varg=""
            [ $v = adaptive ] && varg="--lookup-adaptive --hot-cache-size 256"
            run_pim "varden_mix_64_f${f}_${v}" 64 "$RAW/varden_mix" "uniform_quantile_f${f}_${v}" \
                $WARMUP $LOOK --search-round 10 --query-dist mix --query-file "$V/varden_query.pts" --query-mix $f \
                $varg ${DIST[uniform]}
        done
    done
}

adaptive_rerun() {
    # Re-run only the adaptive variant everywhere it appears (after changing its rule); drops its old rows first
    local V=${VARDEN:-../varden}
    build 16 64
    for f in "$RAW"/contrib_batches.csv "$RAW"/contrib_dpus.csv "$RAW"/varden_hot_batches.csv "$RAW"/varden_hot_dpus.csv \
             "$RAW"/varden_mix_batches.csv "$RAW"/varden_mix_dpus.csv; do
        [ -f "$f" ] && grep -v "_adaptive," "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    done
    local LOOK="-s 1 -S 50000 --search-round 10"
    local A="--lookup-adaptive --hot-cache-size 256"
    for d in 16 64; do
        for q in uniform 0.6 0.8 1.0 1.2 1.5; do
            local qarg="--query-dist zipf --query-zipf-s $q"
            [ "$q" = uniform ] && qarg="--query-dist uniform"
            run_pim "contrib_${d}_q${q}_adaptive" $d "$RAW/contrib" "uniform_quantile_q${q}_adaptive" \
                $WARMUP $LOOK $qarg $A ${DIST[uniform]}
        done
    done
    run_pim "contrib_64_strongdata_q1.0_adaptive" 64 "$RAW/contrib" "strong_quantile_q1.0_adaptive" \
        $WARMUP $LOOK --query-dist zipf --query-zipf-s 1.0 $A ${DIST[strong]}
    for q in uniform 1.0 1.2; do
        local qarg="--query-dist zipf --query-zipf-s $q"
        [ "$q" = uniform ] && qarg="--query-dist uniform"
        run_pim "varden_hot_64_q${q}_adaptive" 64 "$RAW/varden_hot" "varden_quantile_q${q}_adaptive" \
            $WARMUP $LOOK $qarg $A --input "$V/varden_data.pts"
    done
    for f in 0 0.0001 0.001 0.01 0.02 0.05 0.1; do
        run_pim "varden_mix_64_f${f}_adaptive" 64 "$RAW/varden_mix" "uniform_quantile_f${f}_adaptive" \
            $WARMUP $LOOK --query-dist mix --query-file "$V/varden_query.pts" --query-mix $f $A ${DIST[uniform]}
    done
}

varden_cpu() {
    # CPU baselines on the Varden data and the Fig. 9 query streams (20 threads, 3 repetitions)
    local V=${VARDEN:-../varden}
    local LOOK="-s 1 -S 50000"
    rm -f "$RAW"/varden_scaling_cpu.csv "$RAW"/varden_mix_cpu.csv
    for rep in 1 2 3; do
        for tree in pkd zd; do
            PARLAY_NUM_THREADS=20 cpu_baseline/cpu_bench --tree $tree $WARMUP -t 1 -b 50000 -r 3 $LOOK \
                --input "$V/varden_data.pts" --stats-csv "$RAW/varden_scaling" --stats-tag varden --rep $rep \
                > "$LOGS/cpu_varden_${tree}_$rep.log" 2>&1 || echo "FAIL  cpu varden $tree rep $rep"
            for f in 0 0.0001 0.001 0.01 0.02 0.05 0.1; do
                PARLAY_NUM_THREADS=20 cpu_baseline/cpu_bench --tree $tree $WARMUP $LOOK --search-round 10 \
                    --query-dist mix --query-file "$V/varden_query.pts" --query-mix $f ${DIST[uniform]} \
                    --stats-csv "$RAW/varden_mix" --stats-tag "uniform_f$f" --rep $rep \
                    > "$LOGS/cpu_varden_mix_${tree}_f${f}_$rep.log" 2>&1 || echo "FAIL  cpu varden mix $tree f$f rep $rep"
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
    varden) varden ;;
    varden_mix) varden_mix ;;
    adaptive_rerun) adaptive_rerun ;;
    cpu) cpu ;;
    all) cpu; scaling; batch; skewcmp; contrib ;;
    *) echo "usage: $0 [scaling|batch|skewcmp|contrib|varden|varden_mix|adaptive_rerun|cpu|all]"; exit 1 ;;
esac
