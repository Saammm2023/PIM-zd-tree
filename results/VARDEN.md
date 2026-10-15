# Varden benchmark

Varden is the skewed synthetic generator used across this line of work (Pkd-tree, PSI/SpaceTreeLib, and
PIM-zd-tree's skew study, Fig. 9): points come from random walks, so the data forms tight, irregular clusters.
Tables: `results/tables/varden_results.md`; figures: `results/figures/fig_varden_data.*`, `fig_varden_mix.*`.

## How the data was made

```bash
# SpaceTreeLib's generator (g++-14), 3-D, on a 2^21-wide axis = PIM-zd-tree's key resolution per dimension
g++-14 -std=c++20 -O3 -march=native -mcx16 -pthread ../SpaceTreeLib/tests/data_generate.cpp -o ../varden/data_generate \
    -I ../SpaceTreeLib/include -I ../SpaceTreeLib/include/parlaylib/include -I ../SpaceTreeLib/include/libmorton/include -I ../SpaceTreeLib/tests
../varden/data_generate -p ../varden/raw_data  -n 1300000 -d 3 -file_num 1 -varden 1 -axis_max 2097151 -seed 1
../varden/data_generate -p ../varden/raw_query -n 100000  -d 3 -file_num 1 -varden 1 -axis_max 2097151 -seed 7
python3 -I scripts/prepare_points.py ../varden/raw_data/ss_varden_bigint/1300000_3/1.in  ../varden/varden_data.pts  --seed 1
python3 -I scripts/prepare_points.py ../varden/raw_query/ss_varden_bigint/100000_3/1.in ../varden/varden_query.pts --seed 7
VARDEN=../varden scripts/run_sweep.sh varden && ../.venv/bin/python scripts/analyze_varden.py
```

Why a 2^21 axis: Varden clusters are 100–1000 units wide regardless of the axis length. PIM-zd-tree's 64-bit keys keep
21 bits per dimension, so on the default 10^9 axis whole clusters would share one key, and the tree cannot hold more
than 16 copies of a key. On a 2^21 axis Varden's integer grid is exactly the key grid. `prepare_points.py` wraps the
5% of points that random walks push past the axis, drops exact duplicates (78 of 1.3M), shuffles the insertion order
and scales the grid into the 31-bit coordinate space. **Gini over 2048 key ranges: 0.997**, more skewed than our
strong Zipf set (0.978) and the paper's OSM data (0.967).

## 1. Varden as data

800k warm-up points, then 3 × 50k inserts and 50k lookups; all lookups correct (verified per batch on 8 and 64 DPUs).

| | 8 | 16 | 32 | 64 DPUs | Pkd-tree CPU | zd-tree CPU |
|---|---|---|---|---|---|---|
| insert (Mop/s) | 1.66 | 2.95 | 4.80 | 7.35 | 20.73 | 7.64 |
| lookup (Mop/s) | 2.86 | 5.53 | 10.37 | 19.61 | 14.64 | 7.57 |
| lookup imbalance | 1.12× | 1.19× | 1.25× | 1.18× | | |

The same picture as our strong Zipf data: PIM-zd-tree scales with DPUs (lookups 6.9× from 8 to 64 DPUs), passes the
20-thread Pkd-tree on lookups at 64 DPUs, and stays within 1.25× of perfect balance. Range partitioning cannot hold
the 800k-point set (its hottest DPU overflows leaf storage). At 200k points it fits, but its throughput is flat at
0.7 (insert) and 1.1–1.3 (lookup) Mop/s from 8 to 64 DPUs, with imbalance 2.8× → 19.9×.

Our addition on Varden data (hot keys drawn from the Varden points, 64 DPUs): baseline 6.6 / 2.9 Mop/s at Zipf
s = 1.0 / 1.2 → **29.5 / 50.8 Mop/s** with dedup + cache (4.5× / 17.6×). With uniform keys the adaptive variant matches
the baseline (20.7 vs 20.8).

## 2. Paper Fig. 9 for point lookups

Uniform data (800k), 64 DPUs; each lookup comes from the separate Varden query file with probability f, otherwise it is
a stored point chosen uniformly. Varden queries are spatial hot spots: distinct points packed into small regions,
generally not stored (they return "unmatched", which is expected).

| Varden share | 0 | 0.1‰ | 1‰ | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|---|---|
| baseline (Mop/s) | 25.5 | 25.6 | 24.9 | 23.3 | 21.3 | 17.1 | 12.6 |
| imbalance | 1.09× | 1.09× | 1.09× | 1.16× | 1.30× | 1.71× | 2.39× |
| adaptive dedup + cache | 25.5 | 25.2 | 24.3 | 23.0 | 16.2 | 13.6 | 10.8 |
| Pkd-tree CPU | 18.4 | 19.1 | 18.0 | 17.8 | 17.9 | 18.1 | 18.8 |

**Reproduced, qualitatively:** the throughput-optimized PIM-zd-tree loses throughput as spatially skewed queries are
mixed in, because the DPUs owning the hot regions become stragglers (imbalance 2.4× at 10%). It falls below the
Pkd-tree from about 5%. The CPU tree is unaffected. The paper reports a steeper drop (10.66× at 2%) for kNN queries,
which do far more work around each hot spot than our single-leaf point lookups, so they amplify the imbalance.

**A limit of our addition, found by this benchmark:** Varden queries are distinct points, so deduplication and the
hot-key cache find nothing to remove. The adaptive switch still turns the new path on (it sees a hot DPU), and its
extra host work (about 0.8 ms per batch) makes lookups up to 24% slower than the baseline from 2% on. Two ways to fix
it: also require a repeat rate (skip the new path when the last batch that used it saved under ~5% of lookups,
re-checking every few batches), or split hot regions by re-partitioning (the paper's skew-resistant layout).
Re-partitioning is outside what our host-only addition can do.
