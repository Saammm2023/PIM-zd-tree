# Baseline results: comparison with the paper

Generated data: `results/raw/` (CSV), `results/tables/` (summaries, `baseline_results.md`), `results/figures/`
(PDF for LaTeX, PNG for slides). Reproduce with `scripts/run_sweep.sh all` and `scripts/analyze.py`.

## Setup

| | Paper (PPoPP'26) | This reproduction |
|---|---|---|
| PIM system | UPMEM, 2048 DPUs (32 ranks) at 350 MHz, real hardware | UPMEM SDK 2025.1.0 functional simulator, 8–64 DPUs (one rank) |
| PIM time | measured | DPU cycle counters from the simulator / 350 MHz, summed per launch over the slowest DPU |
| CPU–PIM transfers | measured (direct interface) | bytes counted exactly, time from PrIM's measured bandwidth (0.33–6.68 GB/s CPU→DPU, 0.12–4.74 GB/s DPU→CPU for 1–64 DPUs); no launch overhead |
| Host CPU | 2 × Xeon Silver 4216 (32 threads) | i5-13500 (20 threads), host work measured in wall time |
| CPU baselines | Pkd-tree, zd-tree on 2 × Xeon E5-2630 v4 (40 threads) | same two trees (SpaceTreeLib) on the i5-13500, 1/4/20 threads, median of 3 runs |
| Data | 300M uniform warm-up points; COSMOS (Gini 0.287), OSM (Gini 0.967) | 800k warm-up points; uniform, Zipf clusters with Gini 0.28 and 0.98 |
| Operations | insert, box count/fetch, kNN | insert and point lookup only (assignment scope) |
| PIM-zd-tree variant | throughput-optimized (and skew-resistant in Fig. 9) | public throughput-optimized code: L0 key-range split on the CPU, whole subtrees on DPUs; no push-pull, no L1 caching; L0 borders fixed from warm-up quantiles |

Points per DPU at 8 DPUs (≈119k) are comparable to the paper's density (≈146k = 300M / 2048).

## Trends

**1. Throughput grows with the number of PIM cores (Fig. `fig_scaling_throughput`).** From 8 to 64 DPUs,
PIM-zd-tree's modelled throughput rises 4.0–4.5× for inserts and 5.0–5.5× for lookups (ideal 8×). Lookups pass
the 20-thread Pkd-tree at 32–64 DPUs (22.3 vs 15.0 Mop/s, uniform, 64 DPUs). Inserts do not: 7.5 vs 18.3 Mop/s.
The paper reports PIM-zd-tree 1.82× faster than the Pkd-tree on inserts at 2048 DPUs. Our trend points the same way,
but 64 DPUs give 3.1% of the paper's PIM cores and memory bandwidth, so the crossover is not reached here.

**2. Range partitioning fails under skew, PIM-zd-tree does not (Fig. `fig_load_balance`, paper Sec. 3).** With the
public code's equal-width key ranges, strong skew (Gini 0.98) puts most of the data on one DPU: the 800k-point
data set overflows that DPU's leaf storage at every size from 8 to 64 DPUs. At 200k points it fits, but throughput
stays flat at ≈1 Mop/s (lookup) and ≈0.6 Mop/s (insert) for 8–64 DPUs, with load imbalance 3.4× → 24.7×: adding
DPUs does not help, because the hot key range stays on one DPU. With placement by data quantiles (subtree size,
as in the paper's L0 split), imbalance stays at 1.06–1.32× and lookups scale 4.6× from 8 to 64 DPUs.
Moderate skew (Gini 0.28) already costs range partitioning 35% of lookup throughput at 64 DPUs (15.3 vs 23.4 Mop/s).

**3. Skew costs PIM-zd-tree little.** Strong vs uniform at 64 DPUs: lookups 18.1 vs 22.3 Mop/s (−19%), inserts
6.7 vs 7.5 Mop/s (−11%). The remaining loss comes from imbalance growing from 1.09× to 1.24×: our borders are
fixed from the warm-up data, and with few points per DPU the hot clusters do not split evenly. The paper's Fig. 5(b, c) likewise
shows PIM-zd-tree keeping its advantage on the skewed COSMOS and OSM data.

**4. Where the time goes (Fig. `fig_time_breakdown`, paper Fig. 6).** Inserts are the most CPU-heavy operation
(42–45% host CPU at 64 DPUs: sorting, grouping and task packing), lookups are PIM-dominated (61–66% PIM). This is the
paper's pattern. CPU–PIM transfers are 6–8% of the time. As DPUs are added, PIM time shrinks but host time does
not, so the host CPU share rises from ≈11% (8 DPUs) to ≈44% (64 DPUs) for inserts. That host work, not PIM, is
what limits insert scaling here.

**5. Larger batches are faster (Fig. `fig_batch_size`, paper Fig. 7).** At 64 DPUs, insert throughput rises from
4.3 to 10.1 Mop/s and lookups from 12.7 to 29.9 Mop/s between 5k and 200k operations per batch, as fixed per-launch
costs are amortised and per-DPU work evens out. The paper sees the same rise and a drop for very large batches
caused by L3 misses on the host, which our batch sizes (≤200k) do not reach.

**6. Communication volume.** PIM-zd-tree moves 58 B per insert and 16 B per lookup that are actually needed.
Because a parallel UPMEM transfer sends every DPU the largest DPU's size, bytes moved grow with DPU count
(insert: 59 → 63 B/op from 8 to 64 DPUs; `fig_traffic`). The paper's traffic metric also includes CPU–DRAM
traffic measured with hardware counters, which we do not model.

## Where we differ from the paper, and why

- **Absolute throughput and the insert crossover.** We use 3.1% of the paper's PIM cores against a newer desktop CPU,
  so the CPU baselines look relatively strong. Our PIM numbers come from a cost model. It has no kernel-launch
  overhead, so small batches may look better than on real hardware.
- **Simulator timing.** DPU cycles come from the UPMEM functional simulator's pipeline model, not the hardware.
  Relative comparisons between designs on the same simulator are more reliable than absolute numbers.
- **Skew.** Our skew is in the data (Zipf clusters), with queries drawn from the data. The paper's Fig. 9 also
  tests a skewed *query* stream (Varden) against the skew-resistant variant, which is not in the public code.
- **Not reproduced:** thousands of DPUs, box count/fetch and kNN queries, the skew-resistant variant, push-pull
  search and L1 caching (absent from the public code), CPU–DRAM traffic counters, real data sets.

## Code changes needed to get here (see SETUP_SIMULATOR.md)

The public code needed fixes to run correctly at 8–64 DPUs and on skewed data: buffer overflows that only appear
when each DPU gets a large share of a batch, and a host-side bug that lost points on skewed data (two tasks splitting
the same tree edge). Both are fixed and verified with `--verify-each-batch`.
