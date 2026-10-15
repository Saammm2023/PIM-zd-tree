# Our addition: skew-aware lookups for the throughput-optimized PIM-zd-tree

Results: `results/tables/contrib_results.md`, figures `results/figures/fig_contrib_*.pdf|png`.
Reproduce: `scripts/run_sweep.sh contrib`, then `scripts/analyze_contrib.py`.

## The limitation we address

The paper offers two configurations (Table 2). The **throughput-optimized** one is fast, but its allowed skew is
only (P log P, n/P): every lookup goes to the PIM module that owns its key's subtree. The paper's Fig. 9 shows it
falling by up to 10.66× when 2% of a query batch comes from a skewed distribution. Resisting that skew needs the
**skew-resistant** configuration: replicated upper levels, push-pull search, a different chunking. It is slower when
there is no skew, and it is not in the public code.

The public code is the throughput-optimized version. We confirmed the problem in our setup: with Zipf-distributed
lookup keys over 800k uniform points on 64 DPUs, the busiest DPU does 4.5× (s = 1.0) to 21× (s = 1.5) the average
work. Lookup throughput falls from 24.2 to 1.6 Mop/s, below both CPU baselines.

## What we added (host side only; the DPU program is unchanged)

1. **In-batch deduplication.** The host sends each distinct key of a batch once and copies the reply to every
   duplicate. Distinct keys are found with a lock-free open-addressing hash table that persists between batches
   (0.2 ms for 50k keys, about 10× faster than sorting). This removes the BSP straggler that hot keys create.
2. **Hot-key cache.** A small host-side set of keys (256 by default) that recent batches showed to be hot (repeated
   within a batch) and present. Those lookups are answered on the CPU and never reach PIM. Entries cannot go stale:
   a lookup returns the key itself exactly when the key is stored, and the workload has no deletes (a delete would
   remove its key from the cache). Admission uses exponentially decayed in-batch counts, and only the batch's top
   `capacity` repeated keys are considered, so maintenance stays bounded.
3. **Adaptive switching.** Dedup and cache only pay off when a BSP round would wait on a hot DPU *and* the hot work
   repeats. The host knows every lookup's target DPU before sending, so it builds a per-DPU histogram for each batch
   and takes the new path only when the busiest DPU would get more than 1.2× the mean (`--adaptive-threshold`) and the
   last batch that took the new path kept at least 5% of its lookups off the DPUs (`--adaptive-repeat-threshold`,
   re-checked every 8 batches). Otherwise it uses the original path. The repeat check was added after the Varden
   benchmark (`results/VARDEN.md`): hot spots of *distinct* keys have nothing to deduplicate, and the first version of
   the switch lost up to 24% there.

This is the paper's push-pull idea applied at key granularity and over time: hot work is pulled to the CPU. It runs
on the unchanged throughput-optimized structure instead of switching to the skew-resistant layout.

Options: `--lookup-dedup`, `--hot-cache-size N`, `--lookup-adaptive [--adaptive-threshold T]`. Workload options:
`--query-dist uniform|zipf --query-zipf-s s --search-round R`; `cpu_bench` accepts the same, so the CPU baselines see
the identical query stream.

## Evaluation

800k uniform warm-up points (quantile placement), then 10 batches of 50k lookups. Each query is a stored point whose
popularity rank follows Zipf(s); the ranks are scattered over the data set by a fixed permutation. Throughput is
lookups / modelled time over batches 3–10 (two warm-up batches left out; the adaptive switch's probe batches counted),
using the same cost model as the baseline results. Every PIM run returned the correct answer for every lookup.

**Lookup throughput, 64 DPUs (Mop/s; load imbalance in parentheses):**

| Query skew | Baseline | + dedup | + dedup + cache | adaptive | Pkd-tree CPU | zd-tree CPU |
|---|---|---|---|---|---|---|
| uniform | 24.2 (1.10×) | 19.2 | 19.1 | **24.5** | 16.6 | 10.9 |
| s = 0.6 | 23.9 (1.12×) | 20.4 | 19.2 | **24.2** | 16.2 | 11.5 |
| s = 0.8 | 17.5 (1.65×) | 21.3 | 20.9 | **22.0** (1.10×) | 16.3 | 13.4 |
| s = 1.0 | 7.2 (4.45×) | **31.8** | 31.1 | 27.8 (1.13×) | 18.5 | 17.6 |
| s = 1.2 | 3.2 (10.7×) | 40.5 | **48.3** | 44.1 (1.19×) | 20.9 | 20.8 |
| s = 1.5 | 1.6 (21.4×) | 49.2 | **93.3** | 74.9 (1.24×) | 22.0 | 23.1 |

16 DPUs show the same pattern: dedup + cache is 1.40× (s = 0.8) to 50.7× (s = 1.5) faster than the baseline.
With strongly skewed data *and* hot keys (s = 1.0, 64 DPUs), the baseline does 5.7 Mop/s at 5.3× imbalance and dedup
does 29.8 Mop/s (5.2×). With hot keys over Varden data: 6.6 → 29.6 Mop/s (s = 1.0) and 2.9 → 50.6 Mop/s (s = 1.2).

## Findings

- **Imbalance is the problem, and dedup removes it.** With dedup, imbalance stays at 1.10–1.24× for every skew
  (`fig_contrib_balance`, left), against 21× for the baseline. At s = 1.5 only 4% of lookups still reach PIM
  (`fig_contrib_balance`, right).
- **The cache matters at heavy skew.** Dedup sends each hot key once per batch, so the cache's value is removing the
  remaining distinct hot keys from every batch: at s = 1.5 it nearly doubles throughput over dedup alone
  (93.3 vs 49.2 Mop/s). At moderate skew its upkeep costs about as much as it saves (s = 1.0: 31.1 vs 31.8).
- **Cache size has a sweet spot** (`fig_contrib_cache_breakdown`, left; s = 1.2): 64 keys give 52 Mop/s, 256 keys
  48, and 1k–16k keys 44–45. Beyond the truly hot keys, a larger cache adds maintenance and lookup cost on the host
  without catching more hot queries.
- **Neither variant is free.** At 64 DPUs a uniform batch takes about 2 ms, and dedup + cache adds about 0.5 ms of host
  work, which costs 21% when there is no skew. The adaptive switch avoids that: it took the original path in every
  batch for uniform and s = 0.6 queries and the new path in every batch from s = 0.8. Its per-DPU histogram costs up to
  0.2 ms, so it stays within 8% of the baseline without skew, and it trails the best fixed variant at high skew.
- **Time moves from PIM to the host** (`fig_contrib_cache_breakdown`, right). At s = 1.2 a batch drops from 15.9 ms
  (86% PIM) to 1.0 ms, of which the host is the largest share. The PIM side stops being the bottleneck.
- **Versus the CPU.** Hot keys help the CPU trees a little (caches): Pkd-tree 16.6 → 22.0 Mop/s. With our addition,
  PIM-zd-tree passes both CPU trees from s = 0.8 at 64 DPUs. The baseline falls below them from s = 0.8.

## Limits

- Hot *keys* only. Many distinct keys in one small region (Varden hot spots, `results/VARDEN.md`) still overload the
  DPU owning that region, because dedup and the cache only act on repeated keys; the repeat check only keeps the
  switch from making that case worse. It needs re-partitioning (the paper's skew-resistant layout, or adaptive L0
  borders).
- Point lookups only. Dedup carries over to inserts only partly: duplicate inserts of the same point are legal and
  must all be stored.
- The cache assumes no deletes. With deletes, a delete batch must first remove its keys from the cache (one hash
  lookup per deleted key).
- Throughput is modelled (DPU cycles + PrIM transfer bandwidths + measured host time); host time is measured on a
  desktop CPU at the millisecond scale, so we leave out two warm-up batches per run.
