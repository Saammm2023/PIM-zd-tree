# Running PIM-zd-tree on the UPMEM functional simulator (Group 1)

Branch `simulator` of cmuparlay/PIM-zd-tree, tested on Ubuntu 24.04, i5-13500 (no AVX-512), no sudo.

## 1. UPMEM SDK 2025.1.0

`sdk.upmem.com` / `sdk-releases.upmem.com` no longer resolve. We use the Internet Archive copy of the
official Ubuntu 22.04 tarball, whose headers are identical to the authors' fork in `third-party/UPMEM-sdk`:

```
https://web.archive.org/web/20250528120859id_/http://sdk-releases.upmem.com/2025.1.0/ubuntu_22.04/upmem-2025.1.0-Linux-x86_64.tar.gz
```

SHA-1 (base32, matches the archive record): `4OUA3ZMD63JC4HD3EYRQKLO6RDY2OB5Y`.
Extract it to `~/upmem-sdk`.

(CMU-SAFARI/UPMEM ships SDK 2023.1.0; its API headers differ from what this code was written against,
so we did not use it.)

## 2. libpython3.10

`libdpu.so` links against `libpython3.10`; Ubuntu 24.04 only has 3.12. Without sudo, use a standalone build:

```
https://github.com/astral-sh/python-build-standalone/releases/download/20260924/cpython-3.10.21%2B20260924-x86_64-unknown-linux-gnu-install_only.tar.gz
sha256 7e80168bf45472a17da889caa06519024e8589d51656b73cfa6f38cb2236a233
```

Extract it and move the `python` directory to `~/upmem-sdk/python310`.
(With sudo: `apt install libpython3.10` from the deadsnakes PPA instead.)

## 3. Environment

`~/upmem-sdk/env-sim.sh`:

```bash
_UPMEM_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"
source "${_UPMEM_DIR}/upmem_env.sh" simulator > /dev/null
export LD_LIBRARY_PATH="${_UPMEM_DIR}/python310/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${_UPMEM_DIR}/python310/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

Source it in every shell: `source ~/upmem-sdk/env-sim.sh`

## 4. Build (one build directory per DPU count)

```bash
make NR_DPUS=8  BUILDDIR=build8  -j6
make NR_DPUS=64 BUILDDIR=build64 -j6
```

## 5. Run

```bash
./build8/zd_tree_host -i 10000 -I 10 -s 1 -S 20000
```

Expected: `Total dataset size: 100000` and `Match Num: 20000; Unmatch Num: 1`.
The lookup batch has `search_batch_size + 1` queries. The extra query, and any left over when `-S` is not a
multiple of `-I`, is a random point that is not in the dataset, so `Unmatch Num` = 1 + (`-S` mod `-I`).
Checked: `-i 30000 -I 4 -S 2003` gives `Match Num: 2000; Unmatch Num: 4`.

## Data, skew and placement

| Option | Meaning |
|---|---|
| `--dist uniform` | uniform random points (default) |
| `--dist zipf --zipf-s S --clusters K --cluster-width-log2 W` | K cluster centres placed uniformly; cluster rank r is chosen with probability ∝ 1/r^S; points are spread around the centre (triangular, half-width 2^W per dimension, wrapping around the space) |
| `--seed N` | data seed. Points are a pure function of (seed, batch, index), so runs are reproducible |
| `--placement quantile` | default. Each DPU owns a key range holding the same number of warm-up points (theta_L0 = n/P in the paper's throughput-optimized configuration) |
| `--placement equal` | equal-width key ranges: the naive range-partitioned design (paper Sec. 3) |
| `--verify-each-batch` | after each warm-up batch, look up every point inserted so far (slow) |

Calibrated skew levels (Gini of point counts over 2048 equal key ranges, as in paper Sec. 7.2; 1M points):

| Name | Options | Gini | Paper dataset |
|---|---|---|---|
| uniform | `--dist uniform` | 0.03 | — |
| moderate | `--dist zipf --zipf-s 0.6 --cluster-width-log2 28` | 0.28 | COSMOS, 0.287 |
| strong | `--dist zipf --zipf-s 1.5 --cluster-width-log2 22` | 0.98 | OSM, 0.967 |

Each run prints a `Dataset:` line and, with `--stats-csv`, appends it to `<prefix>_dataset.csv`.

Effect of placement, 8 DPUs, 1M points with `--zipf-s 1.0 --cluster-width-log2 22`: lookup load (slowest DPU's cycles / average)
is 1.05 with `quantile` and 1.77 with `equal`. With strong skew, `equal` puts about 43% of the points on one DPU and
overflows its leaf storage.

What the public code does and does not implement: the top layer (L0) is a fixed key-range split done on the CPU, and
each DPU holds the whole subtree for its range. There is no push-pull search and no L1 caching. That matches the paper's
throughput-optimized configuration (theta_L1 = 1, chunk size n/P), except that our borders are fixed from the warm-up
data, while the paper's L0 adapts through promotion and demotion.

## Our addition: skew-aware lookups (see `results/CONTRIBUTION.md`)

| Option | Meaning |
|---|---|
| `--query-dist uniform\|zipf --query-zipf-s s` | lookup keys drawn from the warm-up points, uniformly or with Zipf(s) popularity (hot keys); `sample` (default) is the original query set |
| `--search-round R` | R lookup batches of `-S` queries |
| `--lookup-dedup` | send each distinct key of a batch once |
| `--hot-cache-size N` | host-side cache of N hot, present keys (implies `--lookup-dedup`) |
| `--lookup-adaptive [--adaptive-threshold 1.2] [--adaptive-repeat-threshold 0.05]` | use dedup/cache only when the busiest DPU would get > threshold × mean lookups and the last batch on that path saved ≥ 5% of lookups (re-checked every 8 batches) |

`cpu_baseline/cpu_bench` takes the same `--query-dist`, `--query-zipf-s` and `--search-round`.

## Point files and the Varden benchmark (see `results/VARDEN.md`)

| Option | Meaning |
|---|---|
| `--input file.pts` | read points from a file instead of generating them: warm-up = its first `-i × -I` points, timed inserts = the points after them (host and `cpu_bench`) |
| `--query-dist mix --query-file q.pts --query-mix f` | each lookup comes from `q.pts` with probability f, otherwise from a uniformly chosen stored point (paper Fig. 9) |

`scripts/prepare_points.py` converts a SpaceTreeLib point file (e.g. Varden from `tests/data_generate.cpp`) into this
binary format. `scripts/run_sweep.sh varden` runs the Varden experiments; `scripts/analyze_varden.py` makes the
figures and tables.

## CPU baselines (`cpu_baseline/`)

`cpu_baseline/cpu_bench` runs the paper's two shared-memory competitors from
[SpaceTreeLib](https://github.com/ucrparlay/SpaceTreeLib) (commit `212220f`, header-only, needs g++ ≥ 14):
the Pkd-tree (`psi::kd_tree`, max-stretch dimension and object-median split) and the original zd-tree (`include/baselines/zdtree_3d`).
It uses the same data generator (`host/datagen_core.hpp`) and the same flags as the PIM host, so both see identical
points, insert batches and lookup queries.

```bash
git clone --recursive https://github.com/ucrparlay/SpaceTreeLib.git ../SpaceTreeLib
make -C cpu_baseline
PARLAY_NUM_THREADS=20 cpu_baseline/cpu_bench --tree pkd -i 100000 -I 10 -t 1 -b 50000 -r 2 -s 1 -S 50000 \
    --dist zipf --zipf-s 1.5 --cluster-width-log2 22 --stats-csv results/cpu --stats-tag strong
```

`--tree pkd|zd`. Each batch appends `tree,tag,threads,op,batch_idx,ops,seconds,found` to `<prefix>_cpu.csv`. Times are
wall-clock on the host CPU (20 threads on the i5-13500 by default; set `PARLAY_NUM_THREADS`).

Lookups: the Pkd-tree uses `contains(p)`. The zd-tree wrapper's `range_count` is not `const`, so `contains` does not
compile for it, and it uses a 1-NN search instead (found when the distance is 0).

The bundled zd-tree builds its Morton code from only the low 21 bits of each coordinate. Given our 31-bit coordinates, its curve
order was effectively random and every query scanned most of the tree (about 700 µs per lookup). It now gets `coord >> 10`, the
same 21 bits per dimension that PIM-zd-tree keys use. SpaceTreeLib itself is unmodified.

Smoke test (1M warm-up points, 2 × 50k timed inserts, 50k lookups, 20 threads; every query sampled from the data was found):

| Tree | Data | Insert Mop/s | Lookup Mop/s |
|---|---|---|---|
| Pkd-tree | uniform / moderate / strong | 15.5 / 15.7 / 16.0 | 14.6 / 14.8 / 13.8 |
| zd-tree | uniform / moderate / strong | 6.5 / 7.1 / 6.8 | 7.7 / 8.6 / 7.4 |

## Experiments and figures

```bash
source ~/upmem-sdk/env-sim.sh
make -C cpu_baseline
scripts/run_sweep.sh all          # cpu | scaling | batch | skewcmp | contrib; about 90 minutes in total
../.venv/bin/python scripts/analyze.py
../.venv/bin/python scripts/analyze_contrib.py
```

`analyze.py` needs matplotlib, NumPy and pandas. Without root, create the venv from the standalone Python:
`~/upmem-sdk/python310/bin/python3 -m venv ../.venv && ../.venv/bin/pip install matplotlib numpy pandas`.
Results land in `results/`: raw CSVs, logs, `tables/baseline_results.md`, and `figures/*.pdf|png`. The comparison
with the paper is in `results/ANALYSIS.md`.

## Measurement (`--stats-csv`)

```bash
./build8/zd_tree_host -i 10000 -I 10 -t 1 -b 10000 -r 2 -s 1 -S 20000 --stats-csv results/run --stats-tag uniform
```

Rows are appended (headers written once) to `results/run_batches.csv`, one row per batch, and
`results/run_dpus.csv`, one row per DPU per batch. Batches are tagged `warmup_insert` (the `-i/-I` rounds),
`insert` (the timed `-t 1` rounds) and `lookup` (the `-s 1` batch).

| Column | Meaning |
|---|---|
| `launches` | DPU launches in the batch (insert = search + insert, more if rounds were split) |
| `pim_cycles_bsp` | sum over launches of the slowest DPU's cycles: PIM time = this / 350 MHz |
| `pim_cycles_mean` | average DPU's cycles over the batch; `dpu_cycles_max_total / pim_cycles_mean` = load imbalance |
| `send_bytes`, `recv_bytes` | useful bytes host→DPU and DPU→host, summed over DPUs |
| `*_padded` | bytes actually moved: each parallel transfer sends the largest DPU's size to every DPU |
| `*_max_dpu` | sum over transfers of the per-DPU transfer size (input for a bandwidth model) |
| `wall_s`, `sim_s`, `cpu_s` | host wall time, time inside simulated DPU runs and transfers, and their difference (host CPU work) |

DPU cycles come from `launch_cycles`, set by tasklet 0 in `run()` (`pim_base/dpu/driver.h`): the cycle counter is
reset at the start of each launch and read after all tasklets finish. The per-launch read happens outside the
timed host work.

## Changes made for the simulator

| File | Change |
|---|---|
| `host/host.hpp` | `dpu_control::alloc(NR_DPUS)` instead of `DPU_ALLOCATE_ALL` (the simulator gives only one 64-DPU rank for "all") |
| `host/host.cpp` | default `--interface` is `upmem`; `direct` only works on real UPMEM hardware |
| `pim_base/pim_interface/direct_interface.hpp` | AVX-512 byte interleave compiled only with `__AVX512F__`; otherwise it aborts with a message |
| `host/compile.hpp`, `Makefile` | DPU binary directory comes from `BUILDDIR` (was hard-coded to `build/`) |
| `Makefile` | PAPI optional: `USE_PAPI ?= 0` |
| `pim_base/host/pim_stats.hpp`, `pim_base/dpu/driver.h`, `task_framework_host.hpp`, `host/host.cpp` | per-batch DPU cycles, bytes and host time to CSV (`--stats-csv`, `--stats-tag`) |
| `host/datagen_core.hpp`, `cpu_baseline/` | generator core shared with the CPU baselines; CPU driver for SpaceTreeLib |
| `host/datagen.hpp`, `host/host.cpp` | reproducible uniform / Zipf-cluster generator, dataset Gini, `--placement quantile|equal`, `--verify-each-batch` |
| `host/datagen_core.hpp`, `host/host.cpp`, `cpu_baseline/`, `scripts/prepare_points.py` | point-file input and Varden query mixes |
| `host/operations.hpp` (`search_maximum_match`) | prints only the first 10 mismatched lookups (printing thousands counted as host time) |
| `host/hot_cache.hpp`, `host/operations.hpp` (`search_maximum_match`), `host/datagen_core.hpp` | our addition: hot-key query workloads, lookup dedup, hot-key cache, adaptive switching |
| `host/operations.hpp` (`insert`) | **bug fix**: insert tasks are grouped by target, not by consecutive run in key order (see below) |
| `host/operations.hpp` (`insert`) | insert tasks capped at `MAX_INSERT_TASK_LEN` points; the extra points go into another insert round (see Known limits) |
| `common/configs.h`, `dpu/configs_dpu.h` | `MRAM_BUFFER_SIZE` moved to `common/configs.h`; new `INSERT_SCRATCH_CAPACITY` / `MAX_INSERT_TASK_LEN` |
| `dpu/single_node.h` (`p_insert`) | subtree-build queue bounded to its own quarter of the scratch (it ran into the next tasklet's area); capacity assert (only active with `ZHAOYW_DEBUG`) |
| `host/host.cpp` | the random extra lookup queries are initialised for every `-I` (the loop bound was `search_per_batch + 1`, which left them uninitialised for `-I` > 1) |
| `common/configs.h` | `MAX_TASK_BUFFER_SIZE_PER_DPU` 6.4 MB → 5.75 MB, which grows the DPU reply area from 0.93 MB to 1.9 MB; new `MAX_INSERT_TASKS_PER_DPU` |
| `host/operations.hpp` (`insert`) | also caps insert tasks (leaf groups) per DPU at `MAX_INSERT_TASKS_PER_DPU`; extra groups go into another round |
| `pim_base/host/debug.hpp`, `task_framework_host.hpp` | always-on `CHECK_BOUND` on the host IO buffers (the old `ASSERT`s were compiled out, so overflows corrupted memory silently); on a DPU fault the host prints every DPU's log |
| `pim_base/dpu/debug.h`, `task_framework_dpu.h`, `dpu/storage.h` | always-on `ALWAYS_IN_DPU_ASSERT` for reply-area overflow and full leaf/internal-node storage: the DPU prints the reason, then faults |

## Known limits (measured 2026-10-14)

### Insert-batch crash: cause and fix

`-i 50000 -I 2` on 8 DPUs faulted (`dpu is in fault`, `insert -> insert_vector`), and `-i 100000 -I 1`
returned `Match Num: 0`. The cause was not the 2560-DPU-sized task buffers, the stack, or WRAM:

- The host sends one `SINGLE_INSERT` task for each target leaf, carrying every batch point that lands there.
- `p_insert` builds the new subtree in that tasklet's MRAM scratch, `MRAM_BUFFER_SIZE / NR_TASKLETS` = 128 KB.
  That holds 3072 points (vectors + keys) at most. A longer task silently overwrote the queue and the next
  tasklet's scratch.
- On an empty tree, all of a DPU's points hit one or two root children. The first batch therefore yields
  2 tasks per DPU of about `batch / NR_DPUS / 2` points each: 1983 at 30k (OK), 3211 at 50k (corrupts the tree,
  so the next batch faults), 6371 at 100k (garbage tree, 0 matches).
- With 2560 DPUs and the paper's batch sizes, a task never got near 3072 points. At 8–64 DPUs it does.

Fix: `insert()` caps each task at `MAX_INSERT_TASK_LEN` = 3056 (3072 minus a full leaf). An oversized leaf group
keeps every k-th point in the current round. The rest are re-searched and inserted in another round inside the
same `insert()` call. In practice the first batch into an empty tree takes 3 exec rounds; later uniform batches
take 1 (unchanged). Extra rounds count toward that batch's measured cost, so report them for first-batch or
heavily skewed runs.

Verified after the fix (uniform data, `-s 1 -S 20000`; every dataset query matches, `Unmatch Num: 1`):

| DPUs | runs | largest batch per DPU |
|---|---|---|
| 8  | `-i 50000 -I 2`, `-i 100000 -I 1`, `-i 200000 -I 5`, `-i 800000 -I 1`, `-i 800000 -I 1 -S 800000`, `-i 944000 -I 1` | 118k |
| 16 | `-i 400000 -I 2`, `-i 1000000 -I 1` | 62.5k |
| 32 | `-i 800000 -I 1`, `-i 2000000 -I 1` | 62.5k |
| 64 | `-i 1000000 -I 2`, `-i 2000000 -I 1` | 31k |

### Second round of fixes (larger batches)

Three more limits were hit, all silently, because `ASSERT` and `IN_DPU_ASSERT` are compiled out in normal builds:

1. **Reply area.** Each DPU's 64 MB MRAM holds about 52 MB of static data plus a `MAX_TASK_BUFFER_SIZE_PER_DPU`
   reply scratch (`send_varlen_buffer_tmp`). The heap after that holds the receive area
   (`MAX_TASK_BUFFER_SIZE_PER_DPU`) and then the reply area. At 6.4 MB the reply area was 0.93 MB, about 119k
   8-byte replies, so 968k inserts on 8 DPUs faulted. At 5.75 MB it is 1.9 MB. Fixed: 968k now works.
2. **Tasks per DPU per block** (`MAX_TASK_COUNT_PER_DPU_PER_BLOCK` = 12,800). A later insert batch sends one
   variable-length task per target leaf, and more than 12,800 tasks to one DPU overran a host offsets array.
   That was the host segfault for `-i 400000 -I 5` on 8 DPUs. Fixed: `insert()` defers leaf groups beyond
   `MAX_INSERT_TASKS_PER_DPU` to another round, so batches above about 12.8k points per DPU take 2+ rounds.
3. **Leaf storage** (`P_BUFFER_SIZE` = 36 MB = 65,536 leaves per DPU). This is a capacity limit, not a bug. Now
   reported: `leaf storage full (65536 leaves); store fewer points per DPU`.

Verified (uniform data; every dataset query matches):

| DPUs | runs | stored points per DPU |
|---|---|---|
| 8  | `-i 50000 -I 2 -S 20000`, `-i 200000 -I 5 -S 500000`, `-i 968000 -I 1`, `-i 100000 -I 11`, `-i 100000 -I 10 -S 968000` | up to 137k |
| 16 | `-i 400000 -I 5 -S 1000000`, `-i 1000000 -I 2` | 125k |
| 32 | `-i 2000000 -I 2 -S 2000000` | 125k |
| 64 | `-i 2000000 -I 2 -S 2000000`, `-i 1000000 -I 8 -S 2000000` | 125k |

### Remaining limits

- **Stored points per DPU: at most about 130k** (uniform). 1.1M on 8 DPUs (137k per DPU) works; 1.2M (150k per DPU)
  fills leaf storage. Leaves average only about 3–4 points, so skewed data may fill storage sooner. The paper uses
  about 117k points per DPU (300M on 2560 DPUs).
- **Lookups per DPU per batch: at most about 120k.** 968k on 8 DPUs works. 1.6M (200k per DPU) faults inside the
  lookup kernel on DPU 0 without a message (cause not found). 2.08M is stopped by the host check: lookups queue
  about 37 B per query, which exceeds the 5.75 MB task area.
- **Insert points per DPU per batch:** limited by the task area (24 B per point, about 245k) and by stored
  points per DPU; extra rounds handle task-length and task-count caps.
- Total batch size is capped at `BATCH_SIZE` = 2.1M (host arrays); use more rounds (`-I`) for larger datasets.
- Many identical keys break the tree: a leaf cannot split more than `LEAF_SIZE` (16) copies of one key. Keys keep the
  top 21 bits of each coordinate, so points closer than 1024 in every dimension share a key. An early version of the
  generator clamped wide clusters at the edges of the space; that piled points onto the faces and filled
  internal-node storage. The generator now wraps around.
- `Bnode.subtree_size` is `int16_t`, so it wraps above 32767 points per subtree. It is used only for range-query
  pruning, not for insert or point lookup.
- Memory: 64 DPUs with 8M points peaks at about 3.0 GB RSS; 32 DPUs with 4M at about 2.2 GB.
- Simulator wall-clock time is not a performance result. Use DPU cycle counters and modelled transfer cost.
- Extra insert rounds count toward that batch's measured cost; report them for first-batch or large-batch runs.

### Lost points with skewed data (fixed)

With Zipf-cluster data, lookups of points inserted in earlier batches failed: for example, 27 of 80,000 points were
lost when batch 3 of `-i 20000 -I 5 --zipf-s 1.0` was inserted on 8 DPUs. This was deterministic (it also happened
with one tasklet). The cause was in the authors' host code:
`insert()` sorts points by key and groups *consecutive* points with the same search result into one task. New points
on both sides of an existing subtree C mismatch C's compressed edge, so they all target the same edge above C (same
parent and slot). C's own descendants sit between them in key order, so the host sent two tasks for one edge. The first
task inserted a node N above C. The second treated N as the original child and built another node at a height inside
N's prefix, which disconnected most of C's subtree. Dense clusters make this common, while uniform data rarely triggers it.

Fix: stable-sort the search results by target before building tasks, so each target gets one task with its points in
key order (paper Alg. 2 step 2d: the CPU deduplicates conflicting inserts). Verified with `--verify-each-batch` on 8 and
64 DPUs, for uniform, moderate and strong skew at 1M points: every point is found after every batch.

