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

## Changes made for the simulator

| File | Change |
|---|---|
| `host/host.hpp` | `dpu_control::alloc(NR_DPUS)` instead of `DPU_ALLOCATE_ALL` (the simulator gives only one 64-DPU rank for "all") |
| `host/host.cpp` | default `--interface` is `upmem`; `direct` only works on real UPMEM hardware |
| `pim_base/pim_interface/direct_interface.hpp` | AVX-512 byte interleave compiled only with `__AVX512F__`; otherwise it aborts with a message |
| `host/compile.hpp`, `Makefile` | DPU binary directory comes from `BUILDDIR` (was hard-coded to `build/`) |
| `Makefile` | PAPI optional: `USE_PAPI ?= 0` |
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
- Untested: skewed data with many identical points (more than `LEAF_SIZE` copies of one key). There is no skewed
  generator on this branch yet, and the trie may not be able to split identical keys.
- `Bnode.subtree_size` is `int16_t`, so it wraps above 32767 points per subtree. It is used only for range-query
  pruning, not for insert or point lookup.
- Memory: 64 DPUs with 8M points peaks at about 3.0 GB RSS; 32 DPUs with 4M at about 2.2 GB.
- Simulator wall-clock time is not a performance result. Use DPU cycle counters and modelled transfer cost.
- Extra insert rounds count toward that batch's measured cost; report them for first-batch or large-batch runs.
