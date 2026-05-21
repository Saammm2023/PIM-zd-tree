# PIM-zd-tree

This repository contains an implementation of the **throughput-optimized version of PIM-zd-tree** for the **UPMEM processing-in-memory (PIM) system**.

If you use our code, please cite our paper:

```bibtex
@inproceedings{zhao2026pim,
    author = {Zhao, Yiwei and Kang, Hongbo and Men, Ziyang and Gu, Yan and Blelloch, Guy E. and Dhulipala, Laxman and McGuffey, Charles and Gibbons, Phillip B.},
    title = {PIM-zd-tree: A Fast Space-Partitioning Index Leveraging Processing-in-Memory},
    year = {2026},
    isbn = {9798400723100},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3774934.3786411},
    doi = {10.1145/3774934.3786411},
    booktitle = {Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
    pages = {480–495},
    numpages = {16},
    location = {Sydney, NSW, Australia},
    series = {PPoPP '26}
}
```

## Build

### Prerequisites
- GNU Compiler Collection
- UPMEM SDK
- PAPI installed on your system

### Steps

1. **Configure the PAPI path**

   Edit the Makefile and update the `PAPI_INSTALL_DIR` variable to point to your local PAPI installation.

2. **Build the project**

   Run `make` in the project root directory.

This will compile the host and PIM components and generate the corresponding binaries.

## Usage

```bash
./build/zd_tree_host [options]
```

## Command-Line Options

### Overview of Test and Search Types

Options to select specific test or search modes:

- **Test types (`--test-type`)**:
```
  0. Disabled
  1. Insert
  2. Box count
  3. Box fetch
  4. kNN
```

- **Search types (`--search-type`)**:
```
  0. Disabled
  1. Point find
  2. Box count
  3. Box fetch
  4. kNN
```

### Insert Configuration

| Option                          | Default | Description                         |
| ------------------------------- | ------- | ----------------------------------- |
| `-i, --insert-batch-size <int>` | `50000` | Number of elements per insert batch |
| `-I, --insert-round <int>`      | `10`    | Number of insert rounds/batches     |

### Test Configuration

| Option                          | Default | Description                        |
| ------------------------------- | ------- | ---------------------------------- |
| `-t, --test-type <int>`         | `0`     | Test type selector                 |
| `-b, --test-batch-size <int>`   | `10000` | Number of elements per test batch  |
| `-r, --test-round <int>`        | `2`     | Number of test rounds/batches      |
| `-e, --expected-box-size <int>` | `100`   | Expected box size for test queries |

### Search Configuration

| Option                          | Default | Description                        |
| ------------------------------- | ------- | ---------------------------------- |
| `-s, --search-type <int>`       | `0`     | Search type selector               |
| `-S, --search-batch-size <int>` | `20000` | Number of queries per search batch |

### Runtime / System Options

| Option                      | Default  | Description                 |
| --------------------------- | -------- | --------------------------- |
| `--interface <string>`      | `direct` | Backend interface to use    |
| `--top-level-threads <int>` | `1`      | Number of top-level threads |
| `--debug`                   | `false`  | Enable debug output         |
| `--print-timer`             | `true`   | Print timing information    |

## Examples

Run with all defaults:

```bash
./build/zd_tree_host
```

Customize insert and test behavior:

```bash
./build/zd_tree_host -i 100000 -I 20 -b 20000 -r 5
```

Enable debugging and disable timing output:

```bash
./build/zd_tree_host --debug --print-timer=false
```

Select a different interface and search configuration:

```bash
./build/zd_tree_host --interface upmem -s 1 -S 50000
```
