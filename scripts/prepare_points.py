#!/usr/bin/env python3
"""Convert a SpaceTreeLib point file (text: "n dim" header, then one point per line) into the binary point file that
zd_tree_host and cpu_bench read with --input.

    python3 scripts/prepare_points.py <in.in> <out.pts> [--seed 1] [--grid-bits 21]

* Exact duplicates are dropped: PIM-zd-tree cannot split more than LEAF_SIZE (16) copies of one key.
* Points are shuffled (seeded) into a random insertion order; Varden writes its random walks in order.
* Coordinates are taken modulo 2^grid_bits (generate with -axis_max 2^grid_bits - 1; Varden's walks can step a
  little past it, and those points wrap around). They are scaled into the
  31-bit coordinate space by a left shift; the freed low bits get a fixed per-point hash, which keeps coordinates
  distinct without changing any key (keys keep the top 21 bits of each 31-bit coordinate).

Output: little-endian uint64 n, uint64 dim (= 3), then n * 3 int64 coordinates.
"""
import argparse
import random
import struct
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid-bits", type=int, default=21)
    a = ap.parse_args()

    with open(a.inp) as f:
        n, dim = map(int, f.readline().split())
        if dim != 3:
            sys.exit(f"expected 3-D points, got {dim}")
        pts = [tuple(map(int, line.split())) for line in f if line.strip()]
    if len(pts) != n:
        print(f"warning: header says {n} points, file has {len(pts)}")
    lim = 1 << a.grid_bits
    # Varden's random walks can step slightly past the axis bounds: wrap them around (a torus)
    wrapped = sum(1 for p in pts if min(p) < 0 or max(p) >= lim)
    pts = [tuple(c % lim for c in p) for p in pts]
    uniq = list(dict.fromkeys(pts))
    rng = random.Random(a.seed)
    rng.shuffle(uniq)
    shift = 31 - a.grid_bits
    mask = (1 << shift) - 1

    def jitter(i, k):
        return (hash((a.seed, i, k)) & 0xFFFFFFFF) & mask

    with open(a.out, "wb") as f:
        f.write(struct.pack("<QQ", len(uniq), 3))
        buf = bytearray()
        for i, p in enumerate(uniq):
            buf += struct.pack("<qqq", *[(c << shift) | jitter(i, k) for k, c in enumerate(p)])
        f.write(buf)
    print(f"{a.inp}: {len(pts)} points, {wrapped} wrapped into [0, 2^{a.grid_bits}), "
          f"{len(pts) - len(uniq)} exact duplicates dropped, {len(uniq)} written to {a.out}")


if __name__ == "__main__":
    main()
