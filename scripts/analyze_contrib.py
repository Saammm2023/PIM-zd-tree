#!/usr/bin/env python3
"""Figures and tables for our addition: lookup deduplication, hot-key cache and adaptive switching.

    python scripts/analyze_contrib.py     # reads results/raw/contrib_*.csv (scripts/run_sweep.sh contrib)

Same cost model as scripts/analyze.py. A run's throughput is lookups / modelled time over its batches 3-10: the first
two of its 10 lookup batches are warm-up (first-touch allocations, cache filling) and are left out; the mean over the
rest still counts the adaptive variant's occasional probe batches.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402  (cost model, style, palette)

RAW, TABLES, FIGS = A.RAW, A.TABLES, A.FIGS
WARMUP_BATCHES = 2  # lookup batches 0-1 are warm-up
SKEWS = ["uniform", "0.6", "0.8", "1.0", "1.2", "1.5"]
SKEW_LABEL = {"uniform": "uni.", "0.6": "0.6", "0.8": "0.8", "1.0": "1.0", "1.2": "1.2", "1.5": "1.5"}
# Each variant is its own series: categorical slots 1-4 in fixed order; the CPU references are neutral grays.
VARIANTS = {
    "base": ("PIM-zd-tree (baseline)", dict(color="#2a78d6", marker="o", ls="-")),
    "dedup": ("+ dedup", dict(color="#eb6834", marker="s", ls="-")),
    "cache": ("+ dedup + hot-key cache (256)", dict(color="#1baf7a", marker="^", ls="-")),
    "adaptive": ("adaptive (dedup + cache)", dict(color="#eda100", marker="v", ls=":")),
}
CPU_STYLE = {"pkd-tree": ("Pkd-tree (CPU, 20 thr.)", dict(color="#52514e", marker="D", ls="--")),
             "zd-tree": ("zd-tree (CPU, 20 thr.)", dict(color="#a3a19a", marker="x", ls="--"))}


def load():
    b = pd.read_csv(RAW / "contrib_batches.csv")
    b = b[b.op == "lookup"].copy()
    parts = b["tag"].str.split("_", expand=True)
    b = A.model_pim(b)
    b["data"], b["skew"], b["variant"] = parts[0], parts[2].str[1:], parts[3]
    send = b.groupby(["data", "nr_dpus", "skew", "variant"])["send_bytes"].sum().rename("send")
    w = b[b.batch_idx >= WARMUP_BATCHES]
    g = w.groupby(["data", "nr_dpus", "skew", "variant"]).agg(
        ops=("ops", "mean"), t_total=("t_total", "mean"), t_cpu=("t_cpu", "mean"), t_pim=("t_pim", "mean"),
        t_comm=("t_comm", "mean"), imbalance=("imbalance", "mean")).reset_index()
    g = g.merge(send.reset_index(), on=["data", "nr_dpus", "skew", "variant"])
    g["mops"] = g["ops"] / g["t_total"] / 1e6
    base = g[g.variant == "base"].set_index(["data", "nr_dpus", "skew"])["send"]
    g["pim_fraction"] = [r.send / base.get((r.data, r.nr_dpus, r.skew), np.nan) for r in g.itertuples()]
    c = pd.read_csv(RAW / "contrib_cpu.csv")
    c = c[(c.op == "lookup") & (c.batch_idx >= WARMUP_BATCHES)].copy()
    c["skew"] = c["tag"].str.split("_q").str[1]
    per_rep = c.groupby(["tree", "skew", "rep"]).agg(ops=("ops", "sum"), seconds=("seconds", "sum")).reset_index()
    per_rep["mops"] = per_rep["ops"] / per_rep["seconds"] / 1e6
    cpu = per_rep.groupby(["tree", "skew"])["mops"].median().reset_index()
    return g, cpu


def xpos(skews):
    return [SKEWS.index(s) for s in skews]


def fig_throughput(g, cpu):
    fig, axes = plt.subplots(1, 2, figsize=(A.PAGE_W, 2.4), sharey=False)
    for ax, n in zip(axes, [16, 64]):
        for v, (name, st) in VARIANTS.items():
            e = g[(g.data == "uniform") & (g.nr_dpus == n) & (g.variant == v)].copy()
            e["k"] = xpos(e["skew"])
            e = e.sort_values("k")
            ax.plot(e.k, e.mops, label=name, markeredgecolor="white", markeredgewidth=0.6, **st)
        for tree, (name, st) in CPU_STYLE.items():
            e = cpu[cpu.tree == tree].copy()
            e["k"] = xpos(e["skew"])
            e = e.sort_values("k")
            ax.plot(e.k, e.mops, label=name, lw=1.1, **st)
        ax.set_xticks(range(len(SKEWS)))
        ax.set_xticklabels([SKEW_LABEL[s] for s in SKEWS])
        ax.set_xlabel("Query skew (Zipf exponent over keys)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{n} DPUs", color=A.INK)
    axes[0].set_ylabel("Lookup throughput (Mop/s)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    A.save(fig, "fig_contrib_throughput")


def fig_balance(g):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A.PAGE_W, 2.2))
    for v, (name, st) in VARIANTS.items():
        e = g[(g.data == "uniform") & (g.nr_dpus == 64) & (g.variant == v)].copy()
        e["k"] = xpos(e["skew"])
        e = e.sort_values("k")
        ax1.plot(e.k, e.imbalance, label=name, markeredgecolor="white", markeredgewidth=0.6, **st)
        ax2.plot(e.k, e.pim_fraction, label=name, markeredgecolor="white", markeredgewidth=0.6, **st)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(SKEWS)))
        ax.set_xticklabels([SKEW_LABEL[s] for s in SKEWS])
        ax.set_xlabel("Query skew (Zipf exponent), 64 DPUs")
    ax1.set_yscale("log")
    ax1.set_yticks([1, 2, 5, 10, 20])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_ylabel("Load imbalance\n(slowest DPU / mean)")
    ax1.set_title("Lookup load imbalance", color=A.INK)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_ylabel("Lookups sent to PIM")
    ax2.set_title("Work left for the DPUs", color=A.INK)
    ax1.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    A.save(fig, "fig_contrib_balance")


def fig_cache_and_breakdown(g):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A.PAGE_W, 2.3), gridspec_kw=dict(width_ratios=[1, 1.2]))
    e = g[(g.data == "uniform") & (g.nr_dpus == 64) & (g["skew"] == "1.2")]
    sizes = []
    for v in e.variant:
        if v == "dedup":
            sizes.append(0)
        elif v == "cache":
            sizes.append(256)
        elif v.startswith("cache"):
            sizes.append(int(v[5:]))
        else:
            sizes.append(None)
    e = e.assign(size=sizes).dropna(subset=["size"]).sort_values("size")
    base = g[(g.data == "uniform") & (g.nr_dpus == 64) & (g["skew"] == "1.2") & (g.variant == "base")]["mops"]
    xs = np.arange(len(e))
    ax1.bar(xs, e.mops, color="#1baf7a", edgecolor="white", linewidth=1, width=0.7)
    if len(base):
        ax1.axhline(float(base.iloc[0]), color="#2a78d6", lw=1.2, ls="-",
                    label=f"PIM-zd-tree baseline ({float(base.iloc[0]):.1f})")
        ax1.legend(frameon=False, loc="upper right", fontsize=6, bbox_to_anchor=(1.0, 1.0))
        ax1.set_ylim(0, e.mops.max() * 1.3)
    for x, v in zip(xs, e.mops):
        ax1.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6, color=A.INK2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["0\n(dedup)" if s == 0 else f"{int(s)}" for s in e["size"]])
    ax1.set_xlabel("Hot-key cache size (keys)")
    ax1.set_title("Cache size, 64 DPUs, s = 1.2", color=A.INK)
    ax1.set_ylabel("Lookup throughput (Mop/s)")
    ax1.grid(axis="x", visible=False)

    rows = []
    for data, q in [("uniform", "uniform"), ("uniform", "1.2"), ("strong", "1.0")]:
        for v in ["base", "cache"]:
            r = g[(g.data == data) & (g.nr_dpus == 64) & (g["skew"] == q) & (g.variant == v)]
            if len(r):
                r = r.iloc[0]
                label = f"{'uniform data' if data == 'uniform' else 'strong data'}, " \
                        f"{'uniform q.' if q == 'uniform' else 's=' + q}, {'base' if v == 'base' else '+cache'}"
                rows.append((label, r.t_cpu, r.t_pim, r.t_comm, r.t_total))
    y = np.arange(len(rows))[::-1]
    comp = [("Host CPU", "#2a78d6"), ("PIM (DPU)", "#eb6834"), ("CPU-PIM transfer", "#1baf7a")]
    left = np.zeros(len(rows))
    tot = np.array([r[4] for r in rows]) * 1e3  # mean ms per 50k-lookup batch (after warm-up)
    for k, (name, col) in enumerate(comp):
        vals = np.array([r[1 + k] for r in rows]) * 1e3
        ax2.barh(y, vals, left=left, color=col, edgecolor="white", linewidth=1, height=0.7, label=name)
        left += vals
    for i in range(len(rows)):
        ax2.text(tot[i], y[i], f" {tot[i]:.1f}", va="center", fontsize=6, color=A.INK2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax2.set_xlabel("Time per 50k-lookup batch (ms)")
    ax2.set_title("Where the time goes, 64 DPUs", color=A.INK)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlim(0, tot.max() * 1.15)
    ax2.legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout()
    A.save(fig, "fig_contrib_cache_breakdown")


def write_table(g, cpu):
    A.TABLES.mkdir(parents=True, exist_ok=True)
    g.to_csv(TABLES / "contrib_summary.csv", index=False)
    cpu.to_csv(TABLES / "contrib_cpu_summary.csv", index=False)
    lines = ["# Contribution results (generated by scripts/analyze_contrib.py)", "",
             "Lookup throughput in Mop/s (modelled; lookups / time over batches 3-10 of 10 batches of 50k lookups, "
             "after 800k uniform warm-up points); in parentheses: load imbalance (slowest DPU / mean). "
             "Speedup = dedup + cache (256 keys) / baseline.", ""]
    for n in [16, 64]:
        lines += [f"## {n} DPUs", "", "| Query skew | baseline | + dedup | + dedup + cache | adaptive | speedup "
                  "(cache) | Pkd-tree CPU | zd-tree CPU |", "|---|---|---|---|---|---|---|---|"]
        for q in SKEWS:
            cells = []
            vals = {}
            for v in VARIANTS:
                e = g[(g.data == "uniform") & (g.nr_dpus == n) & (g["skew"] == q) & (g.variant == v)]
                if len(e):
                    vals[v] = e.mops.iloc[0]
                    cells.append(f"{e.mops.iloc[0]:.2f} ({e.imbalance.iloc[0]:.2f}x)")
                else:
                    cells.append("-")
            sp = f"{vals['cache'] / vals['base']:.2f}x" if "cache" in vals and "base" in vals else "-"
            cpus = []
            for tree in CPU_STYLE:
                e = cpu[(cpu.tree == tree) & (cpu["skew"] == q)]
                cpus.append(f"{e.mops.iloc[0]:.2f}" if len(e) else "-")
            lines.append(f"| {q} | " + " | ".join(cells) + f" | {sp} | " + " | ".join(cpus) + " |")
        lines.append("")
    lines += ["## Strong data skew + hot keys (s = 1.0), 64 DPUs", "", "| Variant | Mop/s | imbalance |", "|---|---|---|"]
    for v, (name, _) in VARIANTS.items():
        e = g[(g.data == "strong") & (g.variant == v)]
        if len(e):
            lines.append(f"| {name} | {e.mops.iloc[0]:.2f} | {e.imbalance.iloc[0]:.2f}x |")
    lines += ["", "## Cache size, 64 DPUs, s = 1.2", "", "| Cache keys | Mop/s |", "|---|---|"]
    e = g[(g.data == "uniform") & (g.nr_dpus == 64) & (g["skew"] == "1.2")]
    for v in ["dedup", "cache64", "cache", "cache1024", "cache4096", "cache16384"]:
        r = e[e.variant == v]
        if len(r):
            size = {"dedup": "0 (dedup only)", "cache": "256"}.get(v, v[5:])
            lines.append(f"| {size} | {r.mops.iloc[0]:.2f} |")
    (TABLES / "contrib_results.md").write_text("\n".join(lines) + "\n")
    print((TABLES / "contrib_results.md").read_text())


def main():
    g, cpu = load()
    fig_throughput(g, cpu)
    fig_balance(g)
    fig_cache_and_breakdown(g)
    write_table(g, cpu)


if __name__ == "__main__":
    main()
