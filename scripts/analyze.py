#!/usr/bin/env python3
"""Cost model, summary tables and figures for the PIM-zd-tree reproduction.

    python scripts/analyze.py            # reads results/raw/*.csv, writes results/tables and results/figures

Simulator wall-clock time is not a performance result, so PIM time per batch is modelled from what the
simulator measures exactly (DPU cycles, bytes moved) plus the host CPU time measured on this machine:

    T_batch = T_cpu + T_pim + T_send + T_recv
    T_pim   = sum over launches of the slowest DPU's cycles / 350 MHz          (BSP rounds)
    T_send  = padded host->DPU bytes / BW_to(n),  T_recv = padded DPU->host bytes / BW_from(n)
    BW(n)   = BW(1) * n^a, fitted to the 1-DPU and 64-DPU (one rank) bandwidths measured by PrIM

PrIM: Gomez-Luna et al., "Benchmarking a New Paradigm: Experimental Analysis and Characterization of a Real
Processing-in-Memory System", IEEE Access 2022 (arXiv 2105.03814), Sec. 3.4: parallel CPU->DPU 0.33 GB/s
(1 DPU) and 6.68 GB/s (64 DPUs); DPU->CPU 0.12 GB/s and 4.74 GB/s. DPU clock 350 MHz (Sec. 2.2).
All configurations here (8-64 DPUs) fit in one rank.
"""
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

# ---- cost model parameters ----
F_DPU = 350e6
BW_TO_1, BW_TO_64 = 0.33e9, 6.68e9
BW_FROM_1, BW_FROM_64 = 0.12e9, 4.74e9
LAUNCH_OVERHEAD_S = 0.0  # PrIM reports none; kept explicit so a sensitivity run can set it

A_TO = math.log(BW_TO_64 / BW_TO_1) / math.log(64)
A_FROM = math.log(BW_FROM_64 / BW_FROM_1) / math.log(64)


def bw_to(n):
    return BW_TO_1 * n ** A_TO


def bw_from(n):
    return BW_FROM_1 * n ** A_FROM


# ---- figure style: IEEE two-column paper ----
COL_W, PAGE_W = 3.5, 7.16
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "savefig.dpi": 300, "pdf.fonttype": 42,
})
# Reference categorical palette, fixed order (dataviz skill references/palette.md, light mode)
SERIES = {
    "PIM-zd-tree": dict(color="#2a78d6", marker="o", ls="-"),
    "Range-partitioned PIM": dict(color="#eb6834", marker="s", ls="-"),
    "Pkd-tree (CPU)": dict(color="#1baf7a", marker="^", ls="--"),
    "zd-tree (CPU)": dict(color="#eda100", marker="D", ls="--"),
}
INK, INK2 = "#0b0b0b", "#52514e"
DATASETS = ["uniform", "moderate", "strong"]
DATA_LABEL = {"uniform": "Uniform", "moderate": "Moderate skew (Gini 0.28)", "strong": "Strong skew (Gini 0.98)"}
PLACEMENT_SERIES = {"quantile": "PIM-zd-tree", "equal": "Range-partitioned PIM"}
TREE_SERIES = {"pkd-tree": "Pkd-tree (CPU)", "zd-tree": "zd-tree (CPU)"}


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def model_pim(b):
    """Adds modelled times (s) to a *_batches.csv frame."""
    b = b.copy()
    n = b["nr_dpus"]
    b["t_pim"] = b["pim_cycles_bsp"] / F_DPU + b["launches"] * LAUNCH_OVERHEAD_S
    b["t_send"] = b["send_bytes_padded"] / bw_to(n)
    b["t_recv"] = b["recv_bytes_padded"] / bw_from(n)
    b["t_comm"] = b["t_send"] + b["t_recv"]
    b["t_cpu"] = b["cpu_s"]
    b["t_total"] = b["t_cpu"] + b["t_pim"] + b["t_comm"]
    b["imbalance"] = b["dpu_cycles_max_total"] / b["pim_cycles_mean"]
    b["dataset"] = b["tag"].str.split("_").str[0]
    b["placement"] = b["tag"].str.split("_").str[1]
    return b


def summarise_pim(b):
    keys = ["dataset", "placement", "nr_dpus", "op"]
    g = b[b["op"].isin(["insert", "lookup"])].groupby(keys)
    s = g.agg(ops=("ops", "sum"), t_total=("t_total", "sum"), t_cpu=("t_cpu", "sum"), t_pim=("t_pim", "sum"),
              t_comm=("t_comm", "sum"), imbalance=("imbalance", "mean"), launches=("launches", "mean"),
              bytes_padded=("send_bytes_padded", "sum"), recv_padded=("recv_bytes_padded", "sum"),
              bytes_useful=("send_bytes", "sum"), recv_useful=("recv_bytes", "sum")).reset_index()
    s["mops"] = s["ops"] / s["t_total"] / 1e6
    s["bytes_per_op"] = (s["bytes_padded"] + s["recv_padded"]) / s["ops"]
    s["useful_bytes_per_op"] = (s["bytes_useful"] + s["recv_useful"]) / s["ops"]
    return s


def summarise_cpu(c, keys=("tree", "tag", "threads", "op")):
    """Throughput per repetition (all batches of the op), then the median over repetitions."""
    keys = list(keys)
    c = c[c["op"].isin(["insert", "lookup"])]
    per_rep = c.groupby(keys + ["rep"]).agg(ops=("ops", "sum"), seconds=("seconds", "sum"),
                                            found=("found", "max")).reset_index()
    per_rep["mops"] = per_rep["ops"] / per_rep["seconds"] / 1e6
    s = per_rep.groupby(keys).agg(mops=("mops", "median"), mops_min=("mops", "min"), mops_max=("mops", "max"),
                                  reps=("rep", "nunique"), found=("found", "max")).reset_index()
    return s


def read_failures(prefix):
    p = RAW / f"{prefix}_failures.csv"
    if not p.exists():
        return pd.DataFrame(columns=["tag", "nr_dpus", "status", "reason"])
    return pd.read_csv(p, names=["tag", "nr_dpus", "status", "reason"])


# ---------------------------------------------------------------- figures
def fig_scaling(pim, cpu, failures):
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, 3.9), sharex=True)
    dpus = sorted(pim["nr_dpus"].unique())
    for r, op in enumerate(["insert", "lookup"]):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            ymax = 0
            for pl, name in PLACEMENT_SERIES.items():
                d = pim[(pim.dataset == ds) & (pim.placement == pl) & (pim.op == op)].sort_values("nr_dpus")
                st = SERIES[name]
                ax.plot(d["nr_dpus"], d["mops"], color=st["color"], marker=st["marker"], ls=st["ls"], label=name,
                        markeredgecolor="white", markeredgewidth=0.6)
                ymax = max(ymax, d["mops"].max() if len(d) else 0)
                fails = failures[failures.tag == f"{ds}_{pl}"]
                for _, f in fails.iterrows():
                    ax.plot([f.nr_dpus], [0], marker="x", color=st["color"], ms=5, clip_on=False)
                if len(fails):
                    ax.text(0.97, 0.06, f"x: {name} ran out of\nleaf storage on its hottest DPU", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=6, color=INK2)
            for tree, name in TREE_SERIES.items():
                d = cpu[(cpu.tree == tree) & (cpu.tag == ds) & (cpu.op == op) & (cpu.threads == cpu.threads.max())]
                if len(d):
                    st = SERIES[name]
                    v = float(d["mops"].iloc[0])
                    ax.axhline(v, color=st["color"], ls=st["ls"], lw=1.2, label=name)
                    ymax = max(ymax, v)
            ax.set_xscale("log", base=2)
            ax.set_xticks(dpus)
            ax.set_xticklabels([str(x) for x in dpus])
            ax.set_ylim(0, ymax * 1.15)
            if r == 0:
                ax.set_title(DATA_LABEL[ds], color=INK)
            if c == 0:
                ax.set_ylabel(f"{op.capitalize()} throughput\n(Mop/s)")
            if r == 1:
                ax.set_xlabel("Simulated PIM cores (DPUs)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_scaling_throughput")


def fig_load_balance(pim, skew, skew_dpus):
    """Left: lookup imbalance vs DPUs (uniform/moderate from the scaling runs; strong from the smaller skewcmp
    runs, because the range-partitioned design cannot hold the 800k-point strong data set). Right: per-DPU
    lookup cycles for strong skew at 16 DPUs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PAGE_W, 2.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
    d = pd.concat([pim[(pim.op == "lookup") & (pim.dataset != "strong")], skew[skew.op == "lookup"]])
    for pl, name in PLACEMENT_SERIES.items():
        st = SERIES[name]
        for ds, ls in zip(DATASETS, [":", "--", "-"]):
            e = d[(d.dataset == ds) & (d.placement == pl)].sort_values("nr_dpus")
            ax1.plot(e["nr_dpus"], e["imbalance"], color=st["color"], marker=st["marker"], ls=ls,
                     markeredgecolor="white", markeredgewidth=0.6, label=f"{name}, {ds}")
    ax1.set_xscale("log", base=2)
    xs = sorted(d["nr_dpus"].unique())
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlim(xs[0] / 1.2, xs[-1] * 1.2)
    ax1.set_yscale("log")
    ax1.set_yticks([1, 1.5, 2, 5, 10, 20])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_ylim(0.95, 40)
    ax1.set_xlabel("Simulated PIM cores (DPUs)")
    ax1.set_ylabel("Lookup load imbalance\n(slowest DPU / mean)")
    ax1.set_title("Load imbalance vs. DPUs", color=INK)
    handles = [plt.Line2D([], [], color=SERIES[n]["color"], marker=SERIES[n]["marker"], label=n)
               for n in PLACEMENT_SERIES.values()]
    handles += [plt.Line2D([], [], color=INK2, ls=ls, label=ds) for ds, ls in zip(DATASETS, [":", "--", "-"])]
    ax1.legend(handles=handles, frameon=False, loc="upper left", ncol=2)

    # Per-DPU lookup cycles, strong skew, 16 DPUs
    n = 16
    q = skew_dpus[(skew_dpus.op == "lookup") & (skew_dpus.nr_dpus == n)]
    width = 0.42
    for k, (pl, name) in enumerate(PLACEMENT_SERIES.items()):
        e = q[q.tag == f"strong_{pl}"].sort_values("dpu")
        if not len(e):
            continue
        mean = e["cycles"].mean()
        ax2.bar(e["dpu"] + (k - 0.5) * width, e["cycles"] / mean, width=width, color=SERIES[name]["color"],
                edgecolor="white", linewidth=0.5, label=name)
    ax2.axhline(1, color=INK2, lw=0.8)
    ax2.set_xlabel(f"DPU id ({n} DPUs, strong skew, 200k points)")
    ax2.set_ylabel("Lookup cycles / mean")
    ax2.set_title("Per-DPU lookup work", color=INK)
    ax2.set_xticks(range(0, n, 2))
    ax2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_load_balance")


def fig_breakdown(pim):
    rows = []
    for ds in ["uniform", "strong"]:
        for op in ["insert", "lookup"]:
            for n in [8, 64]:
                e = pim[(pim.dataset == ds) & (pim.placement == "quantile") & (pim.op == op) & (pim.nr_dpus == n)]
                if len(e):
                    e = e.iloc[0]
                    rows.append((f"{op}, {ds}, {n} DPUs", e.t_cpu, e.t_pim, e.t_comm))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    labels = [r[0] for r in rows]
    parts = np.array([r[1:] for r in rows])
    frac = parts / parts.sum(axis=1, keepdims=True)
    left = np.zeros(len(rows))
    comp = [("Host CPU", "#2a78d6"), ("PIM (DPU)", "#eb6834"), ("CPU-PIM transfer", "#1baf7a")]
    y = np.arange(len(rows))[::-1]
    for k, (name, col) in enumerate(comp):
        ax.barh(y, frac[:, k], left=left, color=col, edgecolor="white", linewidth=1, label=name, height=0.7)
        for i in range(len(rows)):
            if frac[i, k] >= 0.12:
                ax.text(left[i] + frac[i, k] / 2, y[i], f"{frac[i, k]:.0%}", ha="center", va="center",
                        fontsize=6, color="white")
        left += frac[:, k]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of modelled batch time (PIM-zd-tree)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0))
    fig.tight_layout()
    save(fig, "fig_time_breakdown")


def fig_batch(pimb, cpub):
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 3.6), sharex=True)
    for r, op in enumerate(["insert", "lookup"]):
        for c, ds in enumerate(["uniform", "strong"]):
            ax = axes[r, c]
            e = pimb[(pimb.dataset == ds) & (pimb.op == op)].sort_values("batch")
            st = SERIES["PIM-zd-tree"]
            ax.plot(e["batch"], e["mops"], color=st["color"], marker=st["marker"], label="PIM-zd-tree (64 DPUs)",
                    markeredgecolor="white", markeredgewidth=0.6)
            for tree, name in TREE_SERIES.items():
                f = cpub[(cpub.tree == tree) & (cpub.tag == ds) & (cpub.op == op)].sort_values("batch")
                st = SERIES[name]
                ax.plot(f["batch"], f["mops"], color=st["color"], marker=st["marker"], ls=st["ls"],
                        label=f"{name}, 20 threads", markeredgecolor="white", markeredgewidth=0.6)
            ax.set_xscale("log")
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(DATA_LABEL[ds], color=INK)
            if c == 0:
                ax.set_ylabel(f"{op.capitalize()} throughput\n(Mop/s)")
            if r == 1:
                ax.set_xlabel("Batch size (operations)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_batch_size")


def fig_traffic(pim):
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for op, ls in [("insert", "-"), ("lookup", "--")]:
        e = pim[(pim.dataset == "uniform") & (pim.placement == "quantile") & (pim.op == op)].sort_values("nr_dpus")
        st = SERIES["PIM-zd-tree"]
        ax.plot(e["nr_dpus"], e["bytes_per_op"], color=st["color"], marker="o", ls=ls, label=f"{op}, moved",
                markeredgecolor="white", markeredgewidth=0.6)
        ax.plot(e["nr_dpus"], e["useful_bytes_per_op"], color=INK2, marker=".", ls=ls, lw=1, label=f"{op}, useful")
    ax.set_xscale("log", base=2)
    xs = sorted(pim["nr_dpus"].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Simulated PIM cores (DPUs)")
    ax.set_ylabel("CPU-PIM bytes per operation")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_traffic")


# ---------------------------------------------------------------- tables
def write_tables(pim, cpu, pimb, cpub, failures, datasets, skew):
    TABLES.mkdir(parents=True, exist_ok=True)
    pim.to_csv(TABLES / "pim_scaling_summary.csv", index=False)
    cpu.to_csv(TABLES / "cpu_scaling_summary.csv", index=False)
    pimb.to_csv(TABLES / "pim_batch_summary.csv", index=False)
    cpub.to_csv(TABLES / "cpu_batch_summary.csv", index=False)
    lines = ["# Baseline results (generated by scripts/analyze.py)", "",
             f"Cost model: DPU {F_DPU/1e6:.0f} MHz; CPU->DPU {BW_TO_1/1e9:.2f}-{BW_TO_64/1e9:.2f} GB/s and "
             f"DPU->CPU {BW_FROM_1/1e9:.2f}-{BW_FROM_64/1e9:.2f} GB/s for 1-64 DPUs (PrIM, power-law fit, "
             f"exponents {A_TO:.3f} / {A_FROM:.3f}); launch overhead {LAUNCH_OVERHEAD_S*1e6:.0f} us.", ""]
    if len(datasets):
        lines += ["## Data sets", "", "| Data | Gini (2048 key ranges) |", "|---|---|"]
        for ds in DATASETS:
            g = datasets[datasets.tag.str.startswith(ds)]["gini_2048"]
            if len(g):
                lines.append(f"| {ds} | {g.iloc[0]:.3f} |")
        lines.append("")
    for op in ["insert", "lookup"]:
        lines += [f"## {op.capitalize()} throughput (Mop/s, modelled for PIM; 3 x 50k inserts / 50k lookups after "
                  "800k warm-up points)", "", "| Data | Design | 8 | 16 | 32 | 64 DPUs | CPU 1 / 4 / 20 threads |",
                  "|---|---|---|---|---|---|---|"]
        for ds in DATASETS:
            for pl, name in PLACEMENT_SERIES.items():
                cells = []
                for n in [8, 16, 32, 64]:
                    e = pim[(pim.dataset == ds) & (pim.placement == pl) & (pim.op == op) & (pim.nr_dpus == n)]
                    if len(e):
                        cells.append(f"{e.mops.iloc[0]:.2f} ({e.imbalance.iloc[0]:.2f}x)")
                    elif len(failures[(failures.tag == f"{ds}_{pl}") & (failures.nr_dpus == n)]):
                        cells.append("storage full")
                    else:
                        cells.append("-")
                lines.append(f"| {ds} | {name} | " + " | ".join(cells) + " | |")
            for tree, name in TREE_SERIES.items():
                e = cpu[(cpu.tree == tree) & (cpu.tag == ds) & (cpu.op == op)].sort_values("threads")
                lines.append(f"| {ds} | {name} | | | | | " + " / ".join(f"{v:.2f}" for v in e.mops) + " |")
        lines += ["", "PIM cells: throughput (load imbalance = slowest DPU's cycles / mean).", ""]
    skew.to_csv(TABLES / "pim_skewcmp_summary.csv", index=False)
    lines += ["## Strong skew, 200k warm-up points (both placements fit)", "",
              "| Op | Design | 8 | 16 | 32 | 64 DPUs |", "|---|---|---|---|---|---|"]
    for op in ["insert", "lookup"]:
        for pl, name in PLACEMENT_SERIES.items():
            cells = []
            for n in [8, 16, 32, 64]:
                e = skew[(skew.placement == pl) & (skew.op == op) & (skew.nr_dpus == n)]
                cells.append(f"{e.mops.iloc[0]:.2f} ({e.imbalance.iloc[0]:.2f}x)" if len(e) else "-")
            lines.append(f"| {op} | {name} | " + " | ".join(cells) + " |")
    lines.append("")
    (TABLES / "baseline_results.md").write_text("\n".join(lines) + "\n")


def main():
    b = pd.read_csv(RAW / "scaling_batches.csv")
    pim = summarise_pim(model_pim(b))
    cpu = summarise_cpu(pd.read_csv(RAW / "scaling_cpu.csv"))
    failures = read_failures("scaling")
    datasets = pd.read_csv(RAW / "scaling_dataset.csv") if (RAW / "scaling_dataset.csv").exists() else pd.DataFrame()

    # Batch-size sweep: one run per (dataset, batch size); a run's batch size is its ops per timed batch
    rows = []
    bb = model_pim(pd.read_csv(RAW / "batch_batches.csv"))
    for (ds, op), g in bb[bb.op.isin(["insert", "lookup"])].groupby(["dataset", "op"]):
        for ops, h in g.groupby("ops"):
            rows.append(dict(dataset=ds, op=op, batch=ops if op == "insert" else ops - 1,
                             mops=h["ops"].sum() / h["t_total"].sum() / 1e6, imbalance=h["imbalance"].mean()))
    pimb = pd.DataFrame(rows)
    c = pd.read_csv(RAW / "batch_cpu.csv")
    c["batch"] = np.where(c.op == "lookup", c.ops - 1, c.ops)
    cpub = summarise_cpu(c, keys=("tree", "tag", "op", "batch"))

    fig_scaling(pim, cpu, failures)
    skew = summarise_pim(model_pim(pd.read_csv(RAW / "skewcmp_batches.csv")))
    fig_load_balance(pim, skew, pd.read_csv(RAW / "skewcmp_dpus.csv"))
    fig_breakdown(pim)
    fig_batch(pimb, cpub)
    fig_traffic(pim)
    write_tables(pim, cpu, pimb, cpub, failures, datasets, skew)
    print((TABLES / "baseline_results.md").read_text())


if __name__ == "__main__":
    sys.exit(main())
