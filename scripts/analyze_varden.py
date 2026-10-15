#!/usr/bin/env python3
"""Figures and tables for the Varden benchmark (scripts/run_sweep.sh varden).

    python scripts/analyze_varden.py

Same cost model as scripts/analyze.py. Multi-batch lookup runs report batch size / median batch time.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

RAW, TABLES = A.RAW, A.TABLES
FRACTIONS = ["0", "0.0001", "0.001", "0.01", "0.02", "0.05", "0.1"]
FRAC_LABEL = {"0": "0", "0.0001": "0.1", "0.001": "1", "0.01": "10", "0.02": "20", "0.05": "50", "0.1": "100"}
CPU_STYLE = {"pkd-tree": ("Pkd-tree (CPU, 20 thr.)", dict(color="#1baf7a", marker="^", ls="--")),
             "zd-tree": ("zd-tree (CPU, 20 thr.)", dict(color="#eda100", marker="D", ls="--"))}


def cpu_median(path, keys):
    c = pd.read_csv(path)
    c = c[c.op.isin(["insert", "lookup"])]
    per_rep = c.groupby(keys + ["rep"]).agg(ops=("ops", "sum"), seconds=("seconds", "sum")).reset_index()
    per_rep["mops"] = per_rep.ops / per_rep.seconds / 1e6
    return per_rep.groupby(keys)["mops"].median().reset_index()


def per_run(b, keys):
    """Throughput per run: sum over batches for inserts (3 timed batches), median batch for multi-batch lookups."""
    rows = []
    for k, g in b[b.op.isin(["insert", "lookup"])].groupby(keys + ["op"]):
        if g.op.iloc[0] == "lookup" and len(g) > 1:
            t, ops = g.t_total.median(), g.ops.median()
        else:
            t, ops = g.t_total.sum(), g.ops.sum()
        rows.append(dict(zip(keys + ["op"], k), mops=ops / t / 1e6, imbalance=g.imbalance.median(),
                         t_cpu=g.t_cpu.median(), t_pim=g.t_pim.median(), t_comm=g.t_comm.median()))
    return pd.DataFrame(rows)


def main():
    sc = per_run(A.model_pim(pd.read_csv(RAW / "varden_scaling_batches.csv")), ["nr_dpus"])
    sm = A.model_pim(pd.read_csv(RAW / "varden_small_batches.csv"))
    sm = per_run(sm, ["placement", "nr_dpus"])
    fails = A.read_failures("varden_small")
    cpu = cpu_median(RAW / "varden_scaling_cpu.csv", ["tree", "op"])

    hot = A.model_pim(pd.read_csv(RAW / "varden_hot_batches.csv"))
    parts = hot.tag.str.split("_", expand=True)
    hot["skew"], hot["variant"] = parts[2].str[1:], parts[3]
    hot = per_run(hot, ["skew", "variant"])

    mx = A.model_pim(pd.read_csv(RAW / "varden_mix_batches.csv"))
    parts = mx.tag.str.split("_", expand=True)
    mx["frac"], mx["variant"] = parts[2].str[1:], parts[3]
    mx = per_run(mx, ["frac", "variant"])
    cmx = pd.read_csv(RAW / "varden_mix_cpu.csv")
    cmx["frac"] = cmx.tag.str.split("_f").str[1]
    cmx = cpu_median_from(cmx, ["tree", "frac", "op"])

    ds = pd.read_csv(RAW / "varden_scaling_dataset.csv")
    gini = float(ds.gini_2048.iloc[0])

    # ---- Figure: Varden as data ----
    fig, axes = plt.subplots(1, 3, figsize=(A.PAGE_W, 2.3))
    dpus = sorted(sc.nr_dpus.unique())
    for ax, op in zip(axes[:2], ["insert", "lookup"]):
        e = sc[sc.op == op].sort_values("nr_dpus")
        st = A.SERIES["PIM-zd-tree"]
        ax.plot(e.nr_dpus, e.mops, color=st["color"], marker=st["marker"], label="PIM-zd-tree",
                markeredgecolor="white", markeredgewidth=0.6)
        for tree, (name, cst) in CPU_STYLE.items():
            v = cpu[(cpu.tree == tree) & (cpu.op == op)].mops
            if len(v):
                ax.axhline(float(v.iloc[0]), color=cst["color"], ls="--", lw=1.2, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dpus)
        ax.set_xticklabels([str(x) for x in dpus])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Simulated PIM cores (DPUs)")
        ax.set_ylabel(f"{op.capitalize()} throughput (Mop/s)")
        ax.set_title(f"{op.capitalize()}, Varden data (800k)", color=A.INK)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.45)
    axes[0].legend(frameon=False, loc="upper left", fontsize=6, ncol=1)
    ax = axes[2]
    for pl, name in A.PLACEMENT_SERIES.items():
        e = sm[(sm.placement == pl) & (sm.op == "lookup")].sort_values("nr_dpus")
        st = A.SERIES[name]
        ax.plot(e.nr_dpus, e.imbalance, color=st["color"], marker=st["marker"], label=name,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dpus)
    ax.set_xticklabels([str(x) for x in dpus])
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Simulated PIM cores (DPUs)")
    ax.set_ylabel("Lookup load imbalance")
    ax.set_title("Placement, Varden data (200k)", color=A.INK)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    A.save(fig, "fig_varden_data")

    # ---- Figure: paper Fig. 9 for point lookups ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A.PAGE_W, 2.3))
    xs = range(len(FRACTIONS))
    var = {"base": ("PIM-zd-tree (baseline)", dict(color="#2a78d6", marker="o", ls="-")),
           "adaptive": ("+ adaptive dedup + cache", dict(color="#eb6834", marker="s", ls="-"))}
    for v, (name, st) in var.items():
        e = mx[(mx.variant == v) & (mx.op == "lookup")].set_index("frac").reindex(FRACTIONS)
        ax1.plot(xs, e.mops, label=name, markeredgecolor="white", markeredgewidth=0.6, **st)
        ax2.plot(xs, e.imbalance, label=name, markeredgecolor="white", markeredgewidth=0.6, **st)
    for tree, (name, cst) in CPU_STYLE.items():
        e = cmx[(cmx.tree == tree) & (cmx.op == "lookup")].set_index("frac").reindex(FRACTIONS)
        ax1.plot(xs, e.mops, label=name, lw=1.1, **cst)
    for ax in (ax1, ax2):
        ax.set_xticks(list(xs))
        ax.set_xticklabels([FRAC_LABEL[f] for f in FRACTIONS])
        ax.set_xlabel("Share of lookups from Varden points (‰), 64 DPUs")
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("Lookup throughput (Mop/s)")
    ax1.set_title("Uniform data, Varden hot-spot queries", color=A.INK)
    ax1.legend(frameon=False, fontsize=6, loc="lower left")
    ax2.set_ylim(0.9, max(3.0, float(mx.imbalance.max()) * 1.15))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax2.set_ylabel("Lookup load imbalance")
    ax2.set_title("Slowest DPU / mean (both variants overlap)", color=A.INK)
    fig.tight_layout()
    A.save(fig, "fig_varden_mix")

    # ---- Tables ----
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, df in [("varden_scaling", sc), ("varden_small", sm), ("varden_hot", hot), ("varden_mix", mx),
                     ("varden_cpu", cpu), ("varden_mix_cpu", cmx)]:
        df.to_csv(TABLES / f"{name}_summary.csv", index=False)
    L = ["# Varden benchmark results (generated by scripts/analyze_varden.py)", "",
         f"Varden data: SpaceTreeLib generator, 3-D, Gini over 2048 key ranges = {gini:.3f} "
         "(our strong Zipf set: 0.978; paper's OSM: 0.967).", "",
         "## Varden as data, 800k warm-up points (PIM-zd-tree; Mop/s, imbalance in parentheses)", "",
         "| Op | 8 | 16 | 32 | 64 DPUs | Pkd-tree CPU | zd-tree CPU |", "|---|---|---|---|---|---|---|"]
    for op in ["insert", "lookup"]:
        cells = [f"{r.mops:.2f} ({r.imbalance:.2f}x)" for r in sc[sc.op == op].sort_values("nr_dpus").itertuples()]
        cpus = [f"{float(cpu[(cpu.tree == t) & (cpu.op == op)].mops.iloc[0]):.2f}" for t in CPU_STYLE]
        L.append(f"| {op} | " + " | ".join(cells) + " | " + " | ".join(cpus) + " |")
    L += ["", "## Varden as data, 200k points: both placements", "",
          "| Op | Design | 8 | 16 | 32 | 64 DPUs |", "|---|---|---|---|---|---|"]
    for op in ["insert", "lookup"]:
        for pl, name in A.PLACEMENT_SERIES.items():
            cells = []
            for n in dpus:
                e = sm[(sm.placement == pl) & (sm.op == op) & (sm.nr_dpus == n)]
                if len(e):
                    cells.append(f"{e.mops.iloc[0]:.2f} ({e.imbalance.iloc[0]:.2f}x)")
                elif len(fails[(fails.tag == f"varden_{pl}") & (fails.nr_dpus == n)]):
                    cells.append("storage full")
                else:
                    cells.append("-")
            L.append(f"| {op} | {name} | " + " | ".join(cells) + " |")
    L += ["", "## Hot keys over Varden data, 64 DPUs (lookup Mop/s, median batch)", "",
          "| Query skew | baseline | + dedup + cache | adaptive |", "|---|---|---|---|"]
    for q in ["uniform", "1.0", "1.2"]:
        cells = []
        for v in ["base", "cache", "adaptive"]:
            e = hot[(hot["skew"] == q) & (hot.variant == v) & (hot.op == "lookup")]
            cells.append(f"{e.mops.iloc[0]:.2f} ({e.imbalance.iloc[0]:.2f}x)" if len(e) else "-")
        L.append(f"| {q} | " + " | ".join(cells) + " |")
    L += ["", "## Paper Fig. 9 for point lookups: uniform data, share of Varden queries, 64 DPUs", "",
          "| Varden share | baseline | adaptive | Pkd-tree CPU | zd-tree CPU |", "|---|---|---|---|---|"]
    for f in FRACTIONS:
        cells = []
        for v in ["base", "adaptive"]:
            e = mx[(mx.frac == f) & (mx.variant == v) & (mx.op == "lookup")]
            cells.append(f"{e.mops.iloc[0]:.2f} ({e.imbalance.iloc[0]:.2f}x)" if len(e) else "-")
        for t in CPU_STYLE:
            e = cmx[(cmx.tree == t) & (cmx.frac == f) & (cmx.op == "lookup")]
            cells.append(f"{e.mops.iloc[0]:.2f}" if len(e) else "-")
        L.append(f"| {FRAC_LABEL[f]}‰ | " + " | ".join(cells) + " |")
    (TABLES / "varden_results.md").write_text("\n".join(L) + "\n")
    print((TABLES / "varden_results.md").read_text())


def cpu_median_from(c, keys):
    c = c[c.op.isin(["insert", "lookup"])]
    per = []
    for k, g in c.groupby(keys + ["rep"]):
        if g.op.iloc[0] == "lookup" and len(g) > 1:
            m = g.ops.median() / g.seconds.median() / 1e6
        else:
            m = g.ops.sum() / g.seconds.sum() / 1e6
        per.append(dict(zip(keys + ["rep"], k), mops=m))
    per = pd.DataFrame(per)
    return per.groupby(keys)["mops"].median().reset_index()


if __name__ == "__main__":
    main()
