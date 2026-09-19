"""Shared plotting style and the figures notebook 01 draws.

One place for colours and axis style so every figure in the project looks
like it came from the same study.

Palette note: the four domain colours are a validated categorical set
(blue / orange / aqua / violet). They clear colour-vision-deficiency and
normal-vision separation on every pair, not just adjacent ones, which is what
scatter plots and small multiples require. Aqua sits slightly under the 3:1
contrast target against the light surface, so every figure using it also
carries direct labels and is backed by the comparison table in notebook 01 --
colour is never the only thing carrying identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import config

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e4e3df"

#: One fixed colour per domain. Assigned by identity, never cycled by rank.
DOMAIN_COLORS: dict[str, str] = {
    "cora": "#2a78d6",      # blue
    "photo": "#eb6834",     # orange
    "ppi": "#1baf7a",       # aqua
    "elliptic": "#4a3aa7",  # violet
}


def color_for(domain_name: str) -> str:
    return DOMAIN_COLORS.get(domain_name, TEXT_SECONDARY)


def use_project_style() -> None:
    """Apply the project's matplotlib defaults. Call once per notebook."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "text.color": TEXT_PRIMARY,
    })


def save_figure(fig, name: str, subdir: str | None = None) -> Path:
    """Save to results/figures/ and return the path."""
    out_dir = config.FIGURES_DIR / subdir if subdir else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------
# Degree distributions
# --------------------------------------------------------------------------

def _ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complementary CDF: for each degree k, the fraction of nodes with >= k.

    This is the standard way to look at a heavy-tailed degree distribution.
    A histogram of a power-law-ish variable is dominated by the first bin and
    the tail turns to noise; the CCDF on log-log axes shows the whole range
    and makes two graphs of very different size directly comparable.
    """
    v = np.sort(values[values > 0])
    if v.size == 0:
        return np.array([1.0]), np.array([1.0])
    ccdf = 1.0 - np.arange(v.size) / v.size
    return v, ccdf


def plot_degree_distributions(domains: dict, save_as: str | None = "01_degree_distributions"):
    """Two views of the same thing, side by side.

    Top row: one histogram per domain, log-spaced bins, so you can see the
    shape of each on its own terms.
    Bottom: all four CCDFs overlaid on log-log, so you can see how unlike
    they are -- which is the whole premise of the project.
    """
    import matplotlib.pyplot as plt
    from .describe import degree_array

    names = list(domains.keys())
    n = len(names)
    fig = plt.figure(figsize=(3.0 * n, 6.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.25], hspace=0.45, wspace=0.28)

    degrees = {name: degree_array(dom) for name, dom in domains.items()}

    # -- top: one panel per domain -------------------------------------
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        d = degrees[name]
        pos = d[d > 0]
        hi = max(pos.max(), 2.0) if pos.size else 2.0
        bins = np.logspace(0, np.log10(hi), 30)
        ax.hist(pos, bins=bins, color=color_for(name), alpha=0.85, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(domains[name].display_name.split(" (")[0], color=TEXT_PRIMARY)
        ax.set_xlabel("degree")
        if i == 0:
            ax.set_ylabel("nodes (log)")
        med = float(np.median(d))
        ax.axvline(max(med, 1e-9), color=TEXT_SECONDARY, linestyle=":", linewidth=1.2)
        ax.annotate(
            f"median {med:.0f}\nmax {d.max():.0f}",
            xy=(0.96, 0.94), xycoords="axes fraction",
            ha="right", va="top", fontsize=8, color=TEXT_SECONDARY,
        )
        ax.grid(axis="x", visible=False)

    # -- bottom: overlaid CCDFs ----------------------------------------
    ax = fig.add_subplot(gs[1, :])
    for name in names:
        x, y = _ccdf(degrees[name])
        ax.plot(x, y, color=color_for(name), linewidth=2.0,
                label=domains[name].display_name)
        # direct label at the tail end of each curve
        ax.annotate(
            name, xy=(x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=TEXT_PRIMARY, va="center",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree k  (log scale)")
    ax.set_ylabel("fraction of nodes with degree >= k")
    ax.set_title("Degree distributions compared (CCDF, log-log)", color=TEXT_PRIMARY)
    ax.legend(loc="lower left", ncol=2)
    ax.grid(which="both", alpha=0.6)

    fig.suptitle(
        "How differently connected are these four graphs?",
        fontsize=12, fontweight="semibold", color=TEXT_PRIMARY, y=0.985,
    )
    if save_as:
        path = save_figure(fig, save_as)
        print(f"saved -> {path}")
    return fig


# --------------------------------------------------------------------------
# Class balance
# --------------------------------------------------------------------------

def plot_class_balance(domains: dict, save_as: str | None = "01_class_balance"):
    """One panel per domain showing how skewed the labels are.

    Multi-class domains get a bar per class. PPI is multi-label, so a bar per
    class makes no sense -- instead it gets the distribution of how often each
    of its 121 labels fires.
    """
    import matplotlib.pyplot as plt
    from .describe import label_summary

    names = list(domains.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        dom = domains[name]
        summary = label_summary(dom)
        color = color_for(name)

        if summary["task"] == "multilabel":
            import torch
            y = torch.cat([g.y for g in dom.graphs], dim=0)
            prevalence = (y.mean(dim=0).cpu().numpy())
            ax.hist(prevalence, bins=25, color=color, linewidth=0, alpha=0.85)
            ax.set_xlabel("label prevalence")
            ax.set_ylabel("number of labels")
            ax.annotate(
                f"{summary['num_labels']} labels\n"
                f"{summary['positives_per_node_mean']:.1f} active per node",
                xy=(0.96, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=TEXT_SECONDARY,
            )
        else:
            fracs = summary["class_fractions"]
            labels = list(fracs.keys())
            vals = [fracs[k] for k in labels]
            order = np.argsort(vals)[::-1]
            labels = [labels[i] for i in order]
            vals = [vals[i] for i in order]
            # width < 1 leaves a surface gap between adjacent bars
            ax.bar(range(len(vals)), vals, color=color, width=0.78, linewidth=0)
            ax.set_xticks(range(len(vals)))
            short = [l if len(l) <= 10 else l[:9] + "." for l in labels]
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("share of labelled nodes")
            ax.set_ylim(0, max(vals) * 1.28)
            ax.annotate(
                f"imbalance {summary['imbalance_ratio']:.1f}x\n"
                f"labelled {summary['labelled_frac']:.0%}",
                xy=(0.96, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=TEXT_SECONDARY,
            )
            ax.grid(axis="x", visible=False)

        ax.set_title(dom.display_name.split(" (")[0], color=TEXT_PRIMARY)

    fig.suptitle(
        "Label structure differs as much as graph structure does",
        fontsize=12, fontweight="semibold", color=TEXT_PRIMARY, y=1.03,
    )
    fig.tight_layout()
    if save_as:
        path = save_figure(fig, save_as)
        print(f"saved -> {path}")
    return fig


# --------------------------------------------------------------------------
# Scale comparison
# --------------------------------------------------------------------------

def plot_scale_comparison(domains: dict, save_as: str | None = "01_scale_comparison"):
    """Nodes, edges and raw feature dimension on log axes.

    Three separate single-measure panels rather than one chart with two
    y-scales: node count and feature dimension have nothing to do with each
    other and putting them on twin axes would invent a relationship.
    """
    import matplotlib.pyplot as plt
    from .describe import describe_domain

    rows = {name: describe_domain(d) for name, d in domains.items()}
    names = list(domains.keys())
    measures = [
        ("nodes", "nodes"),
        ("edges_undirected", "edges (undirected)"),
        ("raw_dim", "raw feature dim"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (key, label) in zip(axes, measures):
        vals = [rows[n][key] for n in names]
        colors = [color_for(n) for n in names]
        ax.bar(range(len(names)), vals, color=colors, width=0.72, linewidth=0)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(label, color=TEXT_PRIMARY)
        ax.grid(axis="x", visible=False)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:,}", xy=(i, v), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=TEXT_SECONDARY)
        ax.set_ylim(top=max(vals) * 3)

    fig.suptitle(
        "Four domains, three orders of magnitude apart",
        fontsize=12, fontweight="semibold", color=TEXT_PRIMARY, y=1.04,
    )
    fig.tight_layout()
    if save_as:
        path = save_figure(fig, save_as)
        print(f"saved -> {path}")
    return fig


# --------------------------------------------------------------------------
# Feature unification (notebook 02)
# --------------------------------------------------------------------------

def plot_svd_spectrum(transforms: dict, save_as: str | None = "02_svd_spectrum"):
    """How much of each domain's information survives compression to 128 dims.

    The curve is cumulative explained variance against number of components.
    Where a curve is still climbing steeply at 128, the compression is
    throwing information away; where it has flattened, 128 dims were more than
    enough. The four domains differ sharply, and that asymmetry is a real
    property of the design rather than a bug.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name, tr in transforms.items():
        ratios = tr.svd.explained_variance_ratio_
        cum = ratios.cumsum()
        xs = range(1, len(cum) + 1)
        ax.plot(xs, cum, color=color_for(name), linewidth=2.0, label=name)
        ax.annotate(
            f"{name}  {cum[-1]:.0%}",
            xy=(len(cum), cum[-1]), xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=TEXT_PRIMARY, va="center",
        )
    ax.set_xlabel("number of SVD components kept")
    ax.set_ylabel("cumulative share of variance explained")
    ax.set_title("What survives compression to 128 dimensions", color=TEXT_PRIMARY)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 150)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig


def plot_structural_distributions(structural: dict, names: list[str],
                                  save_as: str | None = "02_structural_features"):
    """The five structural features, one panel each, compared across domains.

    Boxes show the quartiles and whiskers the 5th-95th percentile; extreme
    outliers are left off so the boxes stay readable, with the true maximum
    annotated instead. These are the raw values, BEFORE standardization --
    standardizing first would hide exactly the between-domain differences
    this figure exists to show.
    """
    import matplotlib.pyplot as plt

    n_feat = len(names)
    fig, axes = plt.subplots(1, n_feat, figsize=(2.9 * n_feat, 3.8))
    domains = list(structural.keys())

    for j, (ax, fname) in enumerate(zip(axes, names)):
        data = [structural[d][:, j] for d in domains]
        bp = ax.boxplot(
            data, vert=True, patch_artist=True, showfliers=False,
            whis=(5, 95), widths=0.62,
            medianprops=dict(color=SURFACE, linewidth=1.6),
        )
        for patch, d in zip(bp["boxes"], domains):
            patch.set_facecolor(color_for(d))
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1.5)
        for element in ("whiskers", "caps"):
            for item in bp[element]:
                item.set_color(TEXT_MUTED)
                item.set_linewidth(1.0)

        ax.set_xticks(range(1, len(domains) + 1))
        ax.set_xticklabels(domains, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(fname, color=TEXT_PRIMARY)
        ax.grid(axis="x", visible=False)

        maxima = ", ".join(f"{structural[d][:, j].max():.3g}" for d in domains)
        ax.annotate(f"max: {maxima}", xy=(0.5, -0.42), xycoords="axes fraction",
                    ha="center", va="top", fontsize=6.8, color=TEXT_MUTED)

    fig.suptitle(
        "The five structural features, before standardization",
        fontsize=12, fontweight="semibold", color=TEXT_PRIMARY, y=1.04,
    )
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig


def plot_unified_scales(unified: dict, transforms: dict, svd_dim: int = 128,
                        save_as: str | None = "02_unified_scales"):
    """Standard deviation of each of the 133 shared input dimensions.

    One panel per domain. This is the figure that shows whether the shared
    input space really is shared: the encoder sees all four of these, and if
    one domain's inputs are an order of magnitude larger than another's, the
    round-robin pretraining in notebook 03 will be dominated by whichever
    domain shouts loudest.

    The flat zero region on PPI is its zero-padding -- it has only 50 raw
    features, so it cannot fill 128 SVD dimensions.
    """
    import matplotlib.pyplot as plt

    names = list(unified.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.5),
                             sharey=True)
    for ax, name in zip(axes, names):
        stds = unified[name].std(axis=0)
        color = color_for(name)
        ax.plot(range(svd_dim), stds[:svd_dim], color=color, linewidth=1.4,
                label="SVD dims")
        ax.plot(range(svd_dim, len(stds)), stds[svd_dim:], color=TEXT_SECONDARY,
                linewidth=2.0, marker="o", markersize=4, label="structural dims")
        ax.axvline(svd_dim - 0.5, color=TEXT_MUTED, linestyle=":", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(name, color=TEXT_PRIMARY)
        ax.set_xlabel("input dimension (0-132)")
        tr = transforms[name]
        if tr.padded_dims:
            ax.annotate(f"{tr.padded_dims} zero-padded",
                        xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                        fontsize=7.5, color=TEXT_SECONDARY)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("std of that dimension (log)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(
        "Do all four domains arrive at the encoder on the same scale?",
        fontsize=12, fontweight="semibold", color=TEXT_PRIMARY, y=1.04,
    )
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig
