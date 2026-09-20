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


# --------------------------------------------------------------------------
# Pretraining (notebook 03)
# --------------------------------------------------------------------------

def plot_pretrain_curve(history, domain_name: str,
                        save_as: str | None = "03_dgi_loss_curve"):
    """DGI loss and discriminator accuracy over training steps.

    Two panels sharing an x axis rather than one chart with two y scales.
    Loss and accuracy have different units and different ranges; putting them
    on twin axes would invent a visual relationship between them.

    Read them together. Loss falling while accuracy climbs to 1.0 means the
    encoder has *solved* the pretext task -- which is not automatically good
    news. A saturated objective supplies no further gradient, so any steps
    after that point are drift rather than learning.
    """
    import matplotlib.pyplot as plt

    color = color_for(domain_name)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.4), sharex=True,
                                   gridspec_kw={"hspace": 0.18})

    ax1.plot(history.steps, history.loss, color=color, linewidth=2.0)
    ax1.set_ylabel("DGI loss")
    ax1.set_yscale("log")
    ax1.set_title(f"Deep Graph Infomax on {domain_name}", color=TEXT_PRIMARY)
    if history.best_step > 0:
        ax1.axvline(history.best_step, color=TEXT_MUTED, linestyle=":", linewidth=1.2)
        ax1.annotate(f"best: step {history.best_step}\nloss {history.best_loss:.4g}",
                     xy=(history.best_step, history.best_loss), xytext=(8, 14),
                     textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)

    ax2.plot(history.steps, history.accuracy, color=color, linewidth=2.0)
    ax2.axhline(0.5, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
    ax2.annotate("chance (0.5)", xy=(history.steps[-1], 0.5), xytext=(-4, 6),
                 textcoords="offset points", ha="right", fontsize=7.5,
                 color=TEXT_MUTED)
    ax2.set_ylabel("discriminator accuracy")
    ax2.set_xlabel("training step")
    ax2.set_ylim(0.3, 1.05)

    solved = next((s for s, a in zip(history.steps, history.accuracy) if a >= 0.99), None)
    if solved:
        for ax in (ax1, ax2):
            ax.axvspan(solved, history.steps[-1], color=TEXT_MUTED, alpha=0.07)
        ax2.annotate(f"task solved from step {solved} on",
                     xy=(solved, 0.62), xytext=(6, 0), textcoords="offset points",
                     fontsize=8, color=TEXT_SECONDARY)

    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig


def plot_tsne_facets(before: np.ndarray, after: np.ndarray, labels: np.ndarray,
                     domain_name: str, class_names: list[str] | None = None,
                     save_as: str | None = "03_tsne_before_after"):
    """t-SNE of the embeddings, one small panel per class.

    Why facets instead of one scatter with seven colours: seven hues in a
    single scatter cannot be told apart reliably by every reader, and colour
    would be the only thing carrying class identity. Here each panel uses ONE
    hue, showing that class against the rest of the cloud in grey, so the
    question "did this class become concentrated?" is answerable panel by
    panel without depending on colour discrimination at all.

    Top row is the untrained encoder, bottom row the pretrained one. If
    pretraining helped, the coloured points should be tighter in the bottom
    row than the top.
    """
    import matplotlib.pyplot as plt

    classes = np.unique(labels)
    n_cls = len(classes)
    color = color_for(domain_name)

    fig, axes = plt.subplots(2, n_cls, figsize=(1.45 * n_cls + 0.8, 3.6),
                             squeeze=False)
    for row, (emb, row_label) in enumerate([(before, "before\n(random init)"),
                                            (after, "after\n(DGI pretrained)")]):
        for col, cls in enumerate(classes):
            ax = axes[row][col]
            mask = labels == cls
            ax.scatter(emb[~mask, 0], emb[~mask, 1], s=1.0, c=GRID,
                       linewidths=0, rasterized=True)
            ax.scatter(emb[mask, 0], emb[mask, 1], s=1.8, c=color,
                       linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if row == 0:
                name = class_names[col] if class_names else f"class {cls}"
                ax.set_title(name, fontsize=8, color=TEXT_PRIMARY)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=8, color=TEXT_SECONDARY)

    fig.suptitle(
        f"Does DGI pretraining concentrate the classes?  ({domain_name})",
        fontsize=11, fontweight="semibold", color=TEXT_PRIMARY, y=1.02,
    )
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig


# --------------------------------------------------------------------------
# Label efficiency (notebooks 04-06)
# --------------------------------------------------------------------------

def plot_label_efficiency(sweep_df, arm_label: str = "",
                          save_as: str | None = "04_label_efficiency"):
    """Score against label fraction, one panel per domain.

    Faceted rather than overlaid because the four domains use three different
    metrics on three different scales -- putting accuracy, micro-F1 and AUC-PR
    on one axis would invite comparisons between numbers that do not mean the
    same thing.

    The x axis is log-spaced: the interesting behaviour is at 1% and 5%, where
    pretraining should help most if it helps at all, and a linear axis would
    squash those points against the origin.
    """
    import matplotlib.pyplot as plt

    domains = [d for d in DOMAIN_COLORS if d in set(sweep_df["domain"])]
    fig, axes = plt.subplots(1, len(domains), figsize=(3.1 * len(domains), 3.5))
    if len(domains) == 1:
        axes = [axes]

    for ax, name in zip(axes, domains):
        sub = sweep_df[sweep_df["domain"] == name].sort_values("fraction")
        color = color_for(name)
        ax.plot(sub["fraction"] * 100, sub["test_score"], color=color,
                linewidth=2.0, marker="o", markersize=5)
        metric = sub["metric"].iloc[0]
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("% of training labels")
        ax.set_title(f"{name}\n({metric})", color=TEXT_PRIMARY, fontsize=9)
        ax.grid(axis="x", visible=False)
        if ax is axes[0]:
            ax.set_ylabel("test score")
        # Direct-label the endpoints so the reader never needs the axis alone.
        for _, row in sub.iloc[[0, -1]].iterrows():
            ax.annotate(f"{row['test_score']:.3f}",
                        xy=(row["fraction"] * 100, row["test_score"]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5, color=TEXT_SECONDARY)
        lo, hi = sub["test_score"].min(), sub["test_score"].max()
        pad = max((hi - lo) * 0.25, 0.02)
        ax.set_ylim(lo - pad, hi + pad * 1.4)

    title = "Label efficiency" + (f" -- {arm_label}" if arm_label else "")
    fig.suptitle(title, fontsize=12, fontweight="semibold",
                 color=TEXT_PRIMARY, y=1.04)
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig


# --------------------------------------------------------------------------
# Results (notebooks 06-07)
# --------------------------------------------------------------------------

ARM_COLORS = {
    "B_random": "#8a8985",     # grey: the floor, deliberately recessive
    "D_expert": "#eb6834",     # orange
    "A_transfer": "#2a78d6",   # blue: the arm under test
    "C_scratch": "#1baf7a",    # aqua: the ceiling
}


def plot_label_efficiency_by_arm(summary, save_as: str | None = "06_label_efficiency"):
    """Every arm's label-efficiency curve, one panel per domain.

    Faceted because the four domains use three different metrics -- accuracy,
    micro-F1 and AUC-PR are not comparable numbers and must never share an
    axis. Each panel carries its own metric in the title.

    Arm B is drawn in grey rather than a hue: it is the floor, and the whole
    question is whether the coloured arms clear it.
    """
    import matplotlib.pyplot as plt

    from .results import ARM_LABELS, ARM_ORDER

    domains = [d for d in DOMAIN_COLORS if d in set(summary["target"])]
    fig, axes = plt.subplots(1, len(domains), figsize=(3.4 * len(domains), 3.9))
    if len(domains) == 1:
        axes = [axes]

    for ax, target in zip(axes, domains):
        block = summary[summary["target"] == target]
        for arm in ARM_ORDER:
            row = block[block["arm"] == arm].sort_values("fraction")
            if row.empty:
                continue
            x = row["fraction"] * 100
            y, err = row["mean"].to_numpy(), row["std"].to_numpy()
            color = ARM_COLORS[arm]
            ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4.5,
                    label=ARM_LABELS[arm], zorder=3)
            # Shaded band = +/- 1 std over seeds. Where bands overlap, the arms
            # are not distinguishable.
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.15, linewidth=0)

        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("% of training labels")
        metric = block["metric"].iloc[0]
        ax.set_title(f"{target}\n({metric})", color=TEXT_PRIMARY, fontsize=9)
        ax.grid(axis="x", visible=False)
        if ax is axes[0]:
            ax.set_ylabel("test score")

    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(
        "Does pretraining help? Shaded bands are +/- 1 std over 3 seeds",
        fontsize=12, fontweight="semibold", color=TEXT_PRIMARY, y=1.04,
    )
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig


def plot_gain_heatmap(gains, column: str = "gain_vs_expert",
                      title: str = "", save_as: str | None = "06_transfer_gain"):
    """Transfer gain per (domain, label fraction), as a diverging heatmap.

    Diverging because the quantity has a meaningful zero: above it pretraining
    helped, below it hurt. Two poles with a neutral midpoint, so "no effect"
    reads as nothing rather than as a colour.

    Cells whose gain is smaller than its own spread across seeds are hatched.
    Those are not evidence of a difference in either direction, and the
    hatching is what stops the eye reading a faint colour as a finding.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    mean_col, std_col = f"{column}_mean", f"{column}_std"
    piv = gains.pivot(index="target", columns="fraction", values=mean_col)
    spread = gains.pivot(index="target", columns="fraction", values=std_col)
    order = [d for d in DOMAIN_COLORS if d in piv.index]
    piv, spread = piv.loc[order], spread.loc[order]

    cmap = LinearSegmentedColormap.from_list(
        "gain", ["#b3261e", "#e7a6a1", "#f0efec", "#9ec5f4", "#184f95"]
    )
    limit = float(np.nanmax(np.abs(piv.to_numpy()))) or 0.01
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(figsize=(1.35 * piv.shape[1] + 3.6,
                                    0.72 * piv.shape[0] + 2.4))
    im = ax.imshow(piv.to_numpy(), cmap=cmap, norm=norm, aspect="auto")

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val, sd = piv.iat[i, j], spread.iat[i, j]
            if np.isnan(val):
                continue
            weak = abs(val) <= sd
            if weak:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor=TEXT_MUTED,
                                           linewidth=0.0, alpha=0.55))
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8.5,
                    color=TEXT_PRIMARY,
                    fontweight="normal" if weak else "semibold")

    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{f:.0%}" for f in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("% of training labels")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("gain (positive = pretraining helped)", fontsize=8)
    cbar.outline.set_visible(False)

    ax.set_title(title or "Transfer gain", color=TEXT_PRIMARY, pad=12)
    ax.annotate(
        "hatched = gain smaller than its spread across seeds, i.e. not evidence",
        xy=(0.0, -0.26), xycoords="axes fraction", fontsize=7.5,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig


def plot_similarity_vs_gain(points, xcol: str = "distance",
                            xlabel: str = "degree-distribution distance (Jensen-Shannon)",
                            save_as: str | None = "07_similarity_vs_gain"):
    """Does structural similarity between source and target predict transfer?

    One point per (source, target) pair. If similar domains transferred
    better, the points would trend downward: more distance, less gain. The
    fitted line and its correlation are drawn so the claim can be judged
    rather than eyeballed -- and with only a handful of pairs, a correlation
    is weak evidence whichever way it points, which the notebook says plainly.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for _, r in points.iterrows():
        ax.scatter(r[xcol], r["gain"], s=64, color=color_for(r["target"]),
                   zorder=3, linewidths=0)
        ax.annotate(f"{r['source']} to {r['target']}",
                    xy=(r[xcol], r["gain"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=7,
                    color=TEXT_SECONDARY)

    x, y = points[xcol].to_numpy(float), points["gain"].to_numpy(float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() >= 3:
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=TEXT_SECONDARY,
                linestyle="--", linewidth=1.4, zorder=2)
        r_val = float(np.corrcoef(x[ok], y[ok])[0, 1])
        ax.annotate(f"Pearson r = {r_val:+.2f}   (n = {int(ok.sum())} pairs)",
                    xy=(0.03, 0.96), xycoords="axes fraction", va="top",
                    fontsize=9, color=TEXT_PRIMARY)

    ax.axhline(0.0, color=TEXT_MUTED, linewidth=1.0, linestyle=":")
    ax.annotate("no gain", xy=(ax.get_xlim()[1], 0.0), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=7.5,
                color=TEXT_MUTED)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("transfer gain over random init")
    ax.set_title("Does domain similarity predict transfer?", color=TEXT_PRIMARY)
    fig.tight_layout()
    if save_as:
        print(f"saved -> {save_figure(fig, save_as)}")
    return fig
