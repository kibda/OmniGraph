"""Descriptive statistics for the loaded domains.

Notebook 01 prints what this returns. Keeping the computation here (not in a
cell) means the same numbers can be re-derived later by scripts, and the
notebook stays a thin, readable presentation layer.

Nothing here is used for training. These are the numbers you look at *before*
modelling, to know what you are dealing with.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .datasets import GraphDomain


# --------------------------------------------------------------------------
# Degrees
# --------------------------------------------------------------------------

def degree_array(domain: GraphDomain) -> np.ndarray:
    """Degree of every node in the domain, pooled across its graphs.

    We count out-edges from edge_index. For an undirected graph stored with
    both directions (the PyG convention) that equals the usual degree.
    """
    import torch
    from torch_geometric.utils import degree

    per_graph = []
    for g in domain.graphs:
        d = degree(g.edge_index[0], num_nodes=g.num_nodes, dtype=torch.float)
        per_graph.append(d.cpu().numpy())
    return np.concatenate(per_graph) if per_graph else np.zeros(0)


def degree_summary(domain: GraphDomain) -> dict[str, float]:
    d = degree_array(domain)
    return {
        "degree_min": float(d.min()),
        "degree_p25": float(np.percentile(d, 25)),
        "degree_median": float(np.median(d)),
        "degree_mean": float(d.mean()),
        "degree_p75": float(np.percentile(d, 75)),
        "degree_p99": float(np.percentile(d, 99)),
        "degree_max": float(d.max()),
        "isolated_nodes": int((d == 0).sum()),
        "isolated_frac": float((d == 0).mean()),
    }


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def feature_summary(domain: GraphDomain) -> dict[str, Any]:
    """What the raw feature matrix looks like before we touch it.

    Sparsity matters a lot here: Cora and Photo are near-binary bag-of-words
    (very sparse), while Elliptic is dense continuous engineered features.
    That difference is exactly why the SVD step in notebook 02 exists.
    """
    import torch

    x = torch.cat([g.x for g in domain.graphs], dim=0)
    nonzero = float((x != 0).float().mean())
    return {
        "raw_dim": int(x.size(1)),
        "dtype": str(x.dtype),
        "nonzero_frac": nonzero,
        "sparsity": 1.0 - nonzero,
        "value_min": float(x.min()),
        "value_max": float(x.max()),
        "value_mean": float(x.mean()),
        "looks_binary": bool(torch.all((x == 0) | (x == 1))),
    }


# --------------------------------------------------------------------------
# Labels and imbalance
# --------------------------------------------------------------------------

def label_summary(domain: GraphDomain) -> dict[str, Any]:
    """Class distribution and how skewed it is.

    `imbalance_ratio` is majority count / minority count over the *labelled*
    nodes. A value near 1 means balanced; Elliptic will be about 9-10x.
    """
    import torch

    y = torch.cat([g.y for g in domain.graphs], dim=0)
    out: dict[str, Any] = {"task": domain.task}

    if domain.task == "multilabel":
        pos_per_label = y.sum(dim=0).cpu().numpy()
        pos_per_node = y.sum(dim=1).cpu().numpy()
        out.update(
            num_labels=int(y.size(1)),
            labelled_nodes=int(y.size(0)),
            labelled_frac=1.0,
            positives_per_node_mean=float(pos_per_node.mean()),
            positives_per_node_median=float(np.median(pos_per_node)),
            label_prevalence_min=float(pos_per_label.min() / y.size(0)),
            label_prevalence_median=float(np.median(pos_per_label) / y.size(0)),
            label_prevalence_max=float(pos_per_label.max() / y.size(0)),
            imbalance_ratio=float(pos_per_label.max() / max(pos_per_label.min(), 1)),
        )
        return out

    total = int(y.numel())
    if domain.has_unlabeled and domain.unlabeled_value is not None:
        labelled_mask = y != domain.unlabeled_value
    else:
        labelled_mask = torch.ones_like(y, dtype=torch.bool)

    y_lab = y[labelled_mask]
    vals, counts = torch.unique(y_lab, return_counts=True)
    counts_np = counts.cpu().numpy()

    out.update(
        total_nodes=total,
        labelled_nodes=int(labelled_mask.sum()),
        labelled_frac=float(labelled_mask.float().mean()),
        unlabelled_nodes=int(total - int(labelled_mask.sum())),
        num_classes=int(vals.numel()),
        class_counts={
            domain.label_semantics.get(int(v), f"class {int(v)}"): int(c)
            for v, c in zip(vals, counts)
        },
        class_fractions={
            domain.label_semantics.get(int(v), f"class {int(v)}"): float(c / counts_np.sum())
            for v, c in zip(vals, counts)
        },
        imbalance_ratio=float(counts_np.max() / max(counts_np.min(), 1)),
        minority_frac=float(counts_np.min() / counts_np.sum()),
    )
    return out


# --------------------------------------------------------------------------
# One row per domain
# --------------------------------------------------------------------------

def describe_domain(domain: GraphDomain) -> dict[str, Any]:
    """Every headline number for one domain, flattened into one dict."""
    n = domain.total_nodes
    e = domain.total_undirected_edges
    row: dict[str, Any] = {
        "domain": domain.name,
        "display_name": domain.display_name,
        "task": domain.task,
        "num_graphs": domain.num_graphs,
        "nodes": n,
        "edges_undirected": e,
        "edge_index_cols": domain.total_directed_edges,
        "avg_degree": domain.avg_degree,
        # density = actual edges / all possible edges in a simple undirected graph
        "density": float(2 * e / (n * (n - 1))) if n > 1 else 0.0,
        "is_undirected": domain.is_undirected,
        "primary_metric": domain.primary_metric,
    }
    row.update(feature_summary(domain))
    row.update(degree_summary(domain))
    row.update(label_summary(domain))
    return row


def describe_all(domains: dict[str, GraphDomain]) -> "Any":
    """A tidy DataFrame, one row per domain, for side-by-side comparison."""
    import pandas as pd

    return pd.DataFrame([describe_domain(d) for d in domains.values()]).set_index("domain")


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def print_domain_card(domain: GraphDomain) -> None:
    """A readable block per domain: what it is, then what it measures."""
    row = describe_domain(domain)
    title = f"  {domain.display_name}  [{domain.name}]  "
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

    print("\nWHAT IT IS")
    for line in _wrap(domain.description, width=76):
        print("  " + line)

    print("\nSHAPE")
    print(f"  graphs in domain    : {row['num_graphs']}")
    print(f"  nodes               : {row['nodes']:,}")
    print(f"  edges (undirected)  : {row['edges_undirected']:,}")
    if row["is_undirected"]:
        print(f"  edge_index columns  : {row['edge_index_cols']:,}   "
              f"(= 2 x edges: both directions stored)")
        print(f"  undirected          : True")
        print(f"  avg degree          : {row['avg_degree']:.2f}")
    else:
        print(f"  edge_index columns  : {row['edge_index_cols']:,}   "
              f"(= 1 x edges: DIRECTED, only one direction stored)")
        print(f"  undirected          : False   <-- unlike the other domains")
        print(f"  avg OUT-degree      : {row['avg_degree']:.2f}   "
              f"(out-degree only; in-degree is counted separately)")
    print(f"  density             : {row['density']:.2e}")

    print("\nRAW FEATURES")
    print(f"  dimension           : {row['raw_dim']}")
    print(f"  dtype               : {row['dtype']}")
    print(f"  non-zero fraction   : {row['nonzero_frac']:.4f}  "
          f"(sparsity {row['sparsity']:.4f})")
    print(f"  value range         : [{row['value_min']:.4g}, {row['value_max']:.4g}]")
    print(f"  strictly 0/1        : {row['looks_binary']}")

    print("\nDEGREES")
    print(f"  min / median / max  : {row['degree_min']:.0f} / "
          f"{row['degree_median']:.0f} / {row['degree_max']:.0f}")
    print(f"  p25 / p75 / p99     : {row['degree_p25']:.0f} / "
          f"{row['degree_p75']:.0f} / {row['degree_p99']:.0f}")
    print(f"  isolated nodes      : {row['isolated_nodes']:,} "
          f"({row['isolated_frac']:.2%})")

    print("\nLABELS")
    print(f"  task                : {row['task']}")
    print(f"  headline metric     : {row['primary_metric']}")
    if row["task"] == "multilabel":
        print(f"  number of labels    : {row['num_labels']}")
        print(f"  positives per node  : mean {row['positives_per_node_mean']:.2f}, "
              f"median {row['positives_per_node_median']:.0f}")
        print(f"  label prevalence    : min {row['label_prevalence_min']:.4f}, "
              f"median {row['label_prevalence_median']:.4f}, "
              f"max {row['label_prevalence_max']:.4f}")
        print(f"  imbalance (labels)  : {row['imbalance_ratio']:.1f}x "
              f"(most common label / least common)")
    else:
        print(f"  classes             : {row['num_classes']}")
        print(f"  labelled nodes      : {row['labelled_nodes']:,} of "
              f"{row['total_nodes']:,} ({row['labelled_frac']:.2%})")
        if row["unlabelled_nodes"]:
            print(f"  UNLABELLED nodes    : {row['unlabelled_nodes']:,} "
                  f"-- these still carry features and edges, so they help "
                  f"message passing, but they can never be scored")
        print("  class distribution  :")
        for cls, cnt in sorted(row["class_counts"].items(), key=lambda kv: -kv[1]):
            frac = row["class_fractions"][cls]
            bar = "#" * max(1, int(frac * 40))
            print(f"      {cls:<22} {cnt:>8,}  {frac:6.2%}  {bar}")
        print(f"  imbalance ratio     : {row['imbalance_ratio']:.1f}x "
              f"(majority / minority)")
        print(f"  minority share      : {row['minority_frac']:.2%}")


def _wrap(text: str, width: int = 76) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width)


#: Columns worth putting in the compact side-by-side comparison table.
COMPARISON_COLUMNS = [
    "task", "num_graphs", "nodes", "edges_undirected", "avg_degree",
    "is_undirected", "raw_dim", "sparsity", "num_classes", "labelled_frac",
    "imbalance_ratio", "primary_metric",
]


def comparison_table(domains: dict[str, GraphDomain]) -> "Any":
    """The compact 4-row table: the one view that shows how unlike they are."""
    df = describe_all(domains)
    cols = [c for c in COMPARISON_COLUMNS if c in df.columns]
    out = df[cols].copy()
    if "num_classes" in out.columns and "num_labels" in df.columns:
        out["num_classes"] = df["num_classes"].fillna(df["num_labels"])
    return out


def print_sample_rows(domain: GraphDomain, n_nodes: int = 3, n_feats: int = 8) -> None:
    """Show the actual tensors: shapes, dtypes, and a few real values.

    Reading a shape is not the same as seeing the numbers. This is what tells
    you Cora's features really are 0/1 counts while Elliptic's are signed
    floats -- the kind of thing a summary table hides.
    """
    import torch

    g = domain.graphs[0]
    tag = f" (graph 0 of {domain.num_graphs})" if domain.num_graphs > 1 else ""
    print(f"\n--- {domain.name}: raw tensors{tag} ---")
    print(f"  x           shape={tuple(g.x.shape)}  dtype={g.x.dtype}")
    print(f"  edge_index  shape={tuple(g.edge_index.shape)}  dtype={g.edge_index.dtype}")
    print(f"  y           shape={tuple(g.y.shape)}  dtype={g.y.dtype}")

    print(f"\n  x[:{n_nodes}, :{n_feats}] (first {n_feats} of {g.x.size(1)} features):")
    with torch.no_grad():
        for i in range(min(n_nodes, g.x.size(0))):
            vals = ", ".join(f"{v:.3g}" for v in g.x[i, :n_feats].tolist())
            print(f"    node {i}: [{vals}, ...]")

    print(f"\n  edge_index[:, :5] (each column is one directed edge u->v):")
    print(f"    source: {g.edge_index[0, :5].tolist()}")
    print(f"    target: {g.edge_index[1, :5].tolist()}")

    print(f"\n  y[:{n_nodes}]:")
    if g.y.ndim == 1:
        for i in range(min(n_nodes, g.y.size(0))):
            code = int(g.y[i])
            meaning = domain.label_semantics.get(code, f"class {code}")
            print(f"    node {i}: {code}  ({meaning})")
    else:
        for i in range(min(n_nodes, g.y.size(0))):
            active = torch.nonzero(g.y[i]).flatten().tolist()
            print(f"    node {i}: {len(active)} labels active, indices {active[:8]}"
                  f"{' ...' if len(active) > 8 else ''}")


def print_ppi_graph_table(domain: GraphDomain, max_rows: int = 8) -> None:
    """PPI is 24 graphs, not one. Show how they vary and which split each is from.

    This matters later: a leave-one-domain-out protocol treats PPI as a single
    domain, so we have to decide how 24 graphs become one set of nodes to
    probe on. Notebook 04 makes that decision; this is the evidence for it.
    """
    import torch
    from torch_geometric.utils import degree

    print(f"\n--- {domain.name}: per-graph breakdown ({domain.num_graphs} graphs) ---")
    print(f"  {'idx':>4}  {'origin':>7}  {'nodes':>7}  {'edges':>9}  {'avg deg':>8}  {'labels/node':>11}")
    rows = []
    for i, g in enumerate(domain.graphs):
        d = degree(g.edge_index[0], num_nodes=g.num_nodes, dtype=torch.float)
        rows.append((
            i, domain.source_split[i] if i < len(domain.source_split) else "?",
            g.num_nodes, g.edge_index.size(1) // 2, float(d.mean()),
            float(g.y.sum(dim=1).float().mean()),
        ))
    for r in rows[:max_rows]:
        print(f"  {r[0]:>4}  {r[1]:>7}  {r[2]:>7,}  {r[3]:>9,}  {r[4]:>8.1f}  {r[5]:>11.2f}")
    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more")

    nodes = np.array([r[2] for r in rows])
    print(f"\n  nodes per graph : min {nodes.min():,}  median {np.median(nodes):,.0f}  max {nodes.max():,}")
    from collections import Counter
    print(f"  origin splits   : {dict(Counter(r[1] for r in rows))}")
    print("  NOTE: these are the dataset's own inductive splits. Our protocol")
    print("        makes its own splits in notebook 04 -- we do not inherit these.")


def edge_direction_report(domains: dict[str, GraphDomain]) -> dict[str, dict[str, Any]]:
    """How directed is each domain, and what would symmetrising change?

    Three of our four domains store every edge twice (u->v and v->u), which is
    how PyTorch Geometric represents an undirected graph. Elliptic does not:
    bitcoin flows one way, so each edge appears once.

    That matters for two concrete reasons, both of which land in notebook 02:

      * A GIN layer aggregates over *incoming* edges. On a directed graph a
        node with no incoming edge receives no messages at all and its
        embedding is a function of its own features only.
      * Clustering coefficient, triangle count and k-core number are defined
        for undirected graphs. Computing them on a directed graph means
        either symmetrising first, or using a directed variant that no longer
        means the same thing as in the other three domains.

    This function reports the numbers the decision should be based on rather
    than making it.
    """
    import torch
    from torch_geometric.utils import degree, is_undirected, to_undirected

    out: dict[str, dict[str, Any]] = {}
    for name, dom in domains.items():
        # Pool across every graph in the domain, not just the first -- PPI has
        # 24, and a statistic from one of them is not the domain's statistic.
        undirected = dom.is_undirected
        cols_now = nodes = no_in = no_out = isolated = 0
        sym_cols = sym_isolated = sym_deg_sum = 0

        for g in dom.graphs:
            n = g.num_nodes
            in_deg = degree(g.edge_index[1], num_nodes=n, dtype=torch.float)
            out_deg = degree(g.edge_index[0], num_nodes=n, dtype=torch.float)

            nodes += n
            cols_now += int(g.edge_index.size(1))
            no_in += int((in_deg == 0).sum())
            no_out += int((out_deg == 0).sum())
            isolated += int(((in_deg == 0) & (out_deg == 0)).sum())

            if not undirected:
                sym = to_undirected(g.edge_index, num_nodes=n)
                sd = degree(sym[0], num_nodes=n, dtype=torch.float)
                sym_cols += int(sym.size(1))
                sym_isolated += int((sd == 0).sum())
                sym_deg_sum += float(sd.sum())

        info: dict[str, Any] = {
            "is_undirected": undirected,
            "num_graphs": dom.num_graphs,
            "nodes": nodes,
            "edge_index_cols": cols_now,
            "avg_degree_now": cols_now / max(nodes, 1),
            "no_incoming_edges": no_in,
            "no_incoming_frac": no_in / max(nodes, 1),
            "no_outgoing_edges": no_out,
            "isolated_both_ways": isolated,
        }
        if not undirected:
            info.update(
                cols_if_symmetrised=sym_cols,
                isolated_if_symmetrised=sym_isolated,
                avg_degree_if_symmetrised=sym_deg_sum / max(nodes, 1),
            )
        out[name] = info
    return out


def print_edge_direction_report(domains: dict[str, GraphDomain]) -> None:
    """Print the directedness comparison as a table."""
    rep = edge_direction_report(domains)
    print("  (pooled over every graph in the domain)\n")
    print(f"  {'domain':<10} {'undirected':>11} {'nodes with no':>16} {'nodes with no':>15}")
    print(f"  {'':<10} {'':>11} {'INCOMING edge':>16} {'edge at all':>15}")
    print("  " + "-" * 55)
    for name, info in rep.items():
        pct = f"({info['no_incoming_frac']:.1%})"
        print(f"  {name:<10} {str(info['is_undirected']):>11} "
              f"{info['no_incoming_edges']:>9,} {pct:>6} "
              f"{info['isolated_both_ways']:>14,}")

    for name, info in rep.items():
        if info["is_undirected"]:
            continue
        print(f"\n  {name} is the odd one out. If its edges were symmetrised:")
        print(f"    edge_index columns   {info['edge_index_cols']:,} "
              f"-> {info['cols_if_symmetrised']:,}")
        print(f"    fully isolated nodes {info['isolated_both_ways']:,} "
              f"-> {info['isolated_if_symmetrised']:,}")
        print(f"    average degree       {info['avg_degree_now']:.2f} "
              f"-> {info['avg_degree_if_symmetrised']:.2f}")
        print(f"\n    Right now {info['no_incoming_frac']:.0%} of its nodes have no incoming edge,")
        print(f"    so a GIN layer would give them an embedding built from their own")
        print(f"    features alone -- no message passing at all. This is an open")
        print(f"    decision for notebook 02, not something to settle silently here.")
