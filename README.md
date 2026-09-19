# OmniGraph — does a cross-domain graph encoder transfer?

> **Status: in progress.** Notebooks 00 and 01 are done. No results yet — the
> Findings section below stays empty until the full run matrix has actually been
> executed. It will report whatever the runs say, including a negative answer.

## The question

Does a single GNN encoder, pretrained self-supervised on several unrelated graph
domains, produce node representations that transfer to a domain it has never
seen — **better than a random-init encoder**, and **better than pretraining on a
single source domain**?

The second comparison is the one that makes the first meaningful. If a
multi-domain encoder beats random init but not a single-domain expert, then the
finding is "pretraining helps", not "cross-domain pretraining helps".

## Method in ten lines

1. Four node-level domains with nothing in common but being graphs: Cora
   (citation), Amazon Photo (co-purchase), PPI (protein, multi-label), Elliptic
   (bitcoin, heavily imbalanced).
2. Unify features: per-domain TruncatedSVD to 128 dims, fit on training nodes only.
3. Append 5 label-free structural features (log-degree, clustering, PageRank,
   k-core, triangles), standardized per domain → **133 dims everywhere**.
4. Encoder: one shared 2-layer GIN, hidden 256, output 128.
5. Pretrain with Deep Graph Infomax — self-supervised, no labels touched.
6. Multi-domain pretraining is round-robin: one batch per domain per step,
   shared encoder, per-domain discriminator heads.
7. Downstream: **freeze** the encoder, train only a linear probe. Frozen is the point.
8. Protocol: leave-one-domain-out, 4 folds × 3 seeds, identical splits across arms.
9. Arms: **A** transfer (3 sources) · **B** random-init · **C** from-scratch
   (upper bound) · **D** single-source expert.
10. Label-efficiency sweep on the target: 1%, 5%, 10%, 50%, 100%.

Metrics: accuracy + macro-F1, AUC-PR for Elliptic, micro-F1 for PPI. Mean ± std
over 3 seeds. Early stopping on validation only, never on test.

## Layout

```
src/         all experiment logic, importable
notebooks/   the teaching layer — imports src, prints shapes, plots what changed
scripts/     run_all.py, the headless resumable run matrix
tests/       leakage test and friends
results/     runs.jsonl (one line per run), table.md, figures/
```

Notebooks are numbered and meant to be read in order.

| Notebook | What it establishes |
|---|---|
| `00_setup` | device, versions, seeding, config hashing |
| `01_datasets` | what the four domains actually contain, and how unalike they are |
| `02_features` | the 133-dim shared input space + the no-leakage test |
| `03_encoder_dgi` | GIN + DGI on one domain, loss curve, t-SNE before/after |
| `04_probe` | splits, linear probe, arms B and C on one fold |
| `05_smoke_test` | one full fold, all arms, tiny budget |
| `06_results` | tables and plots from the full run |
| `07_analysis` | does domain similarity predict transfer gain? |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt  # CPU build
# For a local NVIDIA GPU, install the CUDA build of torch instead:
#   pip install torch==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126
jupyter lab
```

On Kaggle or Colab, open `00_setup.ipynb` and run the install cell — torch is
already present there, so only the graph packages get installed.

Datasets download to `data/` by default. Override with `$OMNIGRAPH_DATA`;
Kaggle and Colab paths are detected automatically.

## Findings

_Empty until the runs exist. It will be filled in from `results/runs.jsonl`,
whatever the answer turns out to be._
