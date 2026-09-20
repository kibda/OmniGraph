# OmniGraph — does a cross-domain graph encoder transfer?

**Answer: pretraining helps. Combining unrelated domains does not.**

A single GIN encoder pretrained on three unrelated graph domains never beats
pretraining on the single best source — 0 wins in 20 conditions under the
stronger of two self-supervised objectives. Two studies, **720 runs**, all
logged in [`results/`](results/).

## The question

Does a single GNN encoder, pretrained self-supervised on several unrelated graph
domains, produce node representations that transfer to a domain it has never
seen — better than a random-init encoder, and better than pretraining on a
single source domain?

The second comparison is what makes the first meaningful. If a multi-domain
encoder beats random init but not a single-domain expert, the finding is
"pretraining helps", not "**cross-domain** pretraining helps".

## Method in ten lines

1. Four node-level domains with nothing in common but being graphs: Cora
   (citation), Amazon Photo (co-purchase), PPI (protein, multi-label), Elliptic
   (bitcoin, 9.2× imbalanced, 77% unlabelled).
2. Unify features: per-domain TruncatedSVD to 128 dims, fit on training nodes only.
3. Append 5 label-free structural features (log-degree, clustering, PageRank,
   k-core, triangles), standardized → **133 dims everywhere**; the SVD block is
   rescaled per domain so content and structure arrive on one footing.
4. Encoder: one shared 2-layer GIN, hidden 256, output 128 (200,322 params).
5. Pretrain self-supervised — **no labels touched**. Two objectives tested
   (see below).
6. Multi-domain pretraining is round-robin: one batch per domain per step,
   shared encoder, per-domain heads.
7. Downstream: **freeze** the encoder, train only a linear probe.
8. Leave-one-domain-out, 4 folds × 3 seeds × 5 label fractions, identical
   splits and scoring code across arms.
9. Arms: **A** transfer (3 sources) · **B** random-init · **C** from-scratch
   (upper bound) · **D** single-source expert, run once per source.
10. Metrics: accuracy (Cora, Photo), micro-F1 (PPI), AUC-PR (Elliptic). Early
    stopping on validation only; test read once per run.

A gain is only counted as real when it **exceeds its own spread across seeds**.
Gains are computed per seed and then averaged, so that spread is the spread of
the *difference*.

## Two studies

| | objective | type | saturates? | where | runs |
|---|---|---|---|---|---|
| **1** | Deep Graph Infomax | contrastive | **yes — step 21 of 300** | local RTX 2050, 6.95 h | 360 |
| **2** | Masked autoencoding (GraphMAE) | generative | no — still improving at 276 | Kaggle T4, 1.86 h | 360 |

Study 1 found nothing transferred. But DGI reached 100% discriminator accuracy
at step 21 and then supplied no gradient, so its flat result was ambiguous:
*pretraining does not transfer*, or *this objective is too weak*? Study 2
settles it with a task that cannot be solved and abandoned.

## Results

Tables: [`results/table.md`](results/table.md) (DGI),
[`results/table_mae.md`](results/table_mae.md) (MAE).
Figures: [`results/figures/`](results/figures/).

Test score at **100% of labels**, mean ± std over 3 seeds, MAE study:

| domain | metric | B random | D best expert | **A transfer** | C from-scratch |
|---|---|---|---|---|---|
| Cora | accuracy | 0.776 ± .004 | 0.796 ± .016 | 0.785 ± .032 | **0.844** ± .011 |
| Photo | accuracy | 0.882 ± .005 | 0.907 ± .012 | 0.904 ± .004 | **0.935** ± .004 |
| PPI | micro-F1 | 0.535 ± .003 | 0.532 ± .001 | 0.535 ± .005 | **0.911** ± .002 |
| Elliptic | AUC-PR | 0.674 ± .016 | 0.749 ± .022 | 0.709 ± .044 | **0.945** ± .005 |

Verdicts across all 20 (domain × label-fraction) cells:

| | beats random-init | beats **best** single source |
|---|---|---|
| **DGI** | 2 positive, 14 tied, 4 negative · mean −0.010 | 2 positive, 9 tied, 9 negative · mean −0.019 |
| **MAE** | **7 positive, 12 tied, 1 negative · mean +0.013** | **0 positive, 11 tied, 9 negative · mean −0.017** |

## Findings

**1. The objective decides whether pretraining works at all.** Under DGI,
pretraining beat random init in 2 of 20 cells. Under MAE — same encoder, same
data, same probe, only the pretext task changed — it wins 7 and loses 1. The
largest single gain is **+0.114 AUC-PR on Elliptic at 1% labels**. A pretext
task that can be solved and abandoned teaches nothing, and DGI solved its task
at step 21 of 300.

**2. Combining unrelated domains still does not help.** Under MAE, three-source
pretraining beats the best single-source expert in **0 of 20 cells** and loses
in 9. Under DGI it at least won twice. Making pretraining work made this answer
*cleaner*, not weaker: picking one good source beats mixing three, at every
domain and every label fraction tested.

**3. Where pretraining helps, it is domain-specific.** Elliptic gains most
(+0.114 at 1% labels, +0.035 at 100%); PPI gains nothing at any fraction
(|gain| ≤ 0.004 everywhere). PPI's frozen arms sit flat at 0.53 micro-F1
*regardless of label budget* while from-scratch reaches 0.911 — a
representation that does not improve with more labels does not contain what the
task needs.

**4. Frozen representations stay far below training on the task.** Arm C wins
all 20 cells in both studies, by +0.021 to +0.376.

**5. Structural similarity does not predict transfer.** Degree-distribution
distance correlates with per-pair gain at r = +0.35 (p = 0.27, n = 12) — not
significant, and the *wrong sign*. The most similar source was the best source
in 1 of 4 folds against a chance rate of ~1.3. The negative result is not an
artefact of badly matched domains.

### What would change this conclusion

Two self-supervised objectives, one architecture (2-layer GIN), four domains,
and a frozen-encoder protocol. Fine-tuning instead of freezing would likely
close much of the gap to arm C — but would stop measuring what the encoder
learned during pretraining, which is the question asked here. More domains
might help a multi-domain encoder that three cannot; nothing here tests that.

### Five bugs that each produced a confident, wrong answer

Each looked like a result:

- **BatchNorm on the encoder output** made the DGI summary a constant vector.
  Loss parked at ln(2) = 0.693 with the discriminator at exactly chance, while
  every curve looked healthy.
- **Uncalibrated BatchNorm in arm B** left a fresh encoder's placeholder
  statistics doing no normalisation. With GIN's sum aggregation over Photo's
  average degree of 31, arm B's embeddings had mean norm 1730 and effective
  rank 5.3 of 128. Fixing it **flipped Cora's headline** from −0.019 to +0.020.
- **An under-trained probe.** At the originally specified 300 epochs validation
  was still improving; Elliptic's AUC-PR read 0.454 against a converged 0.663.
- **"Epoch" meaning one batch in arm C**, so 200 epochs was 200 passes for Cora
  but 8 for PPI — putting the "upper bound" *below* its own floor.
- **A schema change that silently invalidated 360 finished runs.** Adding the
  objective field changed every config hash, so the resumable runner would have
  redone seven hours of work. Fields added after results exist are now excluded
  from the hash at their default, pinned by a test.

All were found by asking "did this converge?" and "is this control fair?",
never "how do I make this number bigger?"

## Layout

```
src/         all experiment logic, importable
notebooks/   the teaching layer — 00 to 07, run in order
scripts/     run_all.py, progress.py, recover_from_log.py
tests/       76 tests: leakage, metrics, reproducibility, aggregation, MAE
results/     runs.jsonl (DGI), runs_mae.jsonl (MAE), tables, figures/
```

| Notebook | What it establishes |
|---|---|
| `00_setup` | device, versions, seeding, config hashing |
| `01_datasets` | what the four domains contain, and how unalike they are |
| `02_features` | the 133-dim shared input space + the no-leakage test |
| `03_encoder_dgi` | GIN + DGI, loss curve, t-SNE before/after |
| `04_probe` | splits, the frozen probe, arms B and C on one fold |
| `05_smoke_test` | all arms, one fold, tiny budget; resumability |
| `06_results` | the tables and figures, and the answer |
| `07_analysis` | does domain similarity predict transfer? |
| `kaggle_run_mae` | self-contained Kaggle runner for the MAE study |

## Reproducing

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install -r requirements.txt       # CPU build
# For an NVIDIA GPU, install the CUDA build first:
#   pip install torch==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126

.venv\Scripts\python.exe -m pytest                              # 76 tests
.venv\Scripts\python.exe scripts\run_all.py                     # DGI matrix
.venv\Scripts\python.exe scripts\run_all.py --objective mae     # MAE matrix
.venv\Scripts\python.exe scripts\progress.py --watch            # live status
```

Both matrices are resumable: every run carries an id derived from its config
hash and seed, and the runner skips ids already logged. Datasets (~1 GB)
download to `data/` on first use; `$OMNIGRAPH_DATA` overrides, and Kaggle and
Colab paths are detected automatically.

Determinism: `src/__init__.py` sets `CUBLAS_WORKSPACE_CONFIG` before the CUDA
context exists, and `set_seed` enables deterministic kernels. Without both,
GIN's scatter-add aggregation is non-deterministic on GPU and the same config
gives different numbers each run. Results are bit-reproducible *within* a
machine — the two studies ran on different GPUs, so their numbers are compared
only to their own baselines.

---

**CV bullets**

- Ran a 720-run leave-one-domain-out study across 4 graph domains, 4 arms, 5
  label fractions and 3 seeds, testing contrastive and generative
  self-supervised objectives, and showed that a cross-domain pretrained GNN
  encoder beats the best single-source expert in **0 of 20** conditions under
  the stronger objective — while a random-init control matched every pretrained
  arm under the weaker one, isolating message passing rather than pretraining
  as the source of performance.
- Diagnosed that the first study's null result came from a saturating pretext
  task (100% discriminator accuracy by **step 21 of 300**), rebuilt the
  experiment around a masked-autoencoder objective that does not saturate, and
  found pretraining then helped in **7 of 20** conditions while the headline
  negative held — alongside four measurement bugs that each produced a
  plausible false result, including an uncalibrated-BatchNorm baseline
  (effective rank **5.3 of 128**) that inverted the headline comparison.
