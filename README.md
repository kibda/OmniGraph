# OmniGraph — does a cross-domain graph encoder transfer?

**Answer: no.** A single GIN encoder pretrained with Deep Graph Infomax on three
unrelated graph domains does not transfer to a fourth better than a
random-initialised encoder, and does not beat pretraining on a single source
domain. All 360 runs are in [`results/runs.jsonl`](results/runs.jsonl).

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
5. Pretrain with Deep Graph Infomax — self-supervised, no labels touched.
6. Multi-domain pretraining is round-robin: one batch per domain per step,
   shared encoder, per-domain discriminator heads.
7. Downstream: **freeze** the encoder, train only a linear probe.
8. Leave-one-domain-out, 4 folds × 3 seeds × 5 label fractions, identical
   splits and scoring code across arms.
9. Arms: **A** transfer (3 sources) · **B** random-init · **C** from-scratch
   (upper bound) · **D** single-source expert, run once per source.
10. Metrics: accuracy (Cora, Photo), micro-F1 (PPI), AUC-PR (Elliptic). Early
    stopping on validation only; test read once per run.

**360 runs, 6.95 GPU-hours, 0 failures** on one RTX 2050. Every run is
bit-reproducible from its config hash and seed.

## Results

Full table: [`results/table.md`](results/table.md). Figures:
[`results/figures/`](results/figures/).

Test score at **100% of labels**, mean ± std over 3 seeds:

| domain | metric | B random | D best expert | **A transfer** | C from-scratch |
|---|---|---|---|---|---|
| Cora | accuracy | 0.763 ± .007 | 0.796 ± .016 | 0.800 ± .012 | **0.846** ± .014 |
| Photo | accuracy | 0.880 ± .011 | 0.881 ± .008 | 0.886 ± .009 | **0.938** ± .006 |
| PPI | micro-F1 | 0.532 ± .003 | 0.531 ± .005 | 0.528 ± .005 | **0.912** ± .004 |
| Elliptic | AUC-PR | 0.675 ± .015 | 0.669 ± .053 | 0.644 ± .031 | **0.949** ± .005 |

Across all 20 (domain × label-fraction) cells, judging a gap real only when it
exceeds its own spread across seeds:

| comparison | positive | indistinguishable | negative |
|---|---|---|---|
| A vs random-init | 2 | 14 | 4 |
| A vs **best** single-source expert | 2 | 9 | 9 |

## Findings

**1. Cross-domain pretraining does not beat single-source pretraining.** Arm A
beat the best single expert in 2 of 20 cells and lost in 9 (mean −0.019).
Against the *mean* expert — biased the other way, since "best of three" is
optimistic — the difference is +0.0004 ± 0.0264 over 60 paired runs. Zero,
either way you frame it.

**2. Pretraining barely beats random initialisation at all.** Arm A beat arm B
in 2 of 20 cells and lost in 4; the mean difference is −0.010. A randomly
initialised GIN with calibrated BatchNorm is as good an encoder as any
pretrained one here. **The message passing does the work; the pretraining does
not.**

**3. Frozen representations fall far short of training on the task.** Arm C
wins every one of the 20 cells, by +0.033 to +0.384. PPI is the extreme case:
all three frozen arms sit flat at 0.49–0.53 micro-F1 *regardless of label
budget*, while from-scratch reaches 0.912. A frozen representation that does
not improve with more labels does not contain what the task needs, and no
amount of labelled data lets a linear probe read what is not there.

**4. Where pretraining does help, it helps where labels are plentiful** — the
opposite of the usual case for self-supervised pretraining. On Cora, arm A is
−0.041 against random init at 1% of labels and +0.036 at 100%.

**5. Structural similarity does not explain the negative result.** Degree-
distribution distance correlates with per-pair transfer gain at r = +0.35
(p = 0.27, n = 12) — not significant, and the *wrong sign*. The most similar
source was the best source in 1 of 4 folds, against a chance rate of ~1.3. So
this is not simply a case of badly matched domains. Only 2 of 12 pair-level
gains exceed their own spread, so there is very little gain for any measure to
predict.

### What would change this conclusion

The result is specific to this setup, and the honest limits are: one
self-supervised objective (DGI), one architecture (2-layer GIN), four domains,
and a frozen-encoder protocol. DGI solves its pretext task completely here —
100% discriminator accuracy by step 21 — which supplies no further gradient, so
a harder objective might behave differently. Fine-tuning instead of freezing
would likely close much of the gap to arm C, but would stop measuring what the
encoder learned during pretraining.

### Four bugs that each produced a confident, wrong answer

Worth recording, because each looked like a result:

- **BatchNorm on the encoder output** made the DGI summary a constant vector.
  Loss parked at ln(2) = 0.693 with the discriminator at exactly chance, while
  every curve looked healthy.
- **Uncalibrated BatchNorm in arm B** left a fresh encoder's placeholder
  statistics (mean 0, var 1) doing no normalisation at all. With GIN's sum
  aggregation over Photo's average degree of 31, arm B's embeddings had mean
  norm 1730 and effective rank 5.3 of 128. Fixing it flipped Cora's headline
  from −0.019 to +0.020.
- **An under-trained probe.** At the originally specified 300 epochs, validation
  was still improving; Elliptic's AUC-PR read 0.454 against a converged 0.663.
- **"Epoch" meaning one batch in arm C**, so 200 epochs was 200 passes for Cora
  but 8 for PPI — putting the "upper bound" *below* its own floor.

All were found by asking "did this converge?" and "is this control fair?",
never by asking "how do I make this number bigger?"

## Layout

```
src/         all experiment logic, importable
notebooks/   the teaching layer — 00 to 07, run in order
scripts/     run_all.py (resumable matrix), progress.py (live status)
tests/       66 tests: leakage, metrics, reproducibility, aggregation
results/     runs.jsonl, table.md, figures/
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

## Reproducing

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install -r requirements.txt       # CPU build
# For an NVIDIA GPU, install the CUDA build first:
#   pip install torch==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126

.venv\Scripts\python.exe -m pytest              # 66 tests
.venv\Scripts\python.exe scripts\run_all.py     # the 360-run matrix, resumable
.venv\Scripts\python.exe scripts\progress.py    # live status, safe to run alongside
```

Datasets (~1 GB) download to `data/` on first use. Override with
`$OMNIGRAPH_DATA`; Kaggle and Colab paths are detected automatically.

Determinism note: `src/__init__.py` sets `CUBLAS_WORKSPACE_CONFIG` before the
CUDA context exists, and `set_seed` enables deterministic kernels. Without both,
GIN's scatter-add aggregation is non-deterministic on GPU and the same config
gives different numbers on each run.

---

**CV bullets**

- Ran a 360-run leave-one-domain-out study (4 graph domains × 4 arms × 5 label
  fractions × 3 seeds, 6.95 GPU-hours, bit-reproducible) showing that a
  DGI-pretrained cross-domain GNN encoder beat a single-source expert in only
  **2 of 20** conditions — and isolated the cause with a random-init control
  that matched every pretrained arm, establishing that message passing rather
  than pretraining carried the performance.
- Found and fixed four measurement bugs that each produced a plausible false
  result, including an uncalibrated BatchNorm baseline whose embeddings had
  **effective rank 5.3 of 128** and which inverted the headline comparison, and
  an under-trained probe that understated Elliptic's AUC-PR by **46%**
  (0.454 vs 0.663 converged).
