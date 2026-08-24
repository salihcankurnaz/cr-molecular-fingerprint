# CR Molecular Fingerprint

> **Research status:** experimental molecular descriptors based on induced-subgraph count histograms. The repository contains promising exploratory ESOL/scaffold experiments, but it does **not** currently support a state-of-the-art or general-superiority claim against GNNs or Morgan/ECFP.

The core idea is to represent a molecular graph by counts of typed induced `k`-atom subgraphs. For `k=3`, the representation contains both connected local motifs and disconnected atom/co-occurrence patterns.

```python
from cr_fingerprint import cr_fingerprint, cr_to_vector
from rdkit import Chem

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
fp = cr_fingerprint(mol, k=3)
vec = cr_to_vector(fp, vocabulary)
```

## Exploratory results in this repository

The scripts report strong results on ESOL and related small molecular-property experiments. Historical README values included approximately:

| Experiment | Script-reported result |
|---|---:|
| ESOL random CV, CR `k=3` + RF | R² around 0.85 |
| ESOL random CV, CR `k=3` + physical descriptors + RF | R² around 0.90 |
| ESOL scaffold experiment, CR `k=3` + RF | R² around 0.72 |
| ESOL scaffold experiment, CR `k=3` + physical descriptors + RF | R² around 0.74 |

These values are retained as **historical script-reported observations**, not publication-grade benchmark claims. The repository does not currently commit a raw, provenance-bound result artifact that independently reconstructs these tables.

## Why the previous “beats/matches SOTA GNN” claim was removed

`cr_vs_gnn.py` computes the repository's RF results locally, but compares them against GNN numbers that are hard-coded from external publications. Those external models are not rerun in the same environment, on the exact same folds, preprocessing, metric implementation, and hyperparameter-selection protocol.

`cr_scaffold_split.py` improves the comparison by using a Bemis–Murcko grouping strategy, but the current implementation has additional methodology limitations:

- scaffold groups are sorted deterministically and the `seed` does not reshuffle those groups, so the five nominal split seeds are not five independent scaffold partitions;
- the CR vocabulary is constructed before the train/test split;
- physical descriptors are standardized on the full dataset before the split;
- published GNN baseline values are still inserted as constants rather than reproduced under the same protocol.

For these reasons, phrases such as **“beats AttentiveFP,” “matches SOTA GNN,”** or **“outperforms Morgan in general”** are not supported by the current evidence.

A fair future comparison should rerun all baselines, or use a benchmark framework with identical fixed splits and metrics, and preserve the exact folds plus raw predictions.

## Connected vs disconnected subgraphs

A useful hypothesis emerging from the repository is that disconnected 3-atom patterns can encode global composition/co-occurrence information that is not identical to a radius-bounded circular fingerprint.

The current ablation scripts report that disconnected components retain more ESOL scaffold-split predictive signal than connected-only components in the tested setup. This is a **dataset-specific descriptive result**. It does not establish that disconnected subgraphs causally explain general scaffold transfer, nor that Morgan fingerprints contain no composition information.

## What CR fingerprints encode

For a molecule with typed atoms and typed bonds:

1. enumerate induced `k`-atom subsets;
2. canonicalize the typed induced subgraph;
3. count each canonical type;
4. convert the histogram to a fixed vocabulary for ML use.

This gives an interpretable count representation with a different inductive bias from hashed circular fingerprints and message-passing models.

## Theoretical scope

The historical README claimed that induced `k`-subgraph histograms are complete graph classifiers for `k >= n-3`. That statement is **not established by this repository** and is not used as a justification for the molecular benchmark.

The general graph-reconstruction problem remains open even for the `(n-1)` vertex-deleted deck. Therefore any universal completeness statement for smaller fixed `k` requires independent proof and specialist review.

For molecular ML, no completeness theorem is needed: the fingerprint can be useful as a predictive descriptor even when it is not injective over all possible graphs.

## Induced Assembly Index (experimental descriptor)

The repository also explores:

```text
IAI_k(mol) = distinct_k_subgraph_types / C(n_atoms, k)
```

Treat this as an experimental structural-diversity descriptor defined by this codebase. It is **not** presented here as an established Assembly Theory quantity, a validated biosignature, or a size-independent molecular-complexity theorem.

## Files

- `cr_fingerprint.py` — core induced-subgraph fingerprint computation
- `cr_benchmark.py` — ESOL experiments
- `cr_vs_gnn.py` — exploratory comparison using hard-coded published GNN reference values
- `cr_scaffold_split.py` — Bemis–Murcko grouped split experiment
- `cr_ablation_connected.py` — connected/disconnected ablation
- `cr_final_ablation.py` — further component ablations
- `cr_theory_analysis.py` — exploratory representation analysis
- `induced_assembly_index.py` — experimental IAI descriptor

## Reproducibility priorities

A publication-quality benchmark should commit:

1. exact dataset/version/checksum;
2. fixed train/validation/test indices;
3. train-only feature/vocabulary fitting;
4. identical baseline evaluation protocol;
5. raw predictions for every seed/fold;
6. environment and package versions;
7. aggregate metrics recomputed from those raw predictions.

## License status

No repository-level `LICENSE` file is currently committed. Until provenance and licensing are explicitly resolved, do not infer reuse rights from earlier README wording.
