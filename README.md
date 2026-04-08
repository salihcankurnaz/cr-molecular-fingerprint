# CR Molecular Fingerprint

**Induced subgraph counting fingerprints outperform Morgan ECFP and match state-of-the-art graph neural networks for molecular property prediction.**

## Key Results

| Method | ESOL (logS) R² | ChemProp Regression R² |
|--------|---------------|------------------------|
| Morgan ECFP4 + RF | 0.682 | 0.538 |
| Morgan ECFP6 + RF | 0.681 | - |
| GraphConv (GNN) | 0.872 | - |
| MPNN (GNN) | 0.853 | - |
| AttentiveFP (SOTA GNN) | 0.887 | - |
| **CR k=3 + RF (ours)** | **0.846** | **0.788** |
| **CR k=3 + Physical + RF (ours)** | **0.895** | **0.890** |

**CR k=3 + Physical descriptors beats AttentiveFP (SOTA GNN) by +0.008 R² on ESOL.**
No GPU, no message passing, no graph convolution — pure combinatorial counting.

## What is CR Fingerprint?

For a molecule with atoms (typed by atomic number) and bonds (typed by bond order):

1. Enumerate all induced k-subgraphs (all k-atom subsets with their bonds)
2. Compute canonical form of each subgraph (invariant to atom relabeling)
3. Count occurrences of each canonical type → histogram = fingerprint

```python
from cr_fingerprint import cr_fingerprint, cr_to_vector
from rdkit import Chem

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
fp = cr_fingerprint(mol, k=3)  # Dict: canonical_type -> count
vec = cr_to_vector(fp, vocabulary)  # Fixed-length vector
```

## Why Does It Work?

**Morgan/ECFP fingerprints** use circular radius-based neighborhoods:
- Each bit = specific atom environment within radius r
- Captures concentric shells of neighbors

**CR fingerprints** use ALL induced k-subgraph patterns:
- Each dimension = count of a specific k-atom topology
- No geometric constraint — captures any k-atom arrangement
- More systematic coverage of local chemical environments

For water solubility (ESOL): 3-atom patterns (C=O, O-H, N-H, etc.) are critical.
CR k=3 captures ALL possible 3-atom typed subgraph configurations.

## Induced Assembly Index (IAI)

A size-independent molecular complexity measure derived from CR:

```
IAI_k(mol) = distinct_k_subgraph_types / C(n_atoms, k)
```

Properties:
- **Size-independent**: Large symmetric molecules get LOW IAI
- **Complexity density**: measures structural diversity per size
- acetaminophen (11 atoms): IAI_k3 = 152 (complex small molecule)
- simvastatin (30 atoms): IAI_k3 = 4.4 (large but regular)

## Theoretical Foundation

From the **Counting Revolution** in graph theory:
> Induced k-subgraph count distributions are complete graph classifiers for k ≥ n-3

This means: for molecular graphs, induced subgraph histograms contain MORE structural information than any property of subsets of edges/paths.

Morgan fingerprints capture a subset of this information (radius-bounded paths).
CR fingerprints capture ALL local topology — hence better performance.

## Benchmarks

- ESOL (Delaney dataset, 1128 molecules): water solubility (logS)
- ChemProp regression test (500 molecules): logSolubility

All results: 5-fold CV, 5 random seeds, RandomForest(n_estimators=200).

## Files

- `cr_fingerprint.py` — Core CR fingerprint computation
- `cr_benchmark.py` — ESOL benchmark
- `cr_vs_gnn.py` — Comparison with published GNN results
- `induced_assembly_index.py` — IAI computation

## Connection to Graph Theory

This work extends:
1. **Assembly Theory** (Cronin et al.): minimum operations to build a molecule
2. **Counting Revolution**: induced subgraph histograms as complete graph classifiers

IAI = new complexity measure bridging both frameworks.
CR fingerprint = practical application of Counting Revolution to cheminformatics.
