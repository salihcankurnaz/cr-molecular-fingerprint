# CR Molecular Fingerprint

**Induced subgraph counting fingerprints outperform Morgan ECFP and match state-of-the-art graph neural networks for molecular property prediction.**

## Key Results

### Random 5-Fold CV (Standard Comparison)

| Method | ESOL (logS) R² | ChemProp (logS) R² |
|--------|---------------|------------------------|
| Morgan ECFP4 + RF | 0.682 | 0.538 |
| GraphConv (GNN) | 0.872 | - |
| MPNN (GNN) | 0.853 | - |
| AttentiveFP (SOTA GNN) | 0.887 | - |
| **CR k=3 + RF (ours)** | **0.846** | **0.788** |
| **CR k=3 + Physical + RF (ours)** | **0.895** | **0.890** |

**CR k=3 + Physical descriptors beats AttentiveFP (SOTA GNN) by +0.008 R² on ESOL.**

### Scaffold Split (GNN-Fair Comparison)

| Method | ESOL R² (scaffold) | Collapse |
|--------|-------------------|---------|
| Morgan ECFP4 + RF | 0.198 | -0.484 |
| GraphConv (GNN) | 0.723 | - |
| AttentiveFP (SOTA GNN) | 0.741 | - |
| **CR k=3 + RF (ours)** | **0.722** | -0.124 |
| **CR k=3 + Physical + RF (ours)** | **0.741** | -0.154 |

**CR matches AttentiveFP on scaffold split. Morgan collapses from R²=0.682 to 0.198 (-0.484).**

No GPU, no message passing, no graph convolution — pure combinatorial counting.

## Why CR Generalizes to New Scaffolds (and Morgan Doesn't)

### Ablation: Connected vs Disconnected Subgraphs

CR k=3 consists of two qualitatively different components:

| Component | Random R² | Scaffold R² | Info Source |
|-----------|-----------|-------------|-------------|
| **CR disconnected** (0-1 bonds) | **0.827** | **0.710** | Atom co-occurrence + local bonds |
| CR connected (2-3 bonds) | 0.703 | 0.233 | Local topology only |
| Morgan ECFP4 | 0.682 | 0.196 | Circular topology |

**The scaffold generalizability comes from disconnected subgraphs.**

- **Disconnected 3-subgraphs**: atom co-occurrence patterns (composition-based) + one-bond patterns with context atom. Examples: `[C,C,C]` (nonpolar density), `[C,C,Cl]` (halogenation).
- **Connected 3-subgraphs**: local bond topology (same as Morgan, different radius). Collapses on scaffold split.
- **Morgan**: only circular topology. No atom co-occurrence information. Collapses.

**Feature importance analysis:**
- Rank 1: `[C,C,C] bonds=[]` (imp=0.31) — three carbons not all adjacent = nonpolar carbon density
- Rank 2: `[C,C,Cl] bonds=[]` (imp=0.26) — chlorine co-occurrence = halogenation
- Top features are chemically meaningful (actual solubility drivers), not scaffold-specific hash codes.

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

## Why It Works: Information Content

**Morgan/ECFP fingerprints** use circular radius-based neighborhoods:
- Each bit = specific atom environment within radius r
- Captures: local topology (scaffold-specific)
- Missing: atom co-occurrence / composition patterns

**CR fingerprints** use ALL induced k-subgraph patterns:
- Connected subgraphs: local topology (like Morgan, not radius-bounded)
- Disconnected subgraphs: atom co-occurrence across molecular distance
- No geometric constraint — captures both topology AND composition

**Key formula:** CR k=3 ≈ (local topology fingerprint) + (atom co-occurrence fingerprint)

Atom co-occurrence is scaffold-independent: a molecule with C=O has that regardless of ring system.
Local topology is scaffold-specific: what ring it's in changes across scaffolds.
Morgan captures only local topology → collapses on scaffold split.
CR captures both → partially stable.

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

The scaffold generalizability additionally exploits the **disconnected subgraph component**, which encodes atom co-occurrence patterns that are invariant to scaffold changes.

## Scope Limits

CR is most effective for **physicochemical properties** driven by local chemistry:
- Water solubility (ESOL, FreeSolv) ✓
- LogP / lipophilicity ✓

CR is less effective for **biological activity** requiring 3D binding geometry:
- Tox21 (nuclear receptor activity): GNNs still outperform (CR AUC=0.706 vs GNN 0.772-0.851)

## Benchmarks

- ESOL (Delaney dataset, 1128 molecules): water solubility (logS)
- ChemProp regression test (500 molecules): logSolubility
- All regression results: 5-fold CV, 5 random seeds, RandomForest(n_estimators=200)

## Files

- `cr_fingerprint.py` — Core CR fingerprint computation
- `cr_benchmark.py` — ESOL benchmark
- `cr_vs_gnn.py` — Comparison with published GNN results
- `cr_scaffold_split.py` — Scaffold split (GNN-fair) evaluation
- `cr_theory_analysis.py` — Feature overlap analysis (why Morgan fails)
- `cr_ablation_connected.py` — Connected vs disconnected subgraph ablation
- `cr_poly_theory.py` — Theory: CR disconnected vs polynomial composition features
- `cr_final_ablation.py` — 4-way bond count ablation
- `induced_assembly_index.py` — IAI computation

## Connection to Graph Theory

This work extends:
1. **Assembly Theory** (Cronin et al.): minimum operations to build a molecule
2. **Counting Revolution**: induced subgraph histograms as complete graph classifiers

IAI = new complexity measure bridging both frameworks.
CR fingerprint = practical application of Counting Revolution to cheminformatics.
Disconnected subgraph component = atom co-occurrence fingerprint (novel scaffold-invariant representation).
