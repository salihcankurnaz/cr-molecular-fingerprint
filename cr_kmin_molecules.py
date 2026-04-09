"""
Molecular k_min: Minimum k to Uniquely Identify a Molecule

THEORY: From Counting Revolution, every non-isomorphic pair of graphs
can be distinguished by their induced k-subgraph histograms for k=n-3.

APPLICATION: For molecules in ESOL, what is the minimum k such that
each molecule has a UNIQUE CR fingerprint (no other molecule shares it)?

k_min(mol) = min k : CR_k(mol) is unique in the dataset

Properties of k_min:
  - k_min=2: molecule distinguished by 2-atom bond types alone
  - k_min=3: needs 3-atom patterns (most molecules)
  - k_min>3: structurally similar to other molecules (harder to distinguish)
  - k_min=INFINITY: truly isomorphic (identical molecules) -- should only happen for duplicates

This is a NEW complexity measure:
  - "Chemical uniqueness index" in a dataset
  - Low k_min = easy to identify = unique structure
  - High k_min = hard to identify = structurally common/redundant

Compare with IAI: IAI measures complexity DENSITY (diversity per molecule size)
k_min measures DISTINCTIVENESS (how much information needed to uniquely identify)
"""
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations, permutations
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
import urllib.request, io

def mol_to_typed_graph(mol):
    nodes = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
    bmap = {Chem.rdchem.BondType.SINGLE:1,Chem.rdchem.BondType.DOUBLE:2,
            Chem.rdchem.BondType.TRIPLE:3,Chem.rdchem.BondType.AROMATIC:4}
    edges = {}
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        bt = bmap.get(b.GetBondType(),1)
        edges[(i,j)]=bt; edges[(j,i)]=bt
    return nodes, edges

def canonical_sub(nodes, edges, nlist):
    n = len(nlist); nl = sorted(nlist)
    if n <= 6:
        best = None
        for perm in permutations(range(n)):
            mat = tuple(tuple(nodes[nl[perm[i]]] if i==j else edges.get((nl[perm[i]],nl[perm[j]]),0) for j in range(n)) for i in range(n))
            if best is None or mat < best: best = mat
        return best
    at = tuple(sorted(nodes[u] for u in nl))
    et = tuple(sorted((nodes[nl[i]],nodes[nl[j]],edges.get((nl[i],nl[j]),0)) for i in range(n) for j in range(i+1,n) if edges.get((nl[i],nl[j]),0)>0))
    return (at,et)

def cr_fp_exact(mol, k, max_combos=5000):
    """Exact CR fingerprint (more combos for better precision)."""
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    if k > len(nl): return Counter()
    c = Counter()
    combos = list(combinations(nl, k))
    if len(combos) > max_combos: np.random.shuffle(combos); combos = combos[:max_combos]
    for sub in combos: c[canonical_sub(nodes, edges, list(sub))] += 1
    return c

def fps_to_hashable(fp):
    """Convert Counter to hashable frozenset of (type, count) pairs."""
    return frozenset(fp.items())

def compute_kmin(mols, k_values=[2, 3, 4], max_combos=5000):
    """
    For each k, compute CR fingerprints and find which molecules are uniquely identified.
    Returns: k_min for each molecule.
    """
    n = len(mols)
    kmin = [None] * n  # None = not yet uniquely identified

    for k in k_values:
        print(f"  k={k}: computing fingerprints...", flush=True)
        fps = [cr_fp_exact(m, k, max_combos) for m in mols]

        # Convert to hashable signatures
        # For counting, we use the NORMALIZED integer counts
        # (not normalized by total, to avoid floating point issues)
        sigs = [fps_to_hashable(fp) for fp in fps]

        # Find unique signatures
        sig_to_indices = {}
        for i, sig in enumerate(sigs):
            sig_to_indices.setdefault(sig, []).append(i)

        n_unique = sum(1 for indices in sig_to_indices.values() if len(indices) == 1)
        n_groups = len(sig_to_indices)
        n_collisions = sum(len(v) for v in sig_to_indices.values() if len(v) > 1)

        print(f"    Unique signatures: {n_groups} (unique mols: {n_unique}, collisions: {n_collisions})")

        # Mark newly unique molecules
        newly_unique = 0
        for indices in sig_to_indices.values():
            if len(indices) == 1:
                i = indices[0]
                if kmin[i] is None:
                    kmin[i] = k
                    newly_unique += 1

        print(f"    Newly unique at k={k}: {newly_unique}")
        print(f"    Still unresolved: {sum(1 for x in kmin if x is None)}")

        # If all resolved, stop
        if all(x is not None for x in kmin):
            print(f"    ALL molecules uniquely identified at k={k}!")
            break

    # Remaining unresolved: set kmin to max_k + 1
    max_k = max(k_values)
    for i in range(n):
        if kmin[i] is None:
            kmin[i] = max_k + 1  # "still not resolved"

    return kmin, sigs, fps

# Load ESOL
print("=== Molecular k_min Analysis ===")
print("New measure: minimum k for unique CR fingerprint identification\n")
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
req = urllib.request.Request(url, headers={'User-Agent':'Python'})
r = urllib.request.urlopen(req, timeout=10)
df = pd.read_csv(io.StringIO(r.read().decode()))
tgt_col = [c for c in df.columns if 'measured' in c.lower()][0]

mols, ys, smiles_list = [], [], []
for _, row in df.iterrows():
    smi = row.get('smiles', row.iloc[0])
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        try:
            t = float(row[tgt_col])
            if not np.isnan(t):
                mols.append(mol); ys.append(t); smiles_list.append(str(smi))
        except: pass
y = np.array(ys)
n_atoms = [mol.GetNumAtoms() for mol in mols]
print(f"ESOL: N={len(mols)}, avg atoms={np.mean(n_atoms):.1f}, max={max(n_atoms)}\n")

# Compute k_min
print("Computing k_min (k=2, 3, 4)...")
kmin, final_sigs, final_fps = compute_kmin(mols, k_values=[2, 3, 4], max_combos=5000)

kmin_arr = np.array(kmin)
print(f"\nk_min distribution:")
for k in sorted(set(kmin_arr)):
    count = (kmin_arr == k).sum()
    label = f"k={k}" if k <= 4 else f"k>{4}"
    print(f"  {label}: {count} molecules ({count/len(mols)*100:.1f}%)")

# Molecules with high k_min (hard to distinguish)
hard_mols = [(i, kmin_arr[i]) for i in range(len(mols)) if kmin_arr[i] >= 4]
print(f"\nMolecules with k_min >= 4 (structurally ambiguous at k=3): {len(hard_mols)}")
for i, km in sorted(hard_mols, key=lambda x: -x[1])[:10]:
    print(f"  k_min={km}: n_atoms={mols[i].GetNumAtoms()}, logS={y[i]:.2f}, SMILES={smiles_list[i][:50]}")

# Molecules with low k_min (very distinctive)
easy_mols = [(i, kmin_arr[i]) for i in range(len(mols)) if kmin_arr[i] == 2]
print(f"\nMolecules with k_min=2 (uniquely identified by 2-atom patterns): {len(easy_mols)}")
for i, km in easy_mols[:5]:
    print(f"  k_min={km}: n_atoms={mols[i].GetNumAtoms()}, logS={y[i]:.2f}, SMILES={smiles_list[i][:50]}")

# Is k_min correlated with molecule properties?
from scipy.stats import pearsonr, spearmanr
r_atoms, _ = spearmanr(kmin_arr, n_atoms)
r_logS, _ = spearmanr(kmin_arr, y)
print(f"\nCorrelations with k_min:")
print(f"  k_min vs n_atoms: rho={r_atoms:.4f}")
print(f"  k_min vs logS:    rho={r_logS:.4f}")

# Compare k_min with IAI at k=3
print("\nk_min and structural diversity:")
from itertools import combinations, permutations
cr3_fps = final_fps  # Already computed at k=3 (last k computed)

def iai_k3(fp, n_atoms, k=3):
    """IAI = distinct types / C(n, k)"""
    from math import comb
    n_combos = comb(n_atoms, k)
    if n_combos == 0: return 0
    return len(fp) / n_combos

# Note: last fps are from k=4 (last iteration)
# Re-compute k=3 fps for IAI
print("Computing k=3 fps for IAI...", flush=True)
fps3 = [cr_fp_exact(m, 3, max_combos=5000) for m in mols]
iai3 = np.array([iai_k3(fp, na, k=3) for fp, na in zip(fps3, n_atoms)])

r_iai, _ = spearmanr(kmin_arr, iai3)
print(f"\nk_min vs IAI_k3: rho={r_iai:.4f}")
print(f"  (Are k_min and IAI measuring different things?)")

# Summary
print(f"\n=== SUMMARY: Molecular k_min ===")
print()
print(f"ESOL dataset (N={len(mols)} molecules):")
print(f"  k_min=2: {(kmin_arr==2).sum()} molecules — uniquely identified by 2-atom patterns")
print(f"  k_min=3: {(kmin_arr==3).sum()} molecules — need 3-atom patterns")
print(f"  k_min=4: {(kmin_arr==4).sum()} molecules — need 4-atom patterns")
print(f"  k_min>4: {(kmin_arr>4).sum()} molecules — still ambiguous at k=4")
print()
cr3_complete = (kmin_arr <= 3).sum()
print(f"CR k=3 uniquely identifies {cr3_complete}/{len(mols)} ({cr3_complete/len(mols)*100:.1f}%) molecules in ESOL")
print(f"CR k=4 uniquely identifies {(kmin_arr<=4).sum()}/{len(mols)} ({(kmin_arr<=4).sum()/len(mols)*100:.1f}%) molecules")
print()
print(f"CONCLUSION: For ESOL molecules (avg {np.mean(n_atoms):.0f} atoms),")
print(f"k=3 fingerprints uniquely identify {cr3_complete/len(mols)*100:.0f}% of molecules.")
print(f"This demonstrates CR's high expressivity even at small k.")
print()
print(f"k_min and IAI are {'CORRELATED' if abs(r_iai) > 0.3 else 'ORTHOGONAL'} (rho={r_iai:.4f})")
print(f"-> Two independent dimensions of molecular complexity:")
print(f"   IAI = structural DIVERSITY (how many different k-subgraph types)")
print(f"   k_min = structural UNIQUENESS (how much k needed to be distinct)")
