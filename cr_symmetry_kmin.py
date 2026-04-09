"""
k_min vs Molecular Symmetry (Automorphism Group Size)

HYPOTHESIS: Highly symmetric molecules have large automorphism groups
and require larger k (higher k_min) to be uniquely identified.

Mathematical reason: If mol M has automorphism sigma, then
  canonical_sub(sigma(v1), sigma(v2), sigma(v3)) = canonical_sub(v1, v2, v3)
-> The histogram is invariant under automorphisms
-> Different subsets of nodes may yield identical canonical forms
-> Fewer distinct canonical forms -> less distinguishing power

PREDICTION: |Aut(G)| correlates POSITIVELY with k_min
"""
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations, permutations
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
import networkx as nx
from networkx.algorithms import isomorphism
from scipy.stats import spearmanr
import urllib.request, io

def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetAtomicNum())
    bmap = {Chem.rdchem.BondType.SINGLE:1,Chem.rdchem.BondType.DOUBLE:2,
            Chem.rdchem.BondType.TRIPLE:3,Chem.rdchem.BondType.AROMATIC:4}
    for bond in mol.GetBonds():
        bt = bmap.get(bond.GetBondType(), 1)
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=bt)
    return G

def count_automorphisms(G, max_count=1000):
    """Count automorphisms of a typed molecular graph."""
    node_match = lambda a, b: a['label'] == b['label']
    edge_match = lambda a, b: a['label'] == b['label']
    em = isomorphism.GraphMatcher(G, G, node_match=node_match, edge_match=edge_match)
    count = 0
    for _ in em.isomorphisms_iter():
        count += 1
        if count >= max_count: break
    return count

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

def cr_fp(mol, k=3, max_combos=5000):
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    if k > len(nl): return Counter()
    c = Counter()
    combos = list(combinations(nl, k))
    if len(combos) > max_combos: np.random.shuffle(combos); combos = combos[:max_combos]
    for sub in combos: c[canonical_sub(nodes, edges, list(sub))] += 1
    return c

# Load ESOL
print("=== k_min vs Molecular Symmetry ===\n")
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
            if not np.isnan(t): mols.append(mol); ys.append(t); smiles_list.append(str(smi))
        except: pass
y = np.array(ys)
print(f"ESOL: N={len(mols)}\n")

# Compute automorphism group sizes (cap at 1000 for speed)
print("Computing automorphism group sizes...")
G_list = [mol_to_nx(m) for m in mols]
auto_sizes = []
for i, G in enumerate(G_list):
    if i % 200 == 0: print(f"  {i}/{len(G_list)}...")
    size = count_automorphisms(G, max_count=500)
    auto_sizes.append(size)
auto_sizes = np.array(auto_sizes, dtype=float)
print(f"Done. Automorphism sizes: min={auto_sizes.min():.0f}, max={auto_sizes.max():.0f}, mean={auto_sizes.mean():.2f}\n")

# Compute CR k=3 fingerprints and k_min
print("Computing CR k=3 fingerprints...")
fps3 = [cr_fp(m, k=3, max_combos=5000) for m in mols]
sigs3 = [frozenset(fp.items()) for fp in fps3]
sig3_groups = {}
for i, s in enumerate(sigs3):
    sig3_groups.setdefault(s, []).append(i)

fps2 = [cr_fp(m, k=2, max_combos=5000) for m in mols]
sigs2 = [frozenset(fp.items()) for fp in fps2]
sig2_groups = {}
for i, s in enumerate(sigs2):
    sig2_groups.setdefault(s, []).append(i)

kmin = []
for i in range(len(mols)):
    if len(sig2_groups[sigs2[i]]) == 1:
        kmin.append(2)
    elif len(sig3_groups[sigs3[i]]) == 1:
        kmin.append(3)
    else:
        kmin.append(4)
kmin = np.array(kmin)

# Also compute "effective distinct types" at k=3 as an IAI-like measure
n_atoms = np.array([mol.GetNumAtoms() for mol in mols])
n_distinct3 = np.array([len(fp) for fp in fps3])

# IAI: distinct types / C(n, 3)
from math import comb
iai3 = np.array([n_distinct3[i] / max(comb(n_atoms[i], 3), 1) for i in range(len(mols))])

# Correlations
r_auto_kmin, _ = spearmanr(auto_sizes, kmin)
r_auto_iai, _ = spearmanr(auto_sizes, iai3)
r_auto_n, _ = spearmanr(auto_sizes, n_atoms)
r_kmin_n, _ = spearmanr(kmin, n_atoms)

print(f"Correlations (Spearman rho):")
print(f"  |Aut(G)| vs k_min:    rho = {r_auto_kmin:.4f}")
print(f"  |Aut(G)| vs IAI_k3:   rho = {r_auto_iai:.4f}")
print(f"  |Aut(G)| vs n_atoms:  rho = {r_auto_n:.4f}")
print(f"  k_min    vs n_atoms:  rho = {r_kmin_n:.4f}")
print()

# Distribution by k_min
print("Automorphism group size by k_min:")
for km in [2, 3, 4]:
    mask = kmin == km
    if km == 4: mask = kmin >= 4
    label = f"k_min={km}" if km < 4 else "k_min>=4"
    if mask.sum() > 0:
        print(f"  {label}: mean|Aut|={auto_sizes[mask].mean():.2f}, median={np.median(auto_sizes[mask]):.1f}, max={auto_sizes[mask].max():.0f}")

# Most symmetric molecules
print(f"\nMost symmetric molecules (largest |Aut|):")
sym_idx = np.argsort(-auto_sizes)[:10]
for i in sym_idx:
    print(f"  |Aut|={auto_sizes[i]:.0f}  k_min={kmin[i]}  n={n_atoms[i]}  IAI={iai3[i]:.4f}  logS={y[i]:.2f}  {smiles_list[i][:40]}")

# Least symmetric (|Aut|=1 means trivial group = most asymmetric)
print(f"\nMost asymmetric molecules (|Aut|=1, trivial group):")
asym_idx = [i for i in range(len(mols)) if auto_sizes[i] == 1]
print(f"  Count: {len(asym_idx)} ({len(asym_idx)/len(mols)*100:.1f}%)")
for i in asym_idx[:5]:
    print(f"  |Aut|=1  k_min={kmin[i]}  n={n_atoms[i]}  {smiles_list[i][:50]}")

# Test: do k_min>=4 mols have larger |Aut|?
high_kmin = auto_sizes[kmin >= 4]
low_kmin = auto_sizes[kmin == 2]
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(high_kmin, low_kmin, alternative='greater')
print(f"\nMann-Whitney test: |Aut| larger for k_min>=4 vs k_min=2")
print(f"  U={stat:.0f}, p={pval:.4f}")
print(f"  Significant (p<0.05): {pval < 0.05}")

print(f"\n=== THEORETICAL SUMMARY ===")
print(f"\nHypothesis: |Aut(G)| positively correlates with k_min")
print(f"Result: rho = {r_auto_kmin:.4f}")
if r_auto_kmin > 0.1:
    print(f"CONFIRMED: More symmetric molecules (larger automorphism group)")
    print(f"require larger k to be uniquely identified in a molecular dataset.")
    print(f"")
    print(f"Mathematical reasoning:")
    print(f"  Automorphism sigma: canonical_sub(sigma(S)) = canonical_sub(S)")
    print(f"  => Symmetric molecules have fewer distinct canonical k-subgraph types")
    print(f"  => Easier to share fingerprints with similar molecules")
    print(f"  => Need larger k to become unique")
elif r_auto_kmin < -0.1:
    print(f"REVERSED: More symmetric molecules are EASIER to identify (k_min lower)")
else:
    print(f"WEAK correlation: Symmetry has little effect on k_min in practice")
    print(f"  (May be dominated by dataset composition effects)")
