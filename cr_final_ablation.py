"""
Final Ablation: 3-way split of k=3 subgraphs

k=3 induced subgraph types (by bond count in induced subgraph):
  0-bond: 3 atoms, no bonds between them -> pure atom co-occurrence (composition)
  1-bond: 2 atoms bonded, 1 isolated -> (bond_type, third_atom) pattern
  2-bond: path A-B-C or cycle-related -> local path topology
  3-bond: triangle (A bonded to B, C; B bonded to C) -> fully connected

"Disconnected" (from prior ablation) = 0-bond + 1-bond (not all 3 connected)
"Connected" = 2-bond + 3-bond

THEORY:
  0-bond: scaffold-independent atom co-occurrence -> stable
  1-bond: scaffold-independent local bond + context -> stable
  2-bond: path A-B-C, depends on scaffold path structure -> scaffold-specific?
  3-bond: triangle (ring fragment) -> scaffold-specific

Prediction: 0-bond and 1-bond are stable; 2-bond and 3-bond collapse.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations, permutations
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
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

def bond_count_in_subgraph(edges, nlist):
    count = 0
    for i in range(len(nlist)):
        for j in range(i+1, len(nlist)):
            if edges.get((nlist[i], nlist[j]), 0) > 0:
                count += 1
    return count

def cr_fp_4way(mol, k=3, max_combos=3000):
    """Split CR into 4 types by bond count."""
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    if k > len(nl): return Counter(), Counter(), Counter(), Counter()
    c0 = Counter(); c1 = Counter(); c2 = Counter(); c3 = Counter()
    combos = list(combinations(nl, k))
    if len(combos) > max_combos: np.random.shuffle(combos); combos = combos[:max_combos]
    for sub in combos:
        canon = canonical_sub(nodes, edges, list(sub))
        bc = bond_count_in_subgraph(edges, list(sub))
        if bc == 0: c0[canon] += 1
        elif bc == 1: c1[canon] += 1
        elif bc == 2: c2[canon] += 1
        else: c3[canon] += 1
    return c0, c1, c2, c3

def scaffold_split(mols, seed=0):
    scaffold_to_idx = defaultdict(list)
    for i, mol in enumerate(mols):
        try: sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except: sc = ''
        scaffold_to_idx[sc].append(i)
    groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)
    n = len(mols)
    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= int(0.8*n): train_idx.extend(g)
        elif len(val_idx) + len(g) <= int(0.1*n): val_idx.extend(g)
        else: test_idx.extend(g)
    return train_idx, val_idx, test_idx

url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
req = urllib.request.Request(url, headers={'User-Agent':'Python'})
r = urllib.request.urlopen(req, timeout=10)
df = pd.read_csv(io.StringIO(r.read().decode()))
tgt_col = [c for c in df.columns if 'measured' in c.lower()][0]

mols, ys = [], []
for _, row in df.iterrows():
    smi = row.get('smiles', row.iloc[0])
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        try:
            t = float(row[tgt_col])
            if not np.isnan(t): mols.append(mol); ys.append(t)
        except: pass
y = np.array(ys)
print(f"ESOL: N={len(mols)}\n")

print("Computing 4-way CR k=3 split...", flush=True)
fps0, fps1, fps2, fps3 = [], [], [], []
for m in mols:
    c0, c1, c2, c3 = cr_fp_4way(m, k=3)
    fps0.append(c0); fps1.append(c1); fps2.append(c2); fps3.append(c3)

def to_matrix(fps):
    vocab = {t:i for i,t in enumerate(set(t for fp in fps for t in fp))}
    X = np.zeros((len(fps), max(len(vocab), 1)))
    for i, fp in enumerate(fps):
        tot = sum(fp.values())
        for t, cnt in fp.items():
            if t in vocab: X[i, vocab[t]] = cnt / max(tot, 1)
    return X, vocab

X0, v0 = to_matrix(fps0)
X1, v1 = to_matrix(fps1)
X2, v2 = to_matrix(fps2)
X3, v3 = to_matrix(fps3)

morgan = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048)) for m in mols],dtype=float)

print(f"  0-bond types: {len(v0)} (pure composition)")
print(f"  1-bond types: {len(v1)} (bond + isolated atom)")
print(f"  2-bond types: {len(v2)} (path/partial ring)")
print(f"  3-bond types: {len(v3)} (triangle/fully connected)")
print()

configs = [
    ("Morgan ECFP4", morgan),
    ("0-bond (composition)", X0),
    ("1-bond (bond+atom)", X1),
    ("2-bond (paths)", X2),
    ("3-bond (triangles)", X3),
    ("0+1 bond (disconnected)", np.hstack([X0, X1])),
    ("2+3 bond (connected)", np.hstack([X2, X3])),
    ("All k=3", np.hstack([X0, X1, X2, X3])),
]

print("=== RANDOM 5-FOLD CV ===")
n_seeds = 5
cv_results = {}
for name, X in configs:
    if X.shape[1] == 0: cv_results[name] = (0.0, 0.0); continue
    r2s = []
    for seed in range(n_seeds):
        cv = KFold(5, shuffle=True, random_state=seed)
        rf = RandomForestRegressor(200, random_state=seed, n_jobs=-1)
        scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
        r2s.append(scores.mean())
    cv_results[name] = (np.mean(r2s), np.std(r2s))
    print(f"  {name:>28s}: R2={np.mean(r2s):.4f} +- {np.std(r2s):.4f}")

print("\n=== SCAFFOLD SPLIT ===")
train_idx, val_idx, test_idx = scaffold_split(mols, seed=0)
test_all = test_idx if test_idx else val_idx
scaffold_results = {}
for name, X in configs:
    if X.shape[1] == 0: scaffold_results[name] = 0.0; continue
    rf = RandomForestRegressor(200, random_state=0, n_jobs=-1)
    rf.fit(X[train_idx], y[train_idx])
    scaffold_results[name] = r2_score(y[test_all], rf.predict(X[test_all]))

print(f"\n{'Method':>28s}  Random  Scaffold  Collapse")
print("-" * 65)
for name, X in configs:
    cv_r2 = cv_results[name][0]
    sc_r2 = scaffold_results[name]
    print(f"  {name:>26s}: {cv_r2:.4f}   {sc_r2:.4f}  ({sc_r2-cv_r2:+.4f})")

print("\n=== FINAL THEORY ===")
print()
r0_cv = cv_results['0-bond (composition)'][0]
r1_cv = cv_results['1-bond (bond+atom)'][0]
r2_cv = cv_results['2-bond (paths)'][0]
r3_cv = cv_results['3-bond (triangles)'][0]
r0_sc = scaffold_results['0-bond (composition)']
r1_sc = scaffold_results['1-bond (bond+atom)']
r2_sc = scaffold_results['2-bond (paths)']
r3_sc = scaffold_results['3-bond (triangles)']

print(f"Bond count  Random  Scaffold  Info source")
print(f"  0-bond:   {r0_cv:.4f}   {r0_sc:.4f}   Atom co-occurrence (composition)")
print(f"  1-bond:   {r1_cv:.4f}   {r1_sc:.4f}   Bond type + third atom context")
print(f"  2-bond:   {r2_cv:.4f}   {r2_sc:.4f}   Path A-B-C (local topology)")
print(f"  3-bond:   {r3_cv:.4f}   {r3_sc:.4f}   Triangle (ring fragment)")
print(f"  Morgan:   {cv_results['Morgan ECFP4'][0]:.4f}   {scaffold_results['Morgan ECFP4']:.4f}   Circular topology (all bond counts)")
print()
print("Scaffold stability analysis:")
for name, X in configs:
    cv_r2 = cv_results[name][0]
    sc_r2 = scaffold_results[name]
    stable = "STABLE" if abs(sc_r2 - cv_r2) < 0.15 else "COLLAPSES"
    print(f"  {name:>26s}: {stable} ({sc_r2-cv_r2:+.4f})")
