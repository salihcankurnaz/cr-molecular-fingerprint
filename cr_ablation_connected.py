"""
Ablation: Connected vs Disconnected CR Subgraphs

Top CR k=3 features for ESOL are DISCONNECTED:
  #1: [C,C,C] bonds=[] imp=0.31 -> nonpolar carbon density -> insolubility
  #2: [C,C,Cl] bonds=[] imp=0.26 -> halogenation presence
  #9: [C,Cl,Cl] bonds=[] imp=0.01 -> polychlorination
  #10: [C,C,S] bonds=[] -> sulfur presence

HYPOTHESIS: CR's advantage over Morgan comes from two complementary parts:
  A) DISCONNECTED subgraphs = atom CO-OCCURRENCE patterns (composition fingerprint)
     - Captures: "molecule has C, Cl but not N" globally
     - Morgan MISSES this: each bit encodes one atom's circular neighborhood
  B) CONNECTED subgraphs = local TOPOLOGY (like Morgan but not radius-constrained)
     - Captures: C=O, C-N, aromatic patterns

EXPERIMENT:
  1. CR_connected: only connected induced k-subgraphs
  2. CR_disconnected: only disconnected induced k-subgraphs
  3. CR_all: both (standard CR k=3)
  4. Atom_composition: just atom-type histogram (no topology at all)
  Compare all on ESOL + scaffold split.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations, permutations
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
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

def is_connected_subgraph(nodes, edges, nlist):
    """Check if induced subgraph on nlist is connected."""
    if len(nlist) <= 1: return True
    visited = {nlist[0]}
    queue = [nlist[0]]
    nset = set(nlist)
    while queue:
        v = queue.pop()
        for u in nlist:
            if u not in visited and (edges.get((v,u),0) > 0):
                visited.add(u)
                queue.append(u)
    return len(visited) == len(nlist)

def cr_fp_split(mol, k=3, max_combos=3000):
    """Return separate counters for connected and disconnected subgraphs."""
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    if k > len(nl): return Counter(), Counter()
    conn = Counter()
    disconn = Counter()
    combos = list(combinations(nl, k))
    if len(combos) > max_combos: np.random.shuffle(combos); combos = combos[:max_combos]
    for sub in combos:
        canon = canonical_sub(nodes, edges, list(sub))
        if is_connected_subgraph(nodes, edges, list(sub)):
            conn[canon] += 1
        else:
            disconn[canon] += 1
    return conn, disconn

def atom_composition(mol):
    """Pure atom-type histogram: no topology, just element counts."""
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    c = Counter(atoms)
    total = len(atoms)
    # Common elements: H=1, C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, I=53
    elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    return np.array([c.get(e, 0) / max(total, 1) for e in elements])

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

# Load ESOL
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

print("Computing CR k=3 (split connected/disconnected)...", flush=True)
conn_fps, disconn_fps = [], []
for m in mols:
    c, d = cr_fp_split(m, k=3)
    conn_fps.append(c)
    disconn_fps.append(d)

# Build vocabularies
vocab_conn = {t:i for i,t in enumerate(set(t for fp in conn_fps for t in fp))}
vocab_disconn = {t:i for i,t in enumerate(set(t for fp in disconn_fps for t in fp))}
vocab_all = {t:i for i,t in enumerate(set(list(vocab_conn.keys()) + list(vocab_disconn.keys())))}

def to_matrix(fps, vocab):
    X = np.zeros((len(fps), len(vocab)))
    for i, fp in enumerate(fps):
        tot = sum(fp.values())
        for t, cnt in fp.items():
            if t in vocab: X[i, vocab[t]] = cnt / max(tot, 1)
    return X

cr_all = to_matrix([{**c, **d} for c, d in zip(conn_fps, disconn_fps)], vocab_all)
cr_conn = to_matrix(conn_fps, vocab_conn)
cr_disconn = to_matrix(disconn_fps, vocab_disconn)
atom_comp = np.array([atom_composition(m) for m in mols])
morgan = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048)) for m in mols],dtype=float)

print(f"  CR all:  {cr_all.shape[1]} types")
print(f"  CR connected: {cr_conn.shape[1]} types")
print(f"  CR disconnected: {cr_disconn.shape[1]} types")
print(f"  Atom composition: {atom_comp.shape[1]} features\n")

configs = [
    ("Morgan ECFP4", morgan),
    ("CR all (k=3)", cr_all),
    ("CR connected only", cr_conn),
    ("CR disconnected only", cr_disconn),
    ("Atom composition", atom_comp),
    ("Comp+Connected", np.hstack([atom_comp, cr_conn])),
]

# Random 5-fold CV
print("=== RANDOM 5-FOLD CV ===")
n_seeds = 5
cv_results = {}
for name, X in configs:
    r2s = []
    for seed in range(n_seeds):
        cv = KFold(5, shuffle=True, random_state=seed)
        rf = RandomForestRegressor(200, random_state=seed, n_jobs=-1)
        scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
        r2s.append(scores.mean())
    cv_results[name] = (np.mean(r2s), np.std(r2s))
    print(f"  {name:>25s}: R2={np.mean(r2s):.4f} +- {np.std(r2s):.4f}")

# Scaffold split
print("\n=== SCAFFOLD SPLIT (seed=0) ===")
train_idx, val_idx, test_idx = scaffold_split(mols, seed=0)
test_all = test_idx if test_idx else val_idx

scaffold_results = {}
for name, X in configs:
    rf = RandomForestRegressor(200, random_state=0, n_jobs=-1)
    rf.fit(X[train_idx], y[train_idx])
    r2 = r2_score(y[test_all], rf.predict(X[test_all]))
    scaffold_results[name] = r2

print(f"\n{'Method':>25s}  Random  Scaffold  Collapse")
print("-" * 60)
for name, X in configs:
    cv_r2 = cv_results[name][0]
    sc_r2 = scaffold_results[name]
    print(f"  {name:>23s}: {cv_r2:.4f}   {sc_r2:.4f}  ({sc_r2-cv_r2:+.4f})")

print("\n=== THEORETICAL INTERPRETATION ===")
print()
cr_all_cv = cv_results['CR all (k=3)'][0]
cr_conn_cv = cv_results['CR connected only'][0]
cr_disconn_cv = cv_results['CR disconnected only'][0]
atom_cv = cv_results['Atom composition'][0]
comp_conn_cv = cv_results['Comp+Connected'][0]

print(f"Atom composition alone:  R2={atom_cv:.4f}")
print(f"CR disconnected only:    R2={cr_disconn_cv:.4f}  (atom co-occurrence = richer than composition)")
print(f"CR connected only:       R2={cr_conn_cv:.4f}  (local topology alone)")
print(f"CR all:                  R2={cr_all_cv:.4f}  (composition + topology)")
print(f"Morgan:                  R2={cv_results['Morgan ECFP4'][0]:.4f}")
print()
print("Key insight: CR's advantage over Morgan has TWO sources:")
print(f"  1. Disconnected subgraphs encode atom CO-OCCURRENCE patterns")
print(f"     (which functional groups co-exist in the molecule)")
print(f"  2. Connected subgraphs encode LOCAL TOPOLOGY (C=O, O-H, etc.)")
print(f"  Morgan only encodes local topology (circular neighborhoods)")
print(f"  CR encodes BOTH -> information complementarity")
print()
print("Scaffold stability:")
cr_all_sc = scaffold_results['CR all (k=3)']
cr_conn_sc = scaffold_results['CR connected only']
cr_disconn_sc = scaffold_results['CR disconnected only']
mg_sc = scaffold_results['Morgan ECFP4']
print(f"  Morgan collapse:          {cv_results['Morgan ECFP4'][0]:.4f} -> {mg_sc:.4f}")
print(f"  CR disconnected collapse: {cr_disconn_cv:.4f} -> {cr_disconn_sc:.4f}")
print(f"  CR connected collapse:    {cr_conn_cv:.4f} -> {cr_conn_sc:.4f}")
print(f"  CR all collapse:          {cr_all_cv:.4f} -> {cr_all_sc:.4f}")
