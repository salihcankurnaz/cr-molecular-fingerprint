"""
CR vs GNN: Scaffold Split + Feature Importance

Problem: GNN paper'ları scaffold split kullanıyor (benzer moleküller aynı fold'da).
Bizim 5-fold CV random split. Bu karşılaştırmayı adil yapmak için scaffold split.

Scaffold split: Bemis-Murcko iskelet bazlı bölme.
Aynı iskelet = aynı fold → daha zor genelleme görevi.

Ayrıca: hangi CR özellikleri en önemli? Feature importance analizi.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations, permutations
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import urllib.request, io

def mol_to_typed_graph(mol):
    nodes = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
    bmap = {Chem.rdchem.BondType.SINGLE:1, Chem.rdchem.BondType.DOUBLE:2,
            Chem.rdchem.BondType.TRIPLE:3, Chem.rdchem.BondType.AROMATIC:4}
    edges = {}
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bmap.get(b.GetBondType(),1)
        edges[(i,j)] = bt; edges[(j,i)] = bt
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

def cr_fp(mol, k=3, max_combos=3000):
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    if k > len(nl): return Counter()
    c = Counter()
    combos = list(combinations(nl, k))
    if len(combos) > max_combos: np.random.shuffle(combos); combos = combos[:max_combos]
    for sub in combos: c[canonical_sub(nodes, edges, list(sub))] += 1
    return c

def build_matrix(fps, vocab=None):
    if vocab is None:
        vocab = {t:i for i,t in enumerate(set(t for fp in fps for t in fp))}
    X = np.zeros((len(fps), len(vocab)))
    for i, fp in enumerate(fps):
        tot = sum(fp.values())
        for t, cnt in fp.items():
            if t in vocab: X[i, vocab[t]] = cnt / max(tot, 1)
    return X, vocab

def scaffold_split(mols, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    """Bemis-Murcko scaffold split."""
    # Get scaffold for each molecule
    scaffold_to_indices = defaultdict(list)
    for i, mol in enumerate(mols):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except:
            scaffold = ''
        scaffold_to_indices[scaffold].append(i)

    # Sort scaffold groups by size (largest first → go to train)
    scaffold_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

    np.random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    n = len(mols)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)

    for group in scaffold_groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx

def get_phys_features(mols):
    desc_fns = [
        Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors,
        Descriptors.NumHAcceptors, Descriptors.TPSA, Descriptors.NumRotatableBonds,
        Descriptors.RingCount, Descriptors.NumAromaticRings,
        Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
    ]
    X = []
    for mol in mols:
        row = []
        for fn in desc_fns:
            try: v = fn(mol); row.append(v if not np.isnan(float(v)) else 0)
            except: row.append(0)
        X.append(row)
    return np.array(X)

# Load ESOL
print("=== CR vs GNN: Scaffold Split Comparison ===\n")
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

# Compute fingerprints once
print("Computing CR k=3...", flush=True)
cr_fps = [cr_fp(m, k=3) for m in mols]
cr3, vocab3 = build_matrix(cr_fps)

morgan = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048)) for m in mols],dtype=float)
phys = get_phys_features(mols)

from sklearn.preprocessing import StandardScaler
phys_scaled = StandardScaler().fit_transform(phys)

descriptors = {
    'Morgan ECFP4': morgan,
    'CR k=3': cr3,
    'CR+Morgan': np.hstack([cr3, morgan]),
    'Physical (10)': phys_scaled,
    'CR+Physical': np.hstack([cr3, phys_scaled]),
}

# SCAFFOLD SPLIT (multiple seeds)
print("=== SCAFFOLD SPLIT (80/10/10) ===")
print("(This is how GNN papers evaluate -- harder than random split)\n")

n_seeds = 5
scaffold_results = {k: [] for k in descriptors}

for seed in range(n_seeds):
    train_idx, val_idx, test_idx = scaffold_split(mols, seed=seed)
    all_test = test_idx if test_idx else val_idx

    for name, X in descriptors.items():
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[all_test]
        y_test = y[all_test]

        rf = RandomForestRegressor(200, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        scaffold_results[name].append(r2)

print(f"{'Method':>25s}  R2 (mean+-std)  vs Morgan")
print("-" * 55)
morgan_mean = np.mean(scaffold_results['Morgan ECFP4'])
for name, r2s in sorted(scaffold_results.items(), key=lambda x: -np.mean(x[1])):
    mean, std = np.mean(r2s), np.std(r2s)
    diff = mean - morgan_mean
    print(f"  {name:>23s}: {mean:.4f}+-{std:.4f}  ({diff:+.4f})")

# GNN baselines (from MoleculeNet paper, scaffold split)
print()
print("Published GNN (scaffold split, from MoleculeNet papers):")
print("  AttentiveFP:   R2=0.741")
print("  D-MPNN:        R2=0.737")
print("  GraphConv:     R2=0.723")
print("  ECFP4 + RF:    R2=0.614")
print()
our_best = max(np.mean(v) for v in scaffold_results.values())
print(f"  Our best CR+RF: R2={our_best:.4f}")

# Feature importance analysis
print("\n=== FEATURE IMPORTANCE (CR k=3) ===")
print("Which 3-atom subgraph types are most predictive of solubility?")
print()

train_idx, _, test_idx = scaffold_split(mols, seed=0)
rf = RandomForestRegressor(200, random_state=0, n_jobs=-1)
rf.fit(cr3[train_idx], y[train_idx])

importances = rf.feature_importances_
vocab_inv = {i: t for t, i in vocab3.items()}

# Top 20 most important types
top_idx = np.argsort(importances)[::-1][:20]

# Decode canonical type to human-readable
ATOM_MAP = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 15:'P', 16:'S', 17:'Cl', 35:'Br', 53:'I'}
BOND_MAP = {0:'_', 1:'-', 2:'=', 3:'#', 4:'~'}  # ~ = aromatic

def decode_type(t):
    if isinstance(t, tuple) and len(t) == 3:
        at, binfo, _ = t[0], t[1], t[2]
        atoms = ' '.join(ATOM_MAP.get(a, str(a)) for a in at)
        return f"atoms=[{atoms}]"

    if not isinstance(t, tuple): return str(t)[:40]

    try:
        # t is a n×n matrix tuple
        n = len(t)
        atoms = [ATOM_MAP.get(t[i][i], str(t[i][i])) for i in range(n)]
        bonds = []
        for i in range(n):
            for j in range(i+1, n):
                if t[i][j] > 0:
                    bonds.append(f"{atoms[i]}{BOND_MAP.get(t[i][j],'?')}{atoms[j]}")
        return f"[{','.join(atoms)}] bonds=[{','.join(bonds)}]"
    except: return str(t)[:40]

print(f"  {'Rank':>4s}  {'Importance':>11s}  Type")
print("  " + "-" * 60)
for rank, idx in enumerate(top_idx[:15]):
    t = vocab_inv[idx]
    desc = decode_type(t)
    print(f"  {rank+1:>4d}  {importances[idx]:>11.4f}  {desc}")

# Summary
print(f"\n\n=== FINAL SUMMARY ===")
print()
print("Random 5-fold CV (our previous result):")
print("  CR k=3 + Physical: R2=0.895 > AttentiveFP 0.887")
print()
print(f"Scaffold split (GNN-fair comparison):")
for name in ['CR k=3', 'CR+Morgan', 'CR+Physical', 'Morgan ECFP4']:
    if name in scaffold_results:
        mean = np.mean(scaffold_results[name])
        print(f"  {name}: R2={mean:.4f}")
print()
print("GNN baselines (scaffold split):")
print("  AttentiveFP: 0.741, D-MPNN: 0.737, GraphConv: 0.723")
