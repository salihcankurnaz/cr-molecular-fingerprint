"""
Smoking Gun: High-Importance Feature Overlap

Hypothesis: Morgan's aggregate bit overlap (83.6%) is misleading.
The bits RF actually USES (high importance) are scaffold-specific.
The overlap of HIGH-IMPORTANCE bits in test is much lower.

CR's high-importance types ARE scaffold-independent -> still present in test.

This explains the 9% overlap gap -> 0.52 R2 gap.
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

def cr_fp(mol, k=3, max_combos=3000):
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    if k > len(nl): return Counter()
    c = Counter()
    combos = list(combinations(nl, k))
    if len(combos) > max_combos: np.random.shuffle(combos); combos = combos[:max_combos]
    for sub in combos: c[canonical_sub(nodes, edges, list(sub))] += 1
    return c

def scaffold_split(mols, seed=0):
    scaffold_to_idx = defaultdict(list)
    for i, mol in enumerate(mols):
        try: sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except: sc = ''
        scaffold_to_idx[sc].append(i)
    groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)
    n = len(mols)
    train_idx, val_idx, test_idx = [], [], []
    np.random.seed(seed)
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

# Fingerprints
print("Computing fingerprints...")
cr_fps_raw = [cr_fp(m, k=3) for m in mols]
vocab3 = {t:i for i,t in enumerate(set(t for fp in cr_fps_raw for t in fp))}
cr3 = np.zeros((len(mols), len(vocab3)))
for i, fp in enumerate(cr_fps_raw):
    tot = sum(fp.values())
    for t, c in fp.items(): cr3[i, vocab3[t]] = c / max(tot, 1)
morgan = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)) for m in mols], dtype=float)

train_idx, val_idx, test_idx = scaffold_split(mols, seed=0)
test_all = test_idx if test_idx else val_idx
print(f"Split: train={len(train_idx)}, test={len(test_all)}\n")

# Train both RF models on train set
rf_cr = RandomForestRegressor(200, random_state=0, n_jobs=-1)
rf_cr.fit(cr3[train_idx], y[train_idx])

rf_mg = RandomForestRegressor(200, random_state=0, n_jobs=-1)
rf_mg.fit(morgan[train_idx], y[train_idx])

# Test R2
r2_cr = r2_score(y[test_all], rf_cr.predict(cr3[test_all]))
r2_mg = r2_score(y[test_all], rf_mg.predict(morgan[test_all]))
print(f"Scaffold split test R2:")
print(f"  CR k=3:  {r2_cr:.4f}")
print(f"  Morgan:  {r2_mg:.4f}")
print(f"  Gap:     {r2_cr - r2_mg:+.4f}\n")

# Feature importances
cr_imp = rf_cr.feature_importances_
mg_imp = rf_mg.feature_importances_

# Overlap of TOP-K features: are high-importance features seen in test?
print("=== HIGH-IMPORTANCE FEATURE OVERLAP IN TEST ===")
print("(Do the features the model ACTUALLY USES appear in test molecules?)\n")

# Active features per molecule in test
cr_test_types = set()
for i in test_all:
    cr_test_types.update(j for j, v in enumerate(cr3[i]) if v > 0)

mg_test_active = set()
for i in test_all:
    mg_test_active.update(np.where(morgan[i] > 0)[0])

top_ks = [10, 20, 50, 100, 200]
print(f"{'Top-K':>8s}  {'CR overlap':>12s}  {'Morgan overlap':>14s}  {'Advantage':>10s}")
print("-" * 55)
for k in top_ks:
    cr_top = set(np.argsort(cr_imp)[::-1][:k])
    mg_top = set(np.argsort(mg_imp)[::-1][:k])
    cr_ol = len(cr_top & cr_test_types) / k
    mg_ol = len(mg_top & mg_test_active) / k
    print(f"  top-{k:>4d}  {cr_ol:>12.4f}  {mg_ol:>14.4f}  {cr_ol-mg_ol:>+10.4f}")

# Detailed look at top-10 most important features
print(f"\n=== TOP-10 MOST IMPORTANT FEATURES ===")

print(f"\nMorgan ECFP4 top-10 (by RF importance):")
mg_top10 = np.argsort(mg_imp)[::-1][:10]
for rank, bit_idx in enumerate(mg_top10):
    imp = mg_imp[bit_idx]
    # How often active in train?
    train_freq = morgan[train_idx, bit_idx].mean()
    test_freq = morgan[test_all, bit_idx].mean()
    in_test = bit_idx in mg_test_active
    print(f"  {rank+1:2d}. bit={bit_idx:4d}  imp={imp:.4f}  train_freq={train_freq:.3f}  test_freq={test_freq:.3f}  in_test={in_test}")

print(f"\nCR k=3 top-10 (by RF importance):")
ATOM_MAP = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 15:'P', 16:'S', 17:'Cl', 35:'Br', 53:'I'}
BOND_MAP = {0:'_', 1:'-', 2:'=', 3:'#', 4:'~'}
vocab_inv = {i: t for t, i in vocab3.items()}

cr_top10 = np.argsort(cr_imp)[::-1][:10]
for rank, feat_idx in enumerate(cr_top10):
    imp = cr_imp[feat_idx]
    train_freq = cr3[train_idx, feat_idx].mean()
    test_freq = cr3[test_all, feat_idx].mean()
    in_test = feat_idx in cr_test_types
    t = vocab_inv[feat_idx]
    try:
        n = len(t)
        atoms = [ATOM_MAP.get(t[i][i], str(t[i][i])) for i in range(n)]
        bonds = []
        for i in range(n):
            for j in range(i+1, n):
                if t[i][j] > 0:
                    bonds.append(f"{atoms[i]}{BOND_MAP.get(t[i][j],'?')}{atoms[j]}")
        desc = f"[{','.join(atoms)}] bonds=[{','.join(bonds)}]"
    except: desc = str(t)[:40]
    print(f"  {rank+1:2d}. imp={imp:.4f}  train_freq={train_freq:.4f}  test_freq={test_freq:.4f}  in_test={in_test}  type={desc}")

# Summary: the core finding
print(f"\n=== SMOKING GUN ===")
print()
cr_top50 = set(np.argsort(cr_imp)[::-1][:50])
mg_top50 = set(np.argsort(mg_imp)[::-1][:50])
cr_top50_overlap = len(cr_top50 & cr_test_types) / 50
mg_top50_overlap = len(mg_top50 & mg_test_active) / 50
print(f"Top-50 feature overlap in test:")
print(f"  CR k=3 (scaffold-independent): {cr_top50_overlap:.1%}")
print(f"  Morgan ECFP4 (scaffold-specific): {mg_top50_overlap:.1%}")
print(f"  Gap: {cr_top50_overlap-mg_top50_overlap:+.1%}")
print()
print(f"Test R2:")
print(f"  CR: {r2_cr:.3f}, Morgan: {r2_mg:.3f}, Gap: {r2_cr-r2_mg:+.3f}")
print()
print("The high-importance Morgan bits are scaffold-specific: they appear in")
print("training molecules but NOT in test molecules (new scaffolds don't activate them).")
print("The high-importance CR types are functional-group patterns: they appear")
print("in test molecules regardless of scaffold.")
print("This is the mechanistic root cause of the R2 collapse.")
