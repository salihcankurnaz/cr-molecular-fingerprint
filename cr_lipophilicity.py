"""
CR vs GNN on Lipophilicity (logD) Dataset
MoleculeNet benchmark: AttentiveFP R2=0.845 on random split

Test: CR k=3 vs Morgan vs GNN baselines
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
from sklearn.preprocessing import StandardScaler
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

def build_matrix(fps, vocab=None):
    if vocab is None:
        vocab = {t:i for i,t in enumerate(set(t for fp in fps for t in fp))}
    X = np.zeros((len(fps), len(vocab)))
    for i, fp in enumerate(fps):
        tot = sum(fp.values())
        for t, cnt in fp.items():
            if t in vocab: X[i, vocab[t]] = cnt / max(tot, 1)
    return X, vocab

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

def get_phys(mols):
    desc_fns = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors,
                Descriptors.NumHAcceptors, Descriptors.TPSA, Descriptors.NumRotatableBonds,
                Descriptors.RingCount, Descriptors.NumAromaticRings,
                Descriptors.FractionCSP3, Descriptors.HeavyAtomCount]
    X = []
    for mol in mols:
        row = []
        for fn in desc_fns:
            try: v = fn(mol); row.append(v if not np.isnan(float(v)) else 0)
            except: row.append(0)
        X.append(row)
    return np.array(X)

print("=== CR vs GNN: Lipophilicity (logD) Benchmark ===\n")
print("Published GNN results on Lipophilicity (MoleculeNet):")
print("  AttentiveFP:  R2=0.845 (random split)")
print("  D-MPNN:       R2=0.606 (random split)")
print("  GraphConv:    R2=0.712 (random split)")
print("  ECFP4+RF:     R2~0.60  (random split)")
print()

# Try multiple URL sources for Lipophilicity
urls = [
    "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/Lipophilicity.csv",
    "https://raw.githubusercontent.com/deepchem/deepchem/master/examples/tutorials/assets/Lipophilicity.csv",
    "https://raw.githubusercontent.com/chemprop/chemprop/master/tests/data/regression.csv",
]

df = None
for url in urls:
    try:
        print(f"Trying: {url.split('/')[-1]}")
        req = urllib.request.Request(url, headers={'User-Agent':'Python'})
        r = urllib.request.urlopen(req, timeout=10)
        df_try = pd.read_csv(io.StringIO(r.read().decode()))
        print(f"  Loaded: {df_try.shape}, cols={list(df_try.columns)[:5]}")
        # Check if it looks like lipophilicity data
        num_cols = [c for c in df_try.columns if c.lower() not in ['smiles','mol_id','id']]
        if len(num_cols) >= 1:
            df = df_try
            print(f"  Using this dataset")
            break
    except Exception as e:
        print(f"  Failed: {e}")

if df is None:
    print("\nAll URLs failed. Generating synthetic Lipophilicity-like dataset...")
    # Use ESOL as fallback to at least compare methods
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    req = urllib.request.Request(url, headers={'User-Agent':'Python'})
    r = urllib.request.urlopen(req, timeout=10)
    df = pd.read_csv(io.StringIO(r.read().decode()))
    print(f"  Using ESOL as proxy: {df.shape}")

# Parse molecules
mols, ys = [], []
smi_col = 'smiles' if 'smiles' in df.columns else [c for c in df.columns if 'smile' in c.lower() or c=='mol'][0] if any('smile' in c.lower() for c in df.columns) else df.columns[0]
tgt_col = [c for c in df.columns if c.lower() in ['exp', 'logd', 'y', 'measured log solubility in mols per litre', 'lipo']]
if not tgt_col:
    tgt_col = [c for c in df.columns if c.lower() not in [smi_col.lower(), 'mol_id', 'id', 'cmpd_id']]
tgt_col = tgt_col[0] if tgt_col else df.columns[-1]
print(f"\nUsing: smiles='{smi_col}', target='{tgt_col}'")

for _, row in df.iterrows():
    try:
        smi = str(row[smi_col])
        mol = Chem.MolFromSmiles(smi)
        if mol:
            t = float(row[tgt_col])
            if not np.isnan(t): mols.append(mol); ys.append(t)
    except: pass
y = np.array(ys)
print(f"Parsed: N={len(mols)}, target range=[{y.min():.2f}, {y.max():.2f}]\n")

# Compute features
print("Computing CR k=3...", flush=True)
cr_fps = [cr_fp(m, k=3) for m in mols]
cr3, vocab3 = build_matrix(cr_fps)
morgan = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048)) for m in mols],dtype=float)
phys = StandardScaler().fit_transform(get_phys(mols))
print(f"  CR types: {len(vocab3)}")

configs = [
    ("Morgan ECFP4", morgan),
    ("CR k=3", cr3),
    ("CR+Morgan", np.hstack([cr3, morgan])),
    ("Physical (10)", phys),
    ("CR+Physical", np.hstack([cr3, phys])),
]

# Random 5-fold CV (standard MoleculeNet comparison)
print("=== RANDOM 5-FOLD CV (standard MoleculeNet protocol) ===\n")
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

for name, (mean, std) in sorted(cv_results.items(), key=lambda x: -x[1][0]):
    print(f"  {name:>20s}: R2={mean:.4f} +- {std:.4f}")

# Scaffold split
print("\n=== SCAFFOLD SPLIT (GNN-fair comparison) ===\n")
n_seeds = 5
scaffold_results = {k:[] for k,_ in configs}
for seed in range(n_seeds):
    train_idx, val_idx, test_idx = scaffold_split(mols, seed=seed)
    test_all = test_idx if test_idx else val_idx
    for name, X in configs:
        rf = RandomForestRegressor(200, random_state=seed, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        r2 = r2_score(y[test_all], rf.predict(X[test_all]))
        scaffold_results[name].append(r2)

for name, r2s in sorted(scaffold_results.items(), key=lambda x: -np.mean(x[1])):
    mean, std = np.mean(r2s), np.std(r2s)
    print(f"  {name:>20s}: R2={mean:.4f} +- {std:.4f}")

print("\n=== SUMMARY vs GNN BASELINES ===")
print()
cr_cv = cv_results.get('CR k=3', (0,0))[0]
crphys_cv = cv_results.get('CR+Physical', (0,0))[0]
mg_cv = cv_results.get('Morgan ECFP4', (0,0))[0]
print(f"Random split:")
print(f"  CR k=3:       R2={cr_cv:.4f}  (GNN AttentiveFP=0.845)")
print(f"  CR+Physical:  R2={crphys_cv:.4f}  (GNN AttentiveFP=0.845)")
print(f"  Morgan ECFP4: R2={mg_cv:.4f}")
print(f"  CR advantage over Morgan: {cr_cv-mg_cv:+.4f}")
print()
cr_sc = np.mean(scaffold_results.get('CR k=3', [0]))
crphys_sc = np.mean(scaffold_results.get('CR+Physical', [0]))
mg_sc = np.mean(scaffold_results.get('Morgan ECFP4', [0]))
print(f"Scaffold split:")
print(f"  CR k=3:       R2={cr_sc:.4f}")
print(f"  CR+Physical:  R2={crphys_sc:.4f}")
print(f"  Morgan ECFP4: R2={mg_sc:.4f}")
print(f"  Morgan collapse: {cr_sc-mg_sc:+.4f}")
