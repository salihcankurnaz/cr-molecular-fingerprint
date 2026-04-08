"""
CR Fingerprint: k Değeri Taraması ve Lipophilicity Benchmark

ESOL'da CR k=4 >> Morgan (+0.164 R²)
Sorular:
  1. k=3,4,5 arasında en iyi hangisi?
  2. Lipophilicity'de de aynı fark var mı? (farklı kimyasal özellik)
  3. CR + RDKit desc birleşimi = SOTA'ya ulaşır mı?
"""
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations, permutations
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import urllib.request, io

LIPO_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/Lipophilicity.csv"
ESOL_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

def download_df(url, name):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Python'})
        r = urllib.request.urlopen(req, timeout=10)
        df = pd.read_csv(io.StringIO(r.read().decode()))
        print(f"  {name}: {len(df)} rows, cols={list(df.columns)[:4]}")
        return df
    except Exception as e:
        print(f"  {name}: failed ({e})")
        return None

def mol_to_typed_graph(mol):
    nodes = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
    bond_map = {Chem.rdchem.BondType.SINGLE:1, Chem.rdchem.BondType.DOUBLE:2,
                Chem.rdchem.BondType.TRIPLE:3, Chem.rdchem.BondType.AROMATIC:4}
    edges = {}
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bond_map.get(b.GetBondType(),1)
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
    else:
        at = tuple(sorted(nodes[u] for u in nl))
        et = tuple(sorted((nodes[nl[i]],nodes[nl[j]],edges.get((nl[i],nl[j]),0)) for i in range(n) for j in range(i+1,n) if edges.get((nl[i],nl[j]),0)>0))
        return (at,et)

def cr_fp(mol, k, max_combos=2000):
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    if k > len(nl): return Counter()
    c = Counter()
    combos = list(combinations(nl, k))
    if len(combos) > max_combos:
        np.random.shuffle(combos); combos = combos[:max_combos]
    for sub in combos:
        c[canonical_sub(nodes, edges, list(sub))] += 1
    return c

def fps_to_matrix(fps, vocab):
    X = np.zeros((len(fps), len(vocab)))
    for i, fp in enumerate(fps):
        tot = sum(fp.values())
        for t, cnt in fp.items():
            if t in vocab: X[i, vocab[t]] = cnt / max(tot, 1)
    return X

def eval_descriptor(X, y, name, cv):
    if X.shape[1] == 0: return None
    rf = RandomForestRegressor(100, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
    print(f"  {name:>28s}: R²={scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()

print("=== CR Fingerprint k-Sweep + Lipophilicity Benchmark ===\n")

# Load datasets
print("Loading datasets...")
df_esol = download_df(ESOL_URL, "ESOL")
df_lipo = download_df(LIPO_URL, "Lipophilicity")

datasets = []
if df_esol is not None:
    smiles = df_esol.get('smiles', df_esol.iloc[:,0]).tolist()
    target_col = [c for c in df_esol.columns if 'measured' in c.lower() or 'solubility' in c.lower()]
    if not target_col: target_col = [df_esol.columns[-1]]
    targets = df_esol[target_col[0]].tolist()
    datasets.append(("ESOL (logS)", smiles, targets))

if df_lipo is not None:
    smiles = df_lipo.get('smiles', df_lipo.iloc[:,0]).tolist()
    target_col = [c for c in df_lipo.columns if 'exp' in c.lower() or 'lipo' in c.lower()]
    if not target_col: target_col = [df_lipo.columns[-1]]
    targets = df_lipo[target_col[0]].tolist()
    datasets.append(("Lipophilicity (logD)", smiles, targets))

print()

for ds_name, smiles_list, target_list in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    mols, ys = [], []
    for smi, tgt in zip(smiles_list, target_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            try:
                t = float(tgt)
                if not np.isnan(t): mols.append(mol); ys.append(t)
            except: pass
    y = np.array(ys)
    print(f"  N={len(mols)}")

    # Morgan
    morgan2 = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048)) for m in mols],dtype=float)
    morgan3 = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,3,2048)) for m in mols],dtype=float)

    # CR k=3,4,5
    cr_fps_k = {}
    for k in [3, 4, 5]:
        print(f"  Computing CR k={k}...", flush=True)
        fps = [cr_fp(m, k) for m in mols]
        vocab = {t: i for i, t in enumerate(set(t for fp in fps for t in fp))}
        X = fps_to_matrix(fps, vocab)
        cr_fps_k[k] = (X, vocab, fps)
        print(f"    k={k}: {len(vocab)} types, matrix {X.shape}")

    cv = KFold(5, shuffle=True, random_state=42)
    print(f"\n  5-fold CV Results:")
    results = {}
    results['Morgan ECFP4'] = eval_descriptor(morgan2, y, "Morgan ECFP4 (r=2)", cv)
    results['Morgan ECFP6'] = eval_descriptor(morgan3, y, "Morgan ECFP6 (r=3)", cv)
    for k in [3, 4, 5]:
        X = cr_fps_k[k][0]
        results[f'CR k={k}'] = eval_descriptor(X, y, f"CR k={k}", cv)

    # Best CR + Morgan combined
    best_k = max([3,4,5], key=lambda k: results.get(f'CR k={k}', -1))
    X_best = cr_fps_k[best_k][0]
    combined = np.hstack([X_best, morgan2])
    results['CR_best + Morgan'] = eval_descriptor(combined, y, f"CR k={best_k} + Morgan", cv)

    print(f"\n  RANKING:")
    for name, r2 in sorted(results.items(), key=lambda x: -(x[1] or -1)):
        if r2 is not None:
            print(f"    {name:>25s}: R²={r2:.4f}")

    cr4 = results.get('CR k=4', 0)
    m2 = results.get('Morgan ECFP4', 0)
    print(f"\n  CR k=4 vs Morgan: {cr4-m2:+.4f}")
    if cr4 > m2 + 0.02:
        print(f"  CONFIRMED on {ds_name}: CR >> Morgan")
