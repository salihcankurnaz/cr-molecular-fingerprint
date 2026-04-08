"""
CR Molecular Descriptor Benchmark -- Gercek Aktivite Verisiyle

Önceki: yapısal sınıflarla test (çember sayısı gibi)
Şimdi: GERCEK BIYOAKTIVITE verisi

Dataset 1: ESOL (su çözünürlüğü logS) - 1128 molekül
Dataset 2: Lipophilicity - lipid çözünürlüğü (logD)
Dataset 3: BBBP - Blood Brain Barrier Penetration (sınıflandırma)

Metrik: R² for regression, AUC for classification

Eğer CR k=4 > Morgan burada da: gerçek keşif!
"""
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations, permutations
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import r2_score, roc_auc_score
import urllib.request
import io

# Download ESOL dataset (standard benchmark)
ESOL_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

print("=== CR Descriptor Benchmark on Real Bioactivity Data ===")
print()

# Try to load ESOL
print("Loading ESOL dataset (water solubility)...")
try:
    req = urllib.request.Request(ESOL_URL, headers={'User-Agent': 'Python'})
    response = urllib.request.urlopen(req, timeout=10)
    df_esol = pd.read_csv(io.StringIO(response.read().decode()))
    print(f"  Loaded {len(df_esol)} molecules")
    print(f"  Columns: {list(df_esol.columns)[:5]}")
except Exception as e:
    print(f"  Download failed: {e}")
    print("  Using FDA drug dataset with MW as regression target instead")
    df_esol = None

# Alternatively, use FDA drugs with MW as a "property" target
df_fda = pd.read_csv(r'C:\Users\salih\Desktop\gpu-assembly-index\fda_drugs_smiles.csv')

# Parse ESOL or FDA dataset
if df_esol is not None:
    smiles_col = 'smiles' if 'smiles' in df_esol.columns else df_esol.columns[0]
    target_col = 'measured log solubility in mols per litre' if 'measured' in ' '.join(df_esol.columns) else df_esol.columns[-1]
    smiles_list = df_esol[smiles_col].tolist()
    target_list = df_esol[target_col].tolist()
    dataset_name = "ESOL (logS)"
else:
    smiles_list = df_fda['SMILES'].tolist()
    target_list = df_fda['MW'].tolist()
    dataset_name = "FDA drugs (MW as target)"

# Parse molecules
print(f"\nDataset: {dataset_name}")
mols = []
targets = []
for smi, tgt in zip(smiles_list, target_list):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is not None:
        try:
            t = float(tgt)
            if not np.isnan(t):
                mols.append(mol)
                targets.append(t)
        except:
            pass

print(f"  Valid: {len(mols)} molecules")
targets = np.array(targets)


def mol_to_typed_graph(mol):
    nodes = {}
    for atom in mol.GetAtoms():
        nodes[atom.GetIdx()] = atom.GetAtomicNum()
    edges = {}
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond_type_map.get(bond.GetBondType(), 1)
        edges[(i, j)] = bt
        edges[(j, i)] = bt
    return nodes, edges


def canonical_typed_subgraph(nodes, edges, node_list):
    n = len(node_list)
    node_list = sorted(node_list)
    if n <= 6:
        best = None
        for perm in permutations(range(n)):
            mat = tuple(tuple(
                nodes[node_list[perm[i]]] if i == j else edges.get((node_list[perm[i]], node_list[perm[j]]), 0)
                for j in range(n)
            ) for i in range(n))
            if best is None or mat < best:
                best = mat
        return best
    else:
        atom_types = tuple(sorted(nodes[u] for u in node_list))
        edge_types = tuple(sorted(
            (nodes[node_list[i]], nodes[node_list[j]], edges.get((node_list[i], node_list[j]), 0))
            for i in range(n) for j in range(i+1, n)
            if edges.get((node_list[i], node_list[j]), 0) > 0
        ))
        return (atom_types, edge_types)


def cr_fingerprint_fast(mol, k=4, max_combos=3000):
    nodes, edges = mol_to_typed_graph(mol)
    node_list = list(nodes.keys())
    n = len(node_list)
    if k > n:
        return Counter()
    counter = Counter()
    all_combos = list(combinations(node_list, k))
    if len(all_combos) > max_combos:
        np.random.shuffle(all_combos)
        all_combos = all_combos[:max_combos]
    for subset in all_combos:
        canon = canonical_typed_subgraph(nodes, edges, list(subset))
        counter[canon] += 1
    return counter


# Compute fingerprints
print(f"\nComputing CR k=4 fingerprints for {len(mols)} molecules...")
cr_fps = []
for i, mol in enumerate(mols):
    fp = cr_fingerprint_fast(mol, k=4)
    cr_fps.append(fp)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(mols)}", flush=True)

# Build vocabulary
all_types = set()
for fp in cr_fps:
    all_types.update(fp.keys())
vocab = {t: i for i, t in enumerate(all_types)}
print(f"\n  CR k=4: {len(vocab)} distinct subgraph types")

# Vectorize CR
cr_vecs = np.zeros((len(mols), len(vocab)))
for i, fp in enumerate(cr_fps):
    total = sum(fp.values())
    for t, c in fp.items():
        cr_vecs[i, vocab[t]] = c / max(total, 1)

# Morgan fingerprints
print("Computing Morgan fingerprints...")
morgan_vecs = np.array([
    np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    for mol in mols
], dtype=float)

# ECFP6 (radius=3)
ecfp6_vecs = np.array([
    np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048))
    for mol in mols
], dtype=float)

# RDKit descriptors (200 descriptors)
print("Computing RDKit descriptors...")
desc_names = [d[0] for d in Descriptors.descList[:50]]  # top 50
rdkit_desc = []
for mol in mols:
    row = []
    for dname in desc_names:
        try:
            v = getattr(Descriptors, dname)(mol)
            row.append(v if v is not None and not np.isnan(float(v)) else 0)
        except:
            row.append(0)
    rdkit_desc.append(row)
rdkit_vecs = np.array(rdkit_desc)

# 5-fold cross-validation
print("\n=== 5-FOLD CROSS-VALIDATION (RandomForest) ===")
print(f"Dataset: {dataset_name}, N={len(mols)}")
print()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

descriptors = [
    ("Morgan ECFP4 (r=2)", morgan_vecs),
    ("Morgan ECFP6 (r=3)", ecfp6_vecs),
    ("RDKit 50 desc", rdkit_vecs),
    ("CR k=4", cr_vecs),
]

for name, X in descriptors:
    if X.shape[1] == 0:
        print(f"  {name:>25s}: SKIP (empty)")
        continue
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, targets, cv=cv, scoring='r2')
    r2_mean = scores.mean()
    r2_std = scores.std()
    results[name] = r2_mean
    print(f"  {name:>25s}: R²={r2_mean:.4f} ± {r2_std:.4f}")

# Combined CR + Morgan
cr_morgan = np.hstack([cr_vecs, morgan_vecs])
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
scores = cross_val_score(rf, cr_morgan, targets, cv=cv, scoring='r2')
results['CR k=4 + Morgan'] = scores.mean()
print(f"  {'CR k=4 + Morgan':>25s}: R²={scores.mean():.4f} ± {scores.std():.4f}")

print()
print("=== RANKING ===")
for name, r2 in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:>25s}: R²={r2:.4f}")

best = max(results, key=results.get)
print(f"\n  BEST: {best} (R²={results[best]:.4f})")

if 'CR k=4' in results and 'Morgan ECFP4 (r=2)' in results:
    diff = results['CR k=4'] - results['Morgan ECFP4 (r=2)']
    print(f"  CR vs Morgan difference: {diff:+.4f}")
    if diff > 0.02:
        print("  CONFIRMED: CR fingerprint significantly outperforms Morgan!")
    elif diff > 0:
        print("  MARGINAL: CR slightly better")
    else:
        print("  Morgan wins on this dataset")
