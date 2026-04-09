"""
CR vs Morgan: Classification Tasks (Tox21 subset)

Published GNN AUC on Tox21 (MoleculeNet):
  AttentiveFP: AUC=0.819
  D-MPNN:      AUC=0.851
  GraphConv:   AUC=0.772
  ECFP4+RF:    AUC~0.761
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

print("=== CR vs Morgan: Classification Tasks ===\n")

# Load Tox21 classification data from ChemProp
url = "https://raw.githubusercontent.com/chemprop/chemprop/master/tests/data/classification.csv"
req = urllib.request.Request(url, headers={'User-Agent':'Python'})
r = urllib.request.urlopen(req, timeout=10)
df = pd.read_csv(io.StringIO(r.read().decode()))
print(f"Dataset: {df.shape}, cols={list(df.columns)[:6]}...")
print()

# Process - use first few tasks with enough data
task_cols = [c for c in df.columns if c != 'smiles']
smi_col = 'smiles'

# Parse valid molecules for each task
print("Published GNN AUC on Tox21 (MoleculeNet, 12 tasks avg):")
print("  D-MPNN:      AUC=0.851")
print("  AttentiveFP: AUC=0.819")
print("  GraphConv:   AUC=0.772")
print("  ECFP4+RF:    AUC~0.761")
print()

# Use all valid data across tasks
mols_all = []
smiles_all = []
for smi in df[smi_col]:
    mol = Chem.MolFromSmiles(str(smi))
    mols_all.append(mol)
    smiles_all.append(smi)

valid_mask = [mol is not None for mol in mols_all]
mols = [m for m, v in zip(mols_all, valid_mask) if v]
df_valid = df[valid_mask].reset_index(drop=True)
print(f"Valid molecules: {len(mols)} / {len(df)}\n")

# Compute features once
print("Computing CR k=3...", flush=True)
cr_fps = [cr_fp(m, k=3) for m in mols]
cr3, vocab3 = build_matrix(cr_fps)
morgan = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048)) for m in mols],dtype=float)
phys = StandardScaler().fit_transform(get_phys(mols))
print(f"  CR types: {len(vocab3)}, Morgan: 2048\n")

configs = [
    ("Morgan ECFP4", morgan),
    ("CR k=3", cr3),
    ("CR+Morgan", np.hstack([cr3, morgan])),
    ("Physical (10)", phys),
    ("CR+Physical", np.hstack([cr3, phys])),
]

# Per-task AUC with 5-fold CV
print("=== PER-TASK AUC (5-fold CV, averaged over 3 seeds) ===\n")
task_results = {name: [] for name, _ in configs}
task_names_used = []

for task in task_cols:
    y_raw = df_valid[task].values
    # Keep only labeled (non-NaN)
    valid = ~pd.isna(y_raw)
    if valid.sum() < 50: continue  # skip sparse tasks
    y_task = y_raw[valid].astype(int)
    if len(np.unique(y_task)) < 2: continue  # skip single-class

    pos_rate = y_task.mean()
    task_names_used.append(task)

    for name, X in configs:
        X_task = X[valid]
        aucs = []
        for seed in range(3):
            try:
                cv = StratifiedKFold(5, shuffle=True, random_state=seed)
                rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1, class_weight='balanced')
                fold_aucs = []
                for tr_idx, te_idx in cv.split(X_task, y_task):
                    rf.fit(X_task[tr_idx], y_task[tr_idx])
                    proba = rf.predict_proba(X_task[te_idx])[:, 1]
                    fold_aucs.append(roc_auc_score(y_task[te_idx], proba))
                aucs.append(np.mean(fold_aucs))
            except Exception as e:
                aucs.append(0.5)
        task_results[name].append(np.mean(aucs))

print(f"Tasks evaluated: {task_names_used}")
print()

# Summary across tasks
print(f"{'Method':>20s}  Mean AUC  vs Morgan")
print("-" * 45)
morgan_mean = np.mean(task_results['Morgan ECFP4'])
for name, aucs in sorted(task_results.items(), key=lambda x: -np.mean(x[1])):
    mean = np.mean(aucs)
    diff = mean - morgan_mean
    print(f"  {name:>18s}: {mean:.4f}  ({diff:+.4f})")

print()
print("GNN Baselines (Tox21, full dataset, 12 tasks):")
print("  D-MPNN:      0.851")
print("  AttentiveFP: 0.819")
print("  GraphConv:   0.772")
print("  ECFP4+RF:    0.761")
print()
print(f"Our best CR: {max(np.mean(aucs) for aucs in task_results.values()):.4f}")
print(f"Our Morgan:  {morgan_mean:.4f}")
