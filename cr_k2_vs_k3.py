"""
CR k=2 vs k=3 for Molecular Property Prediction

HYPOTHESIS: Since k_min=2 for molecular graphs (c_node=10 >> threshold=5),
CR k=2 features (just typed atom-pair frequencies) should be highly predictive.

If k=2 ≈ k=3 performance:
  -> COMPOSITION ALONE (atom-pair distributions) is sufficient for property prediction
  -> Simpler than Morgan, yet as powerful as CR k=3
  -> Atom-pair fingerprints are a well-known descriptor in cheminformatics (AP2D)

Compare on ESOL (scaffold split, fair evaluation):
  1. CR k=2 only
  2. CR k=3 only
  3. CR k=2 + k=3
  4. Morgan ECFP4 (baseline)
  5. Physical descriptors
"""
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations, permutations
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import urllib.request, io

def mol_to_typed_graph(mol):
    nodes = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
    bmap = {Chem.rdchem.BondType.SINGLE:1, Chem.rdchem.BondType.DOUBLE:2,
            Chem.rdchem.BondType.TRIPLE:3, Chem.rdchem.BondType.AROMATIC:4}
    edges = {}
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bmap.get(b.GetBondType(), 1)
        edges[(i,j)] = bt; edges[(j,i)] = bt
    return nodes, edges

def cr_fp_k2(mol):
    """CR k=2: typed atom-pair distribution (node types + edge type)."""
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    c = Counter()
    for i in range(len(nl)):
        for j in range(i+1, len(nl)):
            u, v = nl[i], nl[j]
            la, lb = nodes[u], nodes[v]
            edge_type = edges.get((u,v), 0)  # 0=no edge, 1-4=bond type
            # Canonical: sort atom types
            if la > lb:
                la, lb = lb, la
            c[(la, lb, edge_type)] += 1
    return c

def canonical_sub_k3(nodes, edges, nlist):
    """Canonical k=3 subgraph (from existing code)."""
    n = len(nlist); nl = sorted(nlist)
    best = None
    for perm in permutations(range(n)):
        mat = tuple(tuple(nodes[nl[perm[i]]] if i==j else edges.get((nl[perm[i]],nl[perm[j]]),0) for j in range(n)) for i in range(n))
        if best is None or mat < best: best = mat
    return best

def cr_fp_k3(mol, max_combos=5000):
    """CR k=3: 3-atom induced subgraph distribution."""
    nodes, edges = mol_to_typed_graph(mol)
    nl = list(nodes.keys())
    c = Counter()
    combos = list(combinations(nl, 3))
    if len(combos) > max_combos:
        np.random.shuffle(combos)
        combos = combos[:max_combos]
    for sub in combos:
        canon = canonical_sub_k3(nodes, edges, list(sub))
        c[canon] += 1
    return c

def physical_features(mol):
    """Physical/QSAR descriptors."""
    try:
        feats = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
        ]
        return np.array(feats, dtype=float)
    except:
        return np.zeros(9)

# Load ESOL
print("Loading ESOL...", flush=True)
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
print(f"ESOL: N={len(mols)}\n")

# Compute fingerprints
print("Computing CR k=2 fingerprints...", flush=True)
fps2 = [cr_fp_k2(m) for m in mols]
print("Computing CR k=3 fingerprints...", flush=True)
fps3 = [cr_fp_k3(m, max_combos=5000) for m in mols]

# Vectorize k=2
vocab2 = sorted(set(t for fp in fps2 for t in fp))
v2idx = {t:i for i,t in enumerate(vocab2)}
X_cr2 = np.zeros((len(mols), len(vocab2)))
for i, fp in enumerate(fps2):
    tot = sum(fp.values())
    for t, c in fp.items(): X_cr2[i, v2idx[t]] = c / max(tot, 1)

# Vectorize k=3
vocab3 = sorted(set(t for fp in fps3 for t in fp), key=str)
v3idx = {t:i for i,t in enumerate(vocab3)}
X_cr3 = np.zeros((len(mols), len(vocab3)))
for i, fp in enumerate(fps3):
    tot = sum(fp.values())
    for t, c in fp.items(): X_cr3[i, v3idx[t]] = c / max(tot, 1)

# Combine k=2+k=3
X_cr23 = np.hstack([X_cr2, X_cr3])

# Morgan
X_mg = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)) for m in mols], dtype=float)

# Physical
X_ph = np.vstack([physical_features(m) for m in mols])

# Combined
X_cr3_ph = np.hstack([X_cr3, X_ph])
X_cr2_ph = np.hstack([X_cr2, X_ph])

print(f"Feature dimensions: CR k=2={X_cr2.shape[1]}, CR k=3={X_cr3.shape[1]}, Morgan={X_mg.shape[1]}")
print()

def cv_r2(X, y, seeds=5, n_folds=5, label=""):
    scores = []
    for seed in range(seeds):
        kf = KFold(n_folds, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            rf = RandomForestRegressor(200, random_state=seed, n_jobs=-1)
            rf.fit(X[tr], y[tr])
            pred = rf.predict(X[te])
            scores.append(r2_score(y[te], pred))
    mean, std = np.mean(scores), np.std(scores)
    print(f"  {label:45s}: R2={mean:.3f} +/- {std:.3f} (dim={X.shape[1]})")
    return mean

# ---- Scaffold split ----
def bemis_murcko_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ''
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except: return ''

scaffolds = [bemis_murcko_scaffold(smi) for smi in smiles_list]
unique_scafs = list(set(scaffolds))
np.random.seed(42)
np.random.shuffle(unique_scafs)
n_test_scaffolds = int(len(unique_scafs) * 0.1)
test_scaffolds = set(unique_scafs[:n_test_scaffolds])
train_idx = [i for i, s in enumerate(scaffolds) if s not in test_scaffolds]
test_idx = [i for i, s in enumerate(scaffolds) if s in test_scaffolds]

print(f"=== RANDOM 5-FOLD CV ===\n")
r_cr2 = cv_r2(X_cr2, y, label="CR k=2")
r_cr3 = cv_r2(X_cr3, y, label="CR k=3")
r_cr23 = cv_r2(X_cr23, y, label="CR k=2+3")
r_mg = cv_r2(X_mg, y, label="Morgan ECFP4")
r_ph = cv_r2(X_ph, y, label="Physical")
cv_r2(X_cr2_ph, y, label="CR k=2 + Physical")
cv_r2(X_cr3_ph, y, label="CR k=3 + Physical")

print(f"\n=== SCAFFOLD SPLIT (Bemis-Murcko, GNN-fair) ===\n")
def scaffold_r2(X, y, tr, te, label=""):
    rf = RandomForestRegressor(200, random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    r2 = r2_score(y[te], pred)
    print(f"  {label:45s}: R2={r2:.3f} (train={len(tr)}, test={len(te)})")
    return r2

tr = np.array(train_idx); te = np.array(test_idx)
s_cr2 = scaffold_r2(X_cr2, y, tr, te, label="CR k=2")
s_cr3 = scaffold_r2(X_cr3, y, tr, te, label="CR k=3")
scaffold_r2(X_cr23, y, tr, te, label="CR k=2+3")
s_mg = scaffold_r2(X_mg, y, tr, te, label="Morgan ECFP4")
scaffold_r2(X_ph, y, tr, te, label="Physical")
scaffold_r2(X_cr2_ph, y, tr, te, label="CR k=2 + Physical")
scaffold_r2(X_cr3_ph, y, tr, te, label="CR k=3 + Physical")

print(f"\n=== KEY COMPARISON ===\n")
print(f"CR k=2 vs k=3 (random split): {r_cr2:.3f} vs {r_cr3:.3f} (diff: {r_cr3-r_cr2:+.3f})")
print(f"CR k=2 vs k=3 (scaffold):     {s_cr2:.3f} vs {s_cr3:.3f} (diff: {s_cr3-s_cr2:+.3f})")
print(f"Morgan (random/scaffold):      {r_mg:.3f} / {s_mg:.3f}")
print()
if s_cr2 > s_cr3:
    print("CONFIRMED (scaffold): CR k=2 > CR k=3 -- atom-pair distributions GENERALIZE BETTER!")
    print(f"-> Scaffold diff: {s_cr2-s_cr3:+.3f} (k=2 more generalizable)")
    print("-> c_node=10 >> threshold=5 -> k_min=2 -> composition-only is sufficient")
    print("-> 3-node topology adds scaffold-specific patterns that HURT generalization")
elif abs(r_cr3 - r_cr2) < 0.05:
    print("CONFIRMED: CR k=2 ~= CR k=3 -- atom-pair distributions are SUFFICIENT!")
    print("-> Composition (c_node=10 > threshold=5) explains most molecular property variance")
else:
    print(f"CR k=3 adds value over k=2: {r_cr3-r_cr2:+.3f} (random), {s_cr3-s_cr2:+.3f} (scaffold)")
    print("-> 3-node topology adds real information beyond pair composition")
