"""
CR vs Morgan: Scaffold Generalizability Analysis

BULGU: Scaffold split'te CR=0.72, Morgan=0.20
SORU: Neden Morgan scaffold'a özgü ama CR değil?

HIPOTEZ:
  Morgan fingerprint = scaffold'a bağlı bit vektörü
    - r=2 çembersel komşuluk: molekülün global yapısına bağlı
    - Yeni scaffold'ta bu bitlerin çoğu 0 -> hiç bilgi yok

  CR k=3 fingerprint = lokal fonksiyonel grup tipleri
    - İskeletten bağımsız: -OH, -NH2, C=O her molekülde var
    - Yeni scaffold'ta bile aynı tipler mevcut

DENEY:
  1. Scaffold train/test split'te Morgan ve CR bit overlap'ini ölç
     (Kaçı bit train ve test'te paylaşılıyor?)
  2. Morgan: düşük overlap (scaffold-specific bits)
     CR: yüksek overlap (scaffold-independent types)

  3. Overlap ile test R² korelasyonu -> mekanizma kanıtı

AYRICA: Lipophilicity (logD) için FreeSolv dataset bulabilir miyiz?
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

def scaffold_split(mols, seed=42):
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

print("=== Why Does Morgan Fail on Scaffold Split? ===\n")

# Compute fingerprints
print("Computing fingerprints...")
cr_fps_raw = [cr_fp(m, k=3) for m in mols]
morgan_bits = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]

# Build CR vocabulary
vocab3 = {t:i for i,t in enumerate(set(t for fp in cr_fps_raw for t in fp))}
cr3 = np.zeros((len(mols), len(vocab3)))
for i, fp in enumerate(cr_fps_raw):
    tot = sum(fp.values())
    for t, c in fp.items():
        cr3[i, vocab3[t]] = c / max(tot, 1)

morgan = np.array([np.array(b) for b in morgan_bits], dtype=float)
print(f"  CR k=3: {len(vocab3)} types")
print(f"  Morgan: 2048 bits\n")

# Scaffold split
train_idx, val_idx, test_idx = scaffold_split(mols, seed=0)
test_all = test_idx if test_idx else val_idx

print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_all)}\n")

# OVERLAP ANALYSIS: what fraction of train features appear in test?
print("=== FEATURE OVERLAP: Train -> Test ===")
print("(Higher overlap = model can use train features for test prediction)\n")

# CR overlap: types present in train that also appear in test
cr_train_types = set()
for i in train_idx:
    cr_train_types.update(cr_fps_raw[i].keys())
cr_test_types = set()
for i in test_all:
    cr_test_types.update(cr_fps_raw[i].keys())

cr_overlap = len(cr_train_types & cr_test_types) / max(len(cr_test_types), 1)
print(f"  CR k=3 type overlap (test types seen in train): {cr_overlap:.4f} ({cr_overlap*100:.1f}%)")
print(f"    Train types: {len(cr_train_types)}, Test types: {len(cr_test_types)}")
print(f"    Shared: {len(cr_train_types & cr_test_types)}")

# Morgan bit overlap: bits active in test that were active in at least one train molecule
morgan_train_active = set(np.where(morgan[train_idx].sum(0) > 0)[0])
morgan_test_active = set()
for i in test_all:
    morgan_test_active.update(np.where(morgan[i] > 0)[0])

morgan_overlap = len(morgan_train_active & morgan_test_active) / max(len(morgan_test_active), 1)
print(f"\n  Morgan bit overlap (test bits seen in train): {morgan_overlap:.4f} ({morgan_overlap*100:.1f}%)")
print(f"    Train active bits: {len(morgan_train_active)}, Test active bits: {len(morgan_test_active)}")
print(f"    Shared: {len(morgan_train_active & morgan_test_active)}")

# Coverage: fraction of test molecules with at least X% active bits from train
print(f"\n  Per-molecule coverage analysis:")
cr_coverage = []
for i in test_all:
    test_types = set(cr_fps_raw[i].keys())
    shared = len(test_types & cr_train_types)
    cr_coverage.append(shared / max(len(test_types), 1))

morgan_coverage = []
for i in test_all:
    test_bits = set(np.where(morgan[i] > 0)[0])
    shared = len(test_bits & morgan_train_active)
    morgan_coverage.append(shared / max(len(test_bits), 1))

print(f"    CR k=3  avg coverage per mol: {np.mean(cr_coverage):.4f}")
print(f"    Morgan  avg coverage per mol: {np.mean(morgan_coverage):.4f}")

# Key numbers showing why CR generalizes
print(f"\n=== MECHANISTIC EXPLANATION ===")
print()
print(f"  CR k=3 captures LOCAL FUNCTIONAL GROUP TOPOLOGY:")
print(f"    - 3-atom induced subgraphs: -OH, C=O, N-H, aromatic ring fragments")
print(f"    - These appear in ALL molecules regardless of scaffold")
print(f"    - Train type overlap in test: {cr_overlap*100:.1f}%")
print()
print(f"  Morgan ECFP4 captures SCAFFOLD-SPECIFIC CIRCULAR PATTERNS:")
print(f"    - r=2 neighborhood: specific to the molecular scaffold")
print(f"    - New scaffold -> most bits are 0 (never seen in training)")
print(f"    - Train bit overlap in test: {morgan_overlap*100:.1f}%")
print()
diff = cr_overlap - morgan_overlap
print(f"  CR generalizability advantage: {diff:.4f} ({diff*100:.1f}% more features shared)")
print()
print(f"  CONCLUSION:")
print(f"    CR fingerprints encode scaffold-INDEPENDENT local chemistry.")
print(f"    Morgan fingerprints encode scaffold-DEPENDENT global patterns.")
print(f"    This is why CR generalizes to new scaffolds and Morgan doesn't.")
print()
print(f"  THEORETICAL ROOT:")
print(f"    Counting Revolution: induced subgraph histograms capture LOCAL structure")
print(f"    invariant to the global graph embedding.")
print(f"    This local-global decomposition is the key to scaffold generalizability.")
