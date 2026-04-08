"""
Counting Revolution → Molecular Descriptor

Ana soru: Induced k-subgraph histogramlari, mevcut molekuler parmak
izlerinden (Morgan, MACCS, AssemblyIndex) DAHA IYI bir descriptor mi?

Deney:
  1. Molekulleri atom/bond tipli graflara donustur
  2. Induced k-subgraph histogram hesapla (k=3,4,5)
  3. Bunu Morgan fingerprint ile karsilastir
  4. Metrik: Benzer siniftaki molekuller (antibiotic, antifungal, vb)
     ayni tip mi, yoksa farkli siniflar mi birbirinden uzak mi?

Yeni sorudan dogan hipotez:
  H1: CR fingerprint, aynı ilaç sinifindaki molekulleri Morgan'dan
      daha iyi kumeler
  H2: CR fingerprint + Morgan birlikte en iyi sonucu verir (farkli bilgi)
  H3: CR fingerprint size-normalized IAI olarak molekuler karmasiklik
      olcusu olarak kullanilabilir
"""
import sys
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, permutations
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    HAS_RDKIT = True
except ImportError:
    print("RDKit not found. Using simple SMILES parsing.")
    HAS_RDKIT = False


def mol_to_typed_graph(mol):
    """
    Convert molecule to typed graph:
    - Nodes: atoms with type (atomic number)
    - Edges: bonds with type (1=single, 2=double, 3=triple, 4=aromatic)
    """
    nodes = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        nodes[idx] = atom.GetAtomicNum()  # C=6, N=7, O=8, etc.

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
    """
    Canonical form of induced typed subgraph.
    Returns hashable tuple: sorted atom types + sorted edge types
    """
    n = len(node_list)
    node_list = sorted(node_list)

    # Try all permutations to find canonical form (for n <= 6)
    best = None
    if n <= 7:
        for perm in permutations(range(n)):
            # Build adjacency matrix under this permutation
            mat = []
            for i in range(n):
                row = []
                for j in range(n):
                    u = node_list[perm[i]]
                    v = node_list[perm[j]]
                    if i == j:
                        row.append(nodes[u])  # atom type on diagonal
                    else:
                        row.append(edges.get((u, v), 0))
                mat.append(tuple(row))
            mat_tuple = tuple(mat)
            if best is None or mat_tuple < best:
                best = mat_tuple
    else:
        # For large subgraphs: use sorted atom types + edge multiset (approximate)
        atom_types = tuple(sorted(nodes[u] for u in node_list))
        edge_types = []
        for i in range(len(node_list)):
            for j in range(i+1, len(node_list)):
                et = edges.get((node_list[i], node_list[j]), 0)
                if et > 0:
                    edge_types.append((nodes[node_list[i]], nodes[node_list[j]], et))
        best = (atom_types, tuple(sorted(edge_types)))

    return best


def cr_fingerprint(mol, k_values=[3, 4, 5]):
    """
    Compute CR fingerprint: induced k-subgraph type histogram for each k.

    Returns dict: {k: Counter of canonical subgraph types}
    """
    if mol is None:
        return None

    nodes, edges = mol_to_typed_graph(mol)
    node_list = list(nodes.keys())
    n = len(node_list)

    fingerprint = {}

    for k in k_values:
        if k > n:
            continue

        type_counter = Counter()
        n_combos = 0

        for subset in combinations(node_list, k):
            canon = canonical_typed_subgraph(nodes, edges, list(subset))
            type_counter[canon] += 1
            n_combos += 1
            if n_combos > 5000:  # limit for large molecules
                break

        fingerprint[k] = type_counter

    return fingerprint


def fingerprint_to_vector(fp, k, type_vocab):
    """Convert counter fingerprint to fixed-length vector."""
    if fp is None or k not in fp:
        return np.zeros(len(type_vocab))
    vec = np.zeros(len(type_vocab))
    for t, count in fp[k].items():
        if t in type_vocab:
            vec[type_vocab[t]] = count
    # Normalize by total
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def tanimoto(v1, v2):
    """Tanimoto similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.dot(v1, v1)
    norm2 = np.dot(v2, v2)
    denom = norm1 + norm2 - dot
    return dot / denom if denom > 0 else 0


def intra_inter_ratio(vectors, labels):
    """
    Compute intra-class similarity / inter-class similarity.
    Higher ratio = better class separation.
    """
    intra_sims = []
    inter_sims = []

    unique_labels = list(set(labels))
    label_array = np.array(labels)

    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            sim = tanimoto(vectors[i], vectors[j])
            if labels[i] == labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0

    ratio = intra_mean / (inter_mean + 1e-10)
    return ratio, intra_mean, inter_mean


# Load drugs
print("=== CR Molecular Descriptor vs Morgan Fingerprint ===")
print()

df = pd.read_csv(r'C:\Users\salih\Desktop\gpu-assembly-index\fda_drugs_smiles.csv')
print(f"Loaded {len(df)} molecules")
print()

# Parse molecules
mols = []
names = []
for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    if mol:
        mols.append(mol)
        names.append(row['name'])

print(f"Parsed: {len(mols)} valid molecules")
print()

# Compute CR fingerprints for k=3,4
print("Computing CR fingerprints (k=3,4)...")
k_values = [3, 4]
cr_fps = []
for i, mol in enumerate(mols):
    fp = cr_fingerprint(mol, k_values=k_values)
    cr_fps.append(fp)
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(mols)}", flush=True)

print()

# Build vocabulary of subgraph types
print("Building type vocabularies...")
vocabs = {}
for k in k_values:
    all_types = set()
    for fp in cr_fps:
        if fp and k in fp:
            all_types.update(fp[k].keys())
    vocabs[k] = {t: i for i, t in enumerate(sorted(str(t) for t in all_types))}
    # Remap using string keys
    all_types_list = sorted(all_types, key=str)
    vocabs[k] = {t: i for i, t in enumerate(all_types_list)}
    print(f"  k={k}: {len(vocabs[k])} distinct subgraph types")

# Convert to vectors
cr_vecs_k3 = np.array([fingerprint_to_vector(fp, 3, vocabs[3]) for fp in cr_fps])
cr_vecs_k4 = np.array([fingerprint_to_vector(fp, 4, vocabs[4]) for fp in cr_fps])

# Morgan fingerprints (radius=2, 1024 bits)
print("\nComputing Morgan fingerprints...")
morgan_fps = []
for mol in mols:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    arr = np.array(fp)
    morgan_fps.append(arr)
morgan_vecs = np.array(morgan_fps)

# Molecular weight as baseline
mw_vecs = np.array([[Descriptors.MolWt(mol)] for mol in mols])

print()

# Assign drug classes based on simple heuristics (MW + ring count)
print("Classifying molecules by structure...")
classes = []
for mol in mols:
    n_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
    mw = Descriptors.MolWt(mol)
    n_N = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    n_O = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_S = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)

    # Simple structural classification
    if mw < 200:
        cls = 'small'
    elif n_rings == 0:
        cls = 'acyclic'
    elif n_rings == 1:
        cls = 'monocyclic'
    elif n_rings == 2:
        cls = 'bicyclic'
    elif n_rings >= 3 and n_S > 0:
        cls = 'sulfur_poly'
    elif n_rings >= 3 and n_N >= 2:
        cls = 'nitrogen_poly'
    elif n_rings >= 3:
        cls = 'polycyclic'
    else:
        cls = 'other'
    classes.append(cls)

class_counts = Counter(classes)
print("  Class distribution:")
for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"    {cls:>15s}: {cnt}")

# Keep only classes with >= 5 molecules
valid_classes = {cls for cls, cnt in class_counts.items() if cnt >= 5}
mask = [c in valid_classes for c in classes]
classes_f = [c for c, m in zip(classes, mask) if m]
cr3_f = cr_vecs_k3[mask]
cr4_f = cr_vecs_k4[mask]
morgan_f = morgan_vecs[mask]
print(f"\n  Kept {sum(mask)} molecules with >= 5 in class")

# Compute intra/inter class similarity ratios
print()
print("=== INTRA vs INTER CLASS SIMILARITY ===")
print(f"  (higher ratio = better class separation)")
print()

metrics = [
    ("Morgan (r=2, 1024b)", morgan_f),
    ("CR k=3", cr3_f),
    ("CR k=4", cr4_f),
]

results = {}
for name, vecs in metrics:
    ratio, intra, inter = intra_inter_ratio(vecs, classes_f)
    results[name] = ratio
    print(f"  {name:>25s}: intra={intra:.4f}, inter={inter:.4f}, ratio={ratio:.4f}")

print()
best = max(results, key=results.get)
print(f"  BEST descriptor: {best} (ratio={results[best]:.4f})")

# Combined CR (k=3 + k=4 concatenated)
cr_combined = np.hstack([cr3_f, cr4_f])
if cr_combined.shape[1] > 0:
    ratio_comb, intra_c, inter_c = intra_inter_ratio(cr_combined, classes_f)
    print(f"  {'CR k=3+4 combined':>25s}: intra={intra_c:.4f}, inter={inter_c:.4f}, ratio={ratio_comb:.4f}")
    results['CR combined'] = ratio_comb

print()
print("=== RANKING ===")
for name, ratio in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:>25s}: {ratio:.4f}")

print()
print("=== IAI COMPUTATION FOR MOLECULES ===")
print("(IAI = normalized diversity of induced subgraph types)")
for name, mol in zip(names[:10], mols[:10]):
    fp = cr_fps[names.index(name)] if name in names else None
    if fp and 3 in fp:
        n_distinct_k3 = len(fp[3])
        n_atoms = mol.GetNumAtoms()
        n_combos_k3 = max(1, n_atoms * (n_atoms-1) * (n_atoms-2) // 6)
        iai_k3 = n_distinct_k3 / max(1, n_combos_k3) * 1000
        print(f"  {name:>20s}: n_atoms={n_atoms:3d}, distinct_k3={n_distinct_k3:3d}, IAI_k3={iai_k3:.2f}")
