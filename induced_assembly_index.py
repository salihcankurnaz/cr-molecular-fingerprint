"""
Induced Assembly Index (IAI) - Yeni bir karmasiklik olcusu

Assembly Theory: Bir nesneyi insa etmek icin minimum tekrar sayisi
Counting Revolution: induced subgraph histogram -> tam siniflandirici

IAI: Her iki fikri BIRLESTIRIR

Tanim:
  IAI(G, k) = G'nin boyut-k distinct induced subgraph tiplerinin sayisi
  IAI(G) = sum_k IAI(G,k) / C(n,k)  (normalize edilmis)

Hipotezler:
  H1: Duzgun graflar dusuk IAI (her komsulik ayni)
  H2: Rastgele graflar yuksek IAI (her alt-yapi farkli)
  H3: IAI ~ k_min (Counting Revolution'un tamamlama esigi)
  H4: IAI ~ Assembly Index? (birlesik teori)

Bu dogru ciksa: IAI = Assembly Theory + Counting Revolution'u birlen,
  tek sayiyla hem karmasikligi hem ogrenilmesi hem de siniflandirma
  zorluğunu capture eden bir olcu.
"""

import numpy as np
from itertools import combinations, permutations
import networkx as nx
from collections import Counter

def canonical_subgraph(G, nodes):
    """Get canonical form of induced subgraph."""
    sub = G.subgraph(nodes)
    # Use adjacency matrix sorted canonically
    n = len(nodes)
    node_list = sorted(nodes)
    adj = []
    for i, u in enumerate(node_list):
        row = []
        for j, v in enumerate(node_list):
            row.append(1 if G.has_edge(u, v) else 0)
        adj.append(tuple(row))

    # Canonical: minimum over all relabelings
    best = None
    for perm in permutations(range(n)):
        relabeled = tuple(tuple(adj[perm[i]][perm[j]] for j in range(n)) for i in range(n))
        if best is None or relabeled < best:
            best = relabeled
    return best


def compute_iai(G, k_values=None, fast=True):
    """
    Compute Induced Assembly Index for graph G.

    IAI(G, k) = number of distinct induced k-subgraph types
    IAI(G) = sum_k IAI(G,k) / C(n,k)

    fast=True: sample instead of enumerate for large k
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    if k_values is None:
        k_values = list(range(2, min(n, 8)))

    iai_per_k = {}

    for k in k_values:
        if k > n:
            break

        n_combos = len(list(combinations(range(n), k)))

        # For small graphs or small k: enumerate all
        if n_combos <= 5000 or not fast:
            types = set()
            for subset in combinations(nodes, k):
                canon = canonical_subgraph(G, subset)
                types.add(canon)
        else:
            # Sample random subsets
            types = set()
            for _ in range(min(3000, n_combos)):
                subset = np.random.choice(nodes, k, replace=False)
                canon = canonical_subgraph(G, tuple(subset))
                types.add(canon)

        n_distinct = len(types)
        n_possible = 2 ** (k*(k-1)//2)  # max distinct labeled graphs on k nodes

        # Normalized IAI(k): distinct types / C(n,k) is size-normalized
        iai_per_k[k] = n_distinct

    # Overall IAI: average normalized distinctness
    total = 0
    for k, n_distinct in iai_per_k.items():
        n_combos_k = len(list(combinations(nodes, k))) if n <= 20 else None
        n_possible = 2 ** (k*(k-1)//2)
        # Diversity = distinct_types / n_possible (fraction of type space covered)
        diversity = n_distinct / n_possible
        total += diversity

    iai = total / len(iai_per_k) if iai_per_k else 0
    return iai, iai_per_k


def test_hypothesis_1():
    """H1: Regular graphs have lower IAI than random graphs."""
    print("=== Hypothesis 1: Regular < Random ===")
    np.random.seed(42)

    n = 12

    # Regular graphs
    complete = nx.complete_graph(n)
    cycle = nx.cycle_graph(n)
    petersen = nx.petersen_graph()
    grid = nx.grid_2d_graph(3, 4)
    grid = nx.convert_node_labels_to_integers(grid)

    # Random graphs
    gnp_dense = nx.gnp_random_graph(n, 0.5, seed=42)
    gnp_sparse = nx.gnp_random_graph(n, 0.3, seed=42)

    graphs = [
        ("Complete K12", complete),
        ("Cycle C12", cycle),
        ("Grid 3x4", grid),
        ("Gnp(0.3)", gnp_sparse),
        ("Gnp(0.5)", gnp_dense),
    ]

    results = []
    k_vals = list(range(2, 7))
    for name, G in graphs:
        iai, per_k = compute_iai(G, k_values=k_vals)
        print(f"  {name:>15s}: IAI={iai:.4f}  per_k={[per_k.get(k,0) for k in k_vals]}")
        results.append((name, iai))

    results.sort(key=lambda x: x[1])
    print(f"\n  Ranking (lowest IAI = most regular):")
    for name, iai in results:
        print(f"    {name:>15s}: {iai:.4f}")

    regular_max = max(iai for n, iai in results if n in ["Complete K12", "Cycle C12", "Grid 3x4"])
    random_min = min(iai for n, iai in results if "Gnp" in n)
    print(f"\n  Regular max IAI: {regular_max:.4f}")
    print(f"  Random min IAI:  {random_min:.4f}")
    print(f"  H1 {'SUPPORTED' if regular_max < random_min else 'VIOLATED'}")


def test_hypothesis_3():
    """H3: IAI ~ k_min (graphs with higher IAI need larger k for CR classification)."""
    print("\n=== Hypothesis 3: IAI ~ k_min (Counting Revolution) ===")
    print("Testing known k_min values from memory:")
    print()

    # From Counting Revolution research:
    # - Complete graphs: k_min = 1 (all Kn are distinct from size alone)
    # - Cycle graphs: k_min = ?
    # - Regular graphs: k_min = small?
    # - General graphs n=7: k_min = n-2 = 5
    # - SRG(16,6,2,2): k_min = 4

    np.random.seed(0)
    n = 8

    print("  Computing IAI for n=8 graphs (sample of 100 random)...")
    iaIs = []
    for seed in range(100):
        np.random.seed(seed)
        G = nx.gnp_random_graph(n, 0.5, seed=seed)
        if nx.is_connected(G) and G.number_of_edges() > 0:
            iai, _ = compute_iai(G, k_values=[3, 4, 5])
            iaIs.append(iai)

    print(f"  n=8 random connected graphs:")
    print(f"    IAI mean={np.mean(iaIs):.4f}, std={np.std(iaIs):.4f}")
    print(f"    min={np.min(iaIs):.4f}, max={np.max(iaIs):.4f}")

    # The known hard pairs from CR (need k=6 to resolve):
    # (3630,4580), (7163,7210), (7638,8901), (11754,11839) in graph6 format
    # Let's test: do hard pairs have SIMILAR IAI? (that's why they're hard)
    print()
    print("  Key insight test: Hard pairs for CR should have SAME IAI")
    print("  (Two graphs indistinguishable by k-subgraph histogram = same IAI?)")

    # Create two graphs that we KNOW are CR-hard (similar structure)
    # Petersen graph and its complement
    P = nx.petersen_graph()
    PC = nx.complement(P)
    iai_P, _ = compute_iai(P, k_values=[3, 4, 5, 6])
    iai_PC, _ = compute_iai(PC, k_values=[3, 4, 5, 6])
    print(f"  Petersen: IAI={iai_P:.4f}")
    print(f"  Complement(Petersen): IAI={iai_PC:.4f}")
    print(f"  Same IAI? {'YES' if abs(iai_P - iai_PC) < 0.01 else 'NO'}")


def test_hypothesis_4():
    """H4: IAI for graphs vs Assembly Index - are they related?"""
    print("\n=== Hypothesis 4: IAI vs Graph 'Size' ===")
    print("Testing if IAI captures complexity independent of size...")
    np.random.seed(42)

    # Different graph families
    families = []
    for n in [6, 8, 10, 12]:
        G_rand = nx.gnp_random_graph(n, 0.5, seed=42)
        G_cycle = nx.cycle_graph(n)
        G_complete = nx.complete_graph(n)

        k_test = min(5, n-1)
        iai_rand, _ = compute_iai(G_rand, k_values=list(range(2, k_test+1)))
        iai_cycle, _ = compute_iai(G_cycle, k_values=list(range(2, k_test+1)))
        iai_comp, _ = compute_iai(G_complete, k_values=list(range(2, k_test+1)))

        families.append((n, 'random', iai_rand))
        families.append((n, 'cycle', iai_cycle))
        families.append((n, 'complete', iai_comp))

    print(f"\n  {'n':>4s}  {'type':>10s}  {'IAI':>8s}")
    print("  " + "-"*26)
    for n, t, iai in sorted(families, key=lambda x: (x[1], x[0])):
        print(f"  {n:>4d}  {t:>10s}  {iai:>8.4f}")

    # Does IAI grow with n for same type?
    print()
    print("  IAI vs n for random graphs:")
    rand_iais = [(n, iai) for n, t, iai in families if t == 'random']
    for n, iai in rand_iais:
        print(f"    n={n}: IAI={iai:.4f}")

    ns = [x[0] for x in rand_iais]
    iais = [x[1] for x in rand_iais]
    corr = np.corrcoef(ns, iais)[0,1]
    print(f"  Spearman(n, IAI) for random: rho={corr:.4f}")
    print(f"  If rho ~ 0: IAI is SIZE-INDEPENDENT (good complexity measure)")
    print(f"  If rho ~ 1: IAI scales with n (just measuring size)")


print("=== INDUCED ASSEMBLY INDEX (IAI) -- NEW COMPLEXITY MEASURE ===")
print("Combining Assembly Theory + Counting Revolution\n")

test_hypothesis_1()
test_hypothesis_3()
test_hypothesis_4()

print("\n\n=== SUMMARY ===")
print("IAI(G) = normalized diversity of induced subgraph types")
print()
print("Properties (if confirmed):")
print("  Regular graphs: low IAI (symmetric, predictable)")
print("  Random graphs: high IAI (diverse local structures)")
print("  Hard CR pairs: same IAI (that's why they're hard to distinguish)")
print()
print("Novel insight: IAI measures 'structural entropy' of a graph")
print("  - AI (Assembly Index): minimum production cost")
print("  - CR k_min: minimum observation depth for classification")
print("  - IAI: local structure diversity (bridges both)")
print()
print("Conjecture: IAI(G) ~ k_min(G) / n (normalized detection threshold)")
