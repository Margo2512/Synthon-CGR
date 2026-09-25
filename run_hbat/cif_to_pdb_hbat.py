#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import gemmi

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
HALOGENS = {"F", "Cl", "Br", "I"}

PAIR_MAX = {
    tuple(sorted(k.split("-"))): v
    for k, v in {
        "H-C": 1.25, "H-N": 1.20, "H-O": 1.15, "H-S": 1.45, "H-P": 1.45,
        "C-C": 1.78, "C-N": 1.72, "C-O": 1.68, "C-S": 1.95, "C-P": 1.95,
        "C-F": 1.45, "C-Cl": 1.90, "C-Br": 2.10, "C-I": 2.35,
        "N-N": 1.60, "N-O": 1.55, "N-S": 1.85, "N-P": 1.85,
        "O-O": 1.55, "O-S": 1.80, "O-P": 1.85,
        "S-S": 2.15, "P-P": 2.30, "P-S": 2.20,
        "B-F": 1.50, "B-O": 1.65, "B-N": 1.65, "B-C": 1.70,
        "Si-C": 2.00, "Si-O": 1.85, "Si-N": 1.90, "Si-Cl": 2.20,
    }.items()
}

COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}


@dataclass
class Atom:
    label: str
    element: str
    frac: np.ndarray
    xyz: np.ndarray
    source_index: int
    symop_index: int
    image: Tuple[int, int, int]


@dataclass
class Component:
    indices: List[int]
    centroid: np.ndarray
    formula: str
    distance_to_center: float


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def clean_float(s: str) -> float:
    s = str(s).strip().strip("'\"")
    s = re.sub(r"\([^)]*\)", "", s)
    if s in {"?", ".", ""}:
        raise ValueError("missing numeric CIF value")
    return float(s)


def clean_symbol(s: str) -> str:
    s = str(s).strip().strip("'\"")
    m = re.match(r"([A-Z][a-z]?)", s)
    return m.group(1) if m else s


def get_cell(block) -> gemmi.UnitCell:
    return gemmi.UnitCell(
        clean_float(block.find_value("_cell_length_a")),
        clean_float(block.find_value("_cell_length_b")),
        clean_float(block.find_value("_cell_length_c")),
        clean_float(block.find_value("_cell_angle_alpha")),
        clean_float(block.find_value("_cell_angle_beta")),
        clean_float(block.find_value("_cell_angle_gamma")),
    )


def get_spacegroup(block) -> gemmi.SpaceGroup:
    for key in ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M", "_space_group_name_Hall"):
        v = block.find_value(key)
        if v:
            name = v.strip().strip("'\"")
            sg = gemmi.find_spacegroup_by_name(name)
            if sg is not None:
                return sg
    return gemmi.find_spacegroup_by_name("P 1")


def read_asymmetric_atoms(cif_path: Path):
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()
    cell = get_cell(block)
    sg = get_spacegroup(block)

    labels = list(block.find_loop("_atom_site_label"))
    types = list(block.find_loop("_atom_site_type_symbol"))
    xs = list(block.find_loop("_atom_site_fract_x"))
    ys = list(block.find_loop("_atom_site_fract_y"))
    zs = list(block.find_loop("_atom_site_fract_z"))

    occ_col = list(block.find_loop("_atom_site_occupancy")) if block.find_loop("_atom_site_occupancy") else None
    disorder_col = list(block.find_loop("_atom_site_disorder_group")) if block.find_loop("_atom_site_disorder_group") else None

    asym = []
    for i, (lab, typ, x, y, z) in enumerate(zip(labels, types, xs, ys, zs)):
        if occ_col is not None:
            try:
                if clean_float(occ_col[i]) <= 0:
                    continue
            except Exception:
                pass
        if disorder_col is not None and disorder_col[i] not in {".", "?", "0", "1"}:
            continue
        asym.append((str(lab).strip(), clean_symbol(typ), np.array([clean_float(x), clean_float(y), clean_float(z)], dtype=float)))
    return block, cell, sg, asym


def frac_to_xyz(cell: gemmi.UnitCell, frac: np.ndarray) -> np.ndarray:
    p = cell.orthogonalize(gemmi.Fractional(float(frac[0]), float(frac[1]), float(frac[2])))
    return np.array([p.x, p.y, p.z], dtype=float)


def expand_atoms(asym, cell: gemmi.UnitCell, sg: gemmi.SpaceGroup, images: int) -> List[Atom]:
    ops = list(sg.operations())
    atoms: List[Atom] = []
    translations = range(-images, images + 1)
    seen = set()
    for tx in translations:
        for ty in translations:
            for tz in translations:
                shift = np.array([tx, ty, tz], dtype=float)
                for op_i, op in enumerate(ops):
                    for src_i, (label, element, frac0) in enumerate(asym):
                        f = np.array(op.apply_to_xyz([float(frac0[0]), float(frac0[1]), float(frac0[2])]), dtype=float)
                        f = np.mod(f, 1.0) + shift
                        key = (src_i, tuple(np.round(f, 6)))
                        if key in seen:
                            continue
                        seen.add(key)
                        atoms.append(Atom(label=label, element=element, frac=f, xyz=frac_to_xyz(cell, f), source_index=src_i, symop_index=op_i, image=(tx, ty, tz)))
    return atoms


def covalent_cutoff(el1: str, el2: str, fallback_factor: float, fallback_tolerance: float) -> float | None:
    pair = tuple(sorted((el1, el2)))
    if pair in PAIR_MAX:
        return PAIR_MAX[pair]

    if (el1 in HALOGENS or el2 in HALOGENS):
        return None

    r1 = COVALENT_RADII.get(el1)
    r2 = COVALENT_RADII.get(el2)
    if r1 is None or r2 is None:
        return None
    return fallback_factor * (r1 + r2) + fallback_tolerance


def looks_like_bond(el1: str, el2: str, distance: float, fallback_factor: float, fallback_tolerance: float) -> bool:
    if distance < 0.35:
        return False
    if el1 == "H" and el2 == "H":
        return False
    cutoff = covalent_cutoff(el1, el2, fallback_factor, fallback_tolerance)
    return cutoff is not None and distance <= cutoff


def infer_bonds_and_components(atoms: List[Atom], fallback_factor: float, fallback_tolerance: float):
    coords = np.array([a.xyz for a in atoms], dtype=float)
    n = len(atoms)
    uf = UnionFind(n)
    edges: List[Tuple[int, int]] = []

    max_cutoff = max(PAIR_MAX.values()) + 0.05
    if cKDTree is not None:
        pairs = cKDTree(coords).query_pairs(max_cutoff)
    else:
        pairs = ((i, j) for i in range(n) for j in range(i + 1, n) if np.linalg.norm(coords[i] - coords[j]) <= max_cutoff)

    for i, j in pairs:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if looks_like_bond(atoms[i].element, atoms[j].element, d, fallback_factor, fallback_tolerance):
            uf.union(i, j)
            edges.append((i, j))

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    components = list(groups.values())
    return edges, components


def read_geom_bond_records(block, asym) -> List[Tuple[int, int, float]]:
    col1 = block.find_loop("_geom_bond_atom_site_label_1")
    col2 = block.find_loop("_geom_bond_atom_site_label_2")
    cold = block.find_loop("_geom_bond_distance")
    if not col1 or not col2 or not cold:
        return []

    label_to_src = {label: i for i, (label, _element, _frac) in enumerate(asym)}
    records: List[Tuple[int, int, float]] = []
    seen = set()
    for a, b, d in zip(list(col1), list(col2), list(cold)):
        la = str(a).strip().strip("'\"")
        lb = str(b).strip().strip("'\"")
        if la not in label_to_src or lb not in label_to_src:
            continue
        try:
            dist = clean_float(d)
        except Exception:
            continue
        if dist < 0.35:
            continue
        src1, src2 = label_to_src[la], label_to_src[lb]
        key = (min(src1, src2), max(src1, src2), round(dist, 3))
        if key in seen:
            continue
        seen.add(key)
        records.append((src1, src2, float(dist)))
    return records


def infer_bonds_from_geom_bonds(atoms: List[Atom], bond_records: List[Tuple[int, int, float]], tolerance: float):
    n = len(atoms)
    uf = UnionFind(n)
    edge_set: set[Tuple[int, int]] = set()
    by_src: Dict[int, List[int]] = defaultdict(list)
    for i, atom in enumerate(atoms):
        by_src[atom.source_index].append(i)

    tree_cache = {}
    coords_cache = {}
    idx_cache = {}

    def get_tree(src: int):
        if src not in idx_cache:
            idxs = by_src.get(src, [])
            idx_cache[src] = idxs
            coords_cache[src] = np.array([atoms[i].xyz for i in idxs], dtype=float) if idxs else np.empty((0, 3))
            if cKDTree is not None and len(idxs) > 0:
                tree_cache[src] = cKDTree(coords_cache[src])
            else:
                tree_cache[src] = None
        return idx_cache[src], coords_cache[src], tree_cache[src]

    for src1, src2, expected in bond_records:
        idxs1 = by_src.get(src1, [])
        idxs2, coords2, tree2 = get_tree(src2)
        if not idxs1 or not idxs2:
            continue
        upper = expected + tolerance
        lower = max(0.35, expected - tolerance)
        for i in idxs1:
            xyz = atoms[i].xyz
            if tree2 is not None:
                candidate_positions = tree2.query_ball_point(xyz, upper)
                candidates = [idxs2[pos] for pos in candidate_positions]
            else:
                candidates = [j for j in idxs2 if np.linalg.norm(xyz - atoms[j].xyz) <= upper]
            for j in candidates:
                if i == j:
                    continue
                d = float(np.linalg.norm(xyz - atoms[j].xyz))
                if lower <= d <= upper:
                    a, b = (i, j) if i < j else (j, i)
                    edge_set.add((a, b))

    edges = sorted(edge_set)
    for i, j in edges:
        uf.union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return edges, list(groups.values())



def add_missing_xh_bonds_and_components(
    atoms: List[Atom],
    edges: List[Tuple[int, int]],
    mode: str = "auto",
) -> Tuple[List[Tuple[int, int]], List[List[int]], int]:
    if mode == "no":
        n = len(atoms)
        uf = UnionFind(n)
        for i, j in edges:
            uf.union(i, j)
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)
        return sorted(set(tuple(sorted(e)) for e in edges)), list(groups.values()), 0

    edge_set: set[Tuple[int, int]] = set(tuple(sorted(e)) for e in edges)
    h_has_bond = set()
    for i, j in edge_set:
        if atoms[i].element == "H":
            h_has_bond.add(i)
        if atoms[j].element == "H":
            h_has_bond.add(j)

    heavy_indices = [i for i, a in enumerate(atoms) if a.element != "H"]
    if not heavy_indices:
        n = len(atoms)
        return sorted(edge_set), [[i] for i in range(n)], 0

    heavy_coords = np.array([atoms[i].xyz for i in heavy_indices], dtype=float)
    heavy_tree = cKDTree(heavy_coords) if cKDTree is not None else None
    max_h_cutoff = max(v for pair, v in PAIR_MAX.items() if "H" in pair) + 0.05

    added = 0
    for i, atom in enumerate(atoms):
        if atom.element != "H" or i in h_has_bond:
            continue

        if heavy_tree is not None:
            candidate_positions = heavy_tree.query_ball_point(atom.xyz, max_h_cutoff)
            candidates = [heavy_indices[p] for p in candidate_positions]
        else:
            candidates = [
                j for j in heavy_indices
                if np.linalg.norm(atom.xyz - atoms[j].xyz) <= max_h_cutoff
            ]

        best = None
        best_d = float("inf")
        for j in candidates:
            d = float(np.linalg.norm(atom.xyz - atoms[j].xyz))
            cutoff = covalent_cutoff("H", atoms[j].element, 1.10, 0.10)
            if cutoff is None:
                continue
            if 0.55 <= d <= cutoff + 0.08 and d < best_d:
                best = j
                best_d = d

        if best is not None:
            a, b = (i, best) if i < best else (best, i)
            if (a, b) not in edge_set:
                edge_set.add((a, b))
                added += 1

    n = len(atoms)
    uf = UnionFind(n)
    for i, j in edge_set:
        uf.union(i, j)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return sorted(edge_set), list(groups.values()), added

def component_formula(atoms: List[Atom], indices: Sequence[int]) -> str:
    counts = Counter(atoms[i].element for i in indices)
    order = []
    if "C" in counts:
        order.append("C")
    if "H" in counts:
        order.append("H")
    order.extend(sorted(k for k in counts if k not in {"C", "H"}))
    return "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in order)


def rank_components(atoms: List[Atom], cell: gemmi.UnitCell, components: List[List[int]]) -> List[Component]:
    central_xyz = frac_to_xyz(cell, np.array([0.5, 0.5, 0.5], dtype=float))
    ranked: List[Component] = []
    for comp in components:
        centroid = np.mean([atoms[i].xyz for i in comp], axis=0)
        dist = float(np.linalg.norm(centroid - central_xyz))
        ranked.append(Component(indices=list(comp), centroid=centroid, formula=component_formula(atoms, comp), distance_to_center=dist))
    ranked.sort(key=lambda c: (c.distance_to_center, -len(c.indices), c.formula))
    return ranked


def min_distance_between_components(atoms: List[Atom], a: Sequence[int], b: Sequence[int]) -> float:
    ac = np.array([atoms[i].xyz for i in a], dtype=float)
    bc = np.array([atoms[i].xyz for i in b], dtype=float)
    best = float("inf")
    for i in range(0, len(ac), 128):
        diff = ac[i:i+128, None, :] - bc[None, :, :]
        d = np.linalg.norm(diff, axis=2).min()
        if d < best:
            best = float(d)
    return best


def select_cluster(atoms: List[Atom], ranked: List[Component], central_rank: int, radius: float, max_components: int | None):
    if central_rank < 0 or central_rank >= len(ranked):
        raise ValueError(f"central component rank {central_rank} is out of range 0..{len(ranked)-1}")
    central = ranked[central_rank]
    central_set = set(central.indices)
    selected: List[Tuple[float, Component]] = [(0.0, central)]
    for comp in ranked:
        if set(comp.indices) == central_set:
            continue
        d = min_distance_between_components(atoms, central.indices, comp.indices)
        if d <= radius:
            selected.append((d, comp))
    selected.sort(key=lambda x: (x[0], x[1].distance_to_center, x[1].formula))
    comps = [c for _, c in selected]
    if max_components is not None:
        comps = comps[:max_components]
    return central, comps


def atom_name(label: str, element: str, per_element_count: int) -> str:
    lab = re.sub(r"[^A-Za-z0-9]", "", label).strip()
    if 1 <= len(lab) <= 4 and re.match(r"^[A-Za-z]", lab):
        return lab[:4]
    base = f"{element}{per_element_count}"
    return base[:4]


def pdb_atom_line(serial: int, name: str, resname: str, chain: str, resseq: int, xyz: np.ndarray, element: str) -> str:
    x, y, z = xyz
    return (
        f"HETATM{serial:5d} {name:<4s} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )


def pdb_conect_lines(edges: Iterable[Tuple[int, int]], old_to_serial: Dict[int, int]) -> List[str]:
    neigh: Dict[int, List[int]] = defaultdict(list)
    for i, j in edges:
        if i in old_to_serial and j in old_to_serial:
            si, sj = old_to_serial[i], old_to_serial[j]
            neigh[si].append(sj)
            neigh[sj].append(si)
    lines = []
    for serial in sorted(neigh):
        partners = sorted(set(neigh[serial]))
        for k in range(0, len(partners), 4):
            chunk = partners[k:k+4]
            lines.append("CONECT" + f"{serial:5d}" + "".join(f"{p:5d}" for p in chunk) + "\n")
    return lines



def component_type_signature(
    atoms: List[Atom],
    comp: Component,
    edges: List[Tuple[int, int]],
    mode: str = "connectivity",
):
    if mode == "formula":
        return ("formula", comp.formula)

    comp_set = set(comp.indices)
    element_counts = tuple(sorted(Counter(atoms[i].element for i in comp.indices).items()))

    bond_pair_counts = Counter()
    degree_by_element = Counter()
    for i, j in edges:
        if i in comp_set and j in comp_set:
            e1, e2 = atoms[i].element, atoms[j].element
            bond_pair_counts[tuple(sorted((e1, e2)))] += 1
            degree_by_element[(e1, "degree")] += 1
            degree_by_element[(e2, "degree")] += 1

    return (
        "connectivity",
        element_counts,
        tuple(sorted(bond_pair_counts.items())),
        tuple(sorted(degree_by_element.items())),
    )


def assign_resnames(
    atoms: List[Atom],
    selected: List[Component],
    edges: List[Tuple[int, int]],
    match_mode: str = "connectivity",
) -> List[str]:
    if not selected:
        return []

    central_signature = component_type_signature(atoms, selected[0], edges, match_mode)
    resnames = ["CTR"]
    central_copy_count = 0
    other_count = 0

    for comp in selected[1:]:
        sig = component_type_signature(atoms, comp, edges, match_mode)
        if sig == central_signature:
            central_copy_count += 1
            resnames.append(f"C{central_copy_count % 100:02d}")
        else:
            other_count += 1
            resnames.append(f"M{other_count % 100:02d}")
    return resnames


def write_pdb(
    out_path: Path,
    cell: gemmi.UnitCell,
    atoms: List[Atom],
    selected: List[Component],
    edges: List[Tuple[int, int]],
    center_at_origin: bool,
    type_match_mode: str,
):
    shift = np.zeros(3)
    if center_at_origin:
        all_indices = [i for comp in selected for i in comp.indices]
        shift = np.mean([atoms[i].xyz for i in all_indices], axis=0)

    resnames = assign_resnames(atoms, selected, edges, type_match_mode)

    old_to_serial: Dict[int, int] = {}
    atom_lines: List[str] = []
    serial = 1
    for mol_no, comp in enumerate(selected, start=1):
        chain = CHAIN_IDS[(mol_no - 1) % len(CHAIN_IDS)]
        resseq = mol_no
        resname = resnames[mol_no - 1]
        counts = Counter()
        ordered = sorted(comp.indices, key=lambda i: (atoms[i].element == "H", atoms[i].source_index, atoms[i].label))
        for idx in ordered:
            a = atoms[idx]
            counts[a.element] += 1
            name = atom_name(a.label, a.element, counts[a.element])
            atom_lines.append(pdb_atom_line(serial, name, resname, chain, resseq, a.xyz - shift, a.element))
            old_to_serial[idx] = serial
            serial += 1
        atom_lines.append(f"TER   {serial:5d}      {resname:>3s} {chain:1s}{resseq:4d}\n")
        serial += 1

    with out_path.open("w", encoding="utf-8") as f:
        f.write("REMARK 900 HBAT-FRIENDLY PACKING CLUSTER GENERATED FROM CIF\n")
        f.write("REMARK 900 VERSION 5: CIF GEOM_BOND + SAFE X-H CONNECTIVITY; NONBONDED CONTACTS ARE NOT CONECT\n")
        f.write("REMARK 900 RESNAME CTR IS THE SELECTED CENTRAL MOLECULE\n")
        f.write("REMARK 900 RESNAMES C01,C02,... ARE PERIODIC COPIES MATCHING CTR\n")
        f.write("REMARK 900 RESNAMES M01,M02,... ARE OTHER COFORMER MOLECULES\n")
        f.write("REMARK 900 NEIGHBOURING MOLECULES ARE EXPLICIT PERIODIC COPIES\n")
        f.write(f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} P 1           1\n")
        for line in atom_lines:
            f.write(line)
        for line in pdb_conect_lines(edges, old_to_serial):
            f.write(line)
        f.write("END\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create an HBAT-friendly PDB packing cluster from a small-molecule CIF.")
    ap.add_argument("cif", type=Path, help="Input CIF file")
    ap.add_argument("-o", "--output", type=Path, default=Path("hbat_cluster.pdb"), help="Output PDB file")
    ap.add_argument("-r", "--radius", type=float, default=5.0, help="Keep molecules with any atom within this radius of the central molecule [Å]")
    ap.add_argument("--images", type=int, default=None, help="Number of unit-cell images in +/- a,b,c. Default: automatic from radius, with one extra shell to avoid clipped molecules")
    ap.add_argument("--central-component", type=int, default=0, help="Component rank after sorting by distance to the central unit-cell center")
    ap.add_argument("--max-components", type=int, default=None, help="Optional cap on number of molecules written")
    ap.add_argument("--bond-source", choices=["auto", "geom", "strict"], default="auto", help="Connectivity source: 'geom' uses CIF _geom_bond only; 'strict' uses conservative distance rules; 'auto' uses _geom_bond when available, otherwise strict")
    ap.add_argument("--geom-bond-tolerance", type=float, default=0.25, help="Tolerance in Å when matching expanded atoms to CIF _geom_bond distances")
    ap.add_argument("--attach-hydrogens", choices=["auto", "yes", "no"], default="auto", help="Add missing short X-H covalent bonds when CIF _geom_bond omits hydrogens. Default: auto")
    ap.add_argument("--fallback-bond-factor", type=float, default=1.10, help="Fallback covalent radii multiplier for element pairs not in the strict table, used only with --bond-source strict or auto fallback")
    ap.add_argument("--fallback-bond-tolerance", type=float, default=0.10, help="Fallback bond tolerance in Å, used only with --bond-source strict or auto fallback")
    ap.add_argument("--center-at-origin", action="store_true", help="Translate output cluster centroid to the origin")
    ap.add_argument("--type-match-mode", choices=["connectivity", "formula"], default="connectivity", help="How to decide whether a neighbour is the same molecular type as CTR. Default: connectivity")
    ap.add_argument("--list-components", action="store_true", help="Print the first 30 ranked components and exit without writing PDB")
    args = ap.parse_args()

    block, cell, sg, asym = read_asymmetric_atoms(args.cif)
    if args.images is None:
        min_len = min(cell.a, cell.b, cell.c)
        args.images = max(2, int(math.ceil(args.radius / min_len)) + 1)

    atoms = expand_atoms(asym, cell, sg, args.images)

    bond_records = read_geom_bond_records(block, asym)
    if args.bond_source in {"auto", "geom"} and bond_records:
        edges, raw_components = infer_bonds_from_geom_bonds(atoms, bond_records, args.geom_bond_tolerance)
        bond_source_used = f"CIF _geom_bond ({len(bond_records)} listed bonds)"
    elif args.bond_source == "geom":
        raise ValueError("--bond-source geom was requested, but no usable CIF _geom_bond loop was found")
    else:
        edges, raw_components = infer_bonds_and_components(atoms, args.fallback_bond_factor, args.fallback_bond_tolerance)
        bond_source_used = "strict distance fallback"

    xh_edges_added = 0
    if args.bond_source in {"auto", "geom"} and bond_records:
        edges, raw_components, xh_edges_added = add_missing_xh_bonds_and_components(atoms, edges, args.attach_hydrogens)
        if xh_edges_added:
            bond_source_used += f" + {xh_edges_added} inferred X-H bonds"

    ranked = rank_components(atoms, cell, raw_components)

    if args.list_components:
        print(f"CIF: {args.cif}")
        print(f"Space group: {sg.hm}; asym atoms: {len(asym)}; expanded atoms: {len(atoms)}")
        print("rank\tformula\tatoms\tdistance_to_center_A")
        for i, comp in enumerate(ranked[:30]):
            print(f"{i}\t{comp.formula}\t{len(comp.indices)}\t{comp.distance_to_center:.3f}")
        return

    central, selected = select_cluster(atoms, ranked, args.central_component, args.radius, args.max_components)
    write_pdb(args.output, cell, atoms, selected, edges, args.center_at_origin, args.type_match_mode)

    selected_atom_count = sum(len(c.indices) for c in selected)
    selected_h_count = sum(1 for c in selected for idx in c.indices if atoms[idx].element == "H")
    formula_counts = Counter(c.formula for c in selected)
    resname_counts = Counter(assign_resnames(atoms, selected, edges, args.type_match_mode))

    print(f"Input CIF:              {args.cif}")
    print(f"Output PDB:             {args.output}")
    print(f"Space group:            {sg.hm}")
    print(f"Expanded images:        +/-{args.images} -> {(2*args.images+1)}x{(2*args.images+1)}x{(2*args.images+1)}")
    print(f"Asymmetric-unit atoms:  {len(asym)}")
    print(f"Expanded atoms:         {len(atoms)}")
    print(f"Bond source:            {bond_source_used}")
    print(f"Covalent CONECT edges:  {len(edges)}")
    if xh_edges_added:
        print(f"Added X-H edges:        {xh_edges_added}")
    print(f"Detected components:    {len(ranked)}")
    print(f"Central component:      rank {args.central_component}, formula {central.formula}, atoms {len(central.indices)}")
    print(f"Written molecules:      {len(selected)}")
    print(f"Written atoms:          {selected_atom_count}")
    print(f"Written hydrogens:      {selected_h_count}")
    print("Written formulas:       " + ", ".join(f"{k} x{v}" for k, v in sorted(formula_counts.items())))
    print("Residue names:          " + ", ".join(f"{k} x{v}" for k, v in sorted(resname_counts.items())))
    if selected_h_count == 0:
        print("WARNING: no H atoms were written. HBAT hydrogen-bond detection needs explicit hydrogens.")
    print("Tip: run with --list-components to choose another central molecule if needed.")


if __name__ == "__main__":
    main()
