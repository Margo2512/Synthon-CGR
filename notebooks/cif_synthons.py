from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple, Set
import gemmi
import numpy as np
import networkx as nx
from rdkit import Chem
from itertools import combinations

ATOM_ROLE_RULES = [
    # CARBOXYLIC ACIDS / CARBOXYLATES
    {
        "name": "carboxylic_acid_OH_donor",
        "group": "carboxylic_acid",
        "kind": "donor",
        "element": "O",
        "smarts": ["[OX2H1][CX3](=[OX1])"],
    },
    {
        "name": "carboxylic_acid_carbonyl_O_acceptor",
        "group": "carboxylic_acid",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[CX3]([OX2])[#6]"],
    },
    {
        "name": "carboxylate_O_acceptor",
        "group": "carboxylate",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1-][CX3](=[OX1])", "[OX1]=[CX3][OX1-]"],
    },

    # ALCOHOLS / PHENOLS / ETHERS
    {
        "name": "alcohol_OH_donor",
        "group": "alcohol",
        "kind": "donor",
        "element": "O",
        "smarts": ["[OX2H1][CX4;!$(C=[O,N])]"],
    },
    {
        "name": "alcohol_O_acceptor",
        "group": "alcohol",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX2H1][CX4;!$(C=[O,N])]"],
    },
    {
        "name": "phenol_OH_donor",
        "group": "phenol",
        "kind": "donor",
        "element": "O",
        "smarts": ["[OX2H1][c]"],
    },
    {
        "name": "phenol_O_acceptor",
        "group": "phenol",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX2H1][c]"],
    },
    {
        "name": "ether_O_acceptor",
        "group": "ether",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OD2]([#6])[#6]"],
    },

    # CARBONYL OXYGENS
    {
        "name": "aldehyde_O_acceptor",
        "group": "aldehyde",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[CX3H1]"],
    },
    {
        "name": "ketone_O_acceptor",
        "group": "ketone",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[CX3]([#6])[#6]"],
    },
    {
        "name": "ester_carbonyl_O_acceptor",
        "group": "ester",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[CX3]([#6])[OX2]"],
    },
    {
        "name": "ester_alkoxy_O_acceptor",
        "group": "ester",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX2]([CX3](=[OX1])[#6])[#6]"],
    },

    # AMIDES / UREAS / CARBAMATES / SULFONAMIDES
    {
        "name": "amide_NH_donor",
        "group": "amide",
        "kind": "donor",
        "element": "N",
        "smarts": ["[NX3H1,H2][CX3](=[OX1])"],
    },
    {
        "name": "amide_carbonyl_O_acceptor",
        "group": "amide",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[CX3]([NX3])"],
    },
    {
        "name": "urea_carbonyl_O_acceptor",
        "group": "urea_like",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[CX3]([NX3])[NX3]"],
    },
    {
        "name": "carbamate_carbonyl_O_acceptor",
        "group": "carbamate",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[CX3]([NX3])[OX2]"],
    },
    {
        "name": "sulfonamide_NH_donor",
        "group": "sulfonamide",
        "kind": "donor",
        "element": "N",
        "smarts": ["[NX3H1,H2][SX4](=[OX1])(=[OX1])"],
    },

    # AMINES / IMINES / AMIDINES / GUANIDINES
    {
        "name": "amine_N_acceptor",
        "group": "neutral_amine",
        "kind": "acceptor",
        "element": "N",
        "smarts": [
            "[NX3;H0;!$(N=*);!$(N#*);!$(N[C]=O);!$(N[C]=[N,S])]",
            "[NX3;H1;!$(N=*);!$(N#*);!$(N[C]=O);!$(N[C]=[N,S]);!$([#6]-[NX3H1]-[#6])]",
        ],
    },
    {
        "name": "amine_NH_donor",
        "group": "neutral_amine",
        "kind": "donor",
        "element": "N",
        "smarts": [
            "[NX3H1,H2;!$(N=*);!$(N#*);!$(N[C]=O);!$(N[C]=[N,S])]",
        ],
    },
    {
        "name": "imine_N_acceptor",
        "group": "imine",
        "kind": "acceptor",
        "element": "N",
        "smarts": ["[NX2]=[CX3]"],
    },
    {
        "name": "amidino_imine_N_acceptor",
        "group": "amidine",
        "kind": "acceptor",
        "element": "N",
        "smarts": ["[NX2]=[CX3]([NX3])"],
    },
    {
        "name": "amidino_amino_NH_donor",
        "group": "amidine",
        "kind": "donor",
        "element": "N",
        "smarts": ["[NX3H1,H2][CX3](=[NX2])"],
    },
    {
        "name": "guanidine_imine_N_acceptor",
        "group": "guanidine_like",
        "kind": "acceptor",
        "element": "N",
        "smarts": ["[NX2]=[CX3]([NX3])[NX3]"],
    },
    {
        "name": "guanidine_terminal_NH_donor",
        "group": "guanidine_like",
        "kind": "donor",
        "element": "N",
        "smarts": ["[NX3H1,H2][CX3](=[NX2])"],
    },

    # AROMATIC NITROGENS
    {
        "name": "aza_aromatic_N_acceptor",
        "group": "aza_aromatic_N",
        "kind": "acceptor",
        "element": "N",
        "smarts": ["[n;+0;!H]"],
    },
    {
        "name": "aza_aromatic_NH_donor",
        "group": "aza_aromatic_NH",
        "kind": "donor",
        "element": "N",
        "smarts": ["[nH]"],
    },
    {
        "name": "N_oxide_O_acceptor",
        "group": "N_oxide",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[O-][n+]"],
    },
    {
        "name": "nitrile_N_acceptor",
        "group": "nitrile",
        "kind": "acceptor",
        "element": "N",
        "smarts": ["[NX1]#[CX2]"],
    },

    # PROTONATED NITROGENS
    {
        "name": "ammonium_NH_donor",
        "group": "cationic_NH",
        "kind": "donor",
        "element": "N",
        "smarts": ["[NX4+]"],
    },
    {
        "name": "azolium_NH_donor",
        "group": "cationic_NH",
        "kind": "donor",
        "element": "N",
        "smarts": ["[nH+]"],
    },

    # S / P ACCEPTORS
    {
        "name": "thiocarbonyl_S_acceptor",
        "group": "thiocarbonyl",
        "kind": "acceptor",
        "element": "S",
        "smarts": ["[SX1]=[CX3]"],
    },
    {
        "name": "sulfoxide_O_acceptor",
        "group": "sulfoxide",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[SX3]([#6])[#6]"],
    },
    {
        "name": "sulfone_O_acceptor",
        "group": "sulfone",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[SX4](=[OX1])([#6])[#6]"],
    },
    {
        "name": "phosphine_oxide_O_acceptor",
        "group": "phosphine_oxide",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX1]=[PX4]([#6])([#6])[#6]"],
    },

    # BORONIC ACIDS
    {
        "name": "boronic_acid_OH_donor",
        "group": "boronic_acid",
        "kind": "donor",
        "element": "O",
        "smarts": ["[OX2H1][BX3]([OX2H])"],
    },
    {
        "name": "boronic_acid_O_acceptor",
        "group": "boronic_acid",
        "kind": "acceptor",
        "element": "O",
        "smarts": ["[OX2H1][BX3]([OX2H])"],
    },

    # HALOGEN-BOND-RELATED ROLES
    {
        "name": "iodo_arene_halogenbond_donor",
        "group": "iodo_arene",
        "kind": "halogenbond_donor",
        "element": "I",
        "smarts": ["[I][c]", "[I][C;!$(C([#6])[#6])]"],
    },
    {
        "name": "bromo_arene_halogenbond_donor",
        "group": "bromo_arene",
        "kind": "halogenbond_donor",
        "element": "Br",
        "smarts": ["[Br][c]"],
    },
    {
        "name": "aryl_F_contact_site",
        "group": "aryl_fluoride",
        "kind": "weak_acceptor",
        "element": "F",
        "smarts": ["[F][c]"],
    },
    {
        "name": "pyridine_N_halogen_acceptor",
        "group": "pyridine",
        "kind": "acceptor",
        "element": "N",
        "smarts": ["[n;+0;!H]"],
    },
]

DONOR_PRIORITY = [
    "carboxylic_acid_OH_donor",
    "amide_NH_donor",
    "sulfonamide_NH_donor",
    "amidino_amino_NH_donor",
    "guanidine_terminal_NH_donor",
    "aza_aromatic_NH_donor",
    "ammonium_NH_donor",
    "azolium_NH_donor",
    "amine_NH_donor",
    "alcohol_OH_donor",
    "phenol_OH_donor",
    "boronic_acid_OH_donor",
]

ACCEPTOR_PRIORITY = [
    "carboxylate_O_acceptor",
    "carboxylic_acid_carbonyl_O_acceptor",
    "amide_carbonyl_O_acceptor",
    "urea_carbonyl_O_acceptor",
    "carbamate_carbonyl_O_acceptor",
    "sulfone_O_acceptor",
    "sulfoxide_O_acceptor",
    "phosphine_oxide_O_acceptor",
    "thiocarbonyl_S_acceptor",
    "amidino_imine_N_acceptor",
    "guanidine_imine_N_acceptor",
    "aza_aromatic_N_acceptor",
    "imine_N_acceptor",
    "nitrile_N_acceptor",
    "amine_N_acceptor",
    "aldehyde_O_acceptor",
    "ketone_O_acceptor",
    "ester_carbonyl_O_acceptor",
    "ester_alkoxy_O_acceptor",
    "ether_O_acceptor",
    "alcohol_O_acceptor",
    "phenol_O_acceptor",
    "N_oxide_O_acceptor",
    "pyridine_N_halogen_acceptor",
]

ROLE_PRIORITY = DONOR_PRIORITY + ACCEPTOR_PRIORITY

ROLE_KIND_COMPATIBILITY = {
    "donor": {"acceptor", "weak_acceptor"},
    "acceptor": {"donor", "halogenbond_donor"},
    "weak_acceptor": {"donor", "halogenbond_donor"},
    "halogenbond_donor": {"acceptor", "weak_acceptor"},
}

HALOGEN_DONORS = {"Cl", "Br", "I"}
HALOGEN_ACCEPTORS = {"N", "O", "S"}

@dataclass
class AtomNode:
    idx: int
    label: str
    element: str
    pos: np.ndarray
    neighbors: List[int]
    is_donor: bool = False
    is_acceptor: bool = False
    is_halogen_donor: bool = False
    is_aromatic: bool = False
    group: str = ""
    role: str = ""
    role_element: str = ""
    role_kind: str = ""
    mol_id: int = -1
    asym_id: str = ""
    serial: int = 0
    ring_ids: List[int] = field(default_factory=list)

@dataclass
class Synthon:
    type: str
    donor_group: str
    acceptor_group: str
    donor_atom: str
    acceptor_atom: str
    distance: float
    angle: Optional[float] = None
    has_explicit_h: bool = False
    motif: str = ""
    donor_role: str = ""
    acceptor_role: str = ""
    acceptor_symmetry: str = ""

@dataclass
class HalogenBond:
    type: str = "halogen_bond"
    donor_atom: str = ""
    donor_element: str = ""
    donor_role: str = ""
    acceptor_atom: str = ""
    acceptor_element: str = ""
    acceptor_role: str = ""
    distance: float = 0.0
    angle_c_x_a: Optional[float] = None
    motif: str = ""
    mol_donor: int = -1
    mol_acceptor: int = -1

@dataclass
class PiStacking:
    type: str = "pi_stacking"
    ring1_centroid: np.ndarray = None
    ring2_centroid: np.ndarray = None
    ring1_atoms: List[str] = None
    ring2_atoms: List[str] = None
    distance: float = 0.0
    offset: float = 0.0
    angle: float = 0.0
    motif: str = ""
    mol1: int = -1
    mol2: int = -1

@dataclass
class CHPiInteraction:
    type: str = "ch_pi"
    donor_atom: str = ""
    donor_h: str = ""
    acceptor_ring: List[str] = field(default_factory=list)
    distance: float = 0.0
    angle: float = 0.0
    motif: str = ""
    mol_donor: int = -1
    mol_acceptor: int = -1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cif")
    p.add_argument("--json", action="store_true")
    return p.parse_args()

def expand_atoms(ss):
    atoms = []
    ops = list(ss.spacegroup.operations())
    seen = set()
    
    for site in ss.sites:
        occ_threshold = 0.3 if site.element.name == "H" else 0.5
        if site.occ < occ_threshold:
            continue
            
        f0 = [site.fract.x, site.fract.y, site.fract.z]
        
        for op_idx, op in enumerate(ops):
            f = np.array(op.apply_to_xyz(f0))
            f = f - np.floor(f)
            key = (site.label, op_idx, tuple(np.round(f, 4)))
            
            if key in seen:
                continue
            seen.add(key)
            
            pos = ss.cell.orthogonalize(gemmi.Fractional(*f))
            
            atoms.append(AtomNode(
                idx=len(atoms),
                label=site.label,
                element=site.element.name,
                pos=np.array([pos.x, pos.y, pos.z]),
                neighbors=[],
                asym_id="", 
                serial=0,
                ring_ids=[]
            ))
    return atoms

SMARTS_CACHE = {}

def get_smarts_pattern(smarts_str):
    if smarts_str not in SMARTS_CACHE:
        try:
            pattern = Chem.MolFromSmarts(smarts_str)
            if pattern is None:
                print(f"  Warning: Invalid template SMARTS: {smarts_str}")
                SMARTS_CACHE[smarts_str] = None
            else:
                SMARTS_CACHE[smarts_str] = pattern
        except Exception as e:
            print(f"  Compilation error SMARTS '{smarts_str}': {e}")
            SMARTS_CACHE[smarts_str] = None
    return SMARTS_CACHE[smarts_str]

def fix_mol_ids_after_expansion(atoms):
    g = nx.Graph()
    for a in atoms:
        for n in a.neighbors:
            g.add_edge(a.idx, n)
    
    components = []
    for comp in nx.connected_components(g):
        comp_list = list(comp)
        components.append(comp_list)
    
    for mol_id, comp in enumerate(components):
        for idx in comp:
            atoms[idx].mol_id = mol_id
    
    return components

def build_graph(ss, atoms):
    g = nx.Graph()
    g.add_nodes_from(a.idx for a in atoms)
    
    radii = {
        "H": 0.37, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71,
        "S": 1.03, "P": 1.06, "Cl": 0.99, "Br": 1.14, "I": 1.33
    }
    
    bond_factor = 1.4
    n_atoms = len(atoms)
    bond_count = 0
    
    for i in range(n_atoms):
        a = atoms[i]
        for j in range(i + 1, n_atoms):
            b = atoms[j]
            dist = np.linalg.norm(a.pos - b.pos)
            
            if dist > 2.2:
                continue
            
            if (a.element == "C" and b.element == "F") or (a.element == "F" and b.element == "C"):
                if dist < 1.6:
                    g.add_edge(i, j)
                    a.neighbors.append(j)
                    b.neighbors.append(i)
                    bond_count += 1
                continue
            
            if a.element == "C" and b.element == "C":
                if dist < 1.8:
                    g.add_edge(i, j)
                    a.neighbors.append(j)
                    b.neighbors.append(i)
                    bond_count += 1
                continue
            
            r1 = radii.get(a.element, 1.0)
            r2 = radii.get(b.element, 1.0)
            threshold = bond_factor * (r1 + r2)
            
            if a.element == "F" or b.element == "F":
                threshold = 1.6
            
            if dist < threshold and dist > 0.4:
                g.add_edge(i, j)
                a.neighbors.append(j)
                b.neighbors.append(i)
                bond_count += 1
    
    print(f"A graph is constructed with {bond_count} edges")
    return g

def graph_to_rdkit_mol(atoms, component):
    by_idx = {a.idx: a for a in atoms}
    rdmol = Chem.RWMol()
    idx_map = {}
    
    sorted_indices = sorted(component)
    
    for orig_idx in sorted_indices:
        atom = by_idx[orig_idx]
        rd_atom = Chem.Atom(atom.element)
        idx_map[orig_idx] = rdmol.AddAtom(rd_atom)
    
    for orig_idx in sorted_indices:
        atom = by_idx[orig_idx]
        for nbr in by_idx[orig_idx].neighbors:
            if nbr in component and orig_idx < nbr:
                dist = np.linalg.norm(atom.pos - by_idx[nbr].pos)
                bond_type = Chem.BondType.SINGLE
                
                if (atom.element == 'C' and by_idx[nbr].element == 'N') or \
                   (atom.element == 'N' and by_idx[nbr].element == 'C'):
                    if dist < 1.2:
                        bond_type = Chem.BondType.TRIPLE
                    elif dist < 1.35:
                        bond_type = Chem.BondType.DOUBLE
                elif atom.element == 'C' and by_idx[nbr].element == 'C':
                    if dist < 1.25:
                        bond_type = Chem.BondType.TRIPLE
                    elif dist < 1.45:
                        bond_type = Chem.BondType.DOUBLE
                elif (atom.element == 'C' and by_idx[nbr].element == 'O') or \
                     (atom.element == 'O' and by_idx[nbr].element == 'C'):
                    if dist < 1.3:
                        bond_type = Chem.BondType.DOUBLE
                elif atom.element == 'N' and by_idx[nbr].element == 'N':
                    if dist < 1.3:
                        bond_type = Chem.BondType.DOUBLE
                    elif dist < 1.2:
                        bond_type = Chem.BondType.TRIPLE
                
                rdmol.AddBond(idx_map[orig_idx], idx_map[nbr], bond_type)
    
    mol = rdmol.GetMol()
    
    try:
        Chem.SanitizeMol(mol, 
                       sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                  Chem.SanitizeFlags.SANITIZE_ADJUSTHS |
                                  Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
                       catchErrors=True)
    except:
        pass
    
    mol = Chem.AddHs(mol)
    
    return mol, sorted_indices

def build_role_index(atom_role_rules):
    role_by_name = {rule["name"]: rule for rule in atom_role_rules}
    priority_rank = {name: i for i, name in enumerate(ROLE_PRIORITY)}
    return role_by_name, priority_rank

def choose_primary_role(role_names, priority_rank, atom_element, atom_role_rules):
    if not role_names:
        return None
    
    rule_by_name = {rule["name"]: rule for rule in atom_role_rules}
    
    donor_roles = []
    acceptor_roles = []
    
    for role in role_names:
        rule = rule_by_name.get(role)
        if rule:
            if rule["kind"] == "donor":
                donor_roles.append(role)
            elif rule["kind"] == "acceptor":
                acceptor_roles.append(role)
    
    if atom_element == "N":
        if donor_roles:
            return min(donor_roles, key=lambda x: priority_rank.get(x, 10**9))
        elif acceptor_roles:
            return min(acceptor_roles, key=lambda x: priority_rank.get(x, 10**9))
    
    if donor_roles:
        return min(donor_roles, key=lambda x: priority_rank.get(x, 10**9))
    elif acceptor_roles:
        return min(acceptor_roles, key=lambda x: priority_rank.get(x, 10**9))
    
    return None

def classify_atoms_with_roles(rdmol, atoms, sorted_indices, atom_role_rules, priority_rank):
    by_idx = {a.idx: a for a in atoms}
    atom_roles = {orig_idx: [] for orig_idx in sorted_indices}
    
    try:
        h_counts = {}
        for i, rd_idx in enumerate(range(rdmol.GetNumAtoms())):
            atom = rdmol.GetAtomWithIdx(rd_idx)
            h_counts[sorted_indices[rd_idx]] = atom.GetTotalNumHs()
    except:
        h_counts = {}
    
    for rule in atom_role_rules:
        for smarts_str in rule["smarts"]:
            pattern = get_smarts_pattern(smarts_str)
            if pattern is None:
                continue
            
            try:
                matches = rdmol.GetSubstructMatches(pattern)
                for match in matches:
                    if match:
                        rd_idx = match[0]
                        if rd_idx < len(sorted_indices):
                            orig_idx = sorted_indices[rd_idx]
                            atom = by_idx[orig_idx]
                            
                            if atom.element == rule["element"]:
                                if rule["kind"] == "donor" and atom.element == "N":
                                    if h_counts.get(orig_idx, 0) == 0:
                                        has_h = any(atoms[n].element == "H" for n in atom.neighbors)
                                        if not has_h:
                                            continue
                                atom_roles[orig_idx].append(rule["name"])
            except Exception as e:
                pass
    
    for orig_idx, roles in atom_roles.items():
        if roles:
            unique_roles = list(set(roles))
            atom = by_idx[orig_idx]
            primary_role = choose_primary_role(unique_roles, priority_rank, atom.element, atom_role_rules)
            if primary_role:
                rule = next(r for r in atom_role_rules if r["name"] == primary_role)
                atom.group = rule["group"]
                atom.role = primary_role
                atom.role_element = rule["element"]
                atom.role_kind = rule["kind"]
                
                if rule["kind"] == "donor":
                    atom.is_donor = True
                elif rule["kind"] == "acceptor":
                    atom.is_acceptor = True
                elif rule["kind"] == "halogenbond_donor":
                    atom.is_halogen_donor = True
                elif rule["kind"] == "weak_acceptor":
                    atom.is_acceptor = True
    
    for orig_idx in sorted_indices:
        atom = by_idx[orig_idx]
        if atom.role:
            continue
        
        if atom.element == 'O':
            has_h = any(atoms[n].element == 'H' for n in atom.neighbors)
            if has_h:
                has_c = any(atoms[n].element == 'C' for n in atom.neighbors)
                if has_c:
                    is_carboxylic = False
                    for n_idx in atom.neighbors:
                        nbr = atoms[n_idx]
                        if nbr.element == 'C':
                            for nn_idx in nbr.neighbors:
                                if nn_idx != orig_idx and atoms[nn_idx].element == 'O':
                                    is_carboxylic = True
                                    break
                        if is_carboxylic:
                            break
                    
                    if is_carboxylic:
                        atom.is_donor = True
                        atom.group = "carboxylic_acid"
                        atom.role = "carboxylic_acid_OH_donor"
                        atom.role_kind = "donor"
                    else:
                        atom.is_donor = True
                        atom.group = "alcohol"
                        atom.role = "alcohol_OH_donor"
                        atom.role_kind = "donor"
        
        elif atom.element == 'O' and not atom.role:
            is_carbonyl = False
            for n_idx in atom.neighbors:
                nbr = atoms[n_idx]
                if nbr.element == 'C':
                    dist = np.linalg.norm(atom.pos - nbr.pos)
                    if dist < 1.3:  
                        is_carbonyl = True
                        break
            
            if is_carbonyl:
                is_carboxylic_carbonyl = False
                for n_idx in atom.neighbors:
                    nbr = atoms[n_idx]
                    if nbr.element == 'C':
                        for nn_idx in nbr.neighbors:
                            if atoms[nn_idx].element == 'O' and atoms[nn_idx] != atom:
                                if any(atoms[hh].element == 'H' for hh in atoms[nn_idx].neighbors):
                                    is_carboxylic_carbonyl = True
                                    break
                        if is_carboxylic_carbonyl:
                            break
                
                if is_carboxylic_carbonyl:
                    atom.is_acceptor = True
                    atom.group = "carboxylic_acid"
                    atom.role = "carboxylic_acid_carbonyl_O_acceptor"
                    atom.role_kind = "acceptor"
                else:
                    atom.is_acceptor = True
                    atom.group = "amide"
                    atom.role = "amide_carbonyl_O_acceptor"
                    atom.role_kind = "acceptor"
        
        elif atom.element == 'N' and not atom.role:
            has_h = any(atoms[n].element == 'H' for n in atom.neighbors)
            if has_h:
                atom.is_donor = True
                atom.group = "neutral_amine"
                atom.role = "amine_NH_donor"
                atom.role_kind = "donor"
    
    return atoms

def detect_aromatic_rings(atoms, components):
    by_idx = {a.idx: a for a in atoms}
    ring_counter = 0
    
    for component in components:
        rdmol, sorted_indices = graph_to_rdkit_mol(atoms, component)
        if rdmol is None:
            continue
        
        try:
            ri = rdmol.GetRingInfo()
            atom_rings = ri.AtomRings()
            
            for ring in atom_rings:
                if len(ring) >= 5:
                    ring_id = ring_counter
                    ring_counter += 1
                    
                    is_aromatic = True
                    for rd_idx in ring:
                        atom = by_idx[sorted_indices[rd_idx]]
                        if atom.element not in ["C", "N", "O"]:
                            is_aromatic = False
                            break
                    
                    if is_aromatic:
                        for rd_idx in ring:
                            atom = by_idx[sorted_indices[rd_idx]]
                            atom.is_aromatic = True
                            atom.ring_ids.append(ring_id)
        except:
            continue
    
    return atoms

def read_hbonds_from_cif(cif_file, atoms=None):
    hbonds = []
    try:
        with open(cif_file, 'r') as f:
            lines = f.readlines()
        
        in_hbond_loop = False
        data_start = -1
        col_indices = {}
        
        for i, line in enumerate(lines):
            if 'loop_' in line:
                in_hbond_loop = False
                col_indices = {}
            
            if '_geom_hbond' in line and not in_hbond_loop:
                in_hbond_loop = True
                j = i
                col_idx = 0
                while j < len(lines) and lines[j].startswith('_geom_hbond'):
                    col_name = lines[j].strip().lower()
                    if 'label_d' in col_name or 'donor' in col_name:
                        col_indices['D'] = col_idx
                    elif 'label_h' in col_name or 'donor_h' in col_name:
                        col_indices['H'] = col_idx
                    elif 'label_a' in col_name or 'acceptor' in col_name:
                        col_indices['A'] = col_idx
                    elif 'distance_da' in col_name or 'da' in col_name:
                        col_indices['DA'] = col_idx
                    elif 'angle_dha' in col_name or 'dha' in col_name:
                        col_indices['DHA'] = col_idx
                    elif 'site_symmetry_a' in col_name or 'symmetry' in col_name:
                        col_indices['sym'] = col_idx
                    col_idx += 1
                    j += 1
                data_start = j
                break
        
        if data_start > 0 and col_indices:
            atom_dict = {a.label: a for a in atoms} if atoms else {}
            
            for i in range(data_start, min(data_start+200, len(lines))):
                line = lines[i].strip()
                if not line or line.startswith('PROBLEM') or line.startswith('RESPONSE'):
                    continue
                if line and not line.startswith('_') and not line.startswith('loop'):
                    parts = line.split()
                    
                    if len(parts) < 4:
                        continue
                    
                    d_idx = col_indices.get('D', -1)
                    a_idx = col_indices.get('A', -1)
                    da_idx = col_indices.get('DA', -1)
                    dha_idx = col_indices.get('DHA', -1)
                    
                    sym_idx = col_indices.get('sym', -1)
                    acceptor_sym = ""
                    if sym_idx >= 0 and sym_idx < len(parts):
                        acceptor_sym = parts[sym_idx]
                        
                    if d_idx < 0 or a_idx < 0 or da_idx < 0:
                        continue
                    if d_idx >= len(parts) or a_idx >= len(parts) or da_idx >= len(parts):
                        continue
                    
                    donor = parts[d_idx]
                    acceptor = parts[a_idx]
                    
                    if donor[0].isdigit() or acceptor[0].isdigit():
                        continue
                    
                    dist_str = parts[da_idx]
                    if dist_str == '.' or dist_str == '?' or dist_str == 'PROBLEM':
                        continue
                    try:
                        dist_da = float(dist_str.split('(')[0])
                    except ValueError:
                        continue
                    
                    angle = 0.0
                    if dha_idx >= 0 and dha_idx < len(parts):
                        angle_str = parts[dha_idx]
                        if angle_str != '.' and angle_str != '?':
                            try:
                                angle = float(angle_str.split('(')[0])
                            except ValueError:
                                pass
                    
                    donor_group = atom_dict[donor].group if donor in atom_dict else ""
                    acceptor_group = atom_dict[acceptor].group if acceptor in atom_dict else ""
                    donor_role = atom_dict[donor].role if donor in atom_dict else ""
                    acceptor_role = atom_dict[acceptor].role if acceptor in atom_dict else ""
                    
                    hbonds.append(Synthon(
                        type="hydrogen_bond",
                        donor_group=donor_group,
                        acceptor_group=acceptor_group,
                        donor_atom=donor,
                        acceptor_atom=acceptor,
                        distance=dist_da,
                        angle=angle,
                        has_explicit_h=True,
                        motif="",
                        donor_role=donor_role,
                        acceptor_role=acceptor_role,
                        acceptor_symmetry=acceptor_sym
                    ))
                    print(f"  Read H-bond: {donor} -> {acceptor}, Distance={dist_da:.3f}Å")
            
            print(f"  Total read H-bonds: {len(hbonds)}")
    
    except Exception as e:
        print(f"Couldn't read H-bonds: {e}")
        import traceback
        traceback.print_exc()
    
    return hbonds

def find_hbonds_by_geometry(atoms, max_distance=3.2, min_angle=120):
    hbonds = []
    
    hydrogen_bonds = []  
    for h in atoms:
        if h.element != "H":
            continue
        donor = None
        for n_idx in h.neighbors:
            neighbor = atoms[n_idx]
            if neighbor.element in ["O", "N"]:
                donor = neighbor
                break
        if donor:
            hydrogen_bonds.append((donor, h, donor.element))
    
    acceptors = [a for a in atoms if a.element in ["O", "N"]]
    
    print(f"\nH-bonds geometry search (fallback)")
    print(f"Found {len(hydrogen_bonds)} donor-H pairs")
    
    for donor, h_atom, donor_elem in hydrogen_bonds:
        for acceptor in acceptors:
            if acceptor.idx == donor.idx:
                continue
            if donor.mol_id == acceptor.mol_id:
                continue
            
            h_a_dist = np.linalg.norm(h_atom.pos - acceptor.pos)
            if h_a_dist > max_distance or h_a_dist < 1.2:
                continue
            
            d_h_vec = donor.pos - h_atom.pos
            h_a_vec = acceptor.pos - h_atom.pos
            norm_dh = np.linalg.norm(d_h_vec)
            norm_ha = np.linalg.norm(h_a_vec)
            
            if norm_dh > 1e-6 and norm_ha > 1e-6:
                cos_angle = np.dot(d_h_vec, h_a_vec) / (norm_dh * norm_ha)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                if angle > min_angle:
                    motif = f"{donor_elem}-H···{acceptor.element}"
                    hbonds.append(Synthon(
                        type="hydrogen_bond",
                        donor_group=donor.group if donor.group else donor_elem,
                        acceptor_group=acceptor.group if acceptor.group else acceptor.element,
                        donor_atom=donor.label,
                        acceptor_atom=acceptor.label,
                        distance=h_a_dist,
                        angle=angle,
                        has_explicit_h=True,
                        motif=motif,
                        donor_role=donor.role,
                        acceptor_role=acceptor.role
                    ))
                    print(f"  Found: {donor.label}-H···{acceptor.label} = {h_a_dist:.3f}Å, Angle={angle:.1f}°")
    
    return hbonds

def filter_intermolecular_hbonds(hbonds, atoms):
    by_label = {a.label: a for a in atoms}
    intermolecular = []
    
    identity_sym_codes = {'.', '1_555', '1_655', '1_545', '1_565', '1_545'}
    
    for hbond in hbonds:
        if hbond.acceptor_symmetry and hbond.acceptor_symmetry not in identity_sym_codes:
            intermolecular.append(hbond)
            print(f"  Intermolecular (by symmetry {hbond.acceptor_symmetry}): {hbond.donor_atom} -> {hbond.acceptor_atom}")
            continue
        
        donor_atom = by_label.get(hbond.donor_atom)
        acceptor_atom = by_label.get(hbond.acceptor_atom)
        
        if donor_atom and acceptor_atom:
            if donor_atom.mol_id != acceptor_atom.mol_id:
                intermolecular.append(hbond)
                print(f"  Intermolecular: {hbond.donor_atom}(mol{donor_atom.mol_id}) -> {hbond.acceptor_atom}(mol{acceptor_atom.mol_id})")
            else:
                print(f"  Intramolecular: {hbond.donor_atom}(mol{donor_atom.mol_id}) -> {hbond.acceptor_atom}(mol{acceptor_atom.mol_id})")
        else:
            if hbond.acceptor_symmetry:
                print(f"  Intermolecular (acceptor with symmetry {hbond.acceptor_symmetry}): {hbond.donor_atom} -> {hbond.acceptor_atom}")
                intermolecular.append(hbond)
            else:
                print(f"  Couldn't verify: {hbond.donor_atom} -> {hbond.acceptor_atom}")
    
    return intermolecular

def identify_synthon_motif_from_roles(donor_role: str, acceptor_role: str, donor_group: str, acceptor_group: str, 
                                       donor_atom_label: str, acceptor_atom_label: str, atoms) -> str:
    by_label = {a.label: a for a in atoms}
    donor_atom = by_label.get(donor_atom_label)
    acceptor_atom = by_label.get(acceptor_atom_label)
    
    if not donor_atom or not acceptor_atom:
        return "Hydrogen bond"
    
    donor_role_name = donor_role if donor_role else donor_atom.role
    acceptor_role_name = acceptor_role if acceptor_role else acceptor_atom.role
    
    group_display = {
        "carboxylic_acid": "Carboxylic acid",
        "carboxylate": "Carboxylate",
        "alcohol": "Alcohol",
        "phenol": "Phenol",
        "ether": "Ether",
        "aldehyde": "Aldehyde",
        "ketone": "Ketone",
        "ester": "Ester",
        "amide": "Amide",
        "urea_like": "Urea",
        "carbamate": "Carbamate",
        "sulfonamide": "Sulfonamide",
        "neutral_amine": "Amine",
        "imine": "Imine",
        "amidine": "Amidine",
        "guanidine_like": "Guanidine",
        "aza_aromatic_N": "Aromatic N",
        "aza_aromatic_NH": "Aromatic NH",
        "N_oxide": "N-oxide",
        "nitrile": "Nitrile",
        "cationic_NH": "Cationic NH",
        "thiocarbonyl": "Thiocarbonyl",
        "sulfoxide": "Sulfoxide",
        "sulfone": "Sulfone",
        "phosphine_oxide": "Phosphine oxide",
        "boronic_acid": "Boronic acid",
    }
    
    donor_group_name = donor_group if donor_group else donor_atom.group
    acceptor_group_name = acceptor_group if acceptor_group else acceptor_atom.group
    
    donor_group_display = group_display.get(donor_group_name, donor_group_name.replace('_', ' ').title() if donor_group_name else "Unknown")
    acceptor_group_display = group_display.get(acceptor_group_name, acceptor_group_name.replace('_', ' ').title() if acceptor_group_name else "Unknown")
    
    donor_elem = donor_atom.element
    acceptor_elem = acceptor_atom.element
    
    if donor_elem == "O" and acceptor_elem == "N":
        bond_type = "O−H···N"
    elif donor_elem == "N" and acceptor_elem == "O":
        bond_type = "N−H···O"
    elif donor_elem == "O" and acceptor_elem == "O":
        bond_type = "O−H···O"
    elif donor_elem == "N" and acceptor_elem == "N":
        bond_type = "N−H···N"
    elif donor_elem == "S" and acceptor_elem in ["N", "O"]:
        bond_type = f"{donor_elem}−H···{acceptor_elem}"
    else:
        bond_type = f"{donor_elem}−H···{acceptor_elem}"
    
    if donor_group_display == "Unknown" and acceptor_group_display != "Unknown":
        motif = f"{donor_atom_label} -> {acceptor_group_display} hydrogen bond ({bond_type})"
    elif acceptor_group_display == "Unknown" and donor_group_display != "Unknown":
        motif = f"{donor_group_display} -> {acceptor_atom_label} hydrogen bond ({bond_type})"
    else:
        motif = f"{donor_group_display} -> {acceptor_group_display} hydrogen bond ({bond_type})"
    
    return motif

def separate_hbonds_by_type(hbonds, atoms):
    by_label = {a.label: a for a in atoms}
    
    acid_base = []
    base_base = []
    
    for hbond in hbonds:
        donor_atom = by_label.get(hbond.donor_atom)
        acceptor_atom = by_label.get(hbond.acceptor_atom)
        
        if donor_atom and acceptor_atom:
            if (donor_atom.element == "O" and acceptor_atom.element == "N"):
                acid_base.append(hbond)
                print(f"  Acid-base (O-H···N): {hbond.donor_atom} -> {hbond.acceptor_atom}")
            elif (donor_atom.element == "N" and acceptor_atom.element == "O"):
                acid_base.append(hbond)
                print(f"  Acid-base (N-H···O): {hbond.donor_atom} -> {hbond.acceptor_atom}")
            else:
                base_base.append(hbond)
                print(f"  Base-base/Other: {hbond.donor_atom}({donor_atom.element}) -> {hbond.acceptor_atom}({acceptor_atom.element})")
    
    return acid_base, base_base

def find_halogen_bonds(atoms, max_distance=3.8):
    halogen_bonds = []
    seen_pairs = set()
    
    halogen_donors = [a for a in atoms if a.is_halogen_donor]
    
    if not halogen_donors:
        halogen_donors = [a for a in atoms if a.element in ["I", "Br", "Cl"]]
        print(f"  No halogen donors are detected using SMARTS, we use all I/Br/Cl atoms: {len(halogen_donors)}")
    
    halogen_acceptors = [a for a in atoms if a.element in HALOGEN_ACCEPTORS]
    
    print(f"\nSearch for I···N halogen bonds")
    print(f"Found {len(halogen_donors)} halogen donors, {len(halogen_acceptors)} potential acceptors")
    
    for donor in halogen_donors:
        bonded_atom = None
        for n_idx in donor.neighbors:
            neighbor = atoms[n_idx]
            if neighbor.element in ["C", "S", "P", "N"]:
                dist_to_neighbor = np.linalg.norm(donor.pos - neighbor.pos)
                if dist_to_neighbor < 2.3:
                    bonded_atom = neighbor
                    break
        
        if not bonded_atom:
            continue
        
        bond_vec = bonded_atom.pos - donor.pos
        bond_len = np.linalg.norm(bond_vec)
        
        if bond_len < 0.1:
            continue
        
        for acceptor in halogen_acceptors:
            if donor.mol_id == acceptor.mol_id:
                continue
            
            pair_key = tuple(sorted([donor.label, acceptor.label]))
            if pair_key in seen_pairs:
                continue
            
            dist = np.linalg.norm(donor.pos - acceptor.pos)
            if dist > max_distance or dist < 2.5:
                continue
            
            donor_to_acceptor = acceptor.pos - donor.pos
            acceptor_len = np.linalg.norm(donor_to_acceptor)
            
            cos_angle = np.dot(bond_vec, donor_to_acceptor) / (bond_len * acceptor_len)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            if angle > 150 and dist < max_distance:
                donor_role = donor.role if donor.role else f"{donor.element}_halogen_donor"
                acceptor_role = acceptor.role if acceptor.role else f"{acceptor.element}_acceptor"
                
                temp_hb = HalogenBond(
                    donor_element=donor.element,
                    acceptor_element=acceptor.element,
                    angle_c_x_a=angle
                )
                motif = identify_halogen_bond_motif(temp_hb)
                
                halogen_bonds.append(HalogenBond(
                    donor_atom=donor.label,
                    donor_element=donor.element,
                    donor_role=donor_role,
                    acceptor_atom=acceptor.label,
                    acceptor_element=acceptor.element,
                    acceptor_role=acceptor_role,
                    distance=dist,
                    angle_c_x_a=angle,
                    motif=motif,
                    mol_donor=donor.mol_id,
                    mol_acceptor=acceptor.mol_id
                ))
                seen_pairs.add(pair_key)
                print(f"    Found: {motif}")
    
    print(f"\nTotal number of halogen bonds detected: {len(halogen_bonds)}")
    return halogen_bonds

def identify_halogen_bond_motif(hb: HalogenBond) -> str:
    donor_name = hb.donor_role if hb.donor_role else hb.donor_element
    acceptor_name = hb.acceptor_role if hb.acceptor_role else hb.acceptor_element
    
    donor_name = donor_name.replace('_halogenbond_donor', '').replace('_', ' ').title()
    acceptor_name = acceptor_name.replace('_acceptor', '').replace('_', ' ').title()
    
    return f"Halogen bond {donor_name} -> {acceptor_name}"

def get_halogen_bond_display_string(hb: HalogenBond) -> str:
    donor_name = hb.donor_role if hb.donor_role else hb.donor_element
    acceptor_name = hb.acceptor_role if hb.acceptor_role else hb.acceptor_element
    
    donor_name = donor_name.replace('_halogenbond_donor', '').replace('_', ' ').title()
    acceptor_name = acceptor_name.replace('_acceptor', '').replace('_', ' ').title()
    
    return f"Halogen bond {donor_name} -> {acceptor_name} (C-{hb.donor_element}···{hb.acceptor_element} = {hb.angle_c_x_a:.1f}°)"

def separate_halogen_bonds_by_type(halogen_bonds, atoms):
    strong_donors = {"I", "Br"}
    good_acceptors = {"N", "O", "S"}
    
    acid_base = []
    base_base = []
    
    for hb in halogen_bonds:
        if hb.donor_element in strong_donors and hb.acceptor_element in good_acceptors:
            acid_base.append(hb)
        else:
            base_base.append(hb)
    
    return acid_base, base_base

def find_rings_by_geometry(atoms, component):
    rings = []
    
    if len(component) < 6:
        return rings
    
    G = nx.Graph()
    for idx in component:
        atom = atoms[idx]
        for nbr in atom.neighbors:
            if nbr in component:
                G.add_edge(idx, nbr)
    
    try:
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            if len(cycle) != 6:
                continue
            
            cycle_atoms = [atoms[i] for i in cycle]
            if not all(a.element == 'C' for a in cycle_atoms):
                continue
            
            bond_lengths = []
            for k in range(len(cycle)):
                a1 = atoms[cycle[k]]
                a2 = atoms[cycle[(k+1) % len(cycle)]]
                bond_lengths.append(np.linalg.norm(a1.pos - a2.pos))
            
            avg_bond = np.mean(bond_lengths)
            
            if 1.35 < avg_bond < 1.45:
                rings.append(cycle)
    except:
        pass
    
    return rings

def find_all_rings_robust(atoms, components):
    all_rings = []
    ring_counter = 0
    
    for comp_idx, comp in enumerate(components):
        if len(comp) < 5:
            continue
        
        rdmol, sorted_indices = graph_to_rdkit_mol(atoms, comp)
        rdkit_rings_found = False
        
        if rdmol is not None and rdmol.GetNumAtoms() >= 5:
            try:
                Chem.SanitizeMol(rdmol, 
                               Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                               Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            except:
                pass
            
            ri = rdmol.GetRingInfo()
            
            for ring in ri.AtomRings():
                if len(ring) not in [5, 6, 7]:
                    continue
                
                ring_atoms_idx = [sorted_indices[rd_idx] for rd_idx in ring]
                ring_atoms = [atoms[idx] for idx in ring_atoms_idx]
                
                if not all(a.element in ['C', 'N', 'O', 'S'] for a in ring_atoms):
                    continue
                
                rdkit_rings_found = True
                all_rings.append(create_ring_dict(ring_atoms, ring_counter, atoms))
                ring_counter += 1
        
        if not rdkit_rings_found and len(comp) >= 6:
            geo_rings = find_rings_by_geometry(atoms, comp)
            for ring in geo_rings:
                ring_atoms = [atoms[i] for i in ring]
                all_rings.append(create_ring_dict(ring_atoms, ring_counter, atoms))
                ring_counter += 1
    
    unique_rings = []
    seen_centroids = []
    
    for ring in all_rings:
        is_unique = True
        for seen in seen_centroids:
            if np.linalg.norm(ring['centroid'] - seen) < 0.5:
                is_unique = False
                break
        if is_unique:
            seen_centroids.append(ring['centroid'])
            unique_rings.append(ring)
    
    return unique_rings

def create_ring_dict(ring_atoms, ring_id, atoms):
    centroid = np.mean([a.pos for a in ring_atoms], axis=0)
    
    normal = None
    if len(ring_atoms) >= 3:
        try:
            p1, p2, p3 = ring_atoms[0].pos, ring_atoms[1].pos, ring_atoms[2].pos
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal = normal / norm
        except:
            pass
    
    is_carbocyclic = all(a.element == 'C' for a in ring_atoms)
    has_fluorine = any(any(atoms[n].element == 'F' for n in a.neighbors) for a in ring_atoms)
    
    if has_fluorine:
        ring_type = "fluoroarene"
    elif is_carbocyclic:
        ring_type = "carbocyclic_aromatic"
    else:
        ring_type = "heteroaromatic"
    
    return {
        'id': ring_id,
        'atoms': ring_atoms,
        'atom_indices': [a.idx for a in ring_atoms],
        'centroid': centroid,
        'normal': normal,
        'mol_id': ring_atoms[0].mol_id,
        'type': ring_type,
        'size': len(ring_atoms),
        'is_aromatic': True,
        'has_fluorine': has_fluorine
    }

def find_pi_stacking(atoms, components, max_distance=4.5):
    pi_stackings = []
    
    print(f"\nSearch π-stacking interactions")
    
    all_rings = find_all_rings_robust(atoms, components)
    print(f"Total found {len(all_rings)} rings")
    
    if len(all_rings) < 2:
        print("Not enough rings for π-stacking")
        return pi_stackings
    
    for ring in all_rings:
        print(f"  Ring {ring['id']}: mol={ring['mol_id']}, тип={ring['type']}, размер={ring['size']}, aromatic={ring['is_aromatic']}")
    
    for i in range(len(all_rings)):
        for j in range(i + 1, len(all_rings)):
            ring1 = all_rings[i]
            ring2 = all_rings[j]
            
            if ring1['mol_id'] == ring2['mol_id']:
                continue
            
            dist = np.linalg.norm(ring1['centroid'] - ring2['centroid'])
            
            if dist > max_distance:
                continue
            
            angle = 90.0
            if ring1['normal'] is not None and ring2['normal'] is not None:
                cos_angle = np.clip(np.dot(ring1['normal'], ring2['normal']), -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                if angle > 90:
                    angle = 180 - angle
            
            if angle < 30:  
                if dist < 3.8:
                    motif_type = "face_to_face"
                    motif = f"Face-to-face π-stacking ({ring1['type']} <-> {ring2['type']}, Distance={dist:.2f}Å)"
                else:
                    motif_type = "offset"
                    motif = f"Offset π-stacking ({ring1['type']} <-> {ring2['type']}, Distance={dist:.2f}Å)"
            else:  
                motif_type = "t_shaped"
                motif = f"T-shaped π-interaction ({ring1['type']} <-> {ring2['type']}, Angle={angle:.1f}°, Distance={dist:.2f}Å)"
            
            if ring1['has_fluorine'] and ring2['is_aromatic'] and not ring2['has_fluorine']:
                motif = f"π-hole···π stacking: fluoroarene -> arene (Distance={dist:.2f}Å)"
                motif_type = "pi_hole_pi"
            elif ring2['has_fluorine'] and ring1['is_aromatic'] and not ring1['has_fluorine']:
                motif = f"π-hole···π stacking: fluoroarene -> arene (Distance={dist:.2f}Å)"
                motif_type = "pi_hole_pi"
            
            pi_stackings.append(PiStacking(
                ring1_centroid=ring1['centroid'],
                ring2_centroid=ring2['centroid'],
                ring1_atoms=[a.label for a in ring1['atoms']],
                ring2_atoms=[a.label for a in ring2['atoms']],
                distance=dist,
                offset=0.0,
                angle=angle,
                motif=motif,
                mol1=ring1['mol_id'],
                mol2=ring2['mol_id']
            ))
            print(f"  Found: {motif}")
    
    pi_stackings.sort(key=lambda x: x.distance)
    print(f"\nFound {len(pi_stackings)} π-stacking interactions")
    return pi_stackings

def find_ch_pi_interactions(atoms, all_rings, max_distance=3.5):
    interactions = []
    
    print(f"\nSearch C-H···π interactions ===")
    
    ch_donors = []
    for a in atoms:
        if a.element == 'C':
            has_h = any(atoms[n].element == 'H' for n in a.neighbors)
            if has_h:
                ch_donors.append(a)
    
    print(f"Found {len(ch_donors)} C-H donors")
    print(f"Found {len(all_rings)} rings for potential acceptance")
    
    for donor in ch_donors:
        h_atom = None
        for n_idx in donor.neighbors:
            if atoms[n_idx].element == 'H':
                h_atom = atoms[n_idx]
                break
        
        if h_atom is None:
            continue
        
        for ring in all_rings:
            if donor.mol_id == ring['mol_id']:
                continue
            
            dist = np.linalg.norm(h_atom.pos - ring['centroid'])
            
            if dist > max_distance:
                continue
            
            c_h_vec = donor.pos - h_atom.pos
            h_centroid_vec = ring['centroid'] - h_atom.pos
            
            norm_ch = np.linalg.norm(c_h_vec)
            norm_hc = np.linalg.norm(h_centroid_vec)
            
            if norm_ch > 1e-6 and norm_hc > 1e-6:
                cos_angle = np.dot(c_h_vec, h_centroid_vec) / (norm_ch * norm_hc)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                if angle > 120:
                    motif = f"C-H···π interaction: {donor.label} -> {ring['type']} ring (Distance={dist:.2f}Å, Angle={angle:.1f}°)"
                    interactions.append({
                        'type': 'ch_pi',
                        'donor_atom': donor.label,
                        'donor_h': h_atom.label,
                        'acceptor_ring': [a.label for a in ring['atoms']],
                        'distance': dist,
                        'angle': angle,
                        'motif': motif,
                        'mol_donor': donor.mol_id,
                        'mol_acceptor': ring['mol_id']
                    })
                    print(f"  {motif}")
    
    print(f"Found {len(interactions)} C-H···π interactions")
    return interactions

def print_pi_stacking_summary(pi_stackings):
    if not pi_stackings:
        print(f"\nπ-STACKING INTERACTIONS SUMMARY")
        print(f"\nπ-stacking interactions not found")
        return
    
    print(f"\nπ-STACKING INTERACTIONS SUMMARY")
    
    for i, ps in enumerate(pi_stackings, 1):
        print(f"\n{i}. {ps.motif}")
        print(f"   Rings: {ps.ring1_atoms[:3]}... <-> {ps.ring2_atoms[:3]}...")
        print(f"   Distance: {ps.distance:.3f}Å")

def main():
    args = parse_args()
    
    INCLUDE_CH_PI = False  
    INCLUDE_PI_STACKING = True  

    try:
        ss = gemmi.read_small_structure(args.cif)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    sg_symbol = getattr(ss.spacegroup, 'symbol', str(ss.spacegroup))
    print(f"Space group: {sg_symbol}")
    
    atoms = expand_atoms(ss)
    print(f"Expanded atoms: {len(atoms)}")
    
    g = build_graph(ss, atoms)
    components = fix_mol_ids_after_expansion(atoms)
    
    print(f"Found {len(components)} molecules")
    
    role_by_name, priority_rank = build_role_index(ATOM_ROLE_RULES)
    
    for component in components:
        rdmol, sorted_indices = graph_to_rdkit_mol(atoms, component)
        if rdmol is None:
            continue
        atoms = classify_atoms_with_roles(rdmol, atoms, sorted_indices, ATOM_ROLE_RULES, priority_rank)

    atoms = detect_aromatic_rings(atoms, components)

    total_donors = sum(1 for a in atoms if a.is_donor)
    total_acceptors = sum(1 for a in atoms if a.is_acceptor)
    total_halogen_donors = sum(1 for a in atoms if a.is_halogen_donor)
    print(f"Found {total_donors} donors, {total_acceptors} acceptors, {total_halogen_donors} halogen donors")
    
    role_counts = {}
    for a in atoms:
        if a.role:
            role_counts[a.role] = role_counts.get(a.role, 0) + 1
    
    if role_counts:
        print("\nAtom roles found:")
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {role}: {count}")
    
    hbonds = read_hbonds_from_cif(args.cif, atoms)
    
    if not hbonds:
        print("No H-bonds detected in CIF, calculating geometrically")
        hbonds = find_hbonds_by_geometry(atoms)
    
    print(f"\nTotal H-bonds: {len(hbonds)}")
    
    if hbonds:
        hbonds = filter_intermolecular_hbonds(hbonds, atoms)
    
    acid_base_hbonds, base_base_hbonds = separate_hbonds_by_type(hbonds, atoms)
    
    print(f"Acid-base H-bonds: {len(acid_base_hbonds)}")
    print(f"Base-base H-bonds: {len(base_base_hbonds)}")
    
    hbonds = acid_base_hbonds
    
    for hbond in hbonds:
        if not hbond.motif:
            hbond.motif = identify_synthon_motif_from_roles(
                hbond.donor_role, hbond.acceptor_role, 
                hbond.donor_group, hbond.acceptor_group,
                hbond.donor_atom, hbond.acceptor_atom,
                atoms
            )
    
    halogen_bonds = find_halogen_bonds(atoms)
    print(f"Halogen bonds found: {len(halogen_bonds)}")
    
    halogen_acid_base, halogen_base_base = separate_halogen_bonds_by_type(halogen_bonds, atoms)
    
    print(f"Acid-base halogen bonds: {len(halogen_acid_base)}")
    print(f"Base-base halogen bonds: {len(halogen_base_base)}")
    
    halogen_motifs = {}
    for hb in halogen_bonds:
        if not hb.motif:
            hb.motif = identify_halogen_bond_motif(hb)
        motif_name = hb.motif
        if motif_name not in halogen_motifs:
            halogen_motifs[motif_name] = []
        halogen_motifs[motif_name].append(hb)

    print(f"\nSearch π-interactions")
    
    all_rings = find_all_rings_robust(atoms, components)
    
    pi_stackings = find_pi_stacking(atoms, components)
    
    if INCLUDE_CH_PI:
        ch_pi_interactions = find_ch_pi_interactions(atoms, all_rings)
    else:
        ch_pi_interactions = []
        print(f"\nThe search for C-H···π interactions is disabled (INCLUDE_CH_PI=False)")
    
    print(f"π-stacking interactions found: {len(pi_stackings)}")
    print(f"C-H···π interactions found: {len(ch_pi_interactions)}")
    
    motifs = {}
    for hbond in hbonds:
        motif_name = hbond.motif
        if motif_name not in motifs:
            motifs[motif_name] = []
        motifs[motif_name].append(hbond)
    
    has_acid_base = len(acid_base_hbonds) > 0 or len(halogen_acid_base) > 0
    
    result = {
        "file": args.cif,
        "n_molecules": len(components),
        "has_explicit_hydrogens": any(a.element == "H" for a in atoms),
        "intermolecular_hbonds": [asdict(s) for s in hbonds],
        "halogen_bonds": [asdict(h) for h in halogen_bonds],
        "acid_base_halogen_bonds": [asdict(h) for h in halogen_acid_base],
        "base_base_halogen_bonds": [asdict(h) for h in halogen_base_base],
        "halogen_synthon_motifs": {motif: len(synthons) for motif, synthons in halogen_motifs.items()},
        "pi_stackings": [{
            "type": ps.type, 
            "distance": ps.distance, 
            "angle": ps.angle,
            "motif": ps.motif, 
            "mol1": ps.mol1, 
            "mol2": ps.mol2
        } for ps in pi_stackings] if not has_acid_base else [],
        "ch_pi_interactions": ch_pi_interactions if not has_acid_base else [],
        "synthon_motifs": {motif: len(synthons) for motif, synthons in motifs.items()}
    }
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\nFile: {args.cif}")
        print(f"Molecules: {len(components)}")
        
        print(f"\nINTERMOLECULAR HYDROGEN BONDS: {len(hbonds)}")
        for s in hbonds:
            angle_str = f"{s.angle:.1f}°" if s.angle else "N/A"
            donor_info = f"{s.donor_atom}({s.donor_role})" if s.donor_role else s.donor_atom
            acceptor_info = f"{s.acceptor_atom}({s.acceptor_role})" if s.acceptor_role else s.acceptor_atom
            print(f"  {donor_info} -> {acceptor_info} = {s.distance:.3f}Å (Angle={angle_str})")
            print(f"    -> {s.motif}")
        
        if halogen_bonds:
            print(f"\nHALOGEN BONDS SUMMARY")
            for i, hb in enumerate(halogen_bonds, 1):
                display_str = get_halogen_bond_display_string(hb)
                print(f"\n{i}. {display_str}")
                print(f"   {hb.donor_atom}({hb.donor_element} - {hb.donor_role}) -> {hb.acceptor_atom}({hb.acceptor_element} - {hb.acceptor_role})")
                print(f"   Distance: {hb.distance:.3f}Å, Angle: {hb.angle_c_x_a:.1f}°")
        
        if halogen_motifs:
            print(f"\nHALOGEN BOND SYNTHON MOTIFS SUMMARY")
            for i, (motif_name, synthons) in enumerate(halogen_motifs.items(), 1):
                print(f"\n{i}. {motif_name}")
                print(f"   Total occurrences: {len(synthons)}")
        
        has_acid_base = len(acid_base_hbonds) > 0
        
        if not has_acid_base and pi_stackings:
            print_pi_stacking_summary(pi_stackings)
        
        if not has_acid_base and ch_pi_interactions:
            print(f"\nC-H···π INTERACTIONS SUMMARY")
            for i, interaction in enumerate(ch_pi_interactions, 1):
                print(f"\n{i}. {interaction['motif']}")
                print(f"   {interaction['donor_atom']}-{interaction['donor_h']} -> ring: {interaction['acceptor_ring'][:3]}")
                print(f"   Distance: {interaction['distance']:.3f}Å, Angle: {interaction['angle']:.1f}°")
        
        if has_acid_base:
            print(f"\nπ-stacking and C-H···π interactions were calculated but not shown because acid-base interactions (H-bonds or halogen bonds) are present (they dominate the supramolecular architecture)")
        
        if motifs:
            print(f"\nSUPRAMOLECULAR SYNTHON MOTIFS SUMMARY")
            for i, (motif_name, synthons) in enumerate(motifs.items(), 1):
                print(f"\n{i}. {motif_name}")
                print(f"   Total occurrences: {len(synthons)}")
        elif not hbonds and not halogen_bonds and not pi_stackings and not ch_pi_interactions:
            print("\nNo intermolecular interactions were detected")

if __name__ == "__main__":
    main()