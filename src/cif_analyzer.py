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
import os


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

SMARTS_CACHE = {}

def get_smarts_pattern(smarts_str):
    if smarts_str not in SMARTS_CACHE:
        try:
            pattern = Chem.MolFromSmarts(smarts_str)
            SMARTS_CACHE[smarts_str] = pattern if pattern else None
        except:
            SMARTS_CACHE[smarts_str] = None
    return SMARTS_CACHE[smarts_str]

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

def build_graph(ss, atoms):
    g = nx.Graph()
    g.add_nodes_from(a.idx for a in atoms)
    
    radii = {
        "H": 0.37, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71,
        "S": 1.03, "P": 1.06, "Cl": 0.99, "Br": 1.14, "I": 1.33
    }
    
    bond_factor = 1.4
    n_atoms = len(atoms)
    
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
                continue
            
            if a.element == "C" and b.element == "C":
                if dist < 1.8:
                    g.add_edge(i, j)
                    a.neighbors.append(j)
                    b.neighbors.append(i)
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
    
    return g

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
    donor_roles, acceptor_roles = [], []
    
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
            except:
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
        
        elif atom.element == 'N' and not atom.role:
            has_h = any(atoms[n].element == 'H' for n in atom.neighbors)
            if has_h:
                atom.is_donor = True
                atom.group = "neutral_amine"
                atom.role = "amine_NH_donor"
                atom.role_kind = "donor"
    
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
    except Exception as e:
        pass
    
    return hbonds

def filter_intermolecular_hbonds(hbonds, atoms):
    by_label = {a.label: a for a in atoms}
    intermolecular = []
    identity_sym_codes = {'.', '1_555', '1_655', '1_545', '1_565', '1_545'}
    
    for hbond in hbonds:
        if hbond.acceptor_symmetry and hbond.acceptor_symmetry not in identity_sym_codes:
            intermolecular.append(hbond)
            continue
        
        donor_atom = by_label.get(hbond.donor_atom)
        acceptor_atom = by_label.get(hbond.acceptor_atom)
        
        if donor_atom and acceptor_atom:
            if donor_atom.mol_id != acceptor_atom.mol_id:
                intermolecular.append(hbond)
        else:
            if hbond.acceptor_symmetry:
                intermolecular.append(hbond)
    
    return intermolecular

def extract_hbonds_from_cif(cif_path):
    try:
        ss = gemmi.read_small_structure(cif_path)
        print(f" CIF uploaded: {cif_path}")
    except Exception as e:
        print(f" CIF download error {cif_path}: {e}")
        return []
    
    atoms = expand_atoms(ss)
    build_graph(ss, atoms)
    components = fix_mol_ids_after_expansion(atoms)
    
    role_by_name, priority_rank = build_role_index(ATOM_ROLE_RULES)
    
    for component in components:
        rdmol, sorted_indices = graph_to_rdkit_mol(atoms, component)
        if rdmol is None:
            continue
        atoms = classify_atoms_with_roles(rdmol, atoms, sorted_indices, ATOM_ROLE_RULES, priority_rank)
    
    hbonds = read_hbonds_from_cif(cif_path, atoms)
    
    print(f" Found H-bonds: {len(hbonds)}")

    if not hbonds:
        return []
    
    hbonds = filter_intermolecular_hbonds(hbonds, atoms)
    print(f" Intermolecular H-bonds: {len(hbonds)}")
    
    return hbonds, atoms