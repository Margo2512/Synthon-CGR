import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.rdBase import DisableLog
DisableLog('rdApp.*')

try:
    import gemmi
except ImportError:
    gemmi = None


def get_molecules_from_cif(cif_path):
    if gemmi is None:
        return []
    try:
        ss = gemmi.read_small_structure(cif_path)
    except Exception:
        return []

    cell = ss.cell
    atoms_data = []
    for site in ss.sites:
        if site.element.name == 'H':
            continue
        pos = cell.orthogonalize(
            gemmi.Fractional(site.fract.x, site.fract.y, site.fract.z)
        )
        atoms_data.append({
            'label': site.label,
            'element': site.element.name,
            'pos': np.array([pos.x, pos.y, pos.z]),
        })
    if not atoms_data:
        return []

    n = len(atoms_data)
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(atoms_data[i]['pos'] - atoms_data[j]['pos'])
            if 0.8 < d < 2.2:
                adj[i].append(j); adj[j].append(i)

    visited = set(); comps = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]; comp = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node); comp.append(node)
            for nei in adj[node]:
                if nei not in visited:
                    stack.append(nei)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return [{'atoms': [atoms_data[i] for i in c], 'indices': c} for c in comps]


def create_mol_with_cif_coords(smiles, cif_atoms_data):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol_noH = Chem.RemoveHs(mol)

    cif_elems  = [d['element'] for d in cif_atoms_data]
    cif_labels = [d['label']   for d in cif_atoms_data]
    cif_coords = [d['pos']     for d in cif_atoms_data]

    used = set(); smi_to_cif = {}
    for smi_idx in range(mol_noH.GetNumAtoms()):
        smi_atom = mol_noH.GetAtomWithIdx(smi_idx)
        smi_elem = smi_atom.GetSymbol()
        smi_nbrs = sorted(n.GetSymbol() for n in smi_atom.GetNeighbors())

        best, best_score = None, -1
        for cif_idx, cif_elem in enumerate(cif_elems):
            if cif_idx in used or cif_elem != smi_elem:
                continue
            cif_nbrs = []
            for j, other in enumerate(cif_atoms_data):
                if j == cif_idx:
                    continue
                d = np.linalg.norm(cif_coords[cif_idx] - cif_coords[j])
                if 0.8 < d < 2.2:
                    cif_nbrs.append(other['element'])
            cif_nbrs.sort()
            score = sum(1 for n in smi_nbrs if n in cif_nbrs)
            if score > best_score:
                best_score, best = score, cif_idx
        if best is not None:
            smi_to_cif[smi_idx] = best; used.add(best)

    if len(smi_to_cif) != mol_noH.GetNumAtoms():
        smi_to_cif, used = {}, set()
        for i in range(mol_noH.GetNumAtoms()):
            e = mol_noH.GetAtomWithIdx(i).GetSymbol()
            for j, ce in enumerate(cif_elems):
                if ce == e and j not in used:
                    smi_to_cif[i] = j; used.add(j); break

    mol_out = Chem.Mol(mol_noH)
    conf = Chem.Conformer(mol_noH.GetNumAtoms())
    for si, ci in smi_to_cif.items():
        pos = cif_coords[ci]
        conf.SetAtomPosition(si, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol_out.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol_out)
    except Exception:
        pass

    labels = {si: cif_labels[ci] for si, ci in smi_to_cif.items()}
    return mol_out, labels


from collections import Counter

def match_cif_molecule(smiles, cif_molecules):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_noH = Chem.RemoveHs(mol)
    smi_elems = [a.GetSymbol() for a in mol_noH.GetAtoms()]
    
    for cm in cif_molecules:
        cif_elems = [a['element'] for a in cm['atoms'] if a['element'] != 'H']
        
        if sorted(smi_elems) == sorted(cif_elems):
            return cm
        
        if len(cif_elems) <= len(smi_elems):
            smi_counter = Counter(smi_elems)
            cif_counter = Counter(cif_elems)
            if all(smi_counter[e] >= cif_counter[e] for e in cif_counter):
                continue
        
        if len(cif_elems) > len(smi_elems):
            cif_counter = Counter(cif_elems)
            smi_counter = Counter(smi_elems)
            if all(cif_counter[e] >= smi_counter[e] for e in smi_counter):
                return cm
    
    return None