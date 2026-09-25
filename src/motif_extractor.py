from rdkit import Chem
from .efgs.efgs import get_dec_fgs

def extract_motifs_efgs(mol):
    if mol is None:
        return []
    try:
        _, idx_list, smi_list, _ = get_dec_fgs(mol)

        motifs = []
        used_atoms = set()
        
        # 1. EFGs
        if smi_list is not None and idx_list is not None:
            for idx_set, smi in zip(idx_list, smi_list):
                if smi is None:
                    continue
                motifs.append({
                    'type': smi, 
                    'atom_ids': list(idx_set)
                })
                used_atoms.update(idx_set)
        return motifs
    except Exception as e:
        return []

def create_motif_vocabulary(motifs_list, log_func=None):
    unique_types = set()
    for m in motifs_list:
        if isinstance(m, dict) and 'type' in m:
            unique_types.add(m['type'])

    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    log_func(f"Found unique types of motifs: {len(type_to_idx)}")
    return type_to_idx