import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from .graph_utils import mol_to_graph, mol_to_graph_with_motifs
from .motif_extractor import create_motif_vocabulary
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import sys
import re
import pickle
from rdkit import Chem
import os
from typing import Dict
from .cif_mapping import (
    get_molecules_from_cif, create_mol_with_cif_coords, match_cif_molecule
)

class CocrystalDataset(Dataset):
    def __init__(self, dataframe, use_motifs=False):
        self.data = dataframe.reset_index(drop=True)
        self.use_motifs = use_motifs
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
         
        graph_a = mol_to_graph(row['SMILES1'])
        graph_b = mol_to_graph(row['SMILES2'])
        
        if graph_a is None or graph_b is None:
            return None
        
        label = torch.tensor(row['result'], dtype=torch.float)
        
        result = {
            'graph_a': graph_a,
            'graph_b': graph_b,
            'label': label
        }
        
        if self.use_motifs:
            motifs_a = row.get('motifs_a')
            motifs_b = row.get('motifs_b')
            
            if not isinstance(motifs_a, list):
                motifs_a = []
            if not isinstance(motifs_b, list):
                motifs_b = []
            
            result['motifs_a'] = motifs_a
            result['motifs_b'] = motifs_b
        
        return result

class CocrystalDatasetWithSynthons(CocrystalDataset):
    def __init__(self, dataframe, use_motifs=False, use_masks=True, use_candidates=False):
        super().__init__(dataframe, use_motifs)
        self.has_hbat = 'observed_synthon_mask' in dataframe.columns
        self.use_masks = use_masks
        self.use_candidates = use_candidates
        self.smiles1 = dataframe['SMILES1'].values
        self.smiles2 = dataframe['SMILES2'].values

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        if result is None:
            return None
         
        result['smiles1'] = self.smiles1[idx]
        result['smiles2'] = self.smiles2[idx]
        
        motifs_a = result.get('motifs_a', [])
        motifs_b = result.get('motifs_b', [])
         
        if self.use_candidates and motifs_a and motifs_b:
            from .candidate_generator import generate_candidates
            candidates = generate_candidates(motifs_a, motifs_b, max_candidates=2000)
        else:
            candidates = []

        result['candidates'] = candidates
        has_mask = False
        observed_mask = None
        
        if self.has_hbat and self.use_masks:
            row = self.data.iloc[idx]
            observed_matrix = row.get('observed_synthon_mask')
            if observed_matrix is not None: 
                if not isinstance(observed_matrix, torch.Tensor):
                    observed_matrix = torch.tensor(observed_matrix, dtype=torch.float)
                has_mask = True
                observed_mask = observed_matrix
            else:
                has_mask = False
                observed_mask = None
        else:
            has_mask = False
            observed_mask = None
        
        num_candidates = len(candidates)

        if has_mask and num_candidates > 0:
            observed_vector = np.zeros(num_candidates, dtype=np.float32)   
            
            def get_halogen_weight(motif_type):
                if 'I[R]' in motif_type:
                    return 1.0   
                elif 'Br[R]' in motif_type:
                    return 0.7   
                elif 'Cl[R]' in motif_type:
                    return 0.1   
                elif 'F[R]' in motif_type:
                    return 0.05   
                return None   
                     
            for k, candidate in enumerate(candidates):
                left_id = candidate['left_motif_id']
                right_id = candidate['right_motif_id']
                left_type = candidate.get('left_type', '')
                right_type = candidate.get('right_type', '')
                
                if left_id < observed_matrix.shape[0] and right_id < observed_matrix.shape[1]:
                    base_value = observed_matrix[left_id, right_id].item()
                    
                    if base_value > 0.5:
                        halogen_weight_left = get_halogen_weight(left_type)
                        halogen_weight_right = get_halogen_weight(right_type)
                        
                        if halogen_weight_left is not None or halogen_weight_right is not None:
                            weight = 1.0
                            if halogen_weight_left is not None:
                                weight = min(weight, halogen_weight_left)
                            if halogen_weight_right is not None:
                                weight = min(weight, halogen_weight_right)
                            observed_vector[k] = weight
                        else:
                            observed_vector[k] = base_value
                    else:
                        observed_vector[k] = 0.0
                else:
                    observed_vector[k] = 0.0
                    
            result['observed_synthon_mask'] = torch.tensor(observed_vector, dtype=torch.float)
            result['has_mask'] = True
        else:
            result['observed_synthon_mask'] = torch.zeros(num_candidates, dtype=torch.float)
            result['has_mask'] = False
        return result

class CocrystalDatasetWithHBAT(CocrystalDataset):
    def __init__(self, dataframe, use_motifs=False):
        super().__init__(dataframe, use_motifs)
        self.has_hbat = 'observed_synthon_mask' in dataframe.columns  

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        
        if result is None:
            return None
        
        if self.has_hbat:
            row = self.data.iloc[idx]
             
            observed_mask = row.get('observed_synthon_mask')
            
            if observed_mask is not None:
                result['observed_synthon_mask'] = torch.tensor(observed_mask, dtype=torch.float)
            else: 
                num_motifs_a = len(result.get('motifs_a', []))
                num_motifs_b = len(result.get('motifs_b', []))
                result['observed_synthon_mask'] = torch.zeros(num_motifs_a, num_motifs_b, dtype=torch.float)
        
        return result

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    graphs_a = [b['graph_a'] for b in batch]
    graphs_b = [b['graph_b'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    
    batch_a = Batch.from_data_list(graphs_a)
    batch_b = Batch.from_data_list(graphs_b)
    
    result = {
        'batch_a': batch_a,
        'batch_b': batch_b,
        'labels': labels
    }
    
    if 'motifs_a' in batch[0]:
        result['motifs_a'] = [b['motifs_a'] for b in batch]
        result['motifs_b'] = [b['motifs_b'] for b in batch]
    
    if 'candidates' in batch[0]:
        result['candidates'] = [b['candidates'] for b in batch]
        if 'observed_synthon_mask' in batch[0]:
            result['observed_synthon_mask'] = [b['observed_synthon_mask'] for b in batch]
        else:
            result['observed_synthon_mask'] = [torch.zeros(len(b['candidates']), dtype=torch.float) 
                                            for b in batch]
    else: 
        result['candidates'] = [[] for _ in range(len(batch))]
        result['observed_synthon_mask'] = [torch.zeros(0, dtype=torch.float) for _ in range(len(batch))]
    
    return result

def get_class_weights(df):
    class_counts = df['result'].value_counts().sort_index()
    total = len(df)
    weights = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    return weights

def create_balanced_sampler(dataset):
    labels = dataset.data['result'].values
    class_counts = np.bincount(labels.astype(int))
    weights = 1.0 / class_counts[labels.astype(int)]
    weights = weights / weights.sum()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

def load_data(csv_path, test_size=0.2, val_size=0.1, random_state=42, 
            use_motifs=False, cache_file_path=None, log_func=None):
    df = pd.read_csv(csv_path)

    log_func(f"Loaded {len(df)} str")
    log_func(f"Columns: {df.columns.tolist()}")
    log_func(f"Label distribution:")
    log_func(f"{df['result'].value_counts()}")
    log_func(f"  - 0: {len(df[df['result']==0])} ({(len(df[df['result']==0])/len(df)*100):.1f}%)")
    log_func(f"  - 1: {len(df[df['result']==1])} ({(len(df[df['result']==1])/len(df)*100):.1f}%)")
    
    if 'SMILES1' not in df.columns:
        log_func("Attention: the columns are not called SMILES1/SMILES2")
        log_func("Renaming...")
        smiles_cols = [col for col in df.columns if 'SMILES' in col.upper() or 'smiles' in col.lower()]
        if len(smiles_cols) >= 2:
            df.columns = ['SMILES1', 'SMILES2', 'result'] + list(df.columns[3:])
        else:
            raise ValueError("Columns with SMILES were not found.")

    if use_motifs and cache_file_path and os.path.exists(cache_file_path):
        log_func(f"Loading motifs from the cache: {cache_file_path}")
        with open(cache_file_path, 'rb') as f:
            df['motifs_a'], df['motifs_b'], type_to_idx = pickle.load(f)
    elif use_motifs:
        log_func("\nExtraction of motifs via EFGs (for V1 and V2)...")
        motifs_cache_a = []
        motifs_cache_b = []
        all_motifs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting motifs"):
            _, motifs_a = mol_to_graph_with_motifs(row['SMILES1'])
            _, motifs_b = mol_to_graph_with_motifs(row['SMILES2'])

            motifs_cache_a.append(motifs_a or [])
            motifs_cache_b.append(motifs_b or [])

            if motifs_a is not None and isinstance(motifs_a, list):
                all_motifs.extend(motifs_a)
            if motifs_b is not None and isinstance(motifs_b, list):
                all_motifs.extend(motifs_b)

        df['motifs_a'] = motifs_cache_a
        df['motifs_b'] = motifs_cache_b
        type_to_idx = create_motif_vocabulary(all_motifs, log_func=log_func)
        log_func(f"Found unique types of motifs: {len(type_to_idx)}")

        if cache_file_path:
            with open(cache_file_path, 'wb') as f:
                pickle.dump((df['motifs_a'], df['motifs_b'], type_to_idx), f)
            log_func(f"Motifs save in the cache: {cache_file_path}")
    else:
        type_to_idx = {}
    
    df_train_val, df_test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['result']
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=df_train_val['result']
    )
    
    log_func(f"\nData split:")
    log_func(f"  Train: {len(df_train)} ({(len(df_train)/len(df)*100):.1f}%)")
    log_func(f"  Val:   {len(df_val)} ({(len(df_val)/len(df)*100):.1f}%)")
    log_func(f"  Test:  {len(df_test)} ({(len(df_test)/len(df)*100):.1f}%)")
    
    train_dataset = CocrystalDataset(df_train, use_motifs=use_motifs)
    val_dataset = CocrystalDataset(df_val, use_motifs=use_motifs)
    test_dataset = CocrystalDataset(df_test, use_motifs=use_motifs)

    class_weights = get_class_weights(df_train)
    
    return train_dataset, val_dataset, test_dataset, class_weights, type_to_idx

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, use_balanced_sampler=True, num_workers=8, collate_fn=None):
    if collate_fn is None:
        from .data_utils import collate_fn as default_collate_fn
        collate_fn = default_collate_fn

    if use_balanced_sampler:
        sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers // 2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers // 2
    )
    
    return train_loader, val_loader, test_loader

def collate_fn_with_hbat(batch):
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    result = collate_fn(batch)
    
    if result is None:
        return None
    
    if 'has_mask' in batch[0]:
        result['has_mask'] = [b.get('has_mask', False) for b in batch]
    else:
        result['has_mask'] = [False] * len(batch)
        
    if 'observed_synthon_mask' in batch[0]:
        all_observed_masks = []
        
        for b in batch:
            mask = b.get('observed_synthon_mask')
            
            if mask is None:
                num_candidates = len(b.get('candidates', []))
                mask = torch.zeros(num_candidates, dtype=torch.float)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float)
            
            all_observed_masks.append(mask)
        
        result['observed_synthon_mask'] = all_observed_masks
    else:
        result['observed_synthon_mask'] = [
            torch.zeros(len(b.get('candidates', [])), dtype=torch.float) 
            for b in batch
        ]

    return result

def _normalize_label(label):
    return re.sub(r'[^A-Za-z0-9]', '', str(label))

def map_hbat_to_synthons(smiles_a, smiles_b, hbat_info, motifs_a, motifs_b,
                          cif_path=None):
    import torch
    import os

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if not hbat_info or not motifs_a or not motifs_b:
        return None
    if not cif_path or not os.path.exists(cif_path):
        return None

    cif_mols = get_molecules_from_cif(cif_path)
    if not cif_mols:
        return None

    cm_a = match_cif_molecule(smiles_a, cif_mols)
    cm_b = match_cif_molecule(smiles_b, cif_mols)
    if cm_a is None or cm_b is None:
        return None

    _, labels_a = create_mol_with_cif_coords(smiles_a, cm_a['atoms'])
    _, labels_b = create_mol_with_cif_coords(smiles_b, cm_b['atoms'])
    if not labels_a or not labels_b:
        return None

    lab2idx_a = {}
    for k, v in labels_a.items():
        lab2idx_a[v] = k                       
        lab2idx_a[_normalize_label(v)] = k    

    lab2idx_b = {}
    for k, v in labels_b.items():
        lab2idx_b[v] = k
        lab2idx_b[_normalize_label(v)] = k

    observed = torch.zeros(len(motifs_a), len(motifs_b))

    n_hb_total = len(hbat_info.get('hbonds', []))
    n_hb_skipped = 0    
    n_hb_ok = 0

    for hb in hbat_info.get('hbonds', []):
        d_name = hb.get('donor_label', '')
        a_name = hb.get('acceptor_label', '')
        if not d_name or not a_name:
            continue
        
        if '-' in d_name:
            parts = d_name.split('-')
            for part in parts:
                if part and part[0] in ('I', 'Br', 'Cl', 'F'):
                    d_name = part
                    break
            else:
                d_name = parts[-1]  

        if '-' in a_name:
            parts = a_name.split('-')
            for part in parts:
                if part and part[0] in ('I', 'Br', 'Cl', 'F'):
                    a_name = part
                    break
            else:
                a_name = parts[-1]

        d_norm = _normalize_label(d_name)
        a_norm = _normalize_label(a_name)

        d_a = lab2idx_a.get(d_name) or lab2idx_a.get(d_norm)
        d_b = lab2idx_b.get(d_name) or lab2idx_b.get(d_norm)
        a_a = lab2idx_a.get(a_name) or lab2idx_a.get(a_norm)
        a_b = lab2idx_b.get(a_name) or lab2idx_b.get(a_norm)

        if d_a is not None and a_b is not None:
            n_hb_ok += 1
            d_ms = [i for i, m in enumerate(motifs_a) if d_a in m['atom_ids']]
            a_ms = [j for j, m in enumerate(motifs_b) if a_b in m['atom_ids']]

            if not d_ms and mol_a is not None and d_a < mol_a.GetNumAtoms():
                d_elem = mol_a.GetAtomWithIdx(d_a).GetSymbol()
                d_ms = [i for i, m in enumerate(motifs_a)
                        if any(mol_a.GetAtomWithIdx(aid).GetSymbol() == d_elem
                            for aid in m['atom_ids'] if aid < mol_a.GetNumAtoms())]

            if not d_ms and mol_a is not None:
                HETERO = ('N', 'O', 'S')
                d_ms = [i for i, m in enumerate(motifs_a)
                        if any(mol_a.GetAtomWithIdx(aid).GetSymbol() in HETERO
                            for aid in m['atom_ids'] if aid < mol_a.GetNumAtoms())]

            if not d_ms:
                d_ms = list(range(len(motifs_a)))

            if not a_ms and mol_b is not None and a_b < mol_b.GetNumAtoms():
                a_elem = mol_b.GetAtomWithIdx(a_b).GetSymbol()
                a_ms = [j for j, m in enumerate(motifs_b)
                        if any(mol_b.GetAtomWithIdx(aid).GetSymbol() == a_elem
                            for aid in m['atom_ids'] if aid < mol_b.GetNumAtoms())]

            if not a_ms and mol_b is not None:
                HETERO = ('N', 'O', 'S')
                a_ms = [j for j, m in enumerate(motifs_b)
                        if any(mol_b.GetAtomWithIdx(aid).GetSymbol() in HETERO
                            for aid in m['atom_ids'] if aid < mol_b.GetNumAtoms())]

            if not a_ms:
                a_ms = list(range(len(motifs_b)))

            for i in d_ms:
                for j in a_ms:
                    observed[i, j] = 1.0
        elif d_b is not None and a_a is not None:
            n_hb_ok += 1
            d_ms = [i for i, m in enumerate(motifs_b) if d_b in m['atom_ids']]
            a_ms = [j for j, m in enumerate(motifs_a) if a_a in m['atom_ids']]

            if not d_ms and mol_b is not None and d_b < mol_b.GetNumAtoms():
                d_elem = mol_b.GetAtomWithIdx(d_b).GetSymbol()
                d_ms = [i for i, m in enumerate(motifs_b)
                        if any(mol_b.GetAtomWithIdx(aid).GetSymbol() == d_elem
                            for aid in m['atom_ids'] if aid < mol_b.GetNumAtoms())]

            if not d_ms and mol_b is not None:
                HETERO = ('N', 'O', 'S')
                d_ms = [i for i, m in enumerate(motifs_b)
                        if any(mol_b.GetAtomWithIdx(aid).GetSymbol() in HETERO
                            for aid in m['atom_ids'] if aid < mol_b.GetNumAtoms())]

            if not d_ms:
                d_ms = list(range(len(motifs_b)))

            if not a_ms and mol_a is not None and a_a < mol_a.GetNumAtoms():
                a_elem = mol_a.GetAtomWithIdx(a_a).GetSymbol()
                a_ms = [j for j, m in enumerate(motifs_a)
                        if any(mol_a.GetAtomWithIdx(aid).GetSymbol() == a_elem
                            for aid in m['atom_ids'] if aid < mol_a.GetNumAtoms())]

            if not a_ms and mol_a is not None:
                HETERO = ('N', 'O', 'S')
                a_ms = [j for j, m in enumerate(motifs_a)
                        if any(mol_a.GetAtomWithIdx(aid).GetSymbol() in HETERO
                            for aid in m['atom_ids'] if aid < mol_a.GetNumAtoms())]

            if not a_ms:
                a_ms = list(range(len(motifs_a)))

            for i in a_ms:
                for j in d_ms:
                    observed[i, j] = 1.0

        else:
            n_hb_skipped += 1
            print(f"[skip] cif={cif_path}, d_name='{d_name}', a_name='{a_name}'")
            print(f"  d_a={d_a}, d_b={d_b}, a_a={a_a}, a_b={a_b}")
            print(f"  labels_a keys: {list(labels_a.values())[:10]}")
            print(f"  labels_b keys: {list(labels_b.values())[:10]}")
            continue
    return observed


def find_atom_by_label(mol, label):
    if ':' in label:
        label = label.split(':')[1]
    
    for atom in mol.GetAtoms():
        try:
            if atom.GetProp('_Name') == label:
                return atom.GetIdx()
        except:
            pass
    
    symbol = label[0]
    number = int(label[1:]) if len(label) > 1 else 0
    
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol:
            count += 1
            if count == number:
                return atom.GetIdx()
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol:
            return atom.GetIdx()
    
    return None

def is_matching_atom(atom, label):
    if ':' in label:
        label = label.split(':')[1]
    
    symbol = label[0]
    
    if atom.GetSymbol() != symbol:
        return False
    
    if len(label) > 1:
        try:
            number = int(label[1:])
        except:
            pass
    
    return True

def parse_hbat_file(hbat_path: str) -> Dict[str, Dict]:
    df = pd.read_csv(hbat_path, sep=';')
    
    print(f"  HBAT file: {len(df)} str")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Examples str:")
    for i in range(min(3, len(df))):
        print(f"    {df.iloc[i].to_dict()}")
    
    unique_files = df['File'].unique()
    print(f"  Unique CIF: {len(unique_files)}")
    
    hbat_data = {}
    
    for filename, group in df.groupby('File'):
        donors = []
        acceptors = []
        hbonds = []
        
        for _, row in group.iterrows():
            donor = str(row['Donor']).strip()
            acceptor = str(row['Acceptor']).strip()
            
            donor_label = donor.split(':')[1] if ':' in donor else donor
            acceptor_label = acceptor.split(':')[1] if ':' in acceptor else acceptor
            
            donors.append(donor_label)
            acceptors.append(acceptor_label)
            
            hbonds.append({
                'donor': donor,
                'acceptor': acceptor,
                'donor_label': donor_label,
                'acceptor_label': acceptor_label,
                'distance': float(row.get('Distance (Å)', 0)),
                'angle': float(row.get('Angle (°)', 0)),
                'type': row.get('Type bond', 'H-bond'),
                'motif': row.get('What is formed', '')
            })
        
        filename_str = str(filename)
        hbat_data[filename_str] = {
            'donors': list(set(donors)),
            'acceptors': list(set(acceptors)),
            'hbonds': hbonds
        }
        
        if len(hbat_data) <= 3:
            print(f"  CIF {filename_str}: {len(hbonds)} bonds")
            print(f"    Donors: {list(set(donors))[:3]}...")
            print(f"    Acceptors: {list(set(acceptors))[:3]}...")
    
    return hbat_data

def load_data_with_hbat(csv_path: str, cod_csv_path: str, hbat_path: str,
                        test_size=0.99, val_size=0.005, random_state=42,
                        use_motifs=False, cache_file_path=None, cif_dir=None, log_func=None, use_masks=True, use_candidates=True):
    from .graph_utils import mol_to_graph_with_motifs
    from .motif_extractor import create_motif_vocabulary
    from tqdm import tqdm
    import pickle
    
    df = pd.read_csv(csv_path)
    log_func(f"Uploaded {len(df)} str from the main file")
    
    df_cod = pd.read_csv(cod_csv_path)
    log_func(f"Uploaded by {len(df_cod)} str from COD_our_df.csv")
    
    df['smiles_pair'] = df['SMILES1'] + '|' + df['SMILES2']
    df_cod['smiles_pair'] = df_cod['SMILES1'] + '|' + df_cod['SMILES2']
    
    df = df.merge(df_cod[['smiles_pair', 'CIF_Filename']], on='smiles_pair', how='left')
    df.rename(columns={'CIF_Filename': 'cif_file'}, inplace=True)
    
    log_func(f"Downloading HBAT data from: {hbat_path}")
    hbat_data = parse_hbat_file(hbat_path)
    log_func(f"Uploaded {len(hbat_data)} CIF str from HBAT")
    
    found_cif = df['cif_file'].notna().sum()
    log_func(f"CIF files found for {found_cif}/{len(df)} SMILES pairs")
    
    df['cif_file_clean'] = df['cif_file'].apply(
        lambda x: str(x).replace('.cif', '').strip() if pd.notna(x) else x
    )
    
    cif_in_hbat = df[df['cif_file_clean'].isin(hbat_data.keys())]
    log_func(f"CIF after cleaning, which are available in HBAT: {len(cif_in_hbat)}")
    
    if use_motifs:
        if cache_file_path and os.path.exists(cache_file_path):
            log_func(f"Loading motifs from the cache: {cache_file_path}")
            with open(cache_file_path, 'rb') as f:
                df['motifs_a'], df['motifs_b'], type_to_idx = pickle.load(f)
            df['motifs_a'] = df['motifs_a'].apply(lambda x: x if isinstance(x, list) else [])
            df['motifs_b'] = df['motifs_b'].apply(lambda x: x if isinstance(x, list) else [])
        else:
            log_func("\nMotif Extraction via EFGs...")
            motifs_cache_a = []
            motifs_cache_b = []
            all_motifs = []
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting motifs"):
                _, motifs_a = mol_to_graph_with_motifs(row['SMILES1'])
                _, motifs_b = mol_to_graph_with_motifs(row['SMILES2'])
                motifs_cache_a.append(motifs_a or [])
                motifs_cache_b.append(motifs_b or [])
                if motifs_a is not None and isinstance(motifs_a, list):
                    all_motifs.extend(motifs_a)
                if motifs_b is not None and isinstance(motifs_b, list):
                    all_motifs.extend(motifs_b)
            
            df['motifs_a'] = motifs_cache_a
            df['motifs_b'] = motifs_cache_b
            type_to_idx = create_motif_vocabulary(all_motifs, log_func=log_func)
            log_func(f"Found unique types of motifs: {len(type_to_idx)}")
            
            if cache_file_path:
                with open(cache_file_path, 'wb') as f:
                    pickle.dump((df['motifs_a'], df['motifs_b'], type_to_idx), f)
                log_func(f"Motifs are save in the cache: {cache_file_path}")
    else:
        type_to_idx = {}
    
    log_func("\nCreating an observed_synthon_mask for synthons...")
    df['observed_synthon_mask'] = None

    total_masks = 0
    masks_with_ones = 0
    total_ones = 0

    from .hbond_prior import HBondPrior
    _hbond_prior = HBondPrior()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Create observed_mask"):
        motifs_a = row.get('motifs_a', [])
        motifs_b = row.get('motifs_b', [])
        
        if not motifs_a or not motifs_b:
            continue
        
        if row['result'] == 0:
            num_a = len(motifs_a)
            num_b = len(motifs_b)
            observed_matrix = np.zeros((num_a, num_b), dtype=np.float32)
            df.at[idx, 'observed_synthon_mask'] = observed_matrix
            total_masks += 1
            continue 



        cif_file = row.get('cif_file_clean')
        has_hbat = not pd.isna(cif_file) and cif_file in hbat_data
        
        num_a = len(motifs_a)
        num_b = len(motifs_b)
        
        observed_matrix = np.zeros((num_a, num_b), dtype=np.float32)
        
        if has_hbat and hbat_data[cif_file]:
            cif_dir = getattr(config, 'CIF_DIR', None) if 'config' in dir() else None
            if cif_dir is None:
                cif_dir = os.path.join(os.path.dirname(cod_csv_path), 'cif_files_all_without_duplicates')

            cif_path = os.path.join(cif_dir, f"{cif_file}.cif")
            hbat_matrix = map_hbat_to_synthons(
                row['SMILES1'], row['SMILES2'],
                hbat_data[cif_file],
                motifs_a, motifs_b,
                cif_path=cif_path
            )
            if hbat_matrix is not None:
                observed_matrix = hbat_matrix.numpy()
        
        df.at[idx, 'observed_synthon_mask'] = observed_matrix
        total_masks += 1
        
        if observed_matrix.sum() > 0:
            masks_with_ones += 1
            total_ones += observed_matrix.sum()
            
            if masks_with_ones <= 5:
                log_func(f"\n  Example of a {masks_with_ones} mask with units:")
                log_func(f"    CIF: {cif_file if has_hbat else 'NO_CIF'}")
                log_func(f"    SMILES1: {row['SMILES1'][:50]}...")
                log_func(f"    SMILES2: {row['SMILES2'][:50]}...")
                log_func(f"    observed_matrix shape: {observed_matrix.shape}")
                log_func(f"    observed_matrix sum: {observed_matrix.sum()}")
                log_func(f"    observed_matrix:\n{observed_matrix}")

    log_func(f"\nStatistics observed_synthon_mask:")
    log_func(f"  Total masks created: {total_masks}")
    log_func(f"  Masks with 1: {masks_with_ones}")
    log_func(f"  Total 1 in masks: {total_ones}")
    if total_masks > 0:
        log_func(f"  Average 1 per mask: {total_ones / total_masks:.2f}")

    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['result']
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size_adjusted,
        random_state=random_state, stratify=df_train_val['result']
    )
    
    log_func(f"\nData split:")
    log_func(f"  Train: {len(df_train)}")
    log_func(f"  Val:   {len(df_val)}")
    log_func(f"  Test:  {len(df_test)}")
    
    train_dataset = CocrystalDatasetWithSynthons(
        df_train, 
        use_motifs=use_motifs,
        use_masks=use_masks,
        use_candidates=use_candidates
    )
    val_dataset = CocrystalDatasetWithSynthons(
        df_val, 
        use_motifs=use_motifs,
        use_masks=use_masks,
        use_candidates=use_candidates
    )
    test_dataset = CocrystalDatasetWithSynthons(
        df_test, 
        use_motifs=use_motifs,
        use_masks=use_masks,
        use_candidates=use_candidates
    )
    
    class_weights = get_class_weights(df_train)
    
    return train_dataset, val_dataset, test_dataset, class_weights, type_to_idx