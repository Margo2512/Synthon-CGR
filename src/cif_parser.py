import torch
import numpy as np
from rdkit import Chem
from typing import List, Dict, Tuple, Optional
import os
import sys
from .cif_analyzer import extract_hbonds_from_cif, AtomNode, Synthon


class CIFMaskExtractor:
    def __init__(self):
        self.cif_cache = {}
    
    def extract_mask_from_cif(self, cif_path: str, smiles: str) -> torch.Tensor:
        cache_key = f"{cif_path}_{smiles}"
        if cache_key in self.cif_cache:
            return self.cif_cache[cache_key]
        
        try:
            hbonds, atoms = self._parse_cif(cif_path)
            
            if not hbonds or not atoms:
                return torch.zeros(1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(1)
            
            atom_mapping = self._map_cif_to_rdkit(atoms, mol)
            
            mask = torch.zeros(mol.GetNumAtoms())
            
            for hbond in hbonds:
                donor_label = hbond.donor_atom
                acceptor_label = hbond.acceptor_atom
                
                for rd_idx, cif_label in atom_mapping.items():
                    if cif_label == donor_label or cif_label == acceptor_label:
                        mask[rd_idx] = 1.0
            
            self.cif_cache[cache_key] = mask
            
            return mask
            
        except Exception as e:
            return torch.zeros(1)
    
    def _parse_cif(self, cif_path: str) -> Tuple[List, List]:
        try:
            from .cif_analyzer import extract_hbonds_from_cif
            hbonds, atoms = extract_hbonds_from_cif(cif_path)
            return hbonds, atoms
        except Exception as e:
            return [], []
    
    def _map_cif_to_rdkit(self, cif_atoms: List, rdkit_mol: Chem.Mol) -> Dict[int, str]:
        mapping = {}
        
        cif_counts = {}
        cif_labels_by_type = {}
        
        for atom in cif_atoms:
            element = atom.element
            label = atom.label
            
            if element not in cif_counts:
                cif_counts[element] = 0
                cif_labels_by_type[element] = []
            
            cif_counts[element] += 1
            cif_labels_by_type[element].append(label)
        
        rdkit_counts = {}
        for atom in rdkit_mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in rdkit_counts:
                rdkit_counts[symbol] = 0
            rdkit_counts[symbol] += 1
        
        for element, count in rdkit_counts.items():
            if element in cif_counts and count == cif_counts.get(element, 0):
                rdkit_indices = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() 
                               if atom.GetSymbol() == element]
                cif_labels = cif_labels_by_type.get(element, [])
                
                for i, (rd_idx, cif_label) in enumerate(zip(rdkit_indices, cif_labels)):
                    mapping[rd_idx] = cif_label
        
        return mapping


def create_cif_mask_from_parser(cif_path: str, smiles: str) -> torch.Tensor:
    extractor = CIFMaskExtractor()
    return extractor.extract_mask_from_cif(cif_path, smiles)