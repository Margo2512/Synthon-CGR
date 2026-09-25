import torch
from torch_geometric.data import Data
from rdkit import Chem, RDConfig
from rdkit.Chem import rdchem, ChemicalFeatures
import os
from rdkit.rdBase import DisableLog
DisableLog('rdApp.*')
from .motif_extractor import extract_motifs_efgs


fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def get_donor_acceptor_flags(mol):
    num_atoms = mol.GetNumAtoms()
    donor_flags = [0] * num_atoms
    acceptor_flags = [0] * num_atoms

    feats = factory.GetFeaturesForMol(mol)

    for feat in feats:
        family = feat.GetFamily()
        atom_ids = feat.GetAtomIds()

        if family == "Donor":
            for idx in atom_ids:
                donor_flags[idx] = 1

        elif family == "Acceptor":
            for idx in atom_ids:
                acceptor_flags[idx] = 1

    return donor_flags, acceptor_flags

def mol_to_graph_with_motifs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    donor_flags, acceptor_flags = get_donor_acceptor_flags(mol)

    halogen_properties = {
        'F': {'size': 0.5, 'polarizability': 0.3, 'electronegativity': 3.98},
        'Cl': {'size': 1.0, 'polarizability': 0.8, 'electronegativity': 3.16},
        'Br': {'size': 1.2, 'polarizability': 1.1, 'electronegativity': 2.96},
        'I': {'size': 1.4, 'polarizability': 1.5, 'electronegativity': 2.66},
    }

    atom_features = []
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()

        features = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons()),
            float(atom.GetIsAromatic()),
            float(atom.GetHybridization()),
            float(atom.GetTotalValence()),
            float(atom.GetImplicitValence()),
            float(atom.IsInRing()),
            float(atom.IsInRingSize(6)),
            float(atom.IsInRingSize(5)),
            float(atom.GetExplicitValence()),
            float(atom.GetTotalNumHs()),
            float(atom.GetNumImplicitHs()),
            float(atom.GetNumExplicitHs()),
            float(atom.GetTotalDegree()),
            float(donor_flags[i]),     
            float(acceptor_flags[i]), 
        ]
        
        if symbol in halogen_properties:
            features.append(halogen_properties[symbol]['size'])          
            features.append(halogen_properties[symbol]['polarizability']) 
            features.append(halogen_properties[symbol]['electronegativity']) 
        else:
            features.append(0.0)
            features.append(0.0)
            features.append(0.0)

        atom_features.append(features)
    
    atom_features = torch.tensor(atom_features, dtype=torch.float)
    
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    
    edge_attr = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        edge_feat = [
            1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
        ]
        edge_attr.append(edge_feat)
        edge_attr.append(edge_feat)
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.tensor([], dtype=torch.float)
    
    motifs = extract_motifs_efgs(mol)
    
    data = Data(
        x=atom_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(atom_features)
    )
    
    return data, motifs

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    donor_flags, acceptor_flags = get_donor_acceptor_flags(mol)

    atom_features = []
    for i, atom in enumerate(mol.GetAtoms()):
        features = [
            float(atom.GetAtomicNum()),           
            float(atom.GetDegree()),              
            float(atom.GetFormalCharge()),        
            float(atom.GetNumRadicalElectrons()), 
            float(atom.GetIsAromatic()),          
            float(atom.GetHybridization()),       
            float(atom.GetTotalValence()),           
            float(atom.GetImplicitValence()),         
            float(atom.IsInRing()),               
            float(atom.IsInRingSize(6)),           
            float(atom.IsInRingSize(5)),
            float(atom.GetExplicitValence()),         
            float(atom.GetTotalNumHs()),              
            float(atom.GetNumImplicitHs()),          
            float(atom.GetNumExplicitHs()),          
            float(atom.GetTotalDegree()),            
            float(donor_flags[i]),    
            float(acceptor_flags[i]),   
        ]
        atom_features.append(features)
    
    atom_features = torch.tensor(atom_features, dtype=torch.float)
    
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    
    edge_attr = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        edge_feat = [
            1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
        ]
        edge_attr.append(edge_feat)
        edge_attr.append(edge_feat)  
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.tensor([], dtype=torch.float)
    
    data = Data(
        x=atom_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(atom_features)
    )
    
    return data