"""
Synthon-CGR inference script.

Usage:
    python predict.py --smiles1 "CC(=O)c1ccccn1" --smiles2 "O=C(O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1"
    python predict.py --smiles1 "CC(=O)c1ccccn1" --smiles2 "O=C(O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1" --top_k 20
    python predict.py                    # interactive mode
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
from rdkit import Chem

from src.models import create_model
from src.config import ConfigV3_Stage2_CommonTest
from src.candidate_generator import generate_candidates
from src.graph_utils import get_donor_acceptor_flags
from src.motif_extractor import extract_motifs_efgs
from torch_geometric.data import Data


def mol_to_graph_inference(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

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
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

        bond_type = bond.GetBondType()
        edge_feat = [
            1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
        ]
        edge_attr.append(edge_feat)
        edge_attr.append(edge_feat)

    edge_index = (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index else torch.tensor([[], []], dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_attr, dtype=torch.float)
        if edge_attr else torch.tensor([], dtype=torch.float)
    )

    motifs = extract_motifs_efgs(mol)

    data = Data(
        x=atom_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(atom_features),
    )

    return data, motifs

def load_model(model_path, config, device):
    model = create_model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_cocrystal(smiles1, smiles2, model, config, type_to_idx, device,
                      top_k=10, max_candidates=1000):
    data_a, motifs_a = mol_to_graph_inference(smiles1)
    data_b, motifs_b = mol_to_graph_inference(smiles2)

    if data_a is None:
        raise ValueError(f"Invalid SMILES1: {smiles1}")
    if data_b is None:
        raise ValueError(f"Invalid SMILES2: {smiles2}")

    data_a.batch = torch.zeros(data_a.num_nodes, dtype=torch.long)
    data_b.batch = torch.zeros(data_b.num_nodes, dtype=torch.long)

    candidates = generate_candidates(motifs_a, motifs_b, max_candidates=max_candidates)

    with torch.no_grad():
        result = model(
            data_a.to(device),
            data_b.to(device),
            [motifs_a],
            [motifs_b],
            type_to_idx,
            candidates=[candidates],
            observed_synthon_mask=None,
        )

    probability = torch.sigmoid(result["logits"]).item()
    candidate_scores = result["candidate_scores"]
    flat_candidates = result["candidates"]

    return probability, candidate_scores, flat_candidates

def print_result(smiles1, smiles2, prob, scores, candidates, top_k=10):
    print("=" * 70)
    print("Synthon-CGR prediction")
    print("=" * 70)
    print(f"SMILES1 : {smiles1}")
    print(f"SMILES2 : {smiles2}")
    print("-" * 70)
    print(f"Co-crystal probability: {prob:.4f}")
    print(f"Prediction: {'CO-CRYSTAL' if prob > 0.5 else 'NO CO-CRYSTAL'}")
    print("-" * 70)

    if len(scores) == 0:
        print("No candidate synthons generated.")
        return

    sorted_idx = torch.argsort(scores, descending=True)
    top_k = min(top_k, len(sorted_idx))

    print(f"Top-{top_k} predicted synthons:")
    for i, idx in enumerate(sorted_idx[:top_k]):
        score = scores[idx].item()
        cand = candidates[idx]
        left = cand.get("left_type", "?")
        right = cand.get("right_type", "?")
        print(f"  {i + 1:2d}. {left:35s} <-> {right:35s} : {score:.4f}")

    print("=" * 70)

def interactive_input():
    print("Enter SMILES strings (or 'q' to quit):")
    smiles1 = input("SMILES1: ").strip()
    if smiles1.lower() in ("q", "quit", "exit"):
        sys.exit(0)
    smiles2 = input("SMILES2: ").strip()
    if smiles2.lower() in ("q", "quit", "exit"):
        sys.exit(0)
    return smiles1, smiles2


def main():
    parser = argparse.ArgumentParser(
        description="Synthon-CGR: predict co-crystal formation and "
                    "identify the most probable supramolecular synthons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python predict.py --smiles1 "CC(=O)c1ccccn1" --smiles2 "O=C(O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1"
            python predict.py --smiles1 "CC(=O)c1ccccn1" --smiles2 "O=C(O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1" --top_k 20
            python predict.py                    # interactive mode
                    """,
    )
    parser.add_argument("--smiles1", type=str, default=None,
                        help="SMILES of the first molecule")
    parser.add_argument("--smiles2", type=str, default=None,
                        help="SMILES of the second molecule")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top synthons to display (default: 10)")
    parser.add_argument("--max_candidates", type=int, default=1000,
                        help="Maximum number of candidate synthons (default: 1000)")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/best_model_ConfigV3_Stage2_CommonTest.pt",
                        help="Path to the pretrained model")
    parser.add_argument("--cache_path", type=str,
                        default="cache/motifs_cache_v3s2_common.pkl",
                        help="Path to the motif vocabulary cache")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Synthon-CGR] Using device: {device}")

    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        _, _, type_to_idx = pickle.load(f)
    print(f"[Synthon-CGR] Loaded motif vocabulary: {len(type_to_idx)} types")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    config = ConfigV3_Stage2_CommonTest()
    model = load_model(model_path, config, device)
    print(f"[Synthon-CGR] Model loaded: {model_path}")
    print()

    while True:
        if args.smiles1 is not None and args.smiles2 is not None:
            smiles1, smiles2 = args.smiles1, args.smiles2
            single_run = True
        else:
            smiles1, smiles2 = interactive_input()
            single_run = False

        if Chem.MolFromSmiles(smiles1) is None:
            print(f"ERROR: invalid SMILES1: {smiles1}\n")
            if single_run:
                sys.exit(1)
            continue
        if Chem.MolFromSmiles(smiles2) is None:
            print(f"ERROR: invalid SMILES2: {smiles2}\n")
            if single_run:
                sys.exit(1)
            continue

        try:
            prob, scores, candidates = predict_cocrystal(
                smiles1, smiles2, model, config, type_to_idx, device,
                top_k=args.top_k, max_candidates=args.max_candidates,
            )
        except Exception as e:
            print(f"ERROR during prediction: {e}\n")
            if single_run:
                sys.exit(1)
            continue

        print_result(smiles1, smiles2, prob, scores, candidates, top_k=args.top_k)

        if single_run:
            break

        print()


if __name__ == "__main__":
    main()