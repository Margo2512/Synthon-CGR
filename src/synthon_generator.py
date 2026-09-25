import torch
import torch.nn as nn
import torch.nn.functional as F
from .hbond_prior import HBondPrior
from .motif_encoder import MotifEncoder


class SynthonGenerator(nn.Module):
    def __init__(self, atom_dim=128, synthon_dim=128, max_synthons=20,
                 num_motif_types=1000, use_hbond_prior=True):
        super().__init__()
        self.max_synthons = max_synthons
        self.use_hbond_prior = use_hbond_prior

        self.motif_encoder = MotifEncoder(
            atom_dim=atom_dim,
            motif_embed_dim=synthon_dim,
            num_motif_types=num_motif_types,
        )

        self.attention = nn.MultiheadAttention(
            atom_dim, num_heads=4, batch_first=True
        )

        in_dim = atom_dim + synthon_dim + (1 if use_hbond_prior else 0)
        self.synthon_classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.synthon_mlp = nn.Sequential(
            nn.Linear(atom_dim + synthon_dim, synthon_dim),
            nn.ReLU(),
            nn.Linear(synthon_dim, synthon_dim),
        )

        if use_hbond_prior:
            self.hbond_prior = HBondPrior()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -1.0)

    def _aggregate_motifs_to_atoms(self, motif_emb, mol_motifs, num_atoms, device):
        if motif_emb.shape[0] == 0 or num_atoms == 0:
            return torch.zeros(num_atoms, motif_emb.shape[1] if motif_emb.numel() > 0 else 0,
                               device=device)

        atom_motif_emb = torch.zeros(num_atoms, motif_emb.shape[1], device=device)
        atom_counts = torch.zeros(num_atoms, device=device)

        for i, motif in enumerate(mol_motifs):
            if i >= motif_emb.shape[0]:
                break
            for atom_id in motif.get('atom_ids', []):
                if 0 <= atom_id < num_atoms:
                    atom_motif_emb[atom_id] += motif_emb[i]
                    atom_counts[atom_id] += 1

        atom_counts = atom_counts.clamp(min=1.0).unsqueeze(-1)
        return atom_motif_emb / atom_counts

    def forward(self, atom_embeddings, data_batch, motifs, type_to_idx):
        batch_size = data_batch.max().item() + 1
        device = atom_embeddings.device
        out_dim = self.synthon_mlp[-1].out_features

        synthons_out = []
        scores_out = []

        for b in range(batch_size):
            mask = (data_batch == b)
            atom_emb = atom_embeddings[mask]
            num_atoms = atom_emb.shape[0]

            mol_motifs = motifs[b] if (motifs is not None
                                       and b < len(motifs)
                                       and isinstance(motifs[b], list)) else []

            if num_atoms == 0:
                synthons_out.append(torch.zeros(self.max_synthons, out_dim, device=device))
                scores_out.append(torch.zeros(self.max_synthons, device=device))
                continue

            if mol_motifs:
                motif_emb = self.motif_encoder(atom_emb, mol_motifs, type_to_idx)
            else:
                motif_emb = torch.zeros(0, out_dim, device=device)

            atom_motif_emb = self._aggregate_motifs_to_atoms(
                motif_emb, mol_motifs, num_atoms, device
            )
            if atom_motif_emb.shape[1] != out_dim:
                atom_motif_emb = torch.zeros(num_atoms, out_dim, device=device)

            if self.use_hbond_prior and mol_motifs:
                hbond_prior = torch.zeros(num_atoms, device=device)
                for motif in mol_motifs:
                    motif_type = motif.get('type', '')
                    score = max(
                        self.hbond_prior.get_donor_score(motif_type),
                        self.hbond_prior.get_acceptor_score(motif_type),
                    )
                    for atom_id in motif.get('atom_ids', []):
                        if 0 <= atom_id < num_atoms:
                            hbond_prior[atom_id] = max(hbond_prior[atom_id], score)
                hbond_feat = hbond_prior.unsqueeze(-1)
            else:
                hbond_feat = torch.zeros(num_atoms, 1, device=device)

            if self.use_hbond_prior:
                atom_features = torch.cat([atom_emb, atom_motif_emb, hbond_feat], dim=1)
            else:
                atom_features = torch.cat([atom_emb, atom_motif_emb], dim=1)

            atom_importance = torch.sigmoid(
                self.synthon_classifier(atom_features).squeeze(-1)
            )

            top_k = min(self.max_synthons, num_atoms)
            top_indices = torch.topk(atom_importance, top_k).indices
            selected_atoms = atom_emb[top_indices]
            selected_motif = atom_motif_emb[top_indices]
            selected_scores = atom_importance[top_indices]

            synthon_emb = self.synthon_mlp(
                torch.cat([selected_atoms, selected_motif], dim=1)
            )
            synthon_emb = synthon_emb * selected_scores.unsqueeze(-1)

            if top_k < self.max_synthons:
                pad = self.max_synthons - top_k
                synthon_emb = torch.cat([
                    synthon_emb,
                    torch.zeros(pad, out_dim, device=device)
                ], dim=0)
                selected_scores = torch.cat([
                    selected_scores,
                    torch.zeros(pad, device=device)
                ], dim=0)

            synthons_out.append(synthon_emb)
            scores_out.append(selected_scores)

        return torch.stack(synthons_out, dim=0), torch.stack(scores_out, dim=0)