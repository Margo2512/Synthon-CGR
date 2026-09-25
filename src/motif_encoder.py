import torch
import torch.nn as nn


class MotifEncoder(nn.Module):
    def __init__(self, atom_dim=128, motif_embed_dim=128, num_motif_types=1000,
                 max_atom_idx=512):
        super().__init__()
        type_dim = motif_embed_dim // 4
        pos_dim  = motif_embed_dim // 4
        size_dim = motif_embed_dim // 8

        self.type_embedding = nn.Embedding(num_motif_types, type_dim)
        self.pos_embedding  = nn.Embedding(max_atom_idx + 1, pos_dim)
        self.size_proj      = nn.Linear(1, size_dim)
        self.max_atom_idx   = max_atom_idx

        self.fusion_mlp = nn.Sequential(
            nn.Linear(atom_dim + type_dim + pos_dim + size_dim, motif_embed_dim),
            nn.ReLU(),
            nn.Linear(motif_embed_dim, motif_embed_dim),
        )

    def forward(self, atom_embeddings, motifs, type_to_idx, batch_offset=0):
        device = atom_embeddings.device
        out_dim = self.fusion_mlp[-1].out_features
        if not motifs:
            return torch.zeros((0, out_dim), device=device)

        out = []
        for motif in motifs:
            if not isinstance(motif, dict):
                continue
            atom_ids = motif.get('atom_ids', [])
            valid_local = [a for a in atom_ids
                           if 0 <= a < atom_embeddings.shape[0]]
            if not valid_local:
                continue

            idx_t = torch.tensor(valid_local, device=device, dtype=torch.long)
            atom_embs = atom_embeddings[idx_t]        
            pooled = atom_embs.mean(dim=0)         

            pos_ids = torch.tensor(
                [min(a, self.max_atom_idx) for a in valid_local],
                device=device, dtype=torch.long
            )
            pos_emb = self.pos_embedding(pos_ids).mean(dim=0)

            type_idx = type_to_idx.get(motif.get('type', ''), 0)
            type_idx = min(type_idx, self.type_embedding.num_embeddings - 1)
            type_emb = self.type_embedding(
                torch.tensor([type_idx], device=device, dtype=torch.long)
            ).squeeze(0)

            size_scalar = torch.tensor(
                [[float(len(valid_local))]], device=device
            )
            size_emb = self.size_proj(torch.log1p(size_scalar)).squeeze(0)

            combined = torch.cat([pooled, type_emb, pos_emb, size_emb], dim=-1)
            out.append(self.fusion_mlp(combined))

        if not out:
            return torch.zeros((0, out_dim), device=device)
        return torch.stack(out, dim=0)