import torch
import torch.nn as nn


class SynthonReasoner(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, candidate_emb, candidate_batch_indices=None):
        if candidate_emb.numel() == 0:
            return candidate_emb

        n = candidate_emb.shape[0]

        if candidate_batch_indices is not None:
            if len(candidate_batch_indices) > n:
                candidate_batch_indices = candidate_batch_indices[:n]
            elif len(candidate_batch_indices) < n:
                pad = [0] * (n - len(candidate_batch_indices))
                candidate_batch_indices = list(candidate_batch_indices) + pad

        if candidate_batch_indices is None or n < 2:
            x = candidate_emb
        else:
            x = candidate_emb.clone()
            uniq = sorted(set(int(b) for b in candidate_batch_indices))
            for b in uniq:
                idx = [i for i, bi in enumerate(candidate_batch_indices) if int(bi) == b]
                idx = [i for i in idx if 0 <= i < n]   
                if len(idx) < 2:
                    continue
                sub = x[idx].unsqueeze(0)
                attn_out, _ = self.attention(sub, sub, sub)
                x[idx] = x[idx] + 0.3 * attn_out.squeeze(0)

        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        return self.final_norm(x)