import torch
import torch.nn as nn
from .hbond_prior import HBondPrior as _HP


class CandidateInitializer(nn.Module):
    def __init__(self, motif_dim=128, candidate_dim=128):
        super().__init__()
        extra = 8
        self.mlp = nn.Sequential(
            nn.Linear(motif_dim * 4 + extra, candidate_dim),
            nn.ReLU(),
            nn.Linear(candidate_dim, candidate_dim),
        )
        self._prior = _HP()

    def _motif_feats(self, motif, motif_type):
        d = self._prior.get_donor_score(motif_type)
        a = self._prior.get_acceptor_score(motif_type)
        size = len(motif.get('atom_ids', [])) if isinstance(motif, dict) else 0
        return d, a, size

    def forward(self, motif_emb_a, motif_emb_b, candidates,
                candidate_batch_indices=None, batch_offsets_a=None, batch_offsets_b=None):
        if candidates and isinstance(candidates[0], list):
            flat = []
            for sub in candidates:
                flat.extend(sub if isinstance(sub, list) else [sub])
            candidates = flat

        device = motif_emb_a.device
        out_dim = self.mlp[-1].out_features

        if not candidates:
            return torch.zeros(0, out_dim, device=device)

        if candidate_batch_indices is None:
            candidate_batch_indices = [0] * len(candidates)

        out = []
        for i, cand in enumerate(candidates):
            left_id  = cand.get('left_motif_id', 0)
            right_id = cand.get('right_motif_id', 0)
            b_idx = candidate_batch_indices[i] if i < len(candidate_batch_indices) else 0

            left_gid  = left_id  + (batch_offsets_a[b_idx] if batch_offsets_a is not None else 0)
            right_gid = right_id + (batch_offsets_b[b_idx] if batch_offsets_b is not None else 0)

            if 0 <= left_gid < motif_emb_a.shape[0]:
                l_emb = motif_emb_a[left_gid]
            else:
                l_emb = torch.zeros(motif_emb_a.shape[1] if motif_emb_a.dim() > 1 else 128,
                                    device=device)

            if 0 <= right_gid < motif_emb_b.shape[0]:
                r_emb = motif_emb_b[right_gid]
            else:
                r_emb = torch.zeros(motif_emb_b.shape[1] if motif_emb_b.dim() > 1 else 128,
                                    device=device)

            left_motif  = cand.get('left_motif', {})
            right_motif = cand.get('right_motif', {})
            l_d, l_a, l_sz = self._motif_feats(left_motif,  cand.get('left_type',  ''))
            r_d, r_a, r_sz = self._motif_feats(right_motif, cand.get('right_type', ''))

            scalars = torch.tensor([
                float(cand.get('prior', 0.05)),
                1.0 if cand.get('is_halogen', False) else 0.0,
                float(l_d), float(l_a),
                float(r_d), float(r_a),
                min(l_sz, 20) / 20.0,
                min(r_sz, 20) / 20.0,
            ], device=device, dtype=torch.float)

            combined = torch.cat([
                l_emb, r_emb,
                torch.abs(l_emb - r_emb),
                l_emb * r_emb,
                scalars,
            ], dim=-1)
            out.append(self.mlp(combined))

        return torch.stack(out, dim=0)