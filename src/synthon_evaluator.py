import torch
import torch.nn as nn
import torch.nn.functional as F
from .hbond_prior import HBondPrior


class SynthonEvaluator(nn.Module):
    def __init__(self, synthon_dim=128, hidden_dim=128,
                 use_hbond_prior=True, pooling_mode='mean',
                 prior_weight=0.01, noise_prob=0.3, dropout=0.3,
                 atom_dim=128):
        super().__init__()
        self.use_hbond_prior = use_hbond_prior
        self.prior_weight = prior_weight
        self.pooling_mode = pooling_mode

        pair_in = synthon_dim * 3 + (1 if use_hbond_prior else 0)
        self.norm = nn.LayerNorm(pair_in)

        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.global_head = nn.Sequential(
            nn.Linear(atom_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(-0.5))  

        self.noise_prob = nn.Parameter(torch.tensor(noise_prob))

        if use_hbond_prior:
            self.hbond_prior = HBondPrior()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -1.0)  

    def forward(self, synthons_a, synthons_b, motifs_a, motifs_b, type_to_idx,
            atom_vec_a=None, atom_vec_b=None):
        batch_size = synthons_a.shape[0]
        num_a = synthons_a.shape[1]
        num_b = synthons_b.shape[1]
        device = synthons_a.device

        synthons_a_exp = synthons_a.unsqueeze(2).expand(-1, -1, num_b, -1)
        synthons_b_exp = synthons_b.unsqueeze(1).expand(-1, num_a, -1, -1)

        pair_features = torch.cat([
            synthons_a_exp,
            synthons_b_exp,
            torch.abs(synthons_a_exp - synthons_b_exp),
        ], dim=-1)

        if self.use_hbond_prior:
            hbond_scores = torch.zeros(batch_size, num_a, num_b, device=device)
            for b in range(batch_size):
                mol_a = motifs_a[b] if b < len(motifs_a) else []
                mol_b = motifs_b[b] if b < len(motifs_b) else []
                for i in range(min(num_a, len(mol_a))):
                    for j in range(min(num_b, len(mol_b))):
                        compat = self.hbond_prior.get_compatibility(
                            mol_a[i].get('type', ''),
                            mol_b[j].get('type', '')
                        )
                        hbond_scores[b, i, j] = compat
            hbond_scores = hbond_scores * self.prior_weight
            pair_features = torch.cat([pair_features, hbond_scores.unsqueeze(-1)], dim=-1)
        
        pair_features = self.norm(pair_features)
        pair_logits = self.pair_mlp(pair_features).squeeze(-1) 
        pair_probs = torch.clamp(torch.sigmoid(pair_logits), 0.01, 0.99)

        if not hasattr(self, '_diag_done'):
            print(f"[EVAL] pair_features.std over pairs: {pair_features.std(dim=(1,2)).mean():.4f}")
            print(f"[EVAL] pair_logits.std over pairs: {pair_logits.std(dim=(1,2)).mean():.4f}")
            self._diag_done = True

        flat = pair_probs.reshape(batch_size, -1)
        if self.pooling_mode == 'max':
            p_syn, _ = torch.max(flat, dim=1)
        elif self.pooling_mode == 'noisy_or':
            no_pair = torch.prod(1 - flat, dim=1)
            p_syn = 1 - no_pair * (1 - self.noise_prob)
        else: 
            p_syn = torch.mean(flat, dim=1)

        if atom_vec_a is not None and atom_vec_b is not None:
            sum_v = atom_vec_a + atom_vec_b
            diff_v = torch.abs(atom_vec_a - atom_vec_b)
            prod_v = atom_vec_a * atom_vec_b
            global_feat = torch.cat([sum_v, diff_v, prod_v], dim=1)
            p_global = torch.sigmoid(self.global_head(global_feat).squeeze(-1))
        else:
            p_global = torch.full((batch_size,), 0.5, device=device)

        alpha = torch.sigmoid(self.alpha)
        p_final = (1 - alpha) * p_global + alpha * p_syn
        p_final = torch.clamp(p_final, 0.01, 0.99)

        return p_final, p_syn, p_global, pair_logits, pair_probs