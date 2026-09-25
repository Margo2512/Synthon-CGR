import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import CocrystalModelV2, pool_motifs_per_molecule
from .synthon_generator import SynthonGenerator
from .synthon_evaluator import SynthonEvaluator
from .hbond_prior import HBondPrior
from .candidate_initializer import CandidateInitializer
from .synthon_reasoner import SynthonReasoner
from torch_geometric.nn import global_mean_pool


class CocrystalModelV3(CocrystalModelV2):
    def __init__(self, 
                 input_dim=18, 
                 edge_dim=4, 
                 hidden_dim=128, 
                 synthon_dim=128, 
                 num_layers=2, 
                 gat_heads=4,
                 max_synthons_a=15, 
                 max_synthons_b=15,
                 use_hbond_prior=True, 
                 num_motif_types=1000,
                 model_type='GATv2', 
                 dropout=0.2,
                 pooling_mode='max', 
                 noise_prob=0.3,
                 evaluator_dropout=0.3, 
                 prior_weight=0.01,
                 use_cif=True, 
                 lambda_mask=0.1, 
                 lambda_conflict=0.05):
        
        super().__init__(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            synthon_dim=synthon_dim,
            num_layers=num_layers,
            gat_heads=gat_heads,
            max_synthons_a=max_synthons_a,
            max_synthons_b=max_synthons_b,
            use_hbond_prior=use_hbond_prior,
            num_motif_types=num_motif_types,
            model_type=model_type,
            dropout=dropout,
            pooling_mode=pooling_mode,
            noise_prob=noise_prob,
            evaluator_dropout=evaluator_dropout,
            prior_weight=prior_weight
        )
        
        from .motif_encoder import MotifEncoder
        self.motif_encoder = MotifEncoder(
            atom_dim=hidden_dim,
            motif_embed_dim=hidden_dim,
            num_motif_types=num_motif_types
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.candidate_initializer = CandidateInitializer(
            motif_dim=hidden_dim,
            candidate_dim=synthon_dim
        )
        self.synthon_reasoner = SynthonReasoner(synthon_dim, num_layers=4, dropout=0.3)
        
        self.candidate_head = nn.Sequential(
            nn.Linear(synthon_dim, synthon_dim * 2),
            nn.LayerNorm(synthon_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(synthon_dim * 2, synthon_dim),
            nn.LayerNorm(synthon_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(synthon_dim, 1)
        )

        nn.init.constant_(self.candidate_head[-1].bias, -1.0)
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim * 6, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self.use_cif = use_cif
        self.lambda_mask = lambda_mask
        self.lambda_conflict = lambda_conflict
        self.alpha = nn.Parameter(torch.tensor(0.3))
        
        self.hidden_dim = hidden_dim
        print(f"global_head input dim: {hidden_dim * 6}")
        self.synthon_dim = synthon_dim
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.candidate_head[-1].bias, -1.0)

    def _contrastive_loss(self, candidate_scores, mask_target):
        pos_mask = mask_target > 0.5
        neg_mask = mask_target < 0.1
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_score = candidate_scores[pos_mask].mean()
            neg_score = candidate_scores[neg_mask].mean()
            margin = 0.5
            loss = torch.relu(neg_score - pos_score + margin)
            pos_penalty = torch.relu(0.7 - pos_score).mean() * 2.0
            loss = loss + pos_penalty
            return loss
        return torch.tensor(0.0, device=candidate_scores.device)

    def forward(self, data_a, data_b, motifs_a, motifs_b, type_to_idx,
            candidates=None, observed_synthon_mask=None, has_mask=None):
        batch_size = data_a.batch.max().item() + 1
        device = data_a.x.device
        
        atom_emb_a = self.gnn(data_a)
        atom_emb_b = self.gnn(data_b)
        
        motif_emb_a_list = []
        for b_idx in range(batch_size):
            mask = (data_a.batch == b_idx)
            atom_indices = mask.nonzero(as_tuple=True)[0]
            
            if len(atom_indices) > 0:
                atom_emb_mol = atom_emb_a[atom_indices]
                motifs_mol = motifs_a[b_idx] if b_idx < len(motifs_a) and isinstance(motifs_a[b_idx], list) else []
                if motifs_mol:
                    emb = self.motif_encoder(
                        atom_emb_mol, 
                        motifs_mol, 
                        type_to_idx,
                        batch_offset=atom_indices[0].item()
                    )
                else:
                    emb = torch.zeros(0, self.hidden_dim, device=device)
            else:
                emb = torch.zeros(0, self.hidden_dim, device=device)
            motif_emb_a_list.append(emb)

        motif_emb_b_list = []
        for b_idx in range(batch_size):
            mask = (data_b.batch == b_idx)
            atom_indices = mask.nonzero(as_tuple=True)[0]
            
            if len(atom_indices) > 0:
                atom_emb_mol = atom_emb_b[atom_indices]
                motifs_mol = motifs_b[b_idx] if b_idx < len(motifs_b) and isinstance(motifs_b[b_idx], list) else []
                if motifs_mol:
                    emb = self.motif_encoder(
                        atom_emb_mol, 
                        motifs_mol, 
                        type_to_idx,
                        batch_offset=atom_indices[0].item()
                    )
                else:
                    emb = torch.zeros(0, self.hidden_dim, device=device)
            else:
                emb = torch.zeros(0, self.hidden_dim, device=device)
            motif_emb_b_list.append(emb)

        motif_emb_a = torch.cat(motif_emb_a_list, dim=0) if motif_emb_a_list else torch.zeros(0, self.hidden_dim, device=device)
        motif_emb_b = torch.cat(motif_emb_b_list, dim=0) if motif_emb_b_list else torch.zeros(0, self.hidden_dim, device=device)
        
        if candidates is not None:
            flat_candidates = []
            candidate_batch_indices = []
            
            if candidates and isinstance(candidates[0], list):
                for b_idx, cand_list in enumerate(candidates):
                    if isinstance(cand_list, list):
                        for cand in cand_list:
                            flat_candidates.append(cand)
                            candidate_batch_indices.append(b_idx)
            else:
                for cand in candidates:
                    flat_candidates.append(cand)
                    candidate_batch_indices.append(0)
        else:
            flat_candidates = []
            candidate_batch_indices = []

        p_syn_list = []
        candidate_scores = torch.tensor([], device=device)
        candidate_logits = torch.tensor([], device=device)
        synthon_loss = torch.tensor(0.0, device=device)
        contrastive_loss = torch.tensor(0.0, device=device)

        if flat_candidates:
            batch_offsets_a = []
            batch_offsets_b = []
            off_a = 0
            off_b = 0
            for b_idx in range(batch_size):
                batch_offsets_a.append(off_a)
                batch_offsets_b.append(off_b)
                off_a += motif_emb_a_list[b_idx].shape[0]
                off_b += motif_emb_b_list[b_idx].shape[0]

            candidate_emb = self.candidate_initializer(
                motif_emb_a, motif_emb_b, flat_candidates,
                candidate_batch_indices=candidate_batch_indices,
                batch_offsets_a=batch_offsets_a,  
                batch_offsets_b=batch_offsets_b
            )
            candidate_emb = self.synthon_reasoner(
                candidate_emb,
                candidate_batch_indices=candidate_batch_indices
            )
            candidate_logits = self.candidate_head(candidate_emb).squeeze(-1) / self.temperature
            candidate_scores = torch.sigmoid(candidate_logits)
            candidate_scores = torch.clamp(candidate_scores, 0.01, 0.99)
            
            if observed_synthon_mask is not None:
                valid_masks = 0
                synthon_loss_total = 0.0
                contrastive_loss_total = 0.0
                pos_mask_count = 0
                
                for b_idx in range(batch_size):
                    if b_idx < len(observed_synthon_mask):
                        mask = observed_synthon_mask[b_idx]
                    else:
                        continue
                    
                    if not isinstance(mask, torch.Tensor):
                        mask = torch.tensor(mask, dtype=torch.float, device=device)
                    else:
                        mask = mask.to(device)
                    if mask.numel() == 0 or mask.sum() == 0:
                        continue
                    
                    has_pos = (mask > 0.5).any()
                    if has_pos:
                        pos_mask_count += 1

                    batch_indices = [i for i, idx in enumerate(candidate_batch_indices) if idx == b_idx]
                    if not batch_indices:
                        continue
                    
                    valid_indices = [i for i in batch_indices if i < len(candidate_logits)]
                    if not valid_indices:
                        continue
                    
                    batch_logits = candidate_logits[valid_indices]
                    
                    if len(mask) > len(batch_logits):
                        mask_target = mask[:len(batch_logits)]
                    elif len(mask) < len(batch_logits):
                        mask_target = torch.zeros(len(batch_logits), device=device)
                        mask_target[:len(mask)] = mask
                    else:
                        mask_target = mask
                    
                    num_pos = (mask_target > 0.5).sum().item()
                    num_neg = (mask_target < 0.5).sum().item()

                    if num_pos == 0:
                        bce_all = F.binary_cross_entropy_with_logits(
                            batch_logits, mask_target, reduction='none'
                        )
                        loss = bce_all.mean() * 0.1
                    else:
                        pw = min(10.0, max(2.0, num_neg / max(num_pos, 1)))
                        loss = F.binary_cross_entropy_with_logits(
                            batch_logits, mask_target,
                            pos_weight=torch.tensor([pw], device=device),
                            reduction='mean'
                        )

                    if num_pos > 0 and num_neg > 0:
                        contrast_loss = self._contrastive_loss(
                            torch.sigmoid(batch_logits),
                            mask_target
                        )
                        if contrast_loss > 0:
                            contrastive_loss_total += contrast_loss

                    synthon_loss_total += loss
                    valid_masks += 1
                
                if valid_masks > 0:
                    synthon_loss = synthon_loss_total / valid_masks
                    contrastive_loss = contrastive_loss_total / valid_masks

            for b_idx in range(batch_size):
                indices = [i for i, idx in enumerate(candidate_batch_indices) if idx == b_idx]
                if indices:
                    valid_indices = [i for i in indices if i < len(candidate_scores)]
                    if valid_indices:
                        batch_scores = candidate_scores[valid_indices]
                        k = min(3, len(batch_scores))
                        topk, _ = torch.topk(batch_scores, k=k)
                        p_syn_list.append(torch.mean(topk))
                    else:
                        p_syn_list.append(torch.tensor(0.0, device=device))
                else:
                    p_syn_list.append(torch.tensor(0.0, device=device))
            
            if p_syn_list:
                p_syn = torch.stack(p_syn_list)
            else:
                p_syn = torch.full((batch_size,), 0.0, device=device)
        else:
            p_syn = torch.full((batch_size,), 0.0, device=device)
        
        atom_vec_a = global_mean_pool(atom_emb_a, data_a.batch)
        atom_vec_b = global_mean_pool(atom_emb_b, data_b.batch)
        
        motif_vec_a = pool_motifs_per_molecule(
            motif_emb_a_list, batch_size, self.hidden_dim, device
        )
        motif_vec_b = pool_motifs_per_molecule(
            motif_emb_b_list, batch_size, self.hidden_dim, device
        )
        
        if motif_vec_a.shape[0] > 1:
            print(f"motif_vec_a shape: {motif_vec_a.shape}")
            print(f"motif_vec_a[0] vs motif_vec_a[1] diff: {(motif_vec_a[0] - motif_vec_a[1]).abs().mean().item():.4f}")

        sum_vec = atom_vec_a + atom_vec_b
        diff_vec = torch.abs(atom_vec_a - atom_vec_b)
        prod_vec = atom_vec_a * atom_vec_b

        sum_motif = motif_vec_a + motif_vec_b
        diff_motif = torch.abs(motif_vec_a - motif_vec_b)
        prod_motif = motif_vec_a * motif_vec_b

        global_emb = torch.cat([
            sum_vec, diff_vec, prod_vec,
            sum_motif, diff_motif, prod_motif,
        ], dim=1)
        
        p_global = torch.sigmoid(self.global_head(global_emb).squeeze(-1))
        
        if not self.use_cif:
            p_final = p_global      
        elif len(candidate_scores) > 0:
            alpha = torch.sigmoid(self.alpha)  # 0-1
            p_final = (1 - alpha) * p_global + alpha * p_syn
        else:
            p_final = p_global

        logits = torch.log(p_final / (1 - p_final + 1e-7))
        logits = torch.clamp(logits, -10, 10)

        return {
            'logits': logits,
            'candidate_scores': candidate_scores,
            'candidate_logits': candidate_logits,
            'p_final': p_final,
            'synthon_loss': synthon_loss,
            'contrastive_loss': contrastive_loss,
            'p_syn': p_syn,
            'p_global': p_global,
            'candidates': flat_candidates,
            'candidate_indices': candidate_batch_indices,
        }


def create_model_v3(config):
    return CocrystalModelV3(
        input_dim=config.INPUT_DIM,
        edge_dim=config.EDGE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_motif_types=config.NUM_MOTIF_TYPES,
        gat_heads=config.GAT_HEADS,
        model_type=config.MODEL_TYPE,
        max_synthons_a=config.MAX_SYNTHONS_A,
        max_synthons_b=config.MAX_SYNTHONS_B,
        synthon_dim=config.SYNTHON_DIM,
        use_hbond_prior=config.USE_HBOND_PRIOR,
        dropout=config.DROPOUT,
        pooling_mode=config.POOLING_MODE,
        noise_prob=config.NOISE_PROB,
        evaluator_dropout=config.EVALUATOR_DROPOUT,
        prior_weight=config.PRIOR_WEIGHT,
        use_cif=config.USE_CIF if hasattr(config, 'USE_CIF') else True,
        lambda_mask=config.LAMBDA_MASK if hasattr(config, 'LAMBDA_MASK') else 0.1,
        lambda_conflict=config.LAMBDA_CONFLICT if hasattr(config, 'LAMBDA_CONFLICT') else 0.05
    )