import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_weight = targets * (1 - self.alpha) + (1 - targets) * self.alpha
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CocrystalLossV2(nn.Module):
    def __init__(self, lambda_sparsity=1.0, lambda_diversity=0.01,
                lambda_hbond=0.001, lambda_local=0.1, 
                lambda_synthon=0.3, pos_weight=None):
        super().__init__()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_diversity = lambda_diversity
        self.lambda_hbond = lambda_hbond     
        self.lambda_local = lambda_local
        self.lambda_synthon = lambda_synthon
        self.pos_weight = pos_weight
        
    def forward(self, logits, labels, synthon_scores_a, synthon_scores_b, 
                pair_scores, model=None, hbond_prior=None, motifs_a=None, 
                motifs_b=None, use_focal_loss=False, focal_alpha=0.75, 
                focal_gamma=2.0, pos_weight=None, local_loss=None,
                synthon_loss=None):
        if use_focal_loss:
            probs = torch.sigmoid(logits)
            ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            p_t = probs * labels + (1 - probs) * (1 - labels)
            focal_weight = (1 - p_t) ** focal_gamma
            alpha_weight = labels * (1 - focal_alpha) + (1 - labels) * focal_alpha
            bce_loss = (alpha_weight * focal_weight * ce_loss).mean()
        else:
            if pos_weight is not None:  
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )
            elif self.pos_weight is not None:
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=self.pos_weight
                )
            else:
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        sparsity_loss = (
            torch.mean(synthon_scores_a) + torch.mean(synthon_scores_b)
        ) / 2
        
        diversity_loss = 0
        for batch_idx in range(synthon_scores_a.shape[0]):
            scores_a = synthon_scores_a[batch_idx]
            scores_a_norm = scores_a / (scores_a.sum() + 1e-8)
            entropy = -torch.sum(scores_a_norm * torch.log(scores_a_norm + 1e-8))
            diversity_loss += entropy
        
        diversity_loss = diversity_loss / synthon_scores_a.shape[0]
        
        pair_entropy = 0
        for batch_idx in range(pair_scores.shape[0]):
            scores = pair_scores[batch_idx].flatten()
            scores_norm = F.softmax(scores, dim=-1)
            entropy = -torch.sum(scores_norm * torch.log(scores_norm + 1e-8))
            pair_entropy += entropy
        
        pair_entropy = pair_entropy / pair_scores.shape[0]

        hbond_loss = 0
        if hbond_prior is not None and motifs_a is not None and motifs_b is not None:
            hbond_penalty = 0
            hbond_reward = 0
            count = 0
            
            for batch_idx in range(len(motifs_a)):
                mol_motifs_a = motifs_a[batch_idx] if batch_idx < len(motifs_a) else []
                mol_motifs_b = motifs_b[batch_idx] if batch_idx < len(motifs_b) else []
                
                for i, motif_a in enumerate(mol_motifs_a):
                    for j, motif_b in enumerate(mol_motifs_b):
                        if i >= synthon_scores_a.shape[1] or j >= synthon_scores_b.shape[1]:
                            continue
                            
                        type_a = motif_a.get('type', '')
                        type_b = motif_b.get('type', '')
                        
                        compatibility = hbond_prior.get_compatibility(type_a, type_b)
                        
                        if compatibility > 0.5:
                            score = pair_scores[batch_idx, i, j]
                            hbond_reward += compatibility * score
                            count += 1
                        else:
                            score = pair_scores[batch_idx, i, j]
                            hbond_penalty += score * 0.1
                            count += 1
            
            if count > 0:
                hbond_loss = -(hbond_reward / count) + (hbond_penalty / count)
        
        stability_loss = torch.mean(torch.abs(logits)) * 0.01

        if local_loss is None:
            local_loss = 0

        total_loss = (bce_loss + 
                      self.lambda_sparsity * sparsity_loss - 
                      self.lambda_diversity * diversity_loss +
                      0.01 * pair_entropy +
                      self.lambda_hbond * hbond_loss +
                      stability_loss +
                      self.lambda_local * local_loss)
        
        if synthon_loss is not None:
            total_loss = total_loss + self.lambda_synthon * synthon_loss
            
        loss_dict = {
            'bce_loss': bce_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'pair_entropy_loss': pair_entropy.item(),
            'hbond_loss': hbond_loss.item() if isinstance(hbond_loss, torch.Tensor) else hbond_loss,
            'stability_loss': stability_loss.item() if isinstance(stability_loss, torch.Tensor) else stability_loss,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    