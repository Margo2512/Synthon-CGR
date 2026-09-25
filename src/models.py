import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, GINEConv, global_mean_pool
from .motif_encoder import MotifEncoder
from .synthon_generator import SynthonGenerator
from .synthon_evaluator import SynthonEvaluator
from .hbond_prior import HBondPrior


def pool_motifs_per_molecule(motif_emb_list, batch_size, hidden_dim, device):
    if not motif_emb_list or all(m.shape[0] == 0 for m in motif_emb_list):
        return torch.zeros(batch_size, hidden_dim, device=device)

    non_empty = [m for m in motif_emb_list if m.shape[0] > 0]
    motif_emb_all = torch.cat(non_empty, dim=0)

    indices = []
    for b_idx in range(batch_size):
        n = motif_emb_list[b_idx].shape[0] if b_idx < len(motif_emb_list) else 0
        indices.extend([b_idx] * n)
    motif_batch = torch.tensor(indices, dtype=torch.long, device=device)

    if motif_batch.shape[0] != motif_emb_all.shape[0]:
        return torch.zeros(batch_size, hidden_dim, device=device)

    out = torch.zeros(batch_size, hidden_dim, device=device)
    counts = torch.zeros(batch_size, device=device)

    out.index_add_(0, motif_batch, motif_emb_all)
    counts.index_add_(0, motif_batch, torch.ones_like(motif_batch, dtype=torch.float))

    counts = counts.clamp(min=1.0)
    return out / counts.unsqueeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * (1 - self.alpha) + (1 - targets) * self.alpha
        focal_loss = alpha_weight * focal_weight * ce_loss
        return focal_loss.mean()
    
class SimpleGNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.lin = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        atom_emb = self.lin(x)
        
        return atom_emb

class GATGNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=128, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
        
        self.lin = nn.Linear(hidden_dim * heads, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        atom_emb = self.lin(x)
        return atom_emb
    
class GATv2GNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=128, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        self.lin = nn.Linear(hidden_dim * heads, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)  
            x = F.relu(x)
            x = self.dropout(x)
        
        atom_emb = self.lin(x)
        return atom_emb

class GINGNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.mlps = nn.ModuleList()
        self.mlps.append(self._build_mlp(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.mlps.append(self._build_mlp(hidden_dim, hidden_dim))
        
        self.convs = nn.ModuleList()
        for mlp in self.mlps:
            self.convs.append(GINConv(mlp, train_eps=True))
        
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _build_mlp(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        atom_emb = self.lin(x)
        return atom_emb
    
class GINEGNN(nn.Module):
    def __init__(self, input_dim=6, edge_dim=4, hidden_dim=128, output_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.mlps = nn.ModuleList()
        self.mlps.append(self._build_mlp(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.mlps.append(self._build_mlp(hidden_dim, hidden_dim))
        
        self.convs = nn.ModuleList()
        for mlp in self.mlps:
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_dim))
        
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _build_mlp(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        atom_emb = self.lin(x)
        return atom_emb
    
class CocrystalModelV0(nn.Module):
    def __init__(self, input_dim=6, edge_dim=4, hidden_dim=128, num_layers=2, model_type='GCN', gat_heads=4, dropout=0.2):
        super().__init__()
        
        if model_type == 'GCN':
            self.gnn = SimpleGNN(input_dim, hidden_dim, hidden_dim, num_layers, dropout=dropout)
        elif model_type == 'GAT':
            self.gnn = GATGNN(input_dim, hidden_dim, hidden_dim, num_layers, heads=gat_heads, dropout=dropout)
        elif model_type == 'GATv2':
            self.gnn = GATv2GNN(input_dim, hidden_dim, hidden_dim, num_layers, heads=gat_heads, dropout=dropout)
        elif model_type == 'GIN':
            self.gnn = GINGNN(input_dim, hidden_dim, hidden_dim, num_layers, dropout=dropout)
        elif model_type == 'GINE':
            self.gnn = GINEGNN(input_dim, edge_dim, hidden_dim, hidden_dim, num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, data_a, data_b):
        atom_emb_a = self.gnn(data_a)  
        atom_emb_b = self.gnn(data_b)   
        
        mol_vec_a = global_mean_pool(atom_emb_a, data_a.batch)  
        mol_vec_b = global_mean_pool(atom_emb_b, data_b.batch)   
        
        sum_vec = mol_vec_a + mol_vec_b                    
        diff_vec = torch.abs(mol_vec_a - mol_vec_b)        
        prod_vec = mol_vec_a * mol_vec_b                 
        
        combined = torch.cat([sum_vec, diff_vec, prod_vec], dim=1)  
        
        logits = self.classifier(combined)
        
        return logits.squeeze()

class CocrystalModelV1(nn.Module):
    def __init__(self, input_dim=18, edge_dim=4, hidden_dim=128, num_layers=2, 
                 num_motif_types=100, gat_heads=4, model_type='GATv2', dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        if model_type == 'GATv2':
            self.gnn = GATv2GNN(input_dim, hidden_dim, hidden_dim, num_layers, heads=gat_heads, dropout=dropout)
        elif model_type == 'GINE':
            self.gnn = GINEGNN(input_dim, edge_dim, hidden_dim, hidden_dim, num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.motif_encoder = MotifEncoder(
            atom_dim=hidden_dim,
            motif_embed_dim=hidden_dim,
            num_motif_types=num_motif_types
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.model_type = model_type
        
    def forward(self, data_a, data_b, motifs_a, motifs_b, type_to_idx):
        atom_emb_a = self.gnn(data_a)
        atom_emb_b = self.gnn(data_b)
        
        batch_size = data_a.batch.max().item() + 1
        device = atom_emb_a.device

        motif_embs_a = []
        for b_idx in range(batch_size):
            mask = (data_a.batch == b_idx)
            atom_indices = mask.nonzero(as_tuple=True)[0]
            if len(atom_indices) == 0:
                motif_embs_a.append(torch.zeros(0, self.hidden_dim, device=device))
                continue
            atom_emb_mol = atom_emb_a[atom_indices]
            motifs_mol = motifs_a[b_idx] if b_idx < len(motifs_a) and isinstance(motifs_a[b_idx], list) else []
            if motifs_mol:
                emb = self.motif_encoder(atom_emb_mol, motifs_mol, type_to_idx)
                motif_embs_a.append(emb)
            else:
                motif_embs_a.append(torch.zeros(0, self.hidden_dim, device=device))

        if motif_embs_a:
            motif_emb_a = torch.cat(motif_embs_a, dim=0)
        else:
            motif_emb_a = torch.zeros(0, self.hidden_dim, device=device)

        motif_embs_b = []
        for b_idx in range(batch_size):
            mask = (data_b.batch == b_idx)
            atom_indices = mask.nonzero(as_tuple=True)[0]
            if len(atom_indices) == 0:
                motif_embs_b.append(torch.zeros(0, self.hidden_dim, device=device))
                continue
            atom_emb_mol = atom_emb_b[atom_indices]
            motifs_mol = motifs_b[b_idx] if b_idx < len(motifs_b) and isinstance(motifs_b[b_idx], list) else []
            if motifs_mol:
                emb = self.motif_encoder(atom_emb_mol, motifs_mol, type_to_idx)
                motif_embs_b.append(emb)
            else:
                motif_embs_b.append(torch.zeros(0, self.hidden_dim, device=device))

        if motif_embs_b:
            motif_emb_b = torch.cat(motif_embs_b, dim=0)
        else:
            motif_emb_b = torch.zeros(0, self.hidden_dim, device=device)

        atom_vec_a = global_mean_pool(atom_emb_a, data_a.batch)
        atom_vec_b = global_mean_pool(atom_emb_b, data_b.batch)

        motif_vec_a = pool_motifs_per_molecule(
            motif_embs_a, batch_size, self.hidden_dim, device
        )
        motif_vec_b = pool_motifs_per_molecule(
            motif_embs_b, batch_size, self.hidden_dim, device
        )
        
        mol_vec_a = torch.cat([atom_vec_a, motif_vec_a], dim=1)
        mol_vec_b = torch.cat([atom_vec_b, motif_vec_b], dim=1)
        
        sum_vec = mol_vec_a + mol_vec_b
        diff_vec = torch.abs(mol_vec_a - mol_vec_b)
        prod_vec = mol_vec_a * mol_vec_b
        
        combined = torch.cat([sum_vec, diff_vec, prod_vec], dim=1)  
        logits = self.classifier(combined)
        return logits.squeeze()

class CocrystalModelV2(nn.Module):
    def __init__(self, input_dim=18, edge_dim=4, hidden_dim=128, 
                 synthon_dim=128, num_layers=2, gat_heads=4,
                 max_synthons_a=15, max_synthons_b=15,
                 use_hbond_prior=True, num_motif_types=1000,
                 model_type='GATv2', dropout=0.2,
                 pooling_mode='max', noise_prob=0.3,
                 evaluator_dropout=0.3,
                 prior_weight=0.01,
                 lambda_local=0.1):
        super().__init__()
        
        if model_type == 'GATv2':
            self.gnn = GATv2GNN(input_dim, hidden_dim, hidden_dim, num_layers, heads=gat_heads, dropout=dropout)
        elif model_type == 'GINE':
            self.gnn = GINEGNN(input_dim, edge_dim, hidden_dim, hidden_dim, num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.synthon_generator_a = SynthonGenerator(
            hidden_dim, synthon_dim, max_synthons_a,
            num_motif_types=num_motif_types,
            use_hbond_prior=use_hbond_prior,
        )
        self.synthon_generator_b = SynthonGenerator(
            hidden_dim, synthon_dim, max_synthons_b,
            num_motif_types=num_motif_types,
            use_hbond_prior=use_hbond_prior,
        )
        
        self.evaluator = SynthonEvaluator(
            synthon_dim, hidden_dim,
            use_hbond_prior=use_hbond_prior,
            pooling_mode=pooling_mode,
            prior_weight=prior_weight,
            noise_prob=noise_prob,
            dropout=evaluator_dropout,
            atom_dim=hidden_dim,
        )
        
        self.use_hbond_prior = use_hbond_prior
        if use_hbond_prior:
            self.hbond_prior = HBondPrior()
        
        self.num_motif_types = num_motif_types

        self.lambda_local = lambda_local
        
    def forward(self, data_a, data_b, motifs_a, motifs_b, type_to_idx):
        from torch_geometric.nn import global_mean_pool
        atom_emb_a = self.gnn(data_a)
        atom_emb_b = self.gnn(data_b)

        synthons_a, synthon_scores_a = self.synthon_generator_a(
            atom_emb_a, data_a.batch, motifs_a, type_to_idx
        )
        synthons_b, synthon_scores_b = self.synthon_generator_b(
            atom_emb_b, data_b.batch, motifs_b, type_to_idx
        )

        atom_vec_a = global_mean_pool(atom_emb_a, data_a.batch)
        atom_vec_b = global_mean_pool(atom_emb_b, data_b.batch)

        p_final, p_syn, p_global, pair_scores, pair_probs = self.evaluator(
            synthons_a, synthons_b,
            motifs_a, motifs_b, type_to_idx,
            atom_vec_a=atom_vec_a,
            atom_vec_b=atom_vec_b,
        )

        logits = torch.log(p_final / (1 - p_final + 1e-7))
        logits = torch.clamp(logits, -10, 10)

        local_loss = self._compute_local_loss(synthon_scores_a, synthon_scores_b)

        return {
            'logits': logits,
            'synthon_scores_a': synthon_scores_a,
            'synthon_scores_b': synthon_scores_b,
            'pair_scores': pair_scores,
            'local_loss': local_loss,
            'synthons_a': synthons_a,
            'synthons_b': synthons_b,
            'p_syn': p_syn,
            'p_global': p_global,
            'pair_probs': pair_probs,
        }
    
    def _compute_local_loss(self, scores_a, scores_b):
        batch_size = scores_a.shape[0]
        local_loss = 0
        
        for b in range(batch_size):
            s_a = scores_a[b] / (scores_a[b].sum() + 1e-8)
            s_b = scores_b[b] / (scores_b[b].sum() + 1e-8)
            
            entropy_a = -torch.sum(s_a * torch.log(s_a + 1e-8))
            entropy_b = -torch.sum(s_b * torch.log(s_b + 1e-8))
            
            local_loss = local_loss + entropy_a + entropy_b
        
        return local_loss / batch_size

def create_model(config):
    if config.MODEL_VERSION == 'V0':
        return CocrystalModelV0(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            model_type=config.MODEL_TYPE,
            gat_heads=config.GAT_HEADS,
            dropout=config.DROPOUT
        )
    
    elif config.MODEL_VERSION == 'V1':
        return CocrystalModelV1(
            input_dim=config.INPUT_DIM,
            edge_dim=config.EDGE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_motif_types=config.NUM_MOTIF_TYPES,
            gat_heads=config.GAT_HEADS,
            model_type=config.MODEL_TYPE,
            dropout=config.DROPOUT
        )

    elif config.MODEL_VERSION == 'V2':
        return CocrystalModelV2(
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
            lambda_local=config.LAMBDA_LOCAL if hasattr(config, 'LAMBDA_LOCAL') else 0.1
        )
    elif config.MODEL_VERSION == 'V3':
        from .models_v3 import CocrystalModelV3
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
    else:
        raise ValueError(f"Unknown MODEL_VERSION: {config.MODEL_VERSION}")
