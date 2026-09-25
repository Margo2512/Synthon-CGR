import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .metrics import calculate_metrics, evaluate_with_optimal_threshold_class0, find_best_threshold_for_class0, find_best_threshold
from .losses import CocrystalLossV2, FocalLoss
from .config import Config

def train_epoch(model, loader, optimizer, device, pos_weight=None, type_to_idx=None, criterion=None, config=None, epoch=0):
    model.train()
    total_loss = 0
    loss_dict_avg = {}
    
    is_v3 = hasattr(model, 'candidate_initializer') and hasattr(model, 'motif_encoder')
    is_v2 = hasattr(model, 'synthon_generator_a') or hasattr(model, 'evaluator')
    
    first_batch = True
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        if batch is None:
            continue
        
        if first_batch:
            if 'has_mask' in batch:
                has_masks = batch['has_mask']
                num_with_masks = sum(has_masks)
            first_batch = False
        
        batch_a = batch['batch_a'].to(device)
        batch_b = batch['batch_b'].to(device)
        labels = batch['labels'].to(device)
        
        if is_v3:
            motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
            motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
            candidates = batch.get('candidates', None)
            observed_mask = batch.get('observed_synthon_mask', None)
            has_mask = batch.get('has_mask', None)

            if batch_idx % 10 == 0: 
                if observed_mask is not None:
                    for b_idx, mask in enumerate(observed_mask):
                        if mask.sum() > 0:
                            print(f"  Batch {batch_idx}, molecule {b_idx}: the mask contains {mask.sum().item()} units")
                            print(f"    Mask example: {mask[:10]}")
                            break
            
            if observed_mask is not None:
                has_ones = False
                for mask in observed_mask:
                    if mask.sum() > 0:
                        has_ones = True
                        break
                if not has_ones and batch_idx < 5:
                    print(f"  [train_epoch] WARNING: All masks in the {batch_idx} batch are null!")

            result = model(
                batch_a, batch_b, motifs_a, motifs_b, type_to_idx,
                candidates=candidates,
                observed_synthon_mask=observed_mask,
                has_mask=has_mask
            )

            if batch_idx == 0 and epoch == 0:
                if observed_mask is not None:
                    total_ones = sum(m.sum().item() for m in observed_mask if isinstance(m, torch.Tensor))
                    print(f"[V3 DIAG] observed_mask total ones: {total_ones}")
                    print(f"[V3 DIAG] p_global mean: {result['p_global'].mean():.4f}")
                    print(f"[V3 DIAG] p_syn mean: {result['p_syn'].mean():.4f}")
                    if len(result['candidate_scores']) > 1:
                        print(f"[V3 DIAG] candidate_scores mean: {result['candidate_scores'].mean():.4f}, "
                            f"std: {result['candidate_scores'].std():.4f}")
                    else:
                        print(f"[V3 DIAG] candidate_scores: empty (Stage1)")
                    print(f"[V3 DIAG] synthon_loss: {result['synthon_loss'].item() if isinstance(result['synthon_loss'], torch.Tensor) else result['synthon_loss']:.4f}")
                    print(f"[V3 DIAG] contrastive_loss: {result['contrastive_loss'].item() if isinstance(result['contrastive_loss'], torch.Tensor) else result['contrastive_loss']:.4f}")
                    print(f"[V3 DIAG] p_final mean: {result['p_final'].mean():.4f}, std: {result['p_final'].std():.4f}")
                    print(f"[V3 DIAG] logits mean: {result['logits'].mean():.4f}, std: {result['logits'].std():.4f}")
            direct_loss = result.get('direct_loss', 0)
            logits = result['logits']
            synthon_loss = result.get('synthon_loss', 0)
            contrastive_loss = result.get('contrastive_loss', 0)
            
            candidate_scores = result.get('candidate_scores')

            flat_candidates = result.get('candidates', [])
            candidate_indices = result.get('candidate_indices', [])

            if len(candidate_scores) > 0 and len(flat_candidates) > 0:
                sorted_idx = torch.argsort(candidate_scores, descending=True)
                
                threshold = 0.7 
                filtered = []
                for idx in sorted_idx:
                    if candidate_scores[idx].item() > threshold:
                        filtered.append((idx, candidate_scores[idx].item()))
                
                if filtered:
                    print("Top 5 synthons (score > 0.7):")
                    for i, (score_idx, score) in enumerate(filtered[:5]):
                        cand = flat_candidates[score_idx]
                        print(f"  {i+1}. {cand.get('left_type', '?')} ↔ {cand.get('right_type', '?')} : {score:.4f}")
                else:
                    print("No synthons with score > 0.7")

            
            
            if candidate_scores is not None and len(candidate_scores) > 0:
                if has_mask is not None and any(has_mask):
                    k = min(3, len(candidate_scores))
                    top_k = torch.topk(candidate_scores, k=k).values
                    sparsity_loss = -torch.mean(top_k)
                else:
                    sparsity_loss = torch.zeros(1, device=device, requires_grad=True)
            else:
                sparsity_loss = torch.zeros(1, device=device, requires_grad=True)

            if batch_idx == 0:
                print(f"\n  [V3 Debug] p_global mean: {result['p_global'].mean().item():.4f}")
                print(f"  [V3 Debug] p_syn mean: {result['p_syn'].mean().item():.4f}")
                print(f"  [V3 Debug] p_final mean: {result['p_final'].mean().item():.4f}")
                print(f"  [V3 Debug] logits mean: {logits.mean().item():.4f}")
                print(f"  [V3 Debug] logits min: {logits.min().item():.4f}")
                print(f"  [V3 Debug] logits max: {logits.max().item():.4f}")
                print(f"  [V3 Debug] labels mean: {labels.mean().item():.4f}")

                mask_0 = (labels == 0)
                mask_1 = (labels == 1)
                if mask_0.sum() > 0:
                    print(f"  [V3 Debug] p_global for class 0: {result['p_global'][mask_0].mean().item():.4f}")
                if mask_1.sum() > 0:
                    print(f"  [V3 Debug] p_global for class 1: {result['p_global'][mask_1].mean().item():.4f}")

                if candidates is not None and len(candidates) > 0:
                    print(f"\n  [V3 Debug] The first 5 candidates:")
                    for i, cand_list in enumerate(candidates[:2]): 
                        print(f"    Molecule {i}: {len(cand_list)} candidates")
                        for j, cand in enumerate(cand_list[:3]):    
                            print(f"      {j+1}. left_type={cand.get('left_type', '')[:20]}... "
                                f"right_type={cand.get('right_type', '')[:20]}... "
                                f"prior={cand.get('prior', 0):.3f}")
                    
                    if 'candidate_scores' in result:
                        scores = result['candidate_scores']
                        if len(scores) > 0:
                            print(f"\n  [V3 Debug] Candidate scores (first 10):")
                            for i, score in enumerate(scores[:10]):
                                print(f"    {i}: {score.item():.4f}")
                            print(f"    ... mean={scores.mean().item():.4f}, "
                                f"max={scores.max().item():.4f}, "
                                f"min={scores.min().item():.4f}")
                        else:
                            print(f"\n  [V3 Debug] No candidate scores (tensor null)")

            if config.USE_FOCAL_LOSS:
                focal_loss_fn = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
                bce_loss = focal_loss_fn(logits, labels)
            else:
                if pos_weight is not None:
                    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                else:
                    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

            synthon_weight = 0.3

            if not model.use_cif:
                loss = bce_loss
            else:
                loss = bce_loss + synthon_weight * synthon_loss + 0.001 * sparsity_loss
                if contrastive_loss > 0:
                    loss = loss + 3.0 * contrastive_loss
            
            loss_dict = {
                'bce_loss': bce_loss.item(),
                'synthon_loss': synthon_loss.item() if isinstance(synthon_loss, torch.Tensor) else synthon_loss,
                'sparsity_loss': sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
                'total_loss': loss.item()
            }
            
            for key, value in loss_dict.items():
                if key not in loss_dict_avg:
                    loss_dict_avg[key] = 0
                loss_dict_avg[key] += value
        
        elif is_v2:
            motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
            motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
            
            result = model(batch_a, batch_b, motifs_a, motifs_b, type_to_idx)
            
            if batch_idx == 0 and epoch == 0:
                s_a = result['synthons_a']
                sc_a = result['synthon_scores_a']
                pp = result['pair_probs']
                pg = result['p_global']
                ps = result['p_syn']
                print(f"[DIAG] synthons_a.std(batch)={s_a.std(dim=0).mean():.4f}")
                print(f"[DIAG] synthons_a.std(synthons)={s_a.std(dim=1).mean():.4f}")
                print(f"[DIAG] synthons_a[0,0,:5]={s_a[0,0,:5]}")
                print(f"[DIAG] synthons_a[1,0,:5]={s_a[1,0,:5]}")
                print(f"[DIAG] scores_a[0]={sc_a[0]}")
                print(f"[DIAG] pair_probs unique={torch.unique(pp).shape[0]}")
                print(f"[DIAG] p_global std={pg.std():.4f}, p_syn std={ps.std():.4f}")
                print(f"[DIAG] alpha={torch.sigmoid(model.evaluator.alpha).item():.4f}")

            if isinstance(result, dict):
                logits = result['logits']
                synthon_scores_a = result.get('synthon_scores_a', torch.zeros(1, device=device))
                synthon_scores_b = result.get('synthon_scores_b', torch.zeros(1, device=device))
                pair_scores = result.get('pair_scores', torch.zeros(1, device=device))
                local_loss = result.get('local_loss', 0)
            else:
                logits, synthon_scores_a, synthon_scores_b, pair_scores, local_loss = result
            
            if batch_idx == 0 and epoch == 0:
                synthons_a = result['synthons_a']
                p_syn = result['p_syn']
                p_global = result['p_global']
                pair_probs = result['pair_probs']
                
                print(f"synthons_a std across synthons: {synthons_a.std(dim=1).mean():.4f}")
                print(f"synthons_a std across features: {synthons_a.std(dim=2).mean():.4f}")
                print(f"p_global: mean={p_global.mean():.4f}, std={p_global.std():.4f}")
                print(f"p_syn:    mean={p_syn.mean():.4f}, std={p_syn.std():.4f}")
                print(f"pair_probs unique: {torch.unique(pair_probs).shape[0]}")
                if hasattr(model, 'evaluator') and hasattr(model.evaluator, 'alpha'):
                    print(f"alpha: {torch.sigmoid(model.evaluator.alpha).item():.4f}")

                print(f"pair_scores (logits): min={pair_scores.min():.4f}, "
                    f"max={pair_scores.max():.4f}, sum={pair_scores.sum():.4f}")

            if criterion is not None:
                loss, loss_dict = criterion(
                    logits, labels, 
                    synthon_scores_a, synthon_scores_b,
                    pair_scores,
                    model=model,
                    hbond_prior=model.hbond_prior if hasattr(model, 'hbond_prior') else None,
                    motifs_a=motifs_a,
                    motifs_b=motifs_b,
                    pos_weight=pos_weight,
                    use_focal_loss=config.USE_FOCAL_LOSS if config else False,
                    focal_alpha=config.FOCAL_ALPHA if config else 0.75,
                    focal_gamma=config.FOCAL_GAMMA if config else 2.0,
                    local_loss=local_loss,
                    synthon_loss=0
                )
            else:
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = bce_loss
                loss_dict = {'bce_loss': bce_loss.item(), 'total_loss': loss.item()}

            p_syn_batch = result['p_syn']
            mask_neg = (labels < 0.5)
            if mask_neg.sum() > 0:
                p_syn_neg_penalty = p_syn_batch[mask_neg].mean() * 0.5
                loss = loss + p_syn_neg_penalty
            for key, value in loss_dict.items():
                if key not in loss_dict_avg:
                    loss_dict_avg[key] = 0
                loss_dict_avg[key] += value
        
        else:
            if 'motifs_a' in batch and type_to_idx is not None:
                motifs_a = batch['motifs_a']
                motifs_b = batch['motifs_b']
                logits = model(batch_a, batch_b, motifs_a, motifs_b, type_to_idx)
            else:
                logits = model(batch_a, batch_b)
            
            if pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss_dict = None
        
        optimizer.zero_grad()
        loss.backward()
        if batch_idx == 0 and epoch == 0:
            for name, p in model.named_parameters():
                if 'synthon_classifier' in name:
                    if p.grad is not None:
                        print(f"[GRAD] {name}: norm={p.grad.norm().item():.6f}")
                    else:
                        print(f"[GRAD] {name}: None")

            for name, p in model.named_parameters():
                if p.grad is not None and p.grad.norm().item() < 1e-6:
                    print(f"ZERO GRAD: {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    if loss_dict_avg:
        for key in loss_dict_avg:
            loss_dict_avg[key] /= len(loader)
        return total_loss / len(loader), loss_dict_avg
    
    return total_loss / len(loader)


def evaluate(model, loader, device, return_proba=False, type_to_idx=None, config=None, threshold=None):
    model.eval()
    
    all_labels = []
    all_probas = []
    
    is_v3 = hasattr(model, 'candidate_initializer') and hasattr(model, 'motif_encoder')
    is_v2 = hasattr(model, 'synthon_generator_a') or hasattr(model, 'evaluator')
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                continue
            
            batch_a = batch['batch_a'].to(device)
            batch_b = batch['batch_b'].to(device)
            labels = batch['labels'].to(device)
            
            if is_v3:
                motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
                motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
                candidates = batch.get('candidates', None)
                
                result = model(
                    batch_a, batch_b, motifs_a, motifs_b, type_to_idx,
                    candidates=candidates,
                    observed_synthon_mask=None
                )
                logits = result['logits']
                
            elif is_v2:
                motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
                motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
                result = model(batch_a, batch_b, motifs_a, motifs_b, type_to_idx)
                logits = result['logits'] if isinstance(result, dict) else result[0]
            
            else:
                if 'motifs_a' in batch and type_to_idx is not None:
                    logits = model(batch_a, batch_b, batch['motifs_a'],
                                   batch['motifs_b'], type_to_idx)
                else:
                    logits = model(batch_a, batch_b)
            
            probas = torch.sigmoid(logits)
            
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_proba = np.array(all_probas)

    if threshold is None:
        best_threshold, _ = find_best_threshold(y_true, y_proba)
        threshold_used = best_threshold
    else:
        threshold_used = threshold

    y_pred = (y_proba > threshold_used).astype(int)
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    metrics['threshold_used'] = threshold_used
    
    if return_proba:
        return metrics, y_proba
    
    return metrics


def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, 
                patience=10, class_weights=None, type_to_idx=None, log_func=None,
                weight_decay=0.01, best_model_path='best_model.pt', config=None):
    log_func(f"\nRun study")

    is_v3 = hasattr(model, 'candidate_initializer') and hasattr(model, 'motif_encoder')
    is_v2 = hasattr(model, 'synthon_generator_a') or hasattr(model, 'evaluator')
    is_v1 = hasattr(model, 'motif_encoder') and not is_v2 and not is_v3
    
    if is_v3:
        log_func(f"Model V3: Synthon-CGR + CIF supervision")
        log_func(f"   Use CIF: {model.use_cif if hasattr(model, 'use_cif') else False}")
        log_func(f"   Types of motifs: {model.num_motif_types}")
        criterion = None
    elif is_v2:
        log_func(f"Model V2: GNN + Synthons + Noisy-OR")
        log_func(f"   Use H-bond prior: {model.use_hbond_prior}")
        log_func(f"   Types of motifs: {model.num_motif_types}")
        
        from .losses import CocrystalLossV2
        pos_weight = None
        if class_weights is not None:
            pos_weight = torch.tensor(class_weights[1] / class_weights[0], device=device)
        
        criterion = CocrystalLossV2(
            lambda_sparsity=config.LAMBDA_SPARSITY if config else 1.0,
            lambda_diversity=config.LAMBDA_DIVERSITY if config else 0.01,
            lambda_hbond=config.LAMBDA_HBOND if config else 0.001,
            pos_weight=pos_weight
        )
    elif is_v1:
        log_func(f"Model V1: GNN + Motifs")
        log_func(f"   Types of motifs: {len(type_to_idx) if type_to_idx else 0}")
        criterion = None
    else:
        log_func(f"Model V0: GNN (without motifs)")
        criterion = None
    
    evaluator_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'evaluator' in name:
            evaluator_params.append(param)
        else:
            other_params.append(param)

    if evaluator_params:
        log_func(f"  Found {len(evaluator_params)} params Evaluator")
        log_func(f"  LR for Evaluator: {lr * config.EVALUATOR_LR_MULTIPLIER:.6f}")
        log_func(f"  LR for other: {lr:.6f}")
        
        optimizer = optim.Adam([
            {'params': other_params, 'lr': lr},
            {'params': evaluator_params, 'lr': lr * config.EVALUATOR_LR_MULTIPLIER}
        ], weight_decay=weight_decay)
    else:
        log_func(f"  No Evaluator parameters were found, we use a single LR")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    pos_weight = None
    if class_weights is not None:
        pos_weight = torch.tensor(class_weights[1] / class_weights[0], device=device)
    
    sample_batch = next(iter(train_loader))
    log_func(f"\nChecking the data in the batch:")
    log_func(f"  The keys are in the batch: {list(sample_batch.keys())}")

    if 'motifs_a' in sample_batch:
        log_func(f"The motifs are in the training data")
        
        if is_v2 or is_v1:
            motifs_a = sample_batch['motifs_a']
            total_motifs_a = sum(len(m) for m in motifs_a)
            log_func(f"  Total motifs of A in the first batch: {total_motifs_a}")
            
            if motifs_a and len(motifs_a) > 0:
                first_molecule_motifs = motifs_a[0]
                log_func(f"  Example of motifs for the first molecule:")
                if isinstance(first_molecule_motifs, list):
                    for i, motif in enumerate(first_molecule_motifs[:5]):
                        log_func(f"    {i+1}. {motif}")
                else:
                    log_func(f"    {first_molecule_motifs}")
            
            log_func(f"  Type motifs_a: {type(motifs_a)}")
            log_func(f"  Type first elem: {type(motifs_a[0]) if motifs_a else 'empty'}")
    else:
        log_func(f"There are no motifs in the training data")
        if is_v1 or is_v2:
            log_func(f"   ATTENTION: The { 'V1' if is_v1 else 'V2'} model requires motifs, but they are not found")
    
    best_val_acc = 0
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(epochs):
        if class_weights is not None and not is_v2:
            pos_weight_epoch = torch.tensor(
                [class_weights[1] / class_weights[0]], 
                device=device
            )
        else:
            pos_weight_epoch = pos_weight

        train_result = train_epoch(
            model, train_loader, optimizer, device, 
            pos_weight=pos_weight_epoch, 
            type_to_idx=type_to_idx, 
            criterion=criterion,
            config=config,
            epoch=epoch
        )
        
        if is_v3:
            train_loss, loss_dict = train_result
            if loss_dict:
                log_func(f"  BCE Loss: {loss_dict.get('bce_loss', 0):.4f}")
                log_func(f"  Synthon Loss: {loss_dict.get('synthon_loss', 0):.4f}")
        elif is_v2:
            train_loss, loss_dict = train_result
            if loss_dict:
                log_func(f"  BCE Loss: {loss_dict.get('bce_loss', 0):.4f}")
                log_func(f"  Sparsity: {loss_dict.get('sparsity_loss', 0):.4f}")
                log_func(f"  Diversity: {loss_dict.get('diversity_loss', 0):.4f}")
                log_func(f"  HBond: {loss_dict.get('hbond_loss', 0):.4f}")
        else:
            train_loss = train_result
            loss_dict = None

        train_losses.append(train_loss)
        
        val_metrics = evaluate(model, val_loader, device,
                       type_to_idx=type_to_idx, config=config,
                       threshold=None)
        val_metrics_history.append(val_metrics)
        
        scheduler.step(val_metrics['f1'])
        
        log_func(f"\nEpoch {epoch+1}/{epochs}")
        log_func(f"  Train Loss: {train_loss:.4f}")
        log_func(f"  Val F1:     {val_metrics['f1']:.4f}")
        log_func(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        log_func(f"  Val AUC:    {val_metrics['roc_auc']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            
            best_model_path = f"checkpoints/best_model_{config.__class__.__name__}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_loss': train_loss,
                'config_name': config.__class__.__name__
            }, best_model_path)
            
            log_func(f"    Improvement F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_func(f"\n  Early stopping. The best F1: {best_val_f1:.4f} on epoch {best_epoch}")
                break
        
        log_func("-"*60)
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    log_func(f"\nTraining stop")
    log_func(f"  The best F1: {best_val_f1:.4f} (epoch {best_epoch})")
    log_func(f"  The best Accuracy: {best_val_acc:.4f}")
    
    return model, train_losses, val_metrics_history