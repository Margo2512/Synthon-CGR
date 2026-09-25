import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm
from sklearn.metrics import average_precision_score


def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
     
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0.0
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    precision_class, recall_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    metrics['precision_class0'] = precision_class[0]
    metrics['recall_class0'] = recall_class[0]
    metrics['f1_class0'] = f1_class[0]
    metrics['support_class0'] = support_class[0]
    
    metrics['precision_class1'] = precision_class[1]
    metrics['recall_class1'] = recall_class[1]
    metrics['f1_class1'] = f1_class[1]
    metrics['support_class1'] = support_class[1]
    
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return metrics

def print_metrics(metrics, title="Metrics", log_func=None):
    log = log_func or print
    log_func(f"\n{'='*50}")
    log_func(f"{title}")
    log_func(f"{'='*50}")
    
    log_func(f"Accuracy:           {metrics['accuracy']:.4f}")
    log_func(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    log_func(f"Precision:          {metrics['precision']:.4f}")
    log_func(f"Recall:             {metrics['recall']:.4f}")
    log_func(f"F1-score:           {metrics['f1']:.4f}")
    log_func(f"F1-macro:           {metrics['f1_macro']:.4f}")
    log_func(f"F1-weighted:        {metrics['f1_weighted']:.4f}")
    log_func(f"MCC:                {metrics['mcc']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] > 0:
        log_func(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    
    log_func(f"\nConfusion Matrix:")
    log_func(f"  TP: {metrics['tp']:5d}   FP: {metrics['fp']:5d}")
    log_func(f"  FN: {metrics['fn']:5d}   TN: {metrics['tn']:5d}")
    
    log_func(f"\nMetrics by class:")
    log_func(f"  Class 0 (No Cocrystal, n={metrics['support_class0']}):")
    log_func(f"    Precision: {metrics['precision_class0']:.4f}, Recall: {metrics['recall_class0']:.4f}, F1: {metrics['f1_class0']:.4f}")
    log_func(f"  Class 1 (Cocrystal, n={metrics['support_class1']}):")
    log_func(f"    Precision: {metrics['precision_class1']:.4f}, Recall: {metrics['recall_class1']:.4f}, F1: {metrics['f1_class1']:.4f}")
    
    log_func(f"\nAdditional metrics:")
    log_func(f"  Specificity: {metrics['specificity']:.4f}")
    log_func(f"  NPV:         {metrics['npv']:.4f}")
    log_func(f"  FPR:         {metrics['fpr']:.4f}")
    log_func(f"  FNR:         {metrics['fnr']:.4f}")
    log_func(f"{'='*50}\n")

def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_threshold = 0, 0.5
    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1
    
def find_best_threshold_for_class0(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba > thresh).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1

def evaluate_with_optimal_threshold_class0(model, loader, device, type_to_idx=None, config=None):
    from .train import evaluate as _evaluate
    return _evaluate(model, loader, device,
                     type_to_idx=type_to_idx, config=config,
                     threshold=None)


def evaluate_synthons(model, loader, device, type_to_idx=None,
                     top_k_list=(1, 3, 5, 10, 20), config=None):
    model.eval()
    
    top_k_hits = {k: 0 for k in top_k_list}
    precision_at_k = {k: [] for k in top_k_list}
    recall_at_k = {k: [] for k in top_k_list}
    ndcg_at_k = {k: [] for k in top_k_list}
    map_at_k = {k: [] for k in top_k_list}
    mrr_values = []
    ap_scores = []
    n_positive = 0
    n_with_mask = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            batch_a = batch['batch_a'].to(device)
            batch_b = batch['batch_b'].to(device)
            labels = batch['labels'].to(device)
            motifs_a = batch.get('motifs_a', [])
            motifs_b = batch.get('motifs_b', [])
            candidates = batch.get('candidates', None)
            observed_mask = batch.get('observed_synthon_mask', None)

            if candidates is None or observed_mask is None:
                continue

            result = model(
                batch_a, batch_b, motifs_a, motifs_b, type_to_idx,
                candidates=candidates,
                observed_synthon_mask=None, 
            )
            candidate_scores = result['candidate_scores']
            candidate_indices = result['candidate_indices']

            batch_size = labels.shape[0]

            for b_idx in range(batch_size):
                if labels[b_idx].item() < 0.5:
                    continue

                mask = observed_mask[b_idx]
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.float)
                mask = mask.to(device)

                if mask.numel() == 0 or mask.sum() == 0:
                    continue

                n_with_mask += 1
                
                idxs = [i for i, bi in enumerate(candidate_indices) if bi == b_idx]
                idxs = [i for i in idxs if i < len(candidate_scores)]
                if not idxs:
                    continue
                scores = candidate_scores[idxs].cpu().numpy()

                target = mask.cpu().numpy().astype(float)
                if len(target) > len(scores):
                    target = target[:len(scores)]
                elif len(target) < len(scores):
                    target = np.concatenate([target, np.zeros(len(scores) - len(target))])
                target = (target > 0.5).astype(int)

                if target.sum() == 0:
                    continue

                total_pos = int(target.sum())

                order = np.argsort(-scores)
                ranked_target = target[order]   

                try:
                    ap = average_precision_score(target, scores)
                    ap_scores.append(ap)
                except ValueError:
                    pass

                first_pos = np.where(ranked_target == 1)[0]
                if len(first_pos) > 0:
                    rank_first = first_pos[0] + 1  
                    mrr_values.append(1.0 / rank_first)

                for k in top_k_list:
                    topk = ranked_target[:k]
                    hits = int(topk.sum())

                    if hits > 0:
                        top_k_hits[k] += 1

                    precision_at_k[k].append(hits / max(k, 1))

                    recall_at_k[k].append(hits / total_pos)

                    dcg = _dcg_at_k(topk)
                    idcg = _idcg_at_k(min(total_pos, k))
                    ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0.0)

                    ap_k = _average_precision_at_k(topk, total_pos, k)
                    map_at_k[k].append(ap_k)

    metrics = {'n_positive_with_mask': n_with_mask}
    
    for k in top_k_list:
        metrics[f'hit_rate@{k}'] = top_k_hits[k] / max(n_with_mask, 1)
        metrics[f'recall@{k}'] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0
        metrics[f'precision@{k}'] = float(np.mean(precision_at_k[k])) if precision_at_k[k] else 0.0
        metrics[f'ndcg@{k}'] = float(np.mean(ndcg_at_k[k])) if ndcg_at_k[k] else 0.0
        metrics[f'map@{k}'] = float(np.mean(map_at_k[k])) if map_at_k[k] else 0.0

    metrics['mrr'] = float(np.mean(mrr_values)) if mrr_values else 0.0

    metrics['map'] = float(np.mean(ap_scores)) if ap_scores else 0.0

    metrics['mean_average_precision'] = metrics['map']

    return metrics


def _dcg_at_k(ranked_target):
    if len(ranked_target) == 0:
        return 0.0
    gains = ranked_target / np.log2(np.arange(2, len(ranked_target) + 2))
    return float(gains.sum())


def _idcg_at_k(num_relevant):
    if num_relevant == 0:
        return 0.0
    positions = np.arange(1, num_relevant + 1)
    gains = 1.0 / np.log2(positions + 1)
    return float(gains.sum())


def _average_precision_at_k(ranked_target, total_pos, k):
    if total_pos == 0:
        return 0.0
    topk = ranked_target[:k]
    hits = np.cumsum(topk)               
    positions = np.arange(1, len(topk) + 1)
    precision_at_j = hits / positions   
    ap_k = (precision_at_j * topk).sum() / min(total_pos, k)
    return float(ap_k)

def print_synthon_metrics(metrics, log_func=None):
    log = log_func or print
    log("\n" + "=" * 60)
    log("SYNTHONS METRICS (only positive pairs with HBAT)")
    log("=" * 60)
    log(f"  Pairs with HBAT markup: {metrics['n_positive_with_mask']}")

    log(f"\n  MAP:       {metrics['map']:.4f}")
    log(f"  MRR:                       {metrics['mrr']:.4f}")

    log(f"\n  HitRate@K (the proportion of pairs where at least 1 is correct in the top K):")
    for k in [1, 3, 5, 10, 20]:
        if f'hit_rate@{k}' in metrics:
            log(f"    HitRate@{k:<2}  {metrics[f'hit_rate@{k}']:.4f}")

    log(f"\n  Recall@K (the proportion of the correct ones in the top K):")
    for k in [1, 3, 5, 10, 20]:
        if f'recall@{k}' in metrics:
            log(f"    Recall@{k:<2}   {metrics[f'recall@{k}']:.4f}")

    log(f"\n  nDCG@K (ranking quality based on position):")
    for k in [1, 3, 5, 10, 20]:
        if f'ndcg@{k}' in metrics:
            log(f"    nDCG@{k:<2}     {metrics[f'ndcg@{k}']:.4f}")

    log(f"\n  MAP@K (Average Precision, cut off by K):")
    for k in [1, 3, 5, 10, 20]:
        if f'map@{k}' in metrics:
            log(f"    MAP@{k:<2}      {metrics[f'map@{k}']:.4f}")

    log(f"\n  Precision@K:")
    for k in [1, 3, 5, 10, 20]:
        if f'precision@{k}' in metrics:
            log(f"    Prec@{k:<2}     {metrics[f'precision@{k}']:.4f}")

    log("=" * 60)

def cross_val_threshold(model, val_loader, device, type_to_idx,
                       config, n_splits=5):
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
    import torch
    from tqdm import tqdm

    model.eval()
    is_v3 = hasattr(model, 'candidate_initializer') and hasattr(model, 'motif_encoder')
    is_v2 = hasattr(model, 'synthon_generator_a') or hasattr(model, 'evaluator')

    all_probas, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="CV: collecting probas"):
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
                    observed_synthon_mask=None,
                )
                logits = result['logits']
            elif is_v2:
                motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
                motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
                res = model(batch_a, batch_b, motifs_a, motifs_b, type_to_idx)
                logits = res['logits'] if isinstance(res, dict) else res[0]
            else:
                logits = model(batch_a, batch_b)

            probas = torch.sigmoid(logits).cpu().numpy()
            all_probas.extend(probas.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probas = np.array(all_probas)
    all_labels = np.array(all_labels)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_thresholds = []

    for fold, (train_idx, _) in enumerate(kf.split(all_probas)):
        train_p = all_probas[train_idx]
        train_l = all_labels[train_idx]

        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.02):
            y_pred = (train_p > t).astype(int)
            f1 = f1_score(train_l, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        best_thresholds.append(best_t)
        print(f"  Fold {fold + 1}: threshold = {best_t:.3f}, F1 = {best_f1:.4f}")

    mean_t = float(np.mean(best_thresholds))
    std_t = float(np.std(best_thresholds))
    print(f"\nCV threshold: {mean_t:.3f} ± {std_t:.3f}")
    return mean_t, std_t