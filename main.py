import sys
import os
import time
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.config import (
    COMMON_TEST_CSV,
    Config,
    ConfigV0_CommonTest, ConfigV1_CommonTest, ConfigV2_CommonTest,
    ConfigV3_Stage1_CommonTest, ConfigV3_Stage2_CommonTest
)
from src.models import create_model
from src.data_utils import load_data, create_dataloaders, CocrystalDatasetWithSynthons, collate_fn_with_hbat, load_data_with_hbat, CocrystalDataset, collate_fn
from src.train import train_model, evaluate
from src.metrics import calculate_metrics, print_metrics, evaluate_synthons, print_synthon_metrics, cross_val_threshold, find_best_threshold
import random
import numpy as np


log_func = None

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score



def run_cross_validation(train_val_df, config, type_to_idx, n_splits=5, log_func=None):
    from src.models import create_model
    from src.train import train_model

    log = log_func or print
    log(f"\n{'='*60}")
    log(f"5-FOLD CROSS-VALIDATION on train+val ({len(train_val_df)} pairs)")
    log(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.RANDOM_STATE)

    cv_results = []

    for fold, (tr_idx, vl_idx) in enumerate(
        skf.split(train_val_df, train_val_df['result'])
    ):
        log(f"\n{'─'*60}")
        log(f"FOLD {fold + 1}/{n_splits}")
        log(f"{'─'*60}")

        df_tr = train_val_df.iloc[tr_idx].reset_index(drop=True)
        df_vl = train_val_df.iloc[vl_idx].reset_index(drop=True)

        train_ds = CocrystalDatasetWithSynthons(
            df_tr, use_motifs=True, use_masks=config.USE_MASKS,
        )
        val_ds = CocrystalDatasetWithSynthons(
            df_vl, use_motifs=True, use_masks=config.USE_MASKS,
        )

        from torch.utils.data import DataLoader
        g = torch.Generator()
        g.manual_seed(config.RANDOM_STATE + fold)
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            generator=g,  
            collate_fn=collate_fn_with_hbat, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_with_hbat, num_workers=0,
        )

        model = create_model(config).to(config.DEVICE)

        ckpt_path = f"checkpoints/cv_fold{fold + 1}_{config.__class__.__name__}.pt"

        trained_model, _, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.DEVICE,
            epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            patience=config.PATIENCE,
            class_weights=None,
            type_to_idx=type_to_idx,
            log_func=log,
            weight_decay=config.WEIGHT_DECAY,
            best_model_path=ckpt_path,
            config=config,
        )

        fold_threshold, _ = cross_val_threshold(
            trained_model, val_loader, config.DEVICE,
            type_to_idx=type_to_idx, config=config, n_splits=5,
        )

        fold_metrics = evaluate(
            trained_model, val_loader, config.DEVICE,
            type_to_idx=type_to_idx, config=config,
            threshold=fold_threshold,
        )

        cv_results.append(fold_metrics)
        log(f"\n  Fold {fold + 1} F1: {fold_metrics['f1']:.4f}, "
            f"AUC: {fold_metrics['roc_auc']:.4f}")

    keys = ['accuracy', 'f1', 'roc_auc', 'mcc',
            'precision_class0', 'recall_class0', 'f1_class0',
            'precision_class1', 'recall_class1', 'f1_class1']

    mean_metrics = {k: np.mean([r[k] for r in cv_results]) for k in keys}
    std_metrics  = {k: np.std([r[k] for r in cv_results]) for k in keys}

    log(f"\n{'='*60}")
    log(f"CROSS-VALIDATION RESULTS ({n_splits}-fold)")
    log(f"{'='*60}")
    for k in keys:
        log(f"  {k:<20} {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

    return cv_results, mean_metrics, std_metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def analyze_dataset(df):
    global log_func
    log = log_func

    log(f"\nTotal str: {len(df)}")
    
    class_counts = df['result'].value_counts()
    log(f"\nClass distribution:")
    log(f"  Class 0: {class_counts.get(0, 0):6d} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    log(f"  Class 1: {class_counts.get(1, 0):6d} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    unique_smiles1 = df['SMILES1'].nunique()
    unique_smiles2 = df['SMILES2'].nunique()
    log(f"\nUnique molecules:")
    log(f"  SMILES1: {unique_smiles1}")
    log(f"  SMILES2: {unique_smiles2}")
    
    log(f"\nТоп-5 the most frequent SMILES1:")
    top_smiles1 = df['SMILES1'].value_counts().head(5)
    for smiles, count in top_smiles1.items():
        log(f"  {smiles[:50]}... : {count}")
    
    log(f"\nТоп-5 the most frequent SMILES2:")
    top_smiles2 = df['SMILES2'].value_counts().head(5)
    for smiles, count in top_smiles2.items():
        log(f"  {smiles[:50]}... : {count}")
    
    log(f"\nSMILES Lengths:")
    log(f"  SMILES1 (mean): {df['SMILES1'].str.len().mean():.1f}")
    log(f"  SMILES1 (min-max): {df['SMILES1'].str.len().min()} - {df['SMILES1'].str.len().max()}")
    log(f"  SMILES2 (mean): {df['SMILES2'].str.len().mean():.1f}")
    log(f"  SMILES2 (min-max): {df['SMILES2'].str.len().min()} - {df['SMILES2'].str.len().max()}")

def plot_training_history(train_losses, val_metrics, plot_file_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    acc = [m['accuracy'] for m in val_metrics]
    axes[1].plot(acc)
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    
    f1 = [m['f1'] for m in val_metrics]
    axes[2].plot(f1)
    axes[2].set_title('Validation F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_file_path, dpi=150)
    plt.show()

def test_on_unseen(model, test_loader, device, type_to_idx=None, config=None, threshold=None):
    log = log_func
    log("Testing on test data")
    
    start_time = time.time()
    test_metrics = evaluate(model, test_loader, device, type_to_idx=type_to_idx, config=config, threshold=threshold)
    inference_time = time.time() - start_time

    log(f"  Inference time: {inference_time:.2f} sec")
    print_metrics(test_metrics, "Test metrics", log_func=log)
    
    return test_metrics

def main(config_class):
    global log_func
    config = config_class()

    log_file = open(config.LOG_FILE_PATH, 'w', encoding='utf-8')

    def log(msg):
        log_file.write(msg + '\n')
        log_file.flush()

    log_func = log

    total_start = time.time()

    log("="*60)
    log(f"Model {config.MODEL_VERSION} — {config.MODEL_TYPE}")
    log("="*60)

    log("Configuration:")
    log(f"  Data file: {config.CSV_FILE}")
    log(f"  Batch size: {config.BATCH_SIZE}")
    log(f"  Hidden dim: {config.HIDDEN_DIM}")
    log(f"  Epochs: {config.NUM_EPOCHS}")
    log(f"  Device: {config.DEVICE}")
    
    log("\n1. Dataset analysis")
    df = pd.read_csv(config.CSV_FILE)
    analyze_dataset(df)
    
    log("\n2. Data split")

    use_motifs = config.MODEL_VERSION in ('V1', 'V2', 'V3')

    df_test_common = pd.read_csv(COMMON_TEST_CSV)
    print(f"Common test: {len(df_test_common)} pairs")

    test_keys = set(
        df_test_common['SMILES1'].astype(str) + '|' +
        df_test_common['SMILES2'].astype(str)
    )

    if config.MODEL_VERSION == 'V3':
        use_cif = hasattr(config, 'USE_CIF') and config.USE_CIF
        
        if use_cif:
            if not hasattr(config, 'COD_CSV_PATH') or not hasattr(config, 'HBAT_PATH'):
                log(" Error: V3 requires COD_CSV_PATH and HBAT_PATH in the config")
                raise ValueError("Missing COD_CSV_PATH or HBAT_PATH in config")
            
            log(f"  COD_CSV_PATH: {config.COD_CSV_PATH}")
            log(f"  HBAT_PATH: {config.HBAT_PATH}")
            
            train_dataset, val_dataset, test_dataset, class_weights, type_to_idx = load_data_with_hbat(
                csv_path=config.CSV_FILE,
                cod_csv_path=config.COD_CSV_PATH,
                hbat_path=config.HBAT_PATH,
                cif_dir="Synthon-CGR/data/cif_files_all_without_duplicates",
                test_size=config.TEST_SIZE,
                val_size=config.VAL_SIZE,
                random_state=config.RANDOM_STATE,
                use_motifs=use_motifs,
                cache_file_path=config.CACHE_FILE_PATH,
                log_func=log,
                use_masks=config.USE_MASKS,
                use_candidates=True
            )

            df_train = train_dataset.data
            df_val = val_dataset.data
            df_test = test_dataset.data
            df_train_val = pd.concat([df_train, df_val]).reset_index(drop=True)

            train_ds, val_ds, test_ds, _, _ = load_data_with_hbat(
                csv_path=COMMON_TEST_CSV,                        
                cod_csv_path=config.COD_CSV_PATH,
                hbat_path=config.HBAT_PATH,
                cif_dir="Synthon-CGR/data/cif_files_all_without_duplicates",
                test_size=0.99,                                  
                val_size=0.005,
                random_state=42,
                use_motifs=True,
                cache_file_path="cache/motifs_test_common.pkl",
                log_func=log,
                use_masks=config.USE_MASKS,
                use_candidates=True
            )
            test_dataset = test_ds
            log(f"  Test (common): {len(test_dataset)}")
            log(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

            sample = train_dataset[0]
            log(f"\n=== DATA VERIFICATION ===")
            log(f"motifs_a: {sample.get('motifs_a', [])[:3] if sample.get('motifs_a') else 'EMPTY'}")
            log(f"motifs_b: {sample.get('motifs_b', [])[:3] if sample.get('motifs_b') else 'EMPTY'}")
            log(f"candidates: {len(sample.get('candidates', []))}")
            if sample.get('candidates'):
                log(f"First candidate: {sample['candidates'][0] if sample['candidates'] else 'None'}")
            log(f"has_mask: {sample.get('has_mask', False)}")
            log(f"observed_synthon_mask shape: {sample.get('observed_synthon_mask', torch.tensor([])).shape}")
            log("======================")
            
            train_loader, val_loader, test_loader = create_dataloaders(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                batch_size=config.BATCH_SIZE,
                use_balanced_sampler=config.USE_BALANCED_SAMPLER,
                collate_fn=collate_fn_with_hbat,  
                num_workers=config.NUM_WORKERS
            )
        else:
            train_dataset, val_dataset, test_dataset, class_weights, type_to_idx = load_data(
                csv_path=config.CSV_FILE,
                test_size=config.TEST_SIZE,
                val_size=config.VAL_SIZE,
                random_state=config.RANDOM_STATE,
                use_motifs=use_motifs,
                cache_file_path=config.CACHE_FILE_PATH,
                log_func=log
            )

            train_dataset.use_candidates = False
            val_dataset.use_candidates = False
            test_dataset.use_candidates = False

            # test_dataset = CocrystalDatasetWithSynthons(
            #     df_test_common, use_motifs=use_motifs, use_masks=config.USE_MASKS,
            # )
            
            df_train = train_dataset.data
            df_val = val_dataset.data
            df_test = test_dataset.data
            df_train_val = pd.concat([df_train, df_val]).reset_index(drop=True)
            log(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test (own): {len(df_test)}")
            log(f"  Train+Val: {len(df_train_val)}")

            train_loader, val_loader, test_loader = create_dataloaders(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                batch_size=config.BATCH_SIZE,
                use_balanced_sampler=config.USE_BALANCED_SAMPLER,
                collate_fn=collate_fn_with_hbat,
                num_workers=config.NUM_WORKERS,
            )
            sample = train_dataset[0]
            log(f"\n=== DATA VERIFICATION ===")
            log(f"motifs_a: {sample.get('motifs_a', [])[:3] if sample.get('motifs_a') else 'EMPTY'}")
            log(f"motifs_b: {sample.get('motifs_b', [])[:3] if sample.get('motifs_b') else 'EMPTY'}")
            log(f"candidates: {len(sample.get('candidates', []))}")
            if sample.get('candidates'):
                log(f"First candidate: {sample['candidates'][0] if sample['candidates'] else 'None'}")
            log(f"has_mask: {sample.get('has_mask', False)}")
            log(f"observed_synthon_mask shape: {sample.get('observed_synthon_mask', torch.tensor([])).shape}")
            log("======================")
    elif config.MODEL_VERSION in ('V0', 'V1', 'V2'):
        df_full = pd.read_csv(config.CSV_FILE)
        df_full['key'] = df_full['SMILES1'].astype(str) + '|' + df_full['SMILES2'].astype(str)
        df_train_pool = df_full[~df_full['key'].isin(test_keys)].drop(columns='key')
        log(f"  Train pool (без test_common): {len(df_train_pool)}")
        
        tmp_csv = "/tmp/train_no_test_common.csv"
        df_train_pool.to_csv(tmp_csv, index=False)

        train_dataset, val_dataset, test_dataset, class_weights, type_to_idx = load_data(
            csv_path=tmp_csv,                                 
            test_size=config.TEST_SIZE,                       
            val_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE,
            use_motifs=use_motifs,
            cache_file_path=config.CACHE_FILE_PATH,
            log_func=log,
        )
        

        print(f"[DIAG] type_to_idx size={len(type_to_idx)}")
        print(f"[DIAG] 'N[Car]' in type_to_idx: {'N[Car]' in type_to_idx}")
        print(f"[DIAG] sample keys: {list(type_to_idx.keys())[:5]}")

        if use_motifs:
            from src.graph_utils import mol_to_graph_with_motifs
            from src.motif_extractor import create_motif_vocabulary
            from tqdm import tqdm
            
            motifs_test_a = []
            motifs_test_b = []
            for _, row in tqdm(df_test_common.iterrows(), total=len(df_test_common), desc="Motifs for test"):
                _, m_a = mol_to_graph_with_motifs(row['SMILES1'])
                _, m_b = mol_to_graph_with_motifs(row['SMILES2'])
                motifs_test_a.append(m_a or [])
                motifs_test_b.append(m_b or [])
            
            df_test_common = df_test_common.copy()
            df_test_common['motifs_a'] = motifs_test_a
            df_test_common['motifs_b'] = motifs_test_b

            new_types = set()
            for m_a in motifs_test_a:
                for m in m_a:
                    if m['type'] not in type_to_idx:
                        new_types.add(m['type'])
            for m_b in motifs_test_b:
                for m in m_b:
                    if m['type'] not in type_to_idx:
                        new_types.add(m['type'])

            if new_types:
                log(f"Adding {len(new_types)} new types of motifs in type_to_idx")
                start = len(type_to_idx)
                for i, t in enumerate(sorted(new_types)):
                    type_to_idx[t] = start + i
                log(f"  type_to_idx now contains {len(type_to_idx)} types")
            else:
                log("All types of motifs from test are already in type_to_idx")

            n_unknown = 0
            for m_a in motifs_test_a:
                for m in m_a:
                    if m['type'] not in type_to_idx:
                        n_unknown += 1
            print(f"Unknown types of motifs in test A: {n_unknown}")

        test_dataset = CocrystalDataset(df_test_common, use_motifs=use_motifs)
        
        log(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test (common): {len(test_dataset)}")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config.BATCH_SIZE,
            use_balanced_sampler=config.USE_BALANCED_SAMPLER,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
        )
        
        df_train = train_dataset.data
        df_val = val_dataset.data
        df_test = test_dataset.data
    
    if config.MODEL_VERSION == 'V3':
        log("\n3. Cross-validation (5-fold on train+val)")
        cv_results = []

        log("\n4. Training of the final model on train+val")
        log(f"  Use everything {len(df_train_val)} pairs for training")

        full_train_ds = CocrystalDatasetWithSynthons(
            df_train_val, use_motifs=True, use_masks=config.USE_MASKS, use_candidates=True
        )

        from torch.utils.data import DataLoader
        g = torch.Generator()
        g.manual_seed(config.RANDOM_STATE)
        full_train_loader = DataLoader(
            full_train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            generator=g,  
            collate_fn=collate_fn_with_hbat, num_workers=0,
        )

        log("\n3. Creating a model")
        model_final = create_model(config).to(config.DEVICE)

        is_v3 = hasattr(model_final, 'candidate_initializer') and hasattr(model_final, 'motif_encoder')
        log(f"\nChecking the model version:")
        log(f"  is_v3: {is_v3}")
        log(f"  has candidate_initializer: {hasattr(model_final, 'candidate_initializer')}")
        log(f"  has motif_encoder: {hasattr(model_final, 'motif_encoder')}")
        log(f"  has synthon_generator_a: {hasattr(model_final, 'synthon_generator_a')}")
        log(f"  has evaluator: {hasattr(model_final, 'evaluator')}")


        log(f"  Version: {config.MODEL_VERSION}")
        log(f"  Model: {config.MODEL_TYPE}")
        total_params = sum(p.numel() for p in model_final.parameters())
        log(f"  Params: {total_params:,}")
        
        if hasattr(config, 'LOAD_PRETRAINED') and config.LOAD_PRETRAINED:
            if config.PRETRAINED_MODEL_PATH and os.path.exists(config.PRETRAINED_MODEL_PATH):
                log(f"  Loading pre-trained weights from: {config.PRETRAINED_MODEL_PATH}")
                checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE)
                
                if 'model_state_dict' in checkpoint:
                    model_final.load_state_dict(checkpoint['model_state_dict'])
                    log(f"   The weights are loaded (epoch {checkpoint.get('epoch', '?')})")
                    log(f"   Val F1 when saving: {checkpoint.get('val_metrics', {}).get('f1', 'N/A')}")
                else:
                    model_final.load_state_dict(checkpoint)
                    log("   Weights loaded (model only)")
                
                if hasattr(config, 'FREEZE_SYNTHON_LAYERS') and config.FREEZE_SYNTHON_LAYERS:
                    log("  Freezing layers for synthons...")
                    
                    if hasattr(model_final, 'evaluator'):
                        for param in model_final.evaluator.parameters():
                            param.requires_grad = False
                        log("    Frozen evaluator")
                    
                    if hasattr(model_final, 'candidate_initializer'):
                        for param in model_final.candidate_initializer.parameters():
                            param.requires_grad = False
                        log("    Frozen candidate_initializer")
                    
                    if hasattr(model_final, 'synthon_reasoner'):
                        for param in model_final.synthon_reasoner.parameters():
                            param.requires_grad = False
                        log("    Frozen synthon_reasoner")
                    
                    if hasattr(model_final, 'candidate_head'):
                        for param in model_final.candidate_head.parameters():
                            param.requires_grad = False
                        log("    candidate_head is frozen")
                    
                    log("  Train: GNN, global_head")
            else:
                log(f"   The weight file was not found: {config.PRETRAINED_MODEL_PATH}")

        log("\n4. Checking the forward pass")
        model_final.eval()

        is_v3 = hasattr(model_final, 'candidate_initializer') and hasattr(model_final, 'motif_encoder')
        is_v2 = hasattr(model_final, 'synthon_generator_a') or hasattr(model_final, 'evaluator')
        is_v1 = hasattr(model_final, 'motif_encoder') and not is_v2 and not is_v3

        with torch.no_grad():
            for batch in train_loader:
                if batch is not None:
                    batch_a = batch['batch_a'].to(config.DEVICE)
                    batch_b = batch['batch_b'].to(config.DEVICE)
                    labels = batch['labels'].to(config.DEVICE)
                    
                    if is_v3:
                        motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
                        motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
                        candidates = batch.get('candidates', None)
                        observed_mask = batch.get('observed_synthon_mask', None)
                        
                        if observed_mask is not None:
                            log(f"  Observed synthon mask found")
                            log(f"  Candidates: {len(candidates) if candidates else 0}")
                        else:
                            log("  Observed synthon mask not found")
                        
                        result = model_final(
                            batch_a, batch_b, motifs_a, motifs_b, type_to_idx,
                            candidates=candidates,
                            observed_synthon_mask=observed_mask
                        )
                        
                        if isinstance(result, dict):
                            logits = result['logits']
                            synthon_loss = result.get('synthon_loss', 0)
                            log(f"  Use V3 (with HBAT masks)")
                            log(f"  Synthon loss: {synthon_loss.item() if isinstance(synthon_loss, torch.Tensor) else synthon_loss:.4f}")
                        else:
                            logits = result
                            log("  Error: V3 should return dict")
                    
                    elif is_v2:
                        motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
                        motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
                        
                        result = model_final(batch_a, batch_b, motifs_a, motifs_b, type_to_idx)
                        
                        if isinstance(result, dict):
                            logits = result['logits']
                            log("  Using V3 (with HBAT masks)")
                        else:
                            logits, _, _, _, _ = result
                            log("  Using V2 (with synthons)")
                    
                    elif is_v1:
                        motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
                        motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
                        logits = model_final(batch_a, batch_b, motifs_a, motifs_b, type_to_idx)
                        log("  Use V1 (with motifs)")
                    
                    else:
                        logits = model_final(batch_a, batch_b)
                        log("  Use V0 (without motifs)")
                    
                    probs = torch.sigmoid(logits)
                    
                    log(f"  Batch size: {len(labels)}")
                    log(f"  An example of predictions: {probs[:3].tolist()}")
                    log(f"  Example of labels: {labels[:3].tolist()}")
                    log(f"  Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                    break
        
        log("\n5. Training")

        train_start = time.time()

        val_ds_final = CocrystalDatasetWithSynthons(
            df_val, use_motifs=True, use_masks=config.USE_MASKS, use_candidates=True
        )
        val_loader_final = DataLoader(
            val_ds_final, batch_size=config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_with_hbat, num_workers=0,
        )

        trained_final, train_losses, val_metrics_history = train_model(
            model=model_final,
            train_loader=full_train_loader,
            val_loader=val_loader_final,  
            device=config.DEVICE,
            epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            patience=config.PATIENCE,
            class_weights=class_weights,
            type_to_idx=type_to_idx,
            log_func=log,
            weight_decay=config.WEIGHT_DECAY,
            best_model_path=config.BEST_MODEL_PATH,
            config=config,
        )
        log("\n5. Final score on the test")
        log(f"  Test: {len(df_test)} pair")

        train_time = time.time() - train_start
        log(f"\n  Training time: {train_time:.2f} sec ({train_time/60:.2f} min)")
        
        plot_training_history(train_losses, val_metrics_history, config.PLOT_FILE_PATH)
        
        val_metrics_with_proba = evaluate(
            trained_final, val_loader_final, config.DEVICE,
            type_to_idx=type_to_idx, config=config,
            threshold=None,
            return_proba=True,
        )
        val_metrics, val_probas = val_metrics_with_proba

        val_labels = val_ds_final.data['result'].values
        val_threshold, val_f1 = find_best_threshold(val_labels, val_probas)
        log(f"  Best threshold on val: {val_threshold:.3f} (F1 = {val_f1:.4f})")

        test_metrics = evaluate(
            trained_final, test_loader, config.DEVICE,
            type_to_idx=type_to_idx, config=config,
            threshold=val_threshold,
        )
        print_metrics(test_metrics, f"TEST METRICS (threshold={val_threshold:.3f}, from val)", log_func=log)
        
        log("\n6. Synthons metrics on test")
        synthon_metrics = evaluate_synthons(
            trained_final, test_loader, config.DEVICE,
            type_to_idx=type_to_idx,
            top_k_list=(1, 3, 5, 10, 20),
            config=config,
        )
        print_synthon_metrics(synthon_metrics, log_func=log)



        log("\n" + "="*60)
        log("EXAMPLES OF PREDICTED SYNTHON")
        log("="*60)

        model_final.eval()
        with torch.no_grad():
            example_count = 0
            max_examples = 200  
            
            for batch_idx, batch in enumerate(test_loader):
                if batch is None or example_count >= max_examples:
                    break
                
                batch_a = batch['batch_a'].to(config.DEVICE)
                batch_b = batch['batch_b'].to(config.DEVICE)
                labels = batch['labels']
                motifs_a = batch.get('motifs_a', [[] for _ in range(len(labels))])
                motifs_b = batch.get('motifs_b', [[] for _ in range(len(labels))])
                candidates = batch.get('candidates', None)
                
                for idx in range(min(3, len(labels))):
                    if example_count >= max_examples:
                        break
                    
                    data_list_a = batch_a.to_data_list()
                    data_list_b = batch_b.to_data_list()

                    single_batch_a = data_list_a[idx]
                    single_batch_b = data_list_b[idx]

                    if hasattr(single_batch_a, 'batch'):
                        single_batch_a.batch = torch.zeros(single_batch_a.num_nodes, dtype=torch.long, device=config.DEVICE)
                    if hasattr(single_batch_b, 'batch'):
                        single_batch_b.batch = torch.zeros(single_batch_b.num_nodes, dtype=torch.long, device=config.DEVICE)

                    single_motifs_a = [motifs_a[idx]] if motifs_a and idx < len(motifs_a) else []
                    single_motifs_b = [motifs_b[idx]] if motifs_b and idx < len(motifs_b) else []
                    single_candidates = [candidates[idx]] if candidates and idx < len(candidates) else []
                    
                    row = test_loader.dataset.data.iloc[batch_idx * test_loader.batch_size + idx]
                    smiles1 = row.get('SMILES1', 'N/A')
                    smiles2 = row.get('SMILES2', 'N/A')
                    
                    result = model_final(
                        single_batch_a, single_batch_b,
                        single_motifs_a, single_motifs_b, type_to_idx,
                        candidates=single_candidates,
                        observed_synthon_mask=None
                    )

                    candidate_scores = result['candidate_scores']
                    flat_candidates = result['candidates']
                    
                    log(f"\n--- Example {example_count + 1} ---")
                    log(f"SMILES1: {smiles1[:80]}..." if len(smiles1) > 80 else f"SMILES1: {smiles1}")
                    log(f"SMILES2: {smiles2[:80]}..." if len(smiles2) > 80 else f"SMILES2: {smiles2}")
                    log(f"label: {labels[idx].item()} (1 = co-crystals)")

                    if len(candidate_scores) > 0 and len(flat_candidates) > 0:
                        sorted_idx = torch.argsort(candidate_scores, descending=True)
                        log("Top 5 synthons:")
                        
                        has_valid = False
                        for i, score_idx in enumerate(sorted_idx[:20]):
                            score = candidate_scores[score_idx].item()
                            cand = flat_candidates[score_idx]
                            
                            left_type = cand.get('left_type', '?')
                            right_type = cand.get('right_type', '?')
                            
                            if left_type != '?' and right_type != '?':
                                log(f"  {i+1}. {left_type} ↔ {right_type} : {score:.4f}")
                                has_valid = True
                            else:
                                log(f"  {i+1}. (no data available)")
                        
                        if not has_valid:
                            log("  There are no valid synthons")
                    else:
                        log("  no candidates")
                    
                    example_count += 1

    else:
        log("\n3. Model training")
        
        model = create_model(config).to(config.DEVICE)
        
        if hasattr(config, 'LOAD_PRETRAINED') and config.LOAD_PRETRAINED:
            if config.PRETRAINED_MODEL_PATH and os.path.exists(config.PRETRAINED_MODEL_PATH):
                ckpt = torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state)
                log(f"Stage1 weights loaded:: {config.PRETRAINED_MODEL_PATH}")
                log(f"  Epoch: {ckpt.get('epoch', '?')}, Val F1: {ckpt.get('val_metrics', {}).get('f1', 'N/A')}")
            else:
                log(f"WARNING: PRETRAINED_MODEL_PATH not found: {config.PRETRAINED_MODEL_PATH}")

        trained_model, train_losses, val_metrics_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.DEVICE,
            epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            patience=config.PATIENCE,
            class_weights=class_weights,
            type_to_idx=type_to_idx,
            log_func=log,
            weight_decay=config.WEIGHT_DECAY,
            best_model_path=config.BEST_MODEL_PATH,
            config=config,
        )
        
        plot_training_history(train_losses, val_metrics_history, config.PLOT_FILE_PATH)

        val_metrics_with_proba = evaluate(
            trained_model, val_loader, config.DEVICE,
            type_to_idx=type_to_idx, config=config,
            threshold=None,       
            return_proba=True,
        )
        val_metrics, val_probas = val_metrics_with_proba   

        val_labels = val_dataset.data['result'].values

        val_threshold, _ = find_best_threshold(val_labels, val_probas)
        test_metrics = evaluate(
            trained_model, test_loader, config.DEVICE,
            type_to_idx=type_to_idx, config=config,
            threshold=val_threshold 
        )
        print_metrics(test_metrics, "TEST METRICS", log_func=log)

    log("\n" + "="*60)
    log("Final report")
    log("="*60)
    log(f"Train samples: {len(train_dataset)}")
    log(f"Val samples:   {len(val_dataset)}")
    log(f"Test samples:  {len(test_dataset)}")
    log(f"\nThe best metrics for validation:")
    best_val = max(val_metrics_history, key=lambda x: x['f1'])
    log(f"  Accuracy: {best_val['accuracy']:.4f}")
    log(f"  F1:       {best_val['f1']:.4f}")
    log(f"  AUC:      {best_val['roc_auc']:.4f}")
    log(f"\nMetrics on the test:")
    log(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    log(f"  F1:       {test_metrics['f1']:.4f}")
    log(f"  AUC:      {test_metrics['roc_auc']:.4f}")

    total_time = time.time() - total_start
    log(f"\nTotal lead time: {total_time:.2f} sec ({total_time/60:.2f} min)")

    log("="*60)
    log_file.close()
    

if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config',
                        help='Config class name: Config, ConfigA, ConfigB, ...')
    args = parser.parse_args()
    
    config_class = globals()[args.config]
    main(config_class)