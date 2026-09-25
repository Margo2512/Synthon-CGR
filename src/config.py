import torch
import os
from datetime import datetime

class Config:
    MODEL_VERSION = 'V2' # 'V0', 'V1', 'V2'
    MODEL_TYPE = 'GATv2' # 'GCN', 'GAT', 'GATv2', 'GIN', 'GINE'
    
    HIDDEN_DIM = 128           
    NUM_LAYERS = 2            
    GAT_HEADS = 4             

    MAX_SYNTHONS_A = 100
    MAX_SYNTHONS_B = 100
    SYNTHON_DIM = 128         
    USE_HBOND_PRIOR = True     

    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 200
    PATIENCE = 20            
 
    DROPOUT = 0.2
    EVALUATOR_DROPOUT = 0.3
    WEIGHT_DECAY = 0.001       
 
    LAMBDA_SPARSITY = 1.0      
    LAMBDA_DIVERSITY = 0.01    
    LAMBDA_HBOND = 0.001      
    PRIOR_WEIGHT = 0.01       
 
    USE_FOCAL_LOSS = False
    FOCAL_ALPHA = 0.75          
    FOCAL_GAMMA = 2.0           
    LAMBDA_LOCAL = 0.1
 
    USE_MULTI_POOLING = False
    POOLING_WEIGHTS = [0.4, 0.3, 0.3]  # mean, max, noisy_or
 
    USE_COSINE_ANNEALING = False
    WARMUP_EPOCHS = 5
    MIN_LR = 0.00001
 
    POOLING_MODE = 'mean'
    NOISE_PROB = 0.3        

    EVALUATOR_LR_MULTIPLIER = 1.0
    LAMBDA_SYNTHON = 0.3  
    
    LOAD_PRETRAINED = False           
    PRETRAINED_MODEL_PATH = None     
 
    DATA_PATH = "Synthon-CGR/data"
    CSV_FILE = "Synthon-CGR/data/data_aug.csv"
     
    TEST_SIZE = 0.1
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
 
    INPUT_DIM = 18      
    OUTPUT_DIM = 128
    NUM_MOTIF_TYPES = 1000 
    EDGE_DIM = 4  # single, double, triple, aromatic
    
    USE_CLASS_WEIGHTS = True 
    USE_BALANCED_SAMPLER = False
    POS_WEIGHT = None   

    NUM_WORKERS = 2
    USE_AMP = True 

    USE_MASKS = False
     
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
     
    EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
    PROJECT_ROOT = "Synthon-CGR"
 
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs_new/")
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache/")

    PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/")
 
    LOG_FILE_NAME = f"training_log_{EXP_NAME}.txt"
    LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
 
    BEST_MODEL_NAME = f"best_model_{EXP_NAME}.pt"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
 
    CACHE_FILE_NAME = f"motifs_cache_{RANDOM_STATE}.pkl" 
    CACHE_FILE_PATH = os.path.join(CACHE_DIR, CACHE_FILE_NAME) 
 
    PLOT_FILE_NAME = f"training_history_{EXP_NAME}.png"
    PLOT_FILE_PATH = os.path.join(PLOTS_DIR, PLOT_FILE_NAME)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    def __repr__(self):
        return f"Config(\n  DATA_PATH='{self.DATA_PATH}',\n  BATCH_SIZE={self.BATCH_SIZE},\n  HIDDEN_DIM={self.HIDDEN_DIM},\n  DEVICE='{self.DEVICE}'\n)"

COMMON_TEST_CSV = "Synthon-CGR/data/test_common.csv"
DF_1_TRAIN_CSV  = "Synthon-CGR/data/df_1_train.csv"


class ConfigBase_CommonTest(Config):
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    GAT_HEADS = 4
    DROPOUT = 0.2
    EVALUATOR_DROPOUT = 0.3
    WEIGHT_DECAY = 0.001
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 200
    PATIENCE = 20
    USE_FOCAL_LOSS = False
    USE_CLASS_WEIGHTS = True
    USE_BALANCED_SAMPLER = False
    TEST_SIZE = 0.1 
    VAL_SIZE = 0.1


class ConfigV0_CommonTest(ConfigBase_CommonTest):
    MODEL_VERSION = 'V0'
    MODEL_TYPE = 'GATv2'
    CSV_FILE = "Synthon-CGR/data/data_aug.csv"
    USE_MASKS = False
    CACHE_FILE_PATH = "Synthon-CGR/cache/motifs_cache_v0_common.pkl"

class ConfigV1_CommonTest(ConfigBase_CommonTest):
    MODEL_VERSION = 'V1'
    MODEL_TYPE = 'GATv2'
    CSV_FILE = "Synthon-CGR/data/data_aug.csv"
    USE_MASKS = False
    NUM_MOTIF_TYPES = 1000
    CACHE_FILE_PATH = "Synthon-CGR/cache/motifs_cache_v1_common.pkl"


class ConfigV2_CommonTest(ConfigBase_CommonTest):
    MODEL_VERSION = 'V2'
    MODEL_TYPE = 'GATv2'
    CSV_FILE = "Synthon-CGR/data/data_aug.csv"
    USE_MASKS = False
    NUM_MOTIF_TYPES = 1000
    MAX_SYNTHONS_A = 10
    MAX_SYNTHONS_B = 10
    POOLING_MODE = 'mean'
    USE_HBOND_PRIOR = True
    LAMBDA_HBOND = 0.0001
    CACHE_FILE_PATH = "Synthon-CGR/cache/motifs_cache_v2_common.pkl"

class ConfigV3_Stage1_CommonTest(ConfigBase_CommonTest):
    MODEL_VERSION = 'V3'
    MODEL_TYPE = 'GATv2'
    CSV_FILE = "Synthon-CGR/data/data_aug_without_masked.csv"
    USE_CIF = False
    USE_MASKS = False
    LAMBDA_MASK = 0.0
    LAMBDA_CONFLICT = 0.0
    LOAD_PRETRAINED = False
    MAX_SYNTHONS_A = 100
    MAX_SYNTHONS_B = 100
    PRIOR_WEIGHT = 0.05
    POOLING_MODE = 'mean'
    LAMBDA_SPARSITY = 0.5
    EVALUATOR_LR_MULTIPLIER = 1.0

    COD_CSV_PATH = "Synthon-CGR/data/COD_our_df.csv"
    HBAT_PATH = "Synthon-CGR/data/bonds_ctr_m_final.csv"
    CACHE_FILE_PATH = "Synthon-CGR/cache/motifs_cache_v3s1_common.pkl"

    BEST_MODEL_NAME = "best_model_ConfigV3_Stage1_CommonTest.pt"
    BEST_MODEL_PATH = "Synthon-CGR/checkpoints/best_model_ConfigV3_Stage1_CommonTest.pt"

class ConfigV3_Stage2_CommonTest(ConfigBase_CommonTest):
    MODEL_VERSION = 'V3'
    MODEL_TYPE = 'GATv2'

    CSV_FILE = "Synthon-CGR/data/df_1_train.csv"

    USE_CIF = True
    USE_MASKS = True
    LAMBDA_MASK = 0.5
    LAMBDA_CONFLICT = 0.05
    LOAD_PRETRAINED = True
    PRETRAINED_MODEL_PATH = "Synthon-CGR/checkpoints/best_model_ConfigV3_Stage1_CommonTest.pt"

    MAX_SYNTHONS_A = 100
    MAX_SYNTHONS_B = 100
    PRIOR_WEIGHT = 0.05
    POOLING_MODE = 'mean'
    LAMBDA_SPARSITY = 0.5
    EVALUATOR_LR_MULTIPLIER = 5.0

    COD_CSV_PATH = "Synthon-CGR/data/COD_our_df.csv"
    HBAT_PATH = "Synthon-CGR/data/bonds_ctr_m_final.csv"
    CACHE_FILE_PATH = "Synthon-CGR/cache/motifs_cache_v3s2_common.pkl"