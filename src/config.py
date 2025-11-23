from pathlib import Path

DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "GANAttributes"
OUT_ROOT = Path("model_data")
RUN_DIR = OUT_ROOT / "runs"
INFER_ROOT = DATA_ROOT / "test_images"

ALL_CLASSES = ["ProGAN", "StyleGAN2", "BigGAN", "RealPhotos"]
GAN_CLASSES = ["ProGAN", "StyleGAN2", "BigGAN"]
REAL_FAKE_CLASSES = ["Real", "Fake"]

MAX_CAP = 4000
IMG_SIZE = 256
SEED = 42

NUM_EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 6
LR_GC = 2e-4
LR_D = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 6

# Training - Real vs Fake
RF_NUM_EPOCHS = 12
RF_BATCH_SIZE = 32
RF_LR = 1e-4
RF_PATIENCE = 4

ALPHA_STAMP = 0.20
LAMBDA_ADV = 0.5
LAMBDA_PERC = 0.10
LAMBDA_RES = 0.005
LAMBDA_HIPASS = 0.02
LAMBDA_CLS_R = 1.5
LAMBDA_CLS_S = 1.0

# checkpoints
CHECKPOINT = RUN_DIR / "best_gfd2_model.pt"
RF_CHECKPOINT = RUN_DIR / "best_real_fake_model.pt"

# Output files
METRICS_CSV = RUN_DIR / "metrics.csv"
RF_METRICS_CSV = RUN_DIR / "rf_metrics.csv"
LABEL_MAP = RUN_DIR / "label_map.json"
