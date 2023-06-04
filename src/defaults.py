from pathlib import Path

from src.enums import SupportedLossFunctions, SupportedModels

# ========== PATHS ==========
PATH_DATA = Path("..\\data")

PATH_TRAIN_A = Path(PATH_DATA, "subtaskA_train_data_20000.csv")
PATH_TEST_A = Path(PATH_DATA, "subtaskA_test_data_2000.csv")
PATH_VALIDATION_A = Path(PATH_DATA, "subtaskA_dev_data_1994.csv")
PATH_MODELS = Path("..\\models") #Path("models")

# ========== DEFAULTS ==========

DEFAULT_NUM_LABELS = 2
DEFAULT_TEST_SIZE = 0.2

DEFAULT_TRAIN_BATCH_SIZE = 4
DEFAULT_TEST_BATCH_SIZE = 4
DEFAULT_NUM_EPOCHS = 5

DEFAULT_OPTIM = "adamw_hf"
DEFAULT_LR = 5e-6
DEFAULT_LOSS_FN = SupportedLossFunctions.BCE
DEFAULT_METRIC = "f1"
DEFAULT_METRIC_MODE = "max"
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_ACCUM_STEPS = 1

DEFAULT_MODEL = SupportedModels.DEBERTA_V3_SMALL
DEBERTA_V3_SMALL_TAG = "microsoft/deberta-v3-small"
DEFAULT_PRETRAINED_TAG = DEBERTA_V3_SMALL_TAG
DEFAULT_PRETRAINED_TAG_MAP = {
    SupportedModels.DEBERTA_V3_SMALL.value: DEBERTA_V3_SMALL_TAG,
}
DEFAULT_PROBLEM_TYPE = "single_label_classification"

DEFAULT_AUGMENTATION_THRESHOLD = 0.0
DEFAULT_AUGMENTER = "none"
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_FREEZE_BERT = False
