from pathlib import Path

from src.enums import SupportedLossFunctions, SupportedModels

# ========== PATHS ==========
PATH_DATA = Path("data")

PATH_TASK_A_DATA = Path(PATH_DATA, "subtaskA_trial_data.csv")
PATH_TASK_B_DATA = Path(PATH_DATA, "subtaskB_trial_data.csv")
PATH_TASK_C_DATA = Path(PATH_DATA, "subtaskC_trial_data.csv")

PATH_TASK_A_LABELS = Path(PATH_DATA, "subtaskA_answers.csv")
PATH_TASK_B_LABELS = Path(PATH_DATA, "subtaskB_answers.csv")
PATH_TASK_C_LABELS = Path(PATH_DATA, "subtaskC_answers.csv")

PATH_TRAIN_A = Path(PATH_DATA, "clean_a_train_3232.csv")
PATH_TEST_A = Path(PATH_DATA, "clean_a_test_810.csv")
PATH_MODELS = Path("models")

# ========== DEFAULTS ==========

DEFAULT_NUM_LABELS = 2
DEFAULT_TEST_SIZE = 0.2

DEFAULT_TRAIN_BATCH_SIZE = 4
DEFAULT_TEST_BATCH_SIZE = 4
DEFAULT_NUM_EPOCHS = 15

DEFAULT_OPTIM = "adamw_hf"
DEFAULT_LR = 1e-5
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
