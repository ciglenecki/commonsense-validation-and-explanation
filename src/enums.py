from enum import Enum


class SupportedModels(Enum):
    DEBERTA_V3_SMALL = "deberta-v3-small"


class SupportedLossFunctions(Enum):
    BCE = "bce"
