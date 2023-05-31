import argparse
from pathlib import Path

from src.defaults import (
    DEFAULT_AUGMENTATION_THRESHOLD,
    DEFAULT_GRAD_ACCUM_STEPS,
    DEFAULT_LOSS_FN,
    DEFAULT_LR,
    DEFAULT_METRIC,
    DEFAULT_METRIC_MODE,
    DEFAULT_MODEL,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_LABELS,
    DEFAULT_OPTIM,
    DEFAULT_PRETRAINED_TAG_MAP,
    DEFAULT_PROBLEM_TYPE,
    DEFAULT_TEST_BATCH_SIZE,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    PATH_MODELS,
    PATH_TEST_A,
    PATH_TRAIN_A,
)
from src.enums import SupportedLossFunctions, SupportedModels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=SupportedModels,
        default=DEFAULT_MODEL,
        choices=list(SupportedModels),
    )
    parser.add_argument(
        "--pretrained_tag",
        type=SupportedModels,
        choices=list(SupportedModels),
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=DEFAULT_NUM_LABELS,
    )

    parser.add_argument(
        "--problem_type",
        type=str,
        default=DEFAULT_PROBLEM_TYPE,
        choices=[
            "single_label_classification",
            "multi_label_classification",
            "regression",
        ],
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=PATH_TRAIN_A,
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=PATH_TEST_A,
    )
    parser.add_argument(
        "--experiment_suffix",
        type=str,
        help="Suffix to add to experiment name to differentiate between experiments.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PATH_MODELS,
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--verify_args",
        action="store_true",
        help="Asks for config confirmation before starting training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=DEFAULT_TEST_BATCH_SIZE,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--loss_fn",
        type=SupportedLossFunctions,
        choices=list(SupportedLossFunctions),
        default=DEFAULT_LOSS_FN,
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        default=DEFAULT_METRIC_MODE,
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=DEFAULT_OPTIM,
    )
    parser.add_argument(
        "--augmentation_threshold",
        type=float,
        default=DEFAULT_AUGMENTATION_THRESHOLD,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
    )
    parser.add_argument(
        "-f",
        type=str,
    )

    args = parser.parse_args()

    if args.pretrained_tag is None:
        if args.model.value not in DEFAULT_PRETRAINED_TAG_MAP.keys():
            raise ValueError(
                f"Default pretrained tag not found for model {args.model.value}."
            )
        args.pretrained_tag = DEFAULT_PRETRAINED_TAG_MAP[args.model.value]

    return args
