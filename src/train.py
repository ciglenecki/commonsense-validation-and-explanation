import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DebertaV2Model,
    Trainer,
    TrainingArguments,
)

from src.functions import get_timestamp, random_codeword, stdout_to_file, to_yaml
from src.train_args import parse_args


def compute_metrics(pred):
    print(pred)
    x, y = pred[0].reshape(-1), pred[1].reshape(-1)
    return {
        "accuracy": accuracy_score(x, y),
        "f1": f1_score(x, y),
        "precision": precision_score(x, y),
        "recall": recall_score(x, y),
        "roc_auc": roc_auc_score(x, y),
    }


def dataset_preprocess(examples, tokenizer: AutoTokenizer):
    return tokenizer(
        examples["sentence"],
        padding=True,
        truncation=True,
        truncation_strategy="longest_first",
        return_tensors="pt",
    )


def experiment_setup(args: argparse.Namespace):
    """Create experiment directory."""
    timestamp = get_timestamp()
    experiment_codeword = random_codeword()
    experiment_name_list = [timestamp, experiment_codeword, args.model.value]

    if args.experiment_suffix:
        experiment_name_list.append(args.experiment_suffix)
    experiment_name = "_".join(experiment_name_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    experiment_dir = Path(output_dir, experiment_name)
    experiment_dir.mkdir(exist_ok=True)

    filename_args = Path(experiment_dir, "args.yaml")
    with open(filename_args, "w") as outfile:
        yaml.dump(args, outfile)
    filename_report = Path(output_dir, experiment_name, "log.txt")

    stdout_to_file(filename_report)
    print()
    print("Created experiment directory:", str(experiment_dir))
    print("Created log file:", str(filename_report))
    print()
    print("================== Args ==================\n\n", to_yaml(vars(args)))
    print()
    if args.verify_args:
        input("Review the args above. Press enter if you wish to continue: ")
    return experiment_name, experiment_dir, output_dir


def get_hf_dataset(args: argparse.Namespace):
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    train_hf = Dataset.from_pandas(train_df)
    test_hf = Dataset.from_pandas(test_df)

    dataset = DatasetDict()
    dataset["train"] = train_hf.rename_column("label", "labels")
    dataset["test"] = test_hf.rename_column("label", "labels")

    return dataset


def main():
    args = parse_args()

    experiment_name, experiment_dir, output_dir = experiment_setup(args)

    training_args = TrainingArguments(
        output_dir=experiment_dir,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        optim=args.optim,
        metric_for_best_model=f"val_{args.metric}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=args.use_fp16,
        report_to="tensorboard",
        gradient_accumulation_steps=args.grad_acc,
        logging_steps=True,
        warmup_ratio=0.1,
        logging_first_step=True,
        logging_dir=experiment_dir,
        greater_is_better=args.metric_mode,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_tag,
        use_fast=True,
    )

    print(tokenizer("Tset my sentence haha"))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = get_hf_dataset(args)

    tokenized_dataset = dataset.map(
        partial(dataset_preprocess, tokenizer=tokenizer), batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_tag,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
        state_dict=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training model")
    trainer.train()


if __name__ == "__main__":
    main()
