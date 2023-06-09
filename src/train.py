import argparse
import os
import random
from functools import partial
from pathlib import Path

import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import torch
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
    DebertaV2ForTokenClassification,
    DebertaV2Model,
    Trainer,
    TrainingArguments,
)

from src.functions import get_timestamp, random_codeword, stdout_to_file, to_yaml
from src.train_args import parse_args


def softmax(pred):
    row_max = np.max(pred[0], axis=1, keepdims=True)
    e_x = np.exp(pred[0] - row_max)
    row_sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / row_sum
    return f_x


def compute_metrics(pred):
    #  y_pred_logits, y_true = pred
    #  y_pred = np.argmax(y_pred_logits, axis=-1)
    x = torch.argmax(torch.tensor(softmax(pred)), dim=1).numpy()  # y_pred
    y = pred[1].reshape(-1)  # y_true
    dict = {
        "accuracy": accuracy_score(x, y),
        "f1": f1_score(x, y, labels=[0, 1]),
        "precision": precision_score(x, y, labels=[0, 1]),
        "recall": recall_score(x, y, labels=[0, 1]),
    }
    try:
        dict["roc_auc"] = roc_auc_score(x, y, labels=[0, 1])
    except ValueError:
        pass
    return dict


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


def perform_dataset_augmentation(threshold, dataset):
    dataset_pd = Dataset.to_pandas(dataset)

    def perform_sentence_augmentation(sentence):
        percentage = random.random()
        return (
            naw.RandomWordAug(action="swap").augment(sentence)
            if percentage > threshold
            else [sentence]
        )

    dataset_pd["sentence"] = dataset_pd["sentence"].apply(perform_sentence_augmentation)
    dataset_pd["sentence"] = dataset_pd["sentence"].apply(lambda x: x[0])
    dataset_pd.head()
    return Dataset.from_pandas(dataset_pd)


def perform_batch_augmentation(threshold, augmenter, batch):
    def perform_sentence_augmentation(batch_to_augment):
        examples = zip(batch_to_augment["sentence"], batch_to_augment["labels"])
        augmented_sentences = []

        for example in examples:
            if example[1] == 0:
                augmented_sentences.append(example[0])
            else:
                percentage = random.random()
                augmented_sentences.append(
                    augmenter.augment(example[0])[0]
                ) if percentage < threshold else augmented_sentences.append(example[0])

        return augmented_sentences

    try:
        batch["sentence"] = perform_sentence_augmentation(batch)
    except KeyError:
        return batch

    return batch


def main():
    args = parse_args()

    experiment_name, experiment_dir, output_dir = experiment_setup(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_tag,
        use_fast=True,
    )

    training_args = TrainingArguments(
        output_dir=experiment_dir,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        optim=args.optim,
        metric_for_best_model=f"eval_{args.metric}",
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=2000,
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=args.use_fp16,
        report_to="tensorboard",
        gradient_accumulation_steps=args.grad_acc,
        logging_steps=20,
        warmup_ratio=args.warmup_ratio,
        logging_first_step=True,
        logging_dir=experiment_dir,
        greater_is_better=args.metric_mode,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = get_hf_dataset(args)

    # Augment entire dataset
    # dataset["train"] = perform_dataset_augmentation(
    #     args.augmentation_threshold, dataset["train"]
    # )

    # Augment by batch
    aug = None
    if args.augmenter == "rand":
        aug = naw.RandomWordAug(action="swap")
    elif args.augmenter == "syn_wordnet":
        aug = naw.ContextualWordEmbsAug(
            model_path="bert-base-uncased", action="substitute"
        )
    elif args.augmenter == "syn_ppdb":
        aug = naw.SynonymAug(
            aug_src="ppdb", model_path=os.environ.get("MODEL_DIR") + "ppdb-2.0-s-all"
        )

    transformation = lambda batch: perform_batch_augmentation(
        args.augmentation_threshold, aug, batch
    )

    if aug is not None:
        dataset["train"].set_transform(transformation)

    tokenized_dataset = dataset.map(
        partial(dataset_preprocess, tokenizer=tokenizer), batched=True
    )

    model: DebertaV2ForTokenClassification = (
        AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_tag,
            num_labels=args.num_labels,
            problem_type=args.problem_type,
            state_dict=None,
        )
    ).to(0)

    if args.freeze_bert:
        for param in model.deberta.parameters():
            param.requires_grad = False

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
    trainer.save_model(experiment_dir)


if __name__ == "__main__":
    main()
