import numpy as np
import pandas as pd
import torch
import os
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer, EvalPrediction,
)

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
    x = torch.argmax(torch.tensor(softmax(pred)), dim=1).cpu().numpy()  # y_pred
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


def check_for_cuda():
    cuda_id = 0
    device = torch.device("cuda:%s" % cuda_id if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(cuda_id) if torch.cuda.is_available() else "cpu"
    print("Using the device %s - %s" % (device, device_name))
    torch.set_default_device(device)

    print(f'\tAllocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
    print(f'\tCached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')


def main():
    check_for_cuda()
    args = parse_args()

    model_folder = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_tag,
        use_fast=True,
    )

    validation_df = pd.read_csv(args.validation_csv)
    validation_hf = Dataset.from_pandas(validation_df)

    models_with_average_metrics = {}

    for folder in os.listdir(model_folder):
        try:
            print(f'Validating model {folder} started...')
            model_path = os.path.join(model_folder, folder)

            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(0)

            print(f'\tAllocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
            print(f'\tCached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')

            outputs = []
            for sentence in validation_hf["sentence"]:
                with torch.no_grad():
                    input_ = tokenizer(sentence, return_tensors="pt")
                    output = model(**input_)
                    outputs.append(output.logits.detach().cpu().numpy()[0])

            eval_predictions = EvalPrediction(inputs=None, label_ids=np.array(validation_hf["label"]),
                                              predictions=np.array(outputs))
            metrics = compute_metrics(eval_predictions)
            models_with_average_metrics[folder] = metrics
            print(f'Validating model {folder} finished...')
        except EnvironmentError as ex:
            print(f'Cannot load model {folder}. {ex}')

    for model in models_with_average_metrics.keys():
        print(model)
        print(models_with_average_metrics[model])
        print()


if __name__ == '__main__':
    main()
