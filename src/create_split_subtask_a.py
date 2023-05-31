"""python3 src/create_split_subtask_a.py --data data/train/subtaskA_data_all.csv --labels
data/train/subtaskA_answers_all.csv.

python3 src/create_split_subtask_a.py --data data/test/subtaskA_test_data.csv --labels data/test/subtaskA_gold_answers.csv

python3 src/create_split_subtask_a.py --data data/dev/subtaskA_dev_data.csv --labels data/dev/subtaskA_gold_answers.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data import concat_dataframes_columnwise, train_test_split_by_group
from src.defaults import (
    DEFAULT_TEST_SIZE,
    PATH_DATA,
    PATH_TASK_A_DATA,
    PATH_TASK_A_LABELS,
)
from src.functions import drop_duplicate_columns


def create_subtask_a_df(
    csv_data: Path = PATH_TASK_A_DATA, csv_labels=PATH_TASK_A_LABELS
):
    """Concatenate the data and labels csv files for subtask A.

    Flatten the data so that each row is a sentence and its label. New label:     doesn't make
    sense: 0     makes sense:        1 Create sentence length column.
    """
    df_data = pd.read_csv(csv_data)
    df_labels = pd.read_csv(csv_labels, names=["id", "label"])
    df = concat_dataframes_columnwise(df_data, df_labels)
    df = drop_duplicate_columns(df)

    # Transform setn0 and sent1 into a single column
    new_data = []
    for _, row in df.iterrows():
        # original csv: label==0 -> sentence 1 makes sense
        sent0_value = row["label"]
        sent1_value = 1 - sent0_value
        new_data.append([row["id"], row["sent0"], sent0_value])
        new_data.append([row["id"], row["sent1"], sent1_value])
    df = pd.DataFrame(new_data, columns=["id", "sentence", "label"])
    df["sentence_length"] = df["sentence"].apply(lambda x: len(x.split()))
    return df


def main():
    """Create clean train and test csv files for subtask A.

    The dataset is split so that ids are not shared between the train and test sets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    csv_data = args.data
    csv_labels = args.labels

    df = create_subtask_a_df(csv_data, csv_labels)
    # df_train, df_test = train_test_split_by_group(
    #     df, group_name="id", test_size=DEFAULT_TEST_SIZE
    # )
    path_train = PATH_DATA / f"{csv_data.stem}_{len(df)}.csv"
    # path_test = PATH_DATA / f"clean_a_test_{len(df_test)}.csv"
    print("Saving files to:", path_train, sep="\n")
    df.to_csv(path_train, index=False)
    # df_test.to_csv(path_test, index=False)


if __name__ == "__main__":
    main()
