from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.defaults import DEFAULT_TEST_SIZE, PATH_TASK_B_DATA, PATH_TASK_B_LABELS
from src.functions import drop_duplicate_columns


def concat_dataframes_columnwise(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1, df2], axis=1)


def concat_csvs_columnwise(csv1: Path, csv2: Path) -> pd.DataFrame:
    return concat_dataframes_columnwise(pd.read_csv(csv1), pd.read_csv(csv2))


def train_test_split_by_group(
    df: pd.DataFrame, group_name="id", test_size=DEFAULT_TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and test sets so that values from group_name column (id) are
    not shared between the two sets."""
    train_ids, test_ids = train_test_split(df[group_name].unique(), test_size=test_size)

    # Create the train and test dataframes
    train_df = df[df[group_name].isin(train_ids)]
    test_df = df[df[group_name].isin(test_ids)]

    # Check that the train and test sets are disjoint (no overlap)
    assert (
        set(train_df[group_name].unique().tolist())
        & set(test_df[group_name].unique().tolist())
        == set()
    )
    return train_df, test_df


def get_subtask_b_df(csv_data: Path = PATH_TASK_B_DATA, csv_labels=PATH_TASK_B_LABELS):
    # TODO: not finished
    df_data = pd.read_csv(csv_data)
    df_labels = pd.read_csv(csv_labels, names=["id", "label"])
    df = concat_dataframes_columnwise(df_data, df_labels)
    df = drop_duplicate_columns(df)
    return df


def main():
    pass


if __name__ == "__main__":
    main()
