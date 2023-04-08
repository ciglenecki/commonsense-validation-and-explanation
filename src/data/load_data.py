import pandas as pd

from src.config import PATH_TASK_A_DATA

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def main():
    df_a_data = pd.read_csv(PATH_TASK_A_DATA)
    print(df_a_data.head(n=3))
    print(df_a_data.describe())


if __name__ == "__main__":
    main()
