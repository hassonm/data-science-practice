import pandas as pd


def read_file_as_dataframe(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_dataframe_to_parquet(dataframe: pd.DataFrame, path: str):
    dataframe.to_parquet(path=path, compression="gzip")
