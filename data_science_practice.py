import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from file_system.io import read_file_as_dataframe, write_dataframe_to_parquet


def load_executions() -> pd.DataFrame:
    logging.getLogger().info("Reading executions file")
    return read_file_as_dataframe("data/executions.parquet")


def load_market_data() -> pd.DataFrame:
    logging.getLogger().info("Reading market data file")
    return read_file_as_dataframe("data/marketdata.parquet")


def load_ref_data() -> pd.DataFrame:
    logging.getLogger().info("Reading ref data file")
    return read_file_as_dataframe("data/refdata.parquet")


def calculate_statistics(executions: pd.DataFrame) -> None:
    logging.getLogger().info(f"The number of unique executions is {executions['Trade_id'].nunique()}")
    logging.getLogger().info(f"The number of unique venues is {executions['Venue'].nunique()}")

def get_continuous_trading_executions(executions: pd.DataFrame) -> pd.DataFrame:
    continuous_trades = executions[executions["Phase"] == "CONTINUOUS_TRADING"]
    logging.getLogger().info(f"Number of continuous trades is {continuous_trades['Trade_id'].nunique()}")
    return continuous_trades


def add_trade_side(executions: pd.DataFrame) -> None:
    if 0 in executions["Quantity"].values:
        raise RuntimeError("Field 'Quantity' cannot be zero")

    executions["side"] = np.where(
        executions["Quantity"] > 0, 1, 2
    )


def add_ref_data_to_executions(executions: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(executions, reference_data, how="inner", on="ISIN")

    return merged_df


def join_with_nearest_key(left: pd.DataFrame, right: pd.DataFrame, left_keys: List[str], right_keys: List[str],
                          primary_join_key: str, column_suffixes: Tuple[str, str]):
    return pd.merge_asof(left=left.sort_values(by=left_keys),
                         right=right.sort_values(by=right_keys),
                         by=primary_join_key,
                         left_on=left_keys,
                         right_on=right_keys,
                         direction="backward",
                         suffixes=column_suffixes)


def add_bbo_data(executions: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    logging.getLogger().info("Adding BBO data")

    merged_df = join_with_nearest_key(left=executions,
                                      right=market_data[["primary_mic", "event_timestamp",
                                                         "best_bid_price", "best_ask_price"]],
                                      left_keys=["TradeTime"],
                                      right_keys=["event_timestamp"],
                                      primary_join_key="primary_mic",
                                      column_suffixes=("", ""))
    return merged_df


def add_bbo_plus_one(executions: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    logging.getLogger().info("Getting bid and ask price for one second after execution")

    executions["trade_time_1s"] = executions["TradeTime"] + pd.Timedelta(seconds=1)
    return join_with_nearest_key(left=executions,
                                 right=market_data[["primary_mic", "event_timestamp",
                                                    "best_bid_price", "best_ask_price"]],
                                 left_keys=["trade_time_1s"],
                                 right_keys=["event_timestamp"],
                                 primary_join_key="primary_mic",
                                 column_suffixes=("", "_1s")).drop(["event_timestamp_1s"], axis=1)


def add_bbo_minus_one(executions: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    logging.getLogger().info("Getting bid and ask price for one second after execution")

    executions["trade_time_min_1s"] = executions["TradeTime"] - pd.Timedelta(seconds=1)
    return join_with_nearest_key(left=executions,
                                 right=market_data[["primary_mic", "event_timestamp",
                                                    "best_bid_price", "best_ask_price"]],
                                 left_keys=["trade_time_min_1s"],
                                 right_keys=["event_timestamp"],
                                 primary_join_key="primary_mic",
                                 column_suffixes=("", "_min_1s")).drop(["event_timestamp_min_1s"], axis=1)


def clean_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    logging.getLogger().info("Removing null bid and ask prices")
    return market_data[market_data["best_ask_price"].notna()]


def add_mid_prices(execution_data: pd.DataFrame, column_suffix: str) -> pd.DataFrame:
    logging.getLogger().info("Adding mid prices")
    mid_price_column = "mid_price" + column_suffix
    bid_price_column = "best_bid_price" + column_suffix
    ask_price_column = "best_ask_price" + column_suffix
    execution_data[mid_price_column] = execution_data[[bid_price_column, ask_price_column]].mean(axis=1)

    return execution_data


def calculate_slippage(execution_data: pd.DataFrame) -> pd.DataFrame:
    logging.getLogger().info("Calculating slippage")
    execution_data["slippage"] = np.where(
        execution_data["side"] == 2,
        (execution_data["Price"] - execution_data["best_bid_price"]) / (
                    execution_data["best_ask_price"] - execution_data["best_bid_price"]),
        (execution_data["best_ask_price"] - execution_data["Price"]) / (
                execution_data["best_ask_price"] - execution_data["best_bid_price"])
    )
    return execution_data


def main() -> None:
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    executions: pd.DataFrame = load_executions()
    calculate_statistics(executions=executions)
    continuous_trades = get_continuous_trading_executions(executions=executions)
    add_trade_side(continuous_trades)
    ref_data = load_ref_data()
    trade_data = add_ref_data_to_executions(executions=continuous_trades,
                                            reference_data=ref_data[["ISIN", "primary_ticker", "primary_mic"]])

    market_data = load_market_data()
    market_data = clean_market_data(market_data=market_data)

    trade_data["TradeTime"] = trade_data["TradeTime"].astype("datetime64[ns]")
    trade_data = add_bbo_data(executions=trade_data, market_data=market_data)
    trade_data = add_bbo_plus_one(executions=trade_data, market_data=market_data)
    trade_data = add_bbo_minus_one(executions=trade_data, market_data=market_data)
    trade_data = add_mid_prices(execution_data=trade_data, column_suffix="")
    trade_data = add_mid_prices(execution_data=trade_data, column_suffix="_1s")
    trade_data = add_mid_prices(execution_data=trade_data, column_suffix="_min_1s")
    trade_data = calculate_slippage(execution_data=trade_data)
    write_dataframe_to_parquet(dataframe=trade_data, path="data/trades.parquet")


if __name__ == '__main__':
    main()
