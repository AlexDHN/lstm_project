import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np
from typing import List
import pandas as pd
from utils.strategy_trading import strategy
from utils.params import init_functions, kwargs
from warnings import warn
from time import time
import polars as pl
warn('Running this module requires the package: polars 0.15.14')


class Frequency(Enum):
    HOURLY = "Hourly"
    MONTHLY = "Monthly"
    DAILY = "Daily"


class PositionGenerator:
    def __init__(self, path=r"dataset.csv", random_=False, all_stock=False):
        self.data = pd.read_csv(path, index_col=0)
        self.data.index = pd.to_datetime(self.data.index)
        self.random_arr = random_
        self.all_stock = all_stock

        if self.random_arr:
            self.random = pd.DataFrame(np.random.randint(2, size=self.data.shape))
            self.random.columns = self.data.columns
            self.random.insert(0, 'Date', self.data.index)

        if self.all_stock:
            self.random = pd.DataFrame(np.ones(self.data.shape))
            self.random.columns = self.data.columns
            self.random.insert(0, 'Date', self.data.index)

        if self.random_arr + self.all_stock >= 2:
            raise ValueError("random and all stock cannot be both true")

    def compute_positions(self):
        """
        This method will compute a list of position for the trading period
        Even if we take random prediction or all stock we have to run strategy to grab the timestamp
        -------
        """
        # import data of prediction
        _, df_pred = strategy(self.data, init_functions, **kwargs)
        if self.random_arr + self.all_stock >= 1:
            y_pred = self.random
            # We need to grab the index
            return y_pred.loc[y_pred.Date.isin(df_pred.index)]
        df_pred.reset_index(inplace=True)
        return df_pred


@dataclass
class Config:
    universe: List[str]
    start_ts: datetime
    end_ts: datetime
    strategy_code: str
    frequency: Frequency
    basis: int = 100
    timeserie: np.array = None

    def __post_init__(self):
        if self.start_ts >= self.end_ts:
            raise ValueError("self.start_ts must be before self.end_ts")
        if len(self.universe) == 0:
            raise ValueError("self.universe should contains at least one element")

    @property
    def timedelta(self):
        if self.frequency == Frequency.HOURLY:
            return timedelta(hours=1)
        if self.frequency == Frequency.DAILY:
            return timedelta(days=1)

    def calendar(self, timeserie) -> List[datetime]:
        # renvoyer une liste de date comprise entre start_ts et end_ts.
        if timeserie is None:
            timedelta_ = self.timedelta
            tmp = self.start_ts
            calendar_ = []
            while tmp <= self.end_ts:
                calendar_.append(tmp)
                tmp += timedelta_
            return calendar_
        else:
            return timeserie['Date']


@dataclass
class Quote:
    ts: datetime = None
    close: float = None


@dataclass
class Weight:
    product_code: str = None
    underlying_code: str = None
    ts: datetime = None
    value: float = None


class Backtester:
    def __init__(self, config: Config, y_pred, timeserie=None):
        self._config = config
        self._calendar = config.calendar(timeserie)
        self._universe = config.universe
        self._timedelta = config.timedelta

        self._quote_by_pk = dict()
        self._weight_by_pk = dict()
        self._level_by_ts = dict()

        self.position = y_pred
        if self._config.start_ts != timeserie['Date'][0]:
            raise ValueError(
                "starts_ts and the start date of the raw data have to be the same"
            )
        self._generate_quotes(timeserie)

    def _generate_quotes(self, quote_data):
        """
        quote_data: data from which we create the quote
        """
        for ts in self._calendar:
            for underlying_code in self._universe:
                if quote_data is None:
                    self._quote_by_pk[(underlying_code, ts - self._timedelta)] = Quote(
                        close=100 * (1 + random.random() / 100), ts=ts - self._timedelta
                    )
                else:
                    self._quote_by_pk[(underlying_code, ts - self._timedelta)] = Quote(
                        close=quote_data.filter((pl.col('Date') == ts)).select([str(underlying_code)]),
                        ts=ts - self._timedelta,
                    )

    def _compute_weight(self, ts: datetime, nb_stock):
        """
        nb_stock: we only use the columns 'sum_h' of the dataframe which is the number of stock which we hold
                at time ts ==> allow us to compute the weight
        Function which compute the weight depending on the strategy
        """
        nb_stock_to_hold = nb_stock.filter((pl.col('Date') == ts)).select(['sum_h'])
        # same weight attributed for stock we go long on
        w = nb_stock_to_hold.apply(lambda x: np.divide(1, x)) if nb_stock_to_hold.to_pandas().values[0][0] != 0 else 0

        for underlying_code in self._universe:
            # weight_giver: we multiply the weight given at the underlying by 0 (if not buy) or 1 (if buy)
            weight_given = (
                self.position.filter((pl.col('Date') == ts)).select([str(underlying_code)]) * w
            )
            self._weight_by_pk[
                (self._config.strategy_code, underlying_code, ts)
            ] = Weight(
                product_code=self._config.strategy_code,
                underlying_code=underlying_code,
                ts=ts,
                value=weight_given,
            )

    def _compute_perf(self, ts: datetime) -> float:
        """
        Function which compute the perf of our strategy
        """
        perf_ = 0.0
        for underlying_code in self._universe:
            posi = self.position.filter((pl.col('Date') == ts)).select([str(underlying_code)])
            key = (self._config.strategy_code, underlying_code, ts - self._timedelta)
            day_prev = self.find_closest_closing_price(
                self._weight_by_pk, ts=ts, key=key
            )
            weight = self._weight_by_pk.get(
                (
                    self._config.strategy_code,
                    underlying_code,
                    ts - self._timedelta * day_prev,
                )
            )
            if weight is not None:
                value = weight.value * posi
                current_quote = self._quote_by_pk.get(
                    (underlying_code, ts - self._timedelta)
                )
                key = (underlying_code, ts - self._timedelta)
                day_prev = self.find_closest_closing_price(
                    self._quote_by_pk, ts=ts, key=key, start=2
                )
                # start = 2 because current quote is taken at ts - _timedelta
                previous_quote = self._quote_by_pk.get(
                    (underlying_code, ts - self._timedelta * day_prev)
                )

                if current_quote is not None and previous_quote is not None:
                    perf_ += value * (current_quote.close / previous_quote.close - 1)
                else:
                    print("")
                    raise ValueError(
                        f"missing quote for {underlying_code} at {ts - self._timedelta * day_prev} "
                        f"or {ts - self._timedelta * 2}"
                    )
        return perf_

    def compute_levels(self) -> List[Quote]:
        # compute the number of stock which will have a weight different from 0
        nb_stock_to_hold_per_period = self.position.with_column(pl.fold(0, lambda x, y: x + y, pl.all().exclude('Date'))
                                                                .alias('sum_h'))
        for ts in self._calendar:
            self._compute_weight(ts, nb_stock_to_hold_per_period)
            if ts == self._config.start_ts:
                quote = Quote(close=self._config.basis, ts=ts - self._timedelta)
                self._level_by_ts[ts - self._timedelta] = quote
            else:
                perf = self._compute_perf(ts)
                prev_day = self.find_closest_closing_price(
                    self._level_by_ts, ts=ts, start=2
                )
                close = self._level_by_ts.get(ts - self._timedelta * prev_day).close * (
                    1 + perf
                ).to_pandas().values[0][0]
                quote = Quote(close=close, ts=ts)
                self._level_by_ts[ts - self._timedelta] = quote

        return list(self._level_by_ts.values())

    def find_closest_closing_price(self, dico, ts, start=1, key=None) -> int:
        """
        f: dict
        ts: timestamp of the current time
        start: time from which we have to start looking from to find the closest value
        key: key of the dico

        We try to find the closest timestamp for a key in a dico which can contains quotes, weigths..
        """
        for prev_day in range(start, 30):
            if key is None:
                if ts - self._timedelta * prev_day in dico:
                    return prev_day
            else:
                day_prev_test = ts - self._timedelta * prev_day
                if key[:-1] + tuple([day_prev_test]) in dico:
                    return prev_day


class StatPtf:
    def __init__(self, backtest: pd.DataFrame, rf=0.02):
        self.backtest = backtest
        self.backtest["years"] = self.backtest.ts.dt.year
        self.backtest.columns = ["Date", "close", "years"]
        self.returns_per_year = (
            self.backtest.groupby("years")["close"].agg(["first", "last"]).reset_index()
        )
        self.table = None

        if type(rf) == "int":
            self.rf = rf
        else:
            idx = list(set(self.backtest.years))
            self.rf = rf.loc[idx]
            self.returns_per_year["10yUS"] = self.rf["10yUS"].values / 100

    def compute_sharpe_per_year(self):
        """
        Compute sharpe ratio of a series
        :return: sharpe ratio
        """
        self.returns_per_year["returns"] = (
            self.returns_per_year["last"] / self.returns_per_year["first"]
        ) - 1
        self.returns_per_year["std_ret_year"] = (
            self.backtest.close.diff().groupby(self.backtest.years).std().values
        )
        self.returns_per_year["sharpe"] = (
            self.returns_per_year["returns"] - self.returns_per_year["10yUS"]
        ) / (self.returns_per_year["std_ret_year"])
        self.table = self.returns_per_year[["years", "sharpe"]]
