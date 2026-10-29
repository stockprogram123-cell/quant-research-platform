import math
from dataclasses import asdict, dataclass
from datetime import time

import pandas as pd

from ..data.intraday import attach_daily
from ..strategy import IntradayStrategy
from .engine import BacktestConfig, BacktestResult, _affordable_shares
from .metrics import compute_metrics
from .portfolio import ROUND_TRIP_COLUMNS, Portfolio


@dataclass
class IntradayConfig(BacktestConfig):
    max_entries_per_day: int = 1
    last_entry_time: str = "14:30"  # これより後に始まるバーでは新規に建てない


def run_intraday_backtest(bars: pd.DataFrame, strategy: IntradayStrategy, config: IntradayConfig,
                          daily_ctx: pd.DataFrame | None = None) -> BacktestResult:
    """5 分足で単一銘柄のデイトレードをバックテストする（買い・空売り）。

    バーの終値で出したシグナル（target: 1 = 買い / -1 = 空売り / 0 = ノーポジ）を、
    同じ日の次のバーの始値で執行する（日をまたいで持ち越さない）。
    損切りは、買いなら安値が、空売りなら高値が損切り価格に触れたら損切り価格で
    （窓を開けて越えて始まったら始値で）決済する。各日の最終バーの終値で必ず手仕舞う。
    空売りは買い戻し時に貸株料（1 日分）を払う。equity は日次（各日の最終バー時点）。
    """
    bars = attach_daily(bars, daily_ctx)
    signals = strategy.generate_signals(bars).reindex(bars.index)
    stop = signals["stop"] if "stop" in signals else pd.Series(math.nan, index=bars.index)

    day = pd.Series(bars.index.normalize(), index=bars.index)
    prev_target = signals["target"].fillna(0).groupby(day).shift(1).fillna(0)
    prev_stop = stop.groupby(day).shift(1)
    is_last = day != day.shift(-1)
    last_entry = time.fromisoformat(config.last_entry_time)

    pf = Portfolio(cash=config.initial_cash)
    costs = config.costs
    equity = {}
    current_day = None
    entries = 0
    active_stop = math.nan

    def close(ts, price, reason):
        if pf.shares > 0:
            price = costs.fill_price(price, "sell")
            pf.sell_all(ts, price, costs.commission(pf.shares * price), reason)
        else:
            price = costs.fill_price(price, "buy")
            notional = -pf.shares * price
            pf.cover_all(ts, price, costs.commission(notional), costs.short_fee(notional), reason)

    for ts, o, h, l, c, tgt, stp, last in zip(bars.index, bars["open"], bars["high"], bars["low"], bars["close"],
                                               prev_target, prev_stop, is_last):
        if ts.normalize() != current_day:
            current_day, entries = ts.normalize(), 0

        if (pf.shares > 0 and tgt <= 0) or (pf.shares < 0 and tgt >= 0):
            close(ts, o, "signal")
        elif (pf.shares == 0 and tgt != 0 and not last and entries < config.max_entries_per_day
              and ts.time() <= last_entry and not (o <= stp if tgt > 0 else o >= stp)):
            price = costs.fill_price(o, "buy" if tgt > 0 else "sell")
            shares = _affordable_shares(pf.cash * config.position_size, price, config.lot_size, costs)
            if shares > 0:
                if tgt > 0:
                    pf.buy(ts, shares, price, costs.commission(shares * price))
                else:  # 空売りも買いと同じ金額（資金 × position_size）まで
                    pf.short(ts, shares, price, costs.commission(shares * price))
                entries += 1
                active_stop = stp

        if pf.shares > 0 and l <= active_stop:
            close(ts, min(o, active_stop), "stop")
        elif pf.shares < 0 and h >= active_stop:
            close(ts, max(o, active_stop), "stop")
        if pf.shares != 0 and last:
            close(ts, c, "eod")
        if last:
            equity[current_day] = pf.equity(c)

    equity = pd.Series(equity, name="equity")
    trades = pd.DataFrame([asdict(t) for t in pf.trades],
                          columns=["date", "side", "shares", "price", "commission"])
    round_trips = pd.DataFrame([asdict(r) for r in pf.round_trips], columns=ROUND_TRIP_COLUMNS)
    return BacktestResult(equity, trades, round_trips, compute_metrics(equity, round_trips))
