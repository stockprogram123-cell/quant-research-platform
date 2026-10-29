import math
from dataclasses import asdict, dataclass, field

import pandas as pd

from ..strategy import Strategy
from .costs import CostModel
from .metrics import compute_metrics
from .portfolio import ROUND_TRIP_COLUMNS, Portfolio


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000
    lot_size: int = 100
    position_size: float = 1.0  # エントリー時に使う現金の割合
    costs: CostModel = field(default_factory=CostModel)


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    round_trips: pd.DataFrame
    metrics: dict[str, float]


def _affordable_shares(budget: float, price: float, lot_size: int, costs: CostModel) -> int:
    """手数料込みで budget 内に収まる最大株数（lot_size 単位）。"""
    lots = math.floor(budget / (price * lot_size))
    while lots > 0:
        shares = lots * lot_size
        if shares * price + costs.commission(shares * price) <= budget:
            return shares
        lots -= 1
    return 0


def run_backtest(df: pd.DataFrame, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
    """日足で単一銘柄のバックテストを行う（買い・空売り）。

    当日終値までで算出したシグナル（1 = 買い / -1 = 空売り / 0 = ノーポジ）を、
    翌営業日の始値で執行する（先読み防止）。空売りは買い戻し時に保有した暦日数分の貸株料を払う。
    """
    if hasattr(strategy, "orders"):  # 日中に値段に触れたら約定する戦略
        from .touch_engine import run_touch_backtest
        return run_touch_backtest(df, strategy, config)
    signals = strategy.generate_signals(df)
    reason = pd.Series("", index=df.index)
    if isinstance(signals, pd.DataFrame):  # 手仕舞いの理由（reason 列）を返す戦略もある
        if "reason" in signals:
            reason = signals["reason"].reindex(df.index).fillna("").shift(1).fillna("")
        signals = signals["target"]
    target = signals.reindex(df.index).fillna(0).shift(1).fillna(0)

    pf = Portfolio(cash=config.initial_cash)
    costs = config.costs
    equity = {}

    for date, bar in df.iterrows():
        tgt = target[date]
        if pf.shares > 0 and tgt <= 0:
            price = costs.fill_price(bar["open"], "sell")
            pf.sell_all(date, price, costs.commission(pf.shares * price), reason[date] or "signal")
        elif pf.shares < 0 and tgt >= 0:
            price = costs.fill_price(bar["open"], "buy")
            notional = -pf.shares * price
            days = max((date - pf._entry_date).days, 1)
            pf.cover_all(date, price, costs.commission(notional), costs.short_fee(notional, days),
                         reason[date] or "signal")
        if pf.shares == 0 and tgt != 0:
            price = costs.fill_price(bar["open"], "buy" if tgt > 0 else "sell")
            shares = _affordable_shares(pf.cash * config.position_size, price, config.lot_size, costs)
            if shares > 0:
                if tgt > 0:
                    pf.buy(date, shares, price, costs.commission(shares * price))
                else:  # 空売りも買いと同じ金額（資金 × position_size）まで
                    pf.short(date, shares, price, costs.commission(shares * price))
        equity[date] = pf.equity(bar["close"])

    equity = pd.Series(equity, name="equity")
    trades = pd.DataFrame([asdict(t) for t in pf.trades],
                          columns=["date", "side", "shares", "price", "commission"])
    round_trips = pd.DataFrame([asdict(r) for r in pf.round_trips], columns=ROUND_TRIP_COLUMNS)
    return BacktestResult(equity, trades, round_trips, compute_metrics(equity, round_trips))


def round_trip_table(result: BacktestResult, code: str) -> pd.DataFrame:
    """往復取引に銘柄コード・約定価格・手数料込みリターン（ret）を付ける。

    期間末に決済されていないポジション（日足のみ起こりうる）は含まない。
    """
    rts = result.round_trips.copy()
    if rts.empty:
        return rts
    rts["ret"] = rts["pnl"] / (rts["shares"] * rts["entry_price"])  # 空売りも売った代金に対する割合
    rts.insert(0, "code", code)
    return rts
