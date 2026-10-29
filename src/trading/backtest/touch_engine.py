"""日足で「値段に触れたら約定する注文」（逆指値・指値）を検証するエンジン。

日中の値動きの順番は日足からは分からないので、同じ日に利益確定と損切りの両方の値段に
触れた場合の扱いを fill で選ぶ:
- conservative: 損切りが先。建てた日（寄りで建てた場合を除く）は利益確定しない
- optimistic: 利益確定が先。建てた日も、高値・安値が建てた後についたとみなす
実際の成績はおおむねこの 2 つの間に入る。
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestResult, _affordable_shares
from .limits import LIMIT_COLUMNS
from .metrics import compute_metrics
from .portfolio import ROUND_TRIP_COLUMNS, Portfolio


@dataclass
class TouchTrade:
    entry_i: int
    exit_i: int
    side: int            # 1 = 買い / -1 = 空売り
    entry_price: float   # コストを含まない約定値
    exit_price: float
    reason: str          # "target" / "stop" / "time" / "limit"（ストップで引けて翌日以降に手仕舞い）


def _exit_on_day(side: int, o: float, h: float, lo: float, target: float, stop: float,
                 check_open: bool, target_ok: bool, fill: str) -> tuple[float, str] | None:
    """その日に手仕舞うなら（約定値, 理由）。side = -1 なら高値が不利、安値が有利。"""
    adverse, favorable = (h, lo) if side < 0 else (lo, h)
    if check_open:  # 寄りで損切り・利益確定の値段を越えて始まったら寄りで約定
        if side * (o - stop) <= 0:
            return o, "stop"
        if not math.isnan(target) and side * (o - target) >= 0:
            return o, "target"
    stop_hit = side * (adverse - stop) <= 0
    target_hit = target_ok and not math.isnan(target) and side * (favorable - target) >= 0
    if stop_hit and target_hit:
        return (stop, "stop") if fill == "conservative" else (target, "target")
    if stop_hit:
        return stop, "stop"
    if target_hit:
        return target, "target"
    return None


def touch_trades(o, h, lo, c, short_entry, long_entry, short_target, long_target, can_enter,
                 stop_pct: float, max_hold: int, fill: str = "conservative",
                 day: np.ndarray | None = None, long_stop_pct: float | None = None,
                 long_max_hold: int | None = None, limits: pd.DataFrame | None = None) -> list[TouchTrade]:
    """各日の注文の値段（寄りの前に決まっている値）から取引を作る。配列はすべて同じ長さの numpy 配列。

    1 行 = 1 日（日足）が基本。5 分足で確かめるときは 1 行 = 1 本にし、day に各行の営業日の番号を渡す
    （期限は営業日で数え、その日の最後の行の終値で手仕舞う）。

    - 建て: 高値が short_entry 以上なら空売り、安値が long_entry 以下なら買い。
      寄りで既に越えていたら寄りで、そうでなければその値段で約定する
    - 利益確定: その日の short_target（空売り） / long_target（買い）に触れたら
    - 損切り: 建値から stop_pct 逆行した値段に触れたら
    - 期限: 建てた日から max_hold 営業日後の大引け
    手仕舞った日には建て直さない。データの終わりまで手仕舞わなかった取引は含めない。
    long_stop_pct / long_max_hold を渡すと、買いだけその値を使う（省略時は空売りと同じ）。

    limits（日足のみ。trading.backtest.limits.limit_flags の 4 列）を渡すと値幅制限を考える:
    - 一日中ストップ高（寄らず・張り付き）の日の空売りは、15:20 までに約定しなかった注文を取り消すとみなし
      建てない（大引けの比例配分で約定して、翌日以降の連続ストップ高に捕まるのを避ける）。買いとストップ安も同じ
    - 持っている向きと逆にストップ（空売りならストップ高）で引けた日は大引けで手仕舞えず、
      一日中張り付いた日は日中も手仕舞えない。次に値段がついた日の寄りで手仕舞う（理由 "limit"）
    """
    n = len(o)
    day = np.arange(n) if day is None else day
    with np.errstate(invalid="ignore"):
        short_hit = can_enter & (h >= short_entry)
        long_hit = can_enter & (lo <= long_entry)
    if limits is not None:
        up_close, down_close, up_locked, down_locked = (
            limits[k].to_numpy(bool) for k in ("up_close", "down_close", "up_locked", "down_locked"))
        short_hit &= ~up_locked
        long_hit &= ~down_locked
    else:
        up_close = down_close = up_locked = down_locked = np.zeros(n, bool)
    trades, last_exit_day = [], -1
    for i in np.flatnonzero(short_hit | long_hit):
        if day[i] <= last_exit_day:
            continue
        side = -1 if short_hit[i] else 1
        stuck_close, stuck_day = (up_close, up_locked) if side < 0 else (down_close, down_locked)
        level = short_entry[i] if side < 0 else long_entry[i]
        at_open = side * (o[i] - level) <= 0
        entry = o[i] if at_open else level
        sp = long_stop_pct if side > 0 and long_stop_pct is not None else stop_pct
        hold = long_max_hold if side > 0 and long_max_hold is not None else max_hold
        stop = entry * (1 - side * sp)
        target = short_target if side < 0 else long_target
        # 建てた日: 寄りで建てたなら高値・安値はその後。そうでなければ順番が分からない
        res = _exit_on_day(side, o[i], h[i], lo[i], target[i], stop, check_open=False,
                           target_ok=at_open or fill == "optimistic", fill=fill)
        j, carried = i, False
        if res is None and hold == 0 and (i == n - 1 or day[i + 1] != day[i]):
            # その日の最後の行で建てて、その大引けで手仕舞う（逆向きのストップで引けたら手仕舞えない）
            carried = bool(stuck_close[i])
            res = None if carried else (c[i], "time")
        while res is None and j < n - 1:
            j += 1
            if stuck_day[j]:
                continue
            if carried:
                res = (o[j], "limit")
                break
            res = _exit_on_day(side, o[j], h[j], lo[j], target[j], stop, check_open=True, target_ok=True, fill=fill)
            if res is None and day[j] - day[i] >= hold and (j == n - 1 or day[j + 1] != day[j]):
                carried = bool(stuck_close[j])
                res = None if carried else (c[j], "time")
        if res is None:
            break
        trades.append(TouchTrade(int(i), int(j), side, float(entry), float(res[0]), res[1]))
        last_exit_day = day[j]
    return trades


def strategy_trades(df: pd.DataFrame, strategy) -> list[TouchTrade]:
    """strategy.orders(df) の注文の値段から、1 銘柄の取引（コストを含まない約定値）を作る。

    df に値幅制限の印（LIMIT_COLUMNS。trading.backtest.limits.limit_flags）があれば、touch_trades の limits として使う。
    """
    od = strategy.orders(df)
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return touch_trades(o, h, lo, c, od["short_entry"].to_numpy(float), od["long_entry"].to_numpy(float),
                        od["short_target"].to_numpy(float), od["long_target"].to_numpy(float),
                        od["can_enter"].to_numpy(bool), strategy.stop_pct, strategy.max_hold, strategy.fill,
                        long_stop_pct=getattr(strategy, "long_stop_pct", None),
                        long_max_hold=getattr(strategy, "long_max_hold", None),
                        limits=df[LIMIT_COLUMNS] if set(LIMIT_COLUMNS) <= set(df.columns) else None)


def run_touch_backtest(df: pd.DataFrame, strategy, config: BacktestConfig) -> BacktestResult:
    """strategy.orders(df) の注文の値段で、日中に触れたら約定するバックテスト（単一銘柄）。

    orders は short_entry / long_entry / short_target / long_target / can_enter の列を持つ（取引は strategy_trades）。
    約定値にはスプレッドなどのコストを上乗せし、空売りは保有した暦日数分の貸株料を払う。
    """
    trades = strategy_trades(df, strategy)
    c = df["close"].to_numpy(float)
    by_entry = {t.entry_i: t for t in trades}
    pf = Portfolio(cash=config.initial_cash)
    costs = config.costs
    equity, current = {}, None
    for i, date in enumerate(df.index):
        if current is None and i in by_entry:
            t = by_entry[i]
            price = costs.fill_price(t.entry_price, "buy" if t.side > 0 else "sell")
            shares = _affordable_shares(pf.cash * config.position_size, price, config.lot_size, costs)
            if shares > 0:
                (pf.buy if t.side > 0 else pf.short)(date, shares, price, costs.commission(shares * price))
                current = t
        if current is not None and current.exit_i == i:
            if current.side > 0:
                price = costs.fill_price(current.exit_price, "sell")
                pf.sell_all(date, price, costs.commission(pf.shares * price), current.reason)
            else:
                price = costs.fill_price(current.exit_price, "buy")
                notional = -pf.shares * price
                days = max((date - pf._entry_date).days, 1)
                pf.cover_all(date, price, costs.commission(notional), costs.short_fee(notional, days), current.reason)
            current = None
        equity[date] = pf.equity(c[i])

    equity = pd.Series(equity, name="equity")
    trade_df = pd.DataFrame([asdict(t) for t in pf.trades], columns=["date", "side", "shares", "price", "commission"])
    round_trips = pd.DataFrame([asdict(r) for r in pf.round_trips], columns=ROUND_TRIP_COLUMNS)
    return BacktestResult(equity, trade_df, round_trips, compute_metrics(equity, round_trips))
