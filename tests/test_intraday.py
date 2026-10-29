import numpy as np
import pandas as pd
import pytest

from trading.backtest import CostModel, IntradayConfig, run_intraday_backtest
from trading.data import attach_market, attach_regime, daily_context, market_context
from trading.regime import confirmed_regime, market_regime, zigzag_regime
from trading.data.intraday import session_vwap
from trading.strategy import IntradayStrategy, OpeningGap


class FixedSignals(IntradayStrategy):
    def __init__(self, target, stop=None):
        self.target = target
        self.stop = stop

    def generate_signals(self, bars):
        out = pd.DataFrame({"target": self.target}, index=bars.index)
        out["stop"] = np.nan if self.stop is None else self.stop
        return out


def make_bars(days, opens, highs=None, lows=None, closes=None, volume=1000, start="09:00"):
    """各日 len(opens) 本の 5 分足を作る（同じ値を毎日繰り返す）。"""
    n = len(opens)
    idx = pd.DatetimeIndex([ts for d in pd.bdate_range("2026-06-01", periods=days)
                            for ts in pd.date_range(f"{d.date()} {start}", periods=n, freq="5min")])
    rep = lambda x: np.tile(x if x is not None else opens, days).astype(float)  # noqa: E731
    return pd.DataFrame({"open": rep(opens), "high": rep(highs), "low": rep(lows),
                         "close": rep(closes), "volume": volume}, index=idx)


def config(**kw):
    return IntradayConfig(initial_cash=100_000, lot_size=100, last_entry_time="23:00", **kw)


def test_next_bar_open_and_eod_exit():
    bars = make_bars(1, [100, 101, 102, 103], closes=[100.5, 101.5, 102.5, 103.5])
    result = run_intraday_backtest(bars, FixedSignals([1, 1, 1, 1]), config())
    trades = result.trades
    assert trades["date"].tolist() == [bars.index[1], bars.index[3]]
    assert trades["price"].tolist() == [101, 103.5]  # 次バーの始値で買い、最終バーの終値で売り
    assert result.round_trips["exit_reason"].tolist() == ["eod"]
    assert result.equity.iloc[-1] == 100_000 + 900 * 2.5


def test_signal_not_carried_over_to_next_day():
    bars = make_bars(2, [100, 100, 100])
    target = [0, 0, 1, 0, 0, 0]  # 1日目の最終バーのシグナルは 2 日目に執行しない
    result = run_intraday_backtest(bars, FixedSignals(target), config())
    assert result.trades.empty


def test_signal_exit():
    bars = make_bars(1, [100, 101, 102, 103])
    result = run_intraday_backtest(bars, FixedSignals([1, 0, 0, 0]), config())
    assert result.trades["price"].tolist() == [101, 102]
    assert result.round_trips["exit_reason"].tolist() == ["signal"]


def test_stop_hit_and_gap_down():
    lows = [100, 100, 95, 100]
    bars = make_bars(1, [100, 100, 100, 100], lows=lows)
    result = run_intraday_backtest(bars, FixedSignals([1, 1, 1, 1], stop=97), config())
    assert result.trades["price"].tolist() == [100, 97]
    assert result.round_trips["exit_reason"].tolist() == ["stop"]

    # 損切り価格より下で寄り付いたら始値で決済
    bars = make_bars(1, [100, 100, 96, 100], lows=[100, 100, 95, 100])
    result = run_intraday_backtest(bars, FixedSignals([1, 1, 1, 1], stop=97), config())
    assert result.trades["price"].tolist() == [100, 96]


def test_short_entry_exit_and_fee():
    bars = make_bars(1, [100, 100, 98, 97], closes=[100, 99, 97.5, 96])
    cfg = config(costs=CostModel(spread_yen=1.0, short_fee_rate=0.0365))
    result = run_intraday_backtest(bars, FixedSignals([-1, -1, -1, -1]), cfg)
    rt = result.round_trips.iloc[0]
    # 次バーの始値 100 から 0.5 円不利な 99.5 で空売り、大引け 96 から 0.5 円不利な 96.5 で買い戻し
    assert (rt["side"], rt["entry_price"], rt["exit_price"], rt["exit_reason"]) == ("short", 99.5, 96.5, "eod")
    fee = 96.5 * rt["shares"] * 0.0365 / 365  # 貸株料 1 日分
    assert rt["pnl"] == pytest.approx((99.5 - 96.5) * rt["shares"] - fee)
    assert result.equity.iloc[-1] == pytest.approx(100_000 + rt["pnl"])
    # シグナルで買い戻し
    result = run_intraday_backtest(bars, FixedSignals([-1, 0, 0, 0]), config())
    assert result.trades["side"].tolist() == ["short", "cover"]
    assert result.trades["price"].tolist() == [100, 98]


def test_short_stop_hit_and_gap_up():
    bars = make_bars(1, [100, 100, 100, 100], highs=[100, 100, 104, 100])
    result = run_intraday_backtest(bars, FixedSignals([-1] * 4, stop=103), config())
    assert result.trades["price"].tolist() == [100, 103]
    assert result.round_trips["exit_reason"].tolist() == ["stop"]
    # 損切り価格より上で寄り付いたら始値で買い戻し
    bars = make_bars(1, [100, 100, 104, 100], highs=[100, 100, 105, 100])
    result = run_intraday_backtest(bars, FixedSignals([-1] * 4, stop=103), config())
    assert result.trades["price"].tolist() == [100, 104]
    assert result.round_trips["pnl"].iloc[0] < 0


def test_max_entries_and_last_entry_time():
    bars = make_bars(1, [100] * 6)
    target = [1, 0, 1, 0, 1, 0]
    result = run_intraday_backtest(bars, FixedSignals(target), config(max_entries_per_day=2))
    assert len(result.round_trips) == 2

    cfg = IntradayConfig(initial_cash=100_000, lot_size=100, last_entry_time="09:05")
    result = run_intraday_backtest(bars, FixedSignals([0, 1, 1, 1, 1, 1]), cfg)
    assert result.trades.empty  # 09:10 のバーでの新規建ては禁止


def test_costs_applied():
    bars = make_bars(1, [100, 100, 100])
    cfg = config(costs=CostModel(slippage_bps=100, commission_fixed=10))
    result = run_intraday_backtest(bars, FixedSignals([1, 1, 1]), cfg)
    assert result.trades["price"].tolist() == [pytest.approx(101), pytest.approx(99)]
    assert result.round_trips["pnl"].iloc[0] == pytest.approx(900 * -2 - 20)






def test_session_vwap_resets_daily():
    bars = make_bars(2, [100, 110], volume=1)
    bars["volume"] = [1, 3, 1, 3]
    vwap = session_vwap(bars)
    assert vwap.tolist() == [100, pytest.approx(107.5), 100, pytest.approx(107.5)]












def test_market_context():
    up = make_bars(1, [100, 101, 102], closes=[100, 101, 102])
    down = make_bars(1, [100, 99, 98], closes=[100, 99, 98])
    m = market_context({"A": up, "B": down, "C": up})
    assert m["mkt_breadth"].tolist() == [0, pytest.approx(2 / 3), pytest.approx(2 / 3)]
    gap_up = 102 / session_vwap(up).iloc[-1] - 1
    gap_down = 98 / session_vwap(down).iloc[-1] - 1
    assert m["mkt_gap"].iloc[-1] == pytest.approx((2 * gap_up + gap_down) / 3)
    assert "mkt_gap" in attach_market(up, m)


def test_daily_context_uses_previous_day_only():
    idx = pd.bdate_range("2026-06-01", periods=4)
    daily = pd.DataFrame({"close": [10.0, 20.0, 30.0, 40.0]}, index=idx)
    ctx = daily_context(daily, sma_windows=(2,))
    assert np.isnan(ctx["prev_close"].iloc[0])
    assert ctx["prev_close"].tolist()[1:] == [10, 20, 30]
    assert ctx["sma2"].iloc[2] == 15


def test_opening_gap_short():
    # 1 日目は 100 で終わり、2 日目は 98（-2%）で寄る → 09:00 の足を見て 09:05 に空売り、09:10 の足で手仕舞い
    bars = make_bars(2, [100, 100, 100, 100], start="09:00")
    bars.iloc[4:, :4] = 98
    sig = OpeningGap(min_gap=0.01, exit_time="09:10", regime=None, stop_pct=0.02).generate_signals(bars)
    assert sig["target"].tolist() == [0, 0, 0, 0] + [-1, -1, 0, 0]
    assert sig["stop"].iloc[4] == pytest.approx(98 * 1.02)
    # 窓が小さければ売らない
    assert OpeningGap(min_gap=0.03, regime=None).generate_signals(bars)["target"].sum() == 0
    # 局面: 下げ相場の日だけ
    days = bars.index.normalize().unique()
    up = attach_regime(bars, pd.Series([1, 0], index=days))
    down = attach_regime(bars, pd.Series([0, 1], index=days))
    assert OpeningGap(exit_time="09:10").generate_signals(up)["target"].sum() == 0
    assert OpeningGap(exit_time="09:10").generate_signals(down)["target"].sum() == -2
    with pytest.raises(ValueError):
        OpeningGap().generate_signals(bars)
    # エンジンでは 09:05 の始値で売り、09:15 の始値で買い戻す
    result = run_intraday_backtest(down, OpeningGap(exit_time="09:10"), config())
    assert result.trades["date"].tolist() == [bars.index[5], bars.index[7]]


def test_regime_labels():
    close = pd.Series([100, 110, 120, 105, 100, 106, 112, 125], dtype=float,
                      index=pd.bdate_range("2026-01-01", periods=8))
    # 120 → 100（-17%）が下げ、100 → 125 が上げ。後から見ると高値の日から下げ相場
    assert zigzag_regime(close, 0.10).tolist()[2:] == [1, 1, 0, 0, 0, 0]
    # その日に分かるのは 10% 下げた 105 の日（4 日目）から。上げに戻るのは 10% 戻した 112 の日
    assert confirmed_regime(close, 0.10).tolist()[3:] == [1, 1, 1, 0, 0]
    # 5 分足では前日までに確定した局面を使う
    assert market_regime(close, "confirmed", 0.10).tolist()[4:] == [1, 1, 1, 0]
