import numpy as np
import pandas as pd
import pytest

from trading.backtest import CostModel
from trading.paper import Account, compute_signals, exit_limit, next_orders, step
from trading.strategy.band_touch import band_price


def make_market(n225_close=None):
    """銘柄 A が 31 日目（i=30）に -2σ を割って引け、翌日に戻る。日経平均は寄りも引けも同じ値段。"""
    idx = pd.bdate_range("2024-01-01", periods=33)
    c = np.array([1000.0 + (5 if i % 2 else -5) for i in range(33)])
    c[30], c[31], c[32] = 950.0, 1000.0, 1000.0
    o = np.r_[c[0], c[:-1]]
    o[31] = 960.0
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 1, "low": np.minimum(o, c) - 5, "close": c,
                       "volume": 200_000.0}, index=idx)
    nkc = pd.Series(30_000.0 if n225_close is None else n225_close, index=idx)
    nk = pd.DataFrame({"close": nkc, "open_tradable": nkc.shift(1).fillna(nkc.iloc[0])}, index=idx)
    return {"A": df}, nk, pd.Series(0.6, index=idx)


def start(sig, day, capital=1_000_000.0):
    acct = Account(capital=capital, slots=1, idle=capital, in_index=True)
    acct.equity.append([str(day.date()), capital, float(sig.market.loc[day, "nk_close"])])
    acct.last_date = str(day.date())
    return acct


def test_close_signal_buy_next_open_and_exit_at_minus_one_sigma():
    stocks, nk, p_down = make_market()
    sig = compute_signals(stocks, nk, p_down, [["p_down", "≧", 0.5]], min_stocks=99, index_p_down=0.5)
    days = nk.index
    assert sig.buy["date"].tolist() == [days[30]]   # -2σ 割れは 31 日目だけ（最初の割れ）
    acct = start(sig, days[29])
    step(acct, days[30], sig, CostModel())
    assert acct.orders["buy"] == [{"code": "A", "shares": 1000, "signal_date": str(days[30].date()), "est_price": 950.0}]
    assert not acct.orders["overlay_buy"]
    step(acct, days[31], sig, CostModel())
    # 寄りで 960 円 × 1,000 株を買い（日経平均を売る費用 0.05%）、終値 1,000 円は -1σ 以上なので大引けで売る
    assert acct.positions == []
    t = acct.trades[0]
    assert (t["entry_price"], t["exit_price"], t["exit_reason"], t["pnl"]) == (960.0, 1000.0, "target", 40_000.0)
    assert acct.idle == pytest.approx(1_000_000 - 960_000 * 1.0005 + 1_000_000 * 0.9995)
    assert acct.equity[-1][1] == pytest.approx(acct.idle)


def test_rules_and_one_lot_must_fit():
    stocks, nk, p_down = make_market()
    sig = compute_signals(stocks, nk, p_down, [["p_down", "≧", 0.7]], min_stocks=99, index_p_down=0.5)
    assert sig.buy.empty
    sig = compute_signals(stocks, nk, p_down, [], min_stocks=99, index_p_down=0.5)
    acct = start(sig, sig.market.index[29], capital=90_000)   # 950 円 × 100 株 = 95,000 円は 1 枠に収まらない
    step(acct, sig.market.index[30], sig, CostModel())
    assert acct.orders["buy"] == [] and "収まらず見送り" in acct.events[-1][1]


def test_index_signal_b_uses_n225_close_band():
    nkc = [30_000.0 + (10 if i % 2 else -10) for i in range(33)]
    nkc[30] = 29_000.0   # 25 日のバンドで −2.5σ より下
    stocks, nk, p_down = make_market(nkc)
    sig = compute_signals(stocks, nk, p_down, [], min_stocks=None, index_p_down=0.5, index_signal="B", index_k=2.5)
    days = nk.index
    assert sig.market.loc[days[30], "nk_z"] < -2.5
    assert sig.market["index_signal"].tolist() == [d == days[30] for d in days]
    p_low = pd.Series(0.4, index=days)   # 前日の下げ相場の確率が 0.5 以下なら出ない
    assert not compute_signals(stocks, nk, p_low, [], None, 0.5, index_signal="B", index_k=2.5).market["index_signal"].any()


def test_index_overlay_next_open():
    nkc = [30_000.0 + (10 if i % 2 else -10) for i in range(33)]
    nkc[30], nkc[31], nkc[32] = 29_000.0, 29_500.0, 30_100.0
    stocks, nk, p_down = make_market(nkc)
    sig = compute_signals(stocks, nk, p_down, [["p_down", "≧", 0.9]], min_stocks=1, index_p_down=0.5)
    days = nk.index
    assert sig.market.loc[days[30], "count3"] == 1 and sig.market.loc[days[30], "index_signal"]
    acct = start(sig, days[29])
    step(acct, days[30], sig, CostModel())
    eq30 = acct.equity[-1][1]
    assert acct.orders["overlay_buy"]
    step(acct, days[31], sig, CostModel())
    o = acct.overlay
    # 寄り（前日の引け 29,000）で資産と同じ額を買い増す。建てた日は手仕舞わない
    assert o["entry_price"] == 29_000.0 and o["units"] * 29_000.0 == pytest.approx(eq30)
    assert acct.orders["overlay_buy"] is False   # 持っている間の合図は使わない
    step(acct, days[32], sig, CostModel())
    assert acct.overlay is None and acct.trades[-1]["code"] == "^N225" and acct.trades[-1]["exit_reason"] == "target"
    units = eq30 / 29_000.0
    interest = units * 29_000.0 * 0.028 * (days[32] - days[31]).days / 365
    expected = units * (30_100.0 - 29_000.0) - units * 29_000.0 * 0.0005 - units * 30_100.0 * 0.0005 - interest
    assert acct.trades[-1]["pnl"] == pytest.approx(expected)


def start_w(sig, day, capital=1_000_000.0, w=0.5):
    acct = Account(capital=capital, slots=1, index_weight=w, idle=capital * w, cash=capital * (1 - w), in_index=True)
    acct.equity.append([str(day.date()), capital, float(sig.market.loc[day, "nk_close"])])
    acct.last_date = str(day.date())
    return acct


def test_half_in_index_buys_from_index_and_cash_and_returns_proceeds_by_weight():
    stocks, nk, p_down = make_market()
    sig = compute_signals(stocks, nk, p_down, [], min_stocks=99, index_p_down=0.5)
    days = nk.index
    acct = start_w(sig, days[29])
    step(acct, days[30], sig, CostModel())
    assert acct.orders["buy"][0]["shares"] == 1000   # 1 枠 100 万円。日経平均を売る費用は半分の分だけ
    step(acct, days[31], sig, CostModel())
    # 寄りで 960 円 × 1,000 株を、日経平均（売る費用込み）と現金から半分ずつ。大引けで 1,000 円で売り、代金を半分ずつ戻す
    assert acct.idle == pytest.approx(500_000 - 480_000 * 1.0005 + 500_000 * 0.9995)
    assert acct.cash == pytest.approx(500_000 - 480_000 + 500_000)
    assert acct.equity[-1][1] == pytest.approx(acct.idle + acct.cash)


def test_half_in_index_overlay_is_paid_with_cash_without_interest():
    nkc = [30_000.0 + (10 if i % 2 else -10) for i in range(33)]
    nkc[30], nkc[31], nkc[32] = 29_000.0, 29_500.0, 30_100.0
    stocks, nk, p_down = make_market(nkc)
    sig = compute_signals(stocks, nk, p_down, [["p_down", "≧", 0.9]], min_stocks=1, index_p_down=0.5)
    days = nk.index
    acct = start_w(sig, days[29])
    step(acct, days[30], sig, CostModel())
    eq30 = acct.equity[-1][1]
    step(acct, days[31], sig, CostModel())
    o = acct.overlay
    assert o["units"] * 29_000.0 == pytest.approx(eq30 * 0.5) and o["debt"] == 0.0   # 資産の半分を、現金で買い増す
    step(acct, days[32], sig, CostModel())
    units = eq30 * 0.5 / 29_000.0
    expected = units * (30_100.0 - 29_000.0) - units * 29_000.0 * 0.0005 - units * 30_100.0 * 0.0005   # 金利はかからない
    assert acct.trades[-1]["code"] == "^N225" and acct.trades[-1]["pnl"] == pytest.approx(expected)


def test_exit_limit_is_minus_one_sigma_price():
    c = pd.Series(np.linspace(900, 1100, 40))
    assert exit_limit(c) == pytest.approx(band_price(pd.concat([c, pd.Series([np.nan])], ignore_index=True), 25, -1.0)
                                          .iloc[-1])


def test_next_orders_skip_held_and_full():
    stocks, nk, p_down = make_market()
    sig = compute_signals(stocks, nk, p_down, [], min_stocks=99, index_p_down=0.5)
    day = nk.index[30]
    acct = start(sig, nk.index[29])
    acct.positions.append({"code": "A", "shares": 100, "entry_date": "2024-01-01", "entry_price": 1000.0,
                           "signal_date": "2024-01-01", "days": 1, "exit_next_open": False})
    assert next_orders(acct, day, sig, CostModel(), 1_000_000)["buy"] == []
