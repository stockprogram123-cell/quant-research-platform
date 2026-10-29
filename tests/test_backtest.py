import numpy as np
import pandas as pd
import pytest

from trading.backtest import BacktestConfig, CostModel, compute_metrics, max_drawdown, run_backtest
from trading.strategy import BandTouchFade, Strategy


class FixedSignals(Strategy):
    def __init__(self, signals):
        self.signals = signals

    def generate_signals(self, df):
        return pd.Series(self.signals, index=df.index)


def make_df(opens, closes=None):
    closes = closes if closes is not None else opens
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame({"open": opens, "high": opens, "low": opens,
                         "close": closes, "volume": 1000}, index=idx)


def test_executes_next_day_open():
    df = make_df([100, 200, 300, 400, 500, 600], closes=[110, 210, 310, 410, 510, 610])
    result = run_backtest(df, FixedSignals([0, 1, 1, 0, 0, 1]),
                          BacktestConfig(initial_cash=100_000, lot_size=100))
    trades = result.trades
    # 1日目のシグナル → 2日目(index 2)の寄付で買い、3日目のシグナル → index 4 の寄付で売り
    assert list(trades["side"]) == ["buy", "sell"]
    assert trades["date"].tolist() == [df.index[2], df.index[4]]
    assert trades["price"].tolist() == [300, 500]
    assert trades["shares"].tolist() == [300, 300]  # 100,000 / (300*100) = 3単元
    # 最終日のシグナルは翌日がないので約定しない
    assert result.equity.iloc[-1] == 100_000 + 300 * (500 - 300)
    assert result.round_trips["pnl"].tolist() == [60_000]


def test_lot_size_respects_commission():
    df = make_df([300] * 4)
    costs = CostModel(commission_fixed=100)
    result = run_backtest(df, FixedSignals([1, 1, 1, 1]),
                          BacktestConfig(initial_cash=90_050, lot_size=100, costs=costs))
    # 3単元だと 90,000 + 100 > 90,050 なので 2単元
    assert result.trades["shares"].tolist() == [200]
    assert result.equity.iloc[-1] == pytest.approx(90_050 - 100)


def test_slippage_and_rate_commission():
    costs = CostModel(commission_rate=0.001, commission_min=50, slippage_bps=10)
    assert costs.fill_price(1000, "buy") == pytest.approx(1001)
    assert costs.fill_price(1000, "sell") == pytest.approx(999)
    assert costs.commission(100_000) == pytest.approx(100)
    assert costs.commission(10_000) == pytest.approx(50)  # 最低手数料


def test_spread_cost():
    costs = CostModel(spread_yen=1.0)
    assert costs.fill_price(5000, "buy") == pytest.approx(5000.5)  # 片道でスプレッドの半分
    assert costs.fill_price(5000, "sell") == pytest.approx(4999.5)
    floor = CostModel(spread_yen=1.0, spread_min_tick=True)
    assert floor.fill_price(2000, "buy") == pytest.approx(2000.5)    # 呼値 0.5 円 < 1 円 → 1 円
    assert floor.fill_price(20000, "buy") == pytest.approx(20002.5)  # 呼値 5 円 → スプレッド 5 円
    assert floor.fill_price(50000, "sell") == pytest.approx(49995)   # 呼値 10 円


def test_cannot_afford_one_lot():
    df = make_df([5000] * 3)
    result = run_backtest(df, FixedSignals([1, 1, 1]),
                          BacktestConfig(initial_cash=400_000, lot_size=100))
    assert result.trades.empty
    assert (result.equity == 400_000).all()


def test_metrics():
    idx = pd.bdate_range("2024-01-01", periods=4)
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
    rts = pd.DataFrame({"pnl": [10.0, -5.0, 3.0, 0.0]})
    m = compute_metrics(equity, rts)
    assert m["total_return"] == pytest.approx(0.10)
    assert m["max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert m["num_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.5)
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0




def test_daily_short_and_fee():
    df = make_df([100, 100, 100, 80, 80], closes=[100, 100, 100, 80, 80])
    costs = CostModel(short_fee_rate=0.365)  # 1 日あたり 0.1%
    result = run_backtest(df, FixedSignals([-1, -1, 0, 0, 0]),
                          BacktestConfig(initial_cash=100_000, lot_size=100, costs=costs))
    assert list(result.trades["side"]) == ["short", "cover"]
    assert result.trades["date"].tolist() == [df.index[1], df.index[3]]
    rt = result.round_trips.iloc[0]
    assert rt["side"] == "short"
    days = (df.index[3] - df.index[1]).days
    assert rt["pnl"] == pytest.approx(1000 * 20 - 1000 * 80 * 0.001 * days)


def test_daily_switch_long_to_short():
    df = make_df([100, 100, 110, 120, 120])
    result = run_backtest(df, FixedSignals([1, -1, 0, 0, 0]), BacktestConfig(initial_cash=100_000, lot_size=100))
    assert list(result.trades["side"]) == ["buy", "sell", "short", "cover"]
    assert result.trades["date"].tolist() == [df.index[1], df.index[2], df.index[2], df.index[3]]




def test_band_price_matches_live_band():
    from trading.strategy import band_price
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(0, 2, 40).cumsum())
    for z in (3.0, 2.0, 0.0, -3.0):
        p = band_price(close, 25, z).iloc[-1]
        window = pd.concat([close.iloc[-25:-1], pd.Series([p])])
        assert (p - window.mean()) / window.std(ddof=0) == pytest.approx(z, abs=1e-9)


def test_touch_trades_fill_rules():
    from trading.backtest.touch_engine import touch_trades
    nan = np.nan
    a = lambda *v: np.array(v, dtype=float)
    # 0 日目に 110 で空売り（寄りは 100）、同じ日に安値 95 が目標 100 に届く
    o, h, lo, c = a(100, 100, 100), a(112, 101, 101), a(95, 99, 99), a(105, 100, 100)
    se, le = a(110, nan, nan), a(nan, nan, nan)
    st, lt = a(100, 100, 100), a(nan, nan, nan)
    ok = np.array([True, True, True])
    cons = touch_trades(o, h, lo, c, se, le, st, lt, ok, stop_pct=0.5, max_hold=5, fill="conservative")
    opt = touch_trades(o, h, lo, c, se, le, st, lt, ok, stop_pct=0.5, max_hold=5, fill="optimistic")
    # 保守的: 建てた日は利益確定しない → 翌日の寄り 100 が目標以下なので寄りで買い戻し
    assert (cons[0].entry_price, cons[0].exit_i, cons[0].exit_price, cons[0].reason) == (110, 1, 100, "target")
    # 楽観的: 建てた日のうちに目標 100 で買い戻し
    assert (opt[0].exit_i, opt[0].exit_price) == (0, 100)
    # 寄りで既に越えていたら寄りで建てる。損切りと目標の両方に触れた日は保守的なら損切り
    o, h, lo, c = a(120, 118, 118), a(121, 135, 120), a(119, 99, 117), a(119, 118, 118)
    cons = touch_trades(o, h, lo, c, se, le, a(100, 100, 100), lt, ok, stop_pct=0.1, max_hold=5,
                        fill="conservative")
    assert cons[0].entry_price == 120 and cons[0].reason == "stop" and cons[0].exit_price == pytest.approx(132)
    # 期限: max_hold 日後の大引け
    o, h, lo, c = a(100, 100, 100), a(111, 105, 105), a(100, 104, 104), a(105, 104, 103)
    t = touch_trades(o, h, lo, c, se, le, a(90, 90, 90), lt, ok, stop_pct=0.5, max_hold=2, fill="conservative")
    assert (t[0].exit_i, t[0].exit_price, t[0].reason) == (2, 103, "time")


def test_gap_only_short_open_to_close():
    closes = [100.0, 101.0] * 15
    df = make_df(closes)
    df.loc[df.index[-1], ["open", "high", "low", "close"]] = [130.0, 133.0, 115.0, 120.0]
    strat = BandTouchFade(exit_k=None, stop_pct=0.10, max_hold=0, sides="short", gap_only=True)
    result = run_backtest(df, strat, BacktestConfig(initial_cash=1_000_000, lot_size=100))
    rt = result.round_trips.iloc[0]
    assert (rt["side"], rt["entry_price"], rt["exit_price"], rt["exit_reason"]) == ("short", 130.0, 120.0, "time")
    assert rt["entry_date"] == rt["exit_date"] == df.index[-1]


def test_touch_trades_long_uses_own_stop_and_hold():
    from trading.backtest.touch_engine import touch_trades
    nan = np.nan
    a = lambda *v: np.array(v, dtype=float)
    # 0 日目に寄り 90 で買い（-3σ は 95）。空売りの設定なら当日の大引けで手仕舞うが、買いは 2 日持つ
    o, h, lo, c = a(90, 92, 93), a(91, 93, 94), a(85, 91, 92), a(88, 92, 93)
    t = touch_trades(o, h, lo, c, a(nan, nan, nan), a(95, nan, nan), a(nan, nan, nan), a(nan, nan, nan),
                     np.array([True, True, True]), stop_pct=0.01, max_hold=0, fill="conservative",
                     long_stop_pct=0.10, long_max_hold=2)
    assert (t[0].entry_price, t[0].exit_i, t[0].exit_price, t[0].reason) == (90, 2, 93, "time")


def test_band_touch_long_exit_k():
    strat = BandTouchFade(exit_k=None, long_exit_k=1.0)
    od = strat.orders(make_df([100.0, 101.0] * 15))
    assert od["short_target"].isna().all() and od["long_target"].notna().any()


def test_band_touch_regime_filter():
    df = make_df([100.0, 101.0] * 15)
    p = np.where(np.arange(len(df)) % 2 == 0, 0.8, 0.2)
    od = BandTouchFade(long_min_p_down=0.5, short_max_p_down=0.5).orders(df.assign(p_down_prev=p))
    base = BandTouchFade().orders(df)
    assert od["long_entry"].iloc[24::2].notna().all() and od["long_entry"].iloc[25::2].isna().all()
    assert od["short_entry"].iloc[25::2].notna().all() and od["short_entry"].iloc[24::2].isna().all()
    assert od["long_entry"].iloc[24::2].equals(base["long_entry"].iloc[24::2])
    with pytest.raises(ValueError):
        BandTouchFade(long_min_p_down=0.5).orders(df)


def test_limit_flags():
    from trading.backtest.limits import limit_flags
    # 前日終値 1000 → ストップ高 1300（基準値段 1,000〜1,500 円未満は 300 円）で一日張り付き。
    # 2 日目は動いたが前日と同じ値段で引け、3 日目は前日 1300 → ストップ安 1000 で引け
    df = pd.DataFrame({"open": [1000, 1300, 1000, 1150], "high": [1000, 1300, 1300, 1200],
                       "low": [1000, 1300, 1000, 1000], "close": [1000, 1300, 1300, 1000]},
                      index=pd.date_range("2024-01-01", periods=4), dtype=float)
    f = limit_flags(df)
    assert f["up_locked"].tolist() == [False, True, False, False]
    assert f["up_close"].tolist() == [False, True, False, False]
    assert f["down_close"].tolist() == [False, False, False, True] and not f["down_locked"].any()
    # 分割調整済み（実際は 2 倍の値段）なら値幅も実際の値段で決まる: 前日 2000 → 2500 がストップ高
    # （調整後の値段のままだと前日 1000 → 1300 になり、1250 はストップ高でない）
    half = pd.DataFrame({"open": [1000, 1150], "high": [1000, 1250], "low": [1000, 1150], "close": [1000, 1250]},
                        index=pd.date_range("2024-01-01", periods=2), dtype=float)
    assert not limit_flags(half)["up_close"].iloc[1]
    real = limit_flags(half, pd.Series(2.0, index=half.index))
    assert real["up_close"].iloc[1] and not real["up_locked"].iloc[1]


def test_touch_trades_price_limits():
    from trading.backtest.touch_engine import touch_trades
    nan = np.nan
    a = lambda *v: np.array(v, dtype=float)
    b = lambda *v: np.array(v, dtype=bool)
    ok, none = b(1, 1, 1, 1), a(nan, nan, nan, nan)
    o, h, lo, c = a(130, 150, 160, 150), a(130, 150, 160, 155), a(130, 150, 160, 140), a(130, 150, 160, 145)

    def run(limits):
        return touch_trades(o, h, lo, c, a(120, 120, 120, 120), none, none, none, ok, stop_pct=0.99, max_hold=0,
                            limits=pd.DataFrame(limits))
    # 0 日目は一日中ストップ高（寄らず）→ 注文は取り消し。1 日目に寄りで建てる
    t = run({"up_close": b(1, 0, 0, 0), "up_locked": b(1, 0, 0, 0), "down_close": b(0, 0, 0, 0),
             "down_locked": b(0, 0, 0, 0)})
    assert (t[0].entry_i, t[0].exit_i, t[0].reason) == (1, 1, "time")
    # 建てた 1 日目がストップ高で引け、2 日目も張り付き → 3 日目の寄りで買い戻す
    t = run({"up_close": b(1, 1, 1, 0), "up_locked": b(1, 0, 1, 0), "down_close": b(0, 0, 0, 0),
             "down_locked": b(0, 0, 0, 0)})
    assert (t[0].entry_i, t[0].exit_i, t[0].exit_price, t[0].reason) == (1, 3, 150, "limit")


def test_simulate_slots():
    from trading.backtest.slot_sim import simulate_slots
    idx = pd.bdate_range("2024-01-01", periods=4)
    closes = pd.DataFrame({"A": [100.0, 110, 120, 120], "B": [100.0, 100, 100, 100], "C": [50.0, 50, 50, 50]},
                          index=idx)
    d = lambda i: idx[i]  # noqa: E731
    cands = pd.DataFrame([
        {"code": "A", "entry_date": d(0), "exit_date": d(2), "side": 1, "entry_price": 100.0, "exit_price": 120.0,
         "reason": "target", "priority": 0},
        {"code": "B", "entry_date": d(0), "exit_date": d(1), "side": 1, "entry_price": 100.0, "exit_price": 100.0,
         "reason": "time", "priority": 1},   # 枠が 1 つなので建てられない
        {"code": "C", "entry_date": d(2), "exit_date": d(3), "side": 1, "entry_price": 50.0, "exit_price": 50.0,
         "reason": "time", "priority": 0},   # A を手仕舞った日なので、枠はまだ空いていない
        {"code": "C", "entry_date": d(3), "exit_date": d(3), "side": 1, "entry_price": 50.0, "exit_price": 55.0,
         "reason": "target", "priority": 0},
    ])
    r = simulate_slots(cands, closes, capital=100_000, slots=1, costs=CostModel())
    assert r.skipped == {"full": 2, "price": 0, "day": 0, "rank": 0, "held": 0}
    assert r.trades["code"].tolist() == ["A", "C"]
    assert r.trades["shares"].tolist() == [1000, 2400]  # 2 回目は前日の資産 120,000 円で 50 円 × 2,400 株
    assert r.equity.tolist() == [100_000, 110_000, 120_000, 132_000]
    assert r.exposure.iloc[0] == 1.0 and r.exposure.iloc[2] == 0.0
    r = simulate_slots(cands, closes, capital=100_000, slots=2, costs=CostModel(), max_per_day=1)
    assert r.skipped["day"] == 1 and r.trades["code"].tolist() == ["A", "C", "C"]
    # 既に持っている銘柄は建てない: 3 日目の C は 4 日目まで持つので、4 日目の C は見送る
    r = simulate_slots(cands, closes, capital=100_000, slots=3, costs=CostModel(), skip_held=True)
    assert r.skipped["held"] == 1 and r.trades["code"].tolist() == ["B", "A", "C"]   # 手仕舞った順
    # 空き枠 2 × 0.5 = 1 銘柄まで（max_per_day=1 と同じ）
    r = simulate_slots(cands, closes, capital=100_000, slots=2, costs=CostModel(), day_fraction=0.5)
    assert r.skipped["day"] == 1 and r.trades["code"].tolist() == ["A", "C", "C"]
    # 空き枠 2 × 0.5 = 1 銘柄にだけ注文を出す: 1 日目は順位 1 の B だけ
    ranked = cands.assign(rank=[2, 1, 1, 1])
    r = simulate_slots(ranked, closes, capital=100_000, slots=2, costs=CostModel(), orders_per_slot=0.5)
    assert r.skipped["rank"] == 1 and r.trades["code"].tolist() == ["B", "C", "C"]
    # 使っていない現金を指数で運用: 1 日目は寄りで指数を全部売って A を買うので、寄りまでの -10% だけ付く
    # （A は 100,000 円分買ったので現金は -10,000 円）。4 日目も同じ。建玉のない日がないので指数の騰落は付かない
    idx_ret = pd.Series([0.5, 0.5, 0.5, 0.5], index=idx)
    r = simulate_slots(cands, closes, capital=100_000, slots=1, costs=CostModel(), cash_return=idx_ret,
                       cash_gap=pd.Series(-0.1, index=idx))
    assert r.equity.round(6).tolist() == [90_000, 100_000, 110_000, 110_000]
    # 候補ごとの売買単位（分割調整済みの値段で、実際の 100 株が 1,000 株にあたる銘柄）: 50 円 × 1,000 株単位
    r = simulate_slots(cands.assign(lot=[100, 100, 1000, 1000]), closes, capital=100_000, slots=1, costs=CostModel())
    assert r.trades["shares"].tolist() == [1000, 2000]
    r = simulate_slots(cands.assign(lot=[100, 100, 1000, 3000]), closes, capital=100_000, slots=1, costs=CostModel())
    assert r.skipped["price"] == 1 and r.trades["code"].tolist() == ["A"]   # 50 円 × 3,000 株は 120,000 円を超える
    # 候補がなければ指数の騰落そのもの
    r = simulate_slots(cands.iloc[:0], closes, capital=100_000, slots=1, costs=CostModel(),
                       cash_return=pd.Series([0.1, 0.0, -0.5, 0.0], index=idx))
    assert r.equity.round(6).tolist() == [110_000, 110_000, 55_000, 55_000]


def test_run_name():
    from trading.store import describe_run, run_details, run_name
    p = {"k": 3.0, "exit_k": 1.0, "stop_pct": 0.2, "max_hold": 5, "sides": "long", "gap_only": False,
         "long_min_p_down": 0.5}
    assert run_name("daily", "bb_touch_fade", p, {"universe": "貸借2,147銘柄"}) == "下げ局面の−3σ割れ買い：1銘柄ずつ検証"
    assert run_details("daily", "bb_touch_fade", p, {"universe": "貸借2,147銘柄"}) == {
        "手法": "買い −3σ割れ（寄り＋場中）", "絞り込み": "下げ局面", "手仕舞い": "利確−1σ・損切20%・最長5日",
        "資金": "銘柄ごとに同じ資金で検証して合算", "データ": "貸借2,147銘柄"}

    q = {**p, "k": 2.0, "long_min_p_down": None, "long_min_high250": -0.1516, "long_max_ret5": -0.02279,
         "long_min_sma200": -0.04539}
    combo = {"legs": [p, q], "slots": 5, "capital": 10_000_000, "idle_in": "n225", "switch_cost": 0.0005}
    assert run_name("daily", "combo", combo) == "−3σ割れ＋−2σ押し目の併用：5枠・待機中は日経平均"
    desc = describe_run("daily", "combo", combo, {"focus": "寄りの約定を予想順に建てる"})
    assert desc["name"] == "−3σ割れ＋−2σ押し目の併用：寄りの約定を予想順に建てる"
    assert desc["details"] == {
        "手法1": "買い −3σ割れ（寄り＋場中）", "絞り込み1": "下げ局面",
        "手法2": "買い −2σ割れ（寄り＋場中）", "絞り込み2": "押し目（1年高値から-15%以内・5日で-2.3%以下・200日線-4.5%以上）",
        "手仕舞い": "利確−1σ・損切20%・最長5日", "資金": "1,000万円を5枠に等分", "待機中の資金": "日経平均（乗り換えは片道0.05%）"}

    gap = {"side": "short", "min_gap": 0.01, "regime": "down", "stop_pct": 0.015, "exit_time": "10:55"}
    assert run_name("intraday", "opening_gap", gap, {"regime_mode": "confirmed"}) == \
        "寄りの窓下げを空売り：下げ相場の日だけ（前日までで確定）"

    wait = {**p, "slots": 10, "capital": 10_000_000, "wait_days": 1, "idle_in": "n225"}
    assert run_name("daily", "bb_touch_fade", wait) == "下げ局面の−3σ割れ買い：10枠・局面の初日は見送る"
    assert run_details("daily", "bb_touch_fade", wait)["見送る日"] == "下げ局面の1日目まで"

    idx = {"signal": "A", "k": 3.0, "exit_k": 1.0, "stop_pct": 0.2, "max_hold": 5, "sides": "long", "extra": 1.0,
           "rate": 0.028, "switch_cost": 0.0005, "capital": 10_000_000, "long_min_p_down": 0.5}
    assert run_name("daily", "index_overlay", idx) == "日経平均の−3σ割れで買い増し：下げ局面だけ・最長5日"
    d = run_details("daily", "index_overlay", idx)
    assert d["手仕舞い"] == "利確−1σ・損切20%・最長5日" and d["資金"] == "1,000万円をいつも日経平均で持ち、合図の間は2倍"
    assert run_name("daily", "index_overlay", {**idx, "signal": "C", "min_stocks": 30, "long_min_p_down": None}) == \
        "30銘柄以上が−3σ割れの日に日経平均を買い増し：局面で絞らない・最長5日"

    close = {"k": 2.0, "entry": "引け", "exit": "−1σで引け・最長5日", "first_break": True,
             "rules": [["ret5", "≦", -0.03]], "rules_text": "5日の騰落≦-3%", "pick": "深く下げた順（終値のσが低い順）",
             "slots": 5, "capital": 10_000_000, "idle_in": "n225", "switch_cost": 0.0005}
    assert run_name("daily", "bb_close", close) == "終値−2σ割れを引けで買い（押し目）：深く下げた順に5枠"
    d = run_details("daily", "bb_close", close)
    assert d["手法"] == "終値が−2σ以下で引けた日（最初の割れだけ）に引けの成行で買う"
    assert d["絞り込み"] == "5日の騰落≦-3%" and d["建てる順"] == "深く下げた順（終値のσが低い順）"

    index_c = {**idx, "signal": "C", "min_stocks": 10, "max_hold": 10, "entry": "next_open"}
    small = {"window": 25, "k": 2.0, "exit_k": 1.0, "capital": 2_000_000, "slots": 3, "lot": "実際の100株",
             "parts": [{"strategy": "bb_close", **close}, {"strategy": "index_overlay", **index_c}]}
    assert run_name("daily", "small_account", small) == "−2σ割れの翌朝買い＋日経平均の買い増し：200万円・3枠"
    d = run_details("daily", "small_account", small)
    assert d["手法2"] == "−3σに触れた貸借銘柄（売買代金1億円以上）が10銘柄以上の日の翌日の寄りで買い増す"
    assert d["資金"].startswith("200万円をいつも日経平均で持ち、個別株は3枠に等分して乗り換える（実際の100株単位）")
    half = {**small, "index_weight": 0.5}
    assert run_name("daily", "small_account", half) == "−2σ割れの翌朝買い＋日経平均の買い増し：200万円・3枠・日経平均は50%"
    assert run_details("daily", "small_account", half)["資金"].startswith("200万円のうち個別株に入っていない資金の50%を日経平均、残りを現金で持ち")
    index_b = {k: v for k, v in index_c.items() if k != "min_stocks"} | {"signal": "B", "k": 2.5}
    sig_b = {**half, "parts": [small["parts"][0], {"strategy": "index_overlay", **index_b}]}
    assert run_name("daily", "small_account", sig_b) == \
        "−2σ割れの翌朝買い＋日経平均の買い増し：200万円・3枠・日経平均は50%・買い増しは日経平均−2.5σ"
    assert run_details("daily", "small_account", sig_b)["手法2"] == "日経平均の終値が−2.5σを割ったら翌日の寄りで買い増す"
