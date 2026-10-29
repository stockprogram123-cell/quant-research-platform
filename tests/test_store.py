import numpy as np
import pandas as pd
import pytest

from trading.backtest import BacktestConfig, round_trip_table, run_backtest, summarize
from trading.store import ResultStore
from trading.strategy import Strategy


class FixedSignals(Strategy):
    def __init__(self, signals):
        self.signals = signals

    def generate_signals(self, df):
        return pd.Series(self.signals, index=df.index)


def make_result(signals=(1, 1, 0, 1, 0, 0)):
    idx = pd.bdate_range("2024-01-01", periods=6)
    prices = [100.0, 110, 120, 100, 90, 95]
    df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1}, index=idx)
    result = run_backtest(df, FixedSignals(list(signals)), BacktestConfig(initial_cash=100_000, lot_size=100))
    rts = round_trip_table(result, "7203")
    return result, rts


def test_round_trip_table_and_summary():
    result, rts = make_result()
    assert rts["entry_price"].tolist() == [110, 90]
    assert rts["exit_price"].tolist() == [100, 95]
    assert rts["ret"].tolist() == pytest.approx([100 / 110 - 1, 95 / 90 - 1])
    s = summarize(rts, result.equity)
    assert s["num_trades"] == 2
    assert s["win_rate"] == 0.5
    assert s["total_pnl"] == pytest.approx(rts["pnl"].sum())


def test_round_trip_table_ignores_open_position():
    _, rts = make_result(signals=(1, 1, 0, 1, 1, 1))  # 最後の買いは期間末まで決済されない
    assert rts["entry_price"].tolist() == [110]


def test_save_and_load(tmp_path):
    result, rts = make_result()
    store = ResultStore(tmp_path / "t.db")
    periods = {"ALL": (summarize(rts, result.equity), rts, result.equity)}
    run_id = store.save_run("daily", "test", {"a": 1}, {"x": np.int64(3)}, periods, note="memo", name="テスト",
                            details={"手法": "テストの手法", "資金": "1,000万円を5枠に等分"})

    runs = store.runs()
    assert runs["run_id"].tolist() == [run_id]
    assert runs.loc[0, "num_trades"] == 2
    assert store.run(run_id)["params"] == {"a": 1}
    assert list(store.run(run_id)["details"].items()) == [("手法", "テストの手法"), ("資金", "1,000万円を5枠に等分")]
    trades = store.trades(run_id)
    assert trades["code"].tolist() == ["7203", "7203"]
    assert trades["pnl"].sum() == pytest.approx(rts["pnl"].sum())
    assert len(store.equity(run_id)) == len(result.equity)

    store.update_note(run_id, "changed")
    assert store.run(run_id)["note"] == "changed"
    assert store.runs().loc[0, "name"] == "テスト"
    store.update_name(run_id, "別の名前")
    assert store.run(run_id)["name"] == "別の名前" and store.run(run_id)["details"]["手法"] == "テストの手法"
    store.update_name(run_id, "別の名前", {"手法": "変えた手法"})
    assert store.run(run_id)["details"] == {"手法": "変えた手法"}
    # ペーパートレード（PT）は同じ実行の期間として毎日入れ替える（ほかの期間はそのまま）
    one = rts.iloc[:1]
    store.replace_periods(run_id, {"PT": (summarize(one, result.equity), one, result.equity)})
    store.replace_periods(run_id, {"PT": (summarize(rts, result.equity), rts, result.equity)})
    trades = store.trades(run_id)
    assert (trades["period"] == "PT").sum() == 2 and (trades["period"] == "ALL").sum() == 2
    assert set(store.runs()["period"]) == {"ALL", "PT"}
    store.update_config(run_id, {**store.run(run_id)["config"], "paper": {"start": "2023-03-31"}})
    assert store.run(run_id)["config"] == {"x": 3, "paper": {"start": "2023-03-31"}}
    store.delete_run(run_id)
    assert store.runs().empty
    assert store.trades(run_id).empty


def test_empty_period_is_saved(tmp_path):
    store = ResultStore(tmp_path / "t.db")
    empty = pd.DataFrame()
    run_id = store.save_run("intraday", "opening_gap", {}, {}, {"OOS": (summarize(empty, pd.Series(dtype=float)),
                                                               empty, pd.Series(dtype=float))})
    row = store.runs().iloc[0]
    assert row["run_id"] == run_id and row["num_trades"] == 0 and pd.isna(row["win_rate"])
