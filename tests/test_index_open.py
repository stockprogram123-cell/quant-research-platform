import numpy as np
import pandas as pd

from trading.data.index_open import tradable_open


def make(index_close, etf_open, etf_close, start="2024-06-03", volume=None):
    idx = pd.bdate_range(start, periods=len(index_close))
    etf = pd.DataFrame({"open": etf_open, "close": etf_close,
                        "volume": volume if volume is not None else [100.0] * len(index_close)}, index=idx)
    return pd.Series(index_close, index=idx, dtype=float), etf


def test_normal_day_uses_etf_gap_from_previous_close():
    c, etf = make([100.0, 98.0], [103.0, 99.91], [103.0, 100.94])   # ETF は指数の 1.03 倍で、今日は寄りで -3%
    assert np.isclose(tradable_open(c, etf).iloc[1], 100.0 * 99.91 / 103.0)


def test_crash_day_with_etf_close_off_from_index_still_uses_gap():
    # 前日（1 日目）は ETF と指数がそろっている。今日は ETF の引けが指数より 2.6% 高いが、寄りの窓は使える（以前は大引けにしていた）
    c, etf = make([35910.0, 35910.0, 31458.0], [37040.0, 37040.0, 35260.0], [37040.0, 37040.0, 33400.0])
    got = tradable_open(c, etf).iloc[2]
    assert np.isclose(got, 35910.0 * 35260.0 / 37040.0)
    assert got > 34000.0


def test_ex_dividend_day_in_july_uses_same_day_ratio():
    # 7 月に ETF だけ配当（2）の分下がって寄り・引ける。指数は変わらない
    c, etf = make([100.0, 100.0, 100.0], [102.0, 102.0, 100.0], [102.0, 102.0, 100.0], start="2024-07-03")
    assert np.isclose(tradable_open(c, etf).iloc[2], 100.0)   # 配当の分を寄りの下げとみなさない


def test_day_after_off_close_uses_same_day_ratio():
    # 2 日目に ETF の引けが指数からずれた（ETF -9.8%・指数 -12.4%）。3 日目は前日の ETF の終値を使わない
    c, etf = make([35910.0, 31458.0, 34675.0], [37040.0, 35260.0, 34750.0], [37040.0, 33400.0, 35110.0])
    assert np.isclose(tradable_open(c, etf).iloc[2], 34675.0 * 34750.0 / 35110.0)


def test_no_etf_trade_falls_back_to_close():
    c, etf = make([100.0, 97.0], [103.0, 0.0], [103.0, 100.0], volume=[100.0, 0.0])
    assert tradable_open(c, etf).iloc[1] == 97.0
