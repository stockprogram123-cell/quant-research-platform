"""日経平均を寄りで実際に売買できる値段の目安（ETF の寄りから求める）。"""
import pandas as pd


def tradable_open(index_close: pd.Series, etf: pd.DataFrame, tol: float = 0.01) -> pd.Series:
    """指数の始値は 9:00 にまだ寄っていない銘柄を前日の終値のまま計算するので、大きく窓を開ける日ほど実際に売買できる値段から離れる。
    そこで ETF（調整しない値段の open / close / volume）の寄りから、指数の単位に直した寄りの値段を出す。

    1. 前日と今日に ETF が売買され、前日の ETF と指数の騰落のずれが tol 未満（または前日が配当落ち）で、今日が配当落ちでなければ、
       前日の指数の終値 × ETF の寄りの窓（ETF の寄り ÷ ETF の前日の終値）。
    2. そうでなくても今日 ETF が売買されていれば、今日の指数の終値 × ETF の寄り ÷ ETF の今日の終値（配当落ちの日は寄りも引けも
       配当の分だけ下がるので、その比は配当に左右されない。前日の ETF の終値が指数からずれていた日もこちら）。
    3. ETF の値がない日は今日の大引けで売買したとみなす。
    配当落ちの日: 7 月で、ETF の騰落が指数より tol 以上低い日（1330 は 7 月に配当落ち。Yahoo の配当の調整は日付がずれているので使わない）。
    以前は「今日の ETF と指数の騰落が tol 以上ずれる日」を 3. にしていたが、2024-08-05 のような急落の日に安値の大引けで
    売買したことになるので改めた。"""
    etf = etf.reindex(index_close.index)
    traded = (etf["volume"] > 0) & (etf["open"] > 0) & (etf["close"] > 0)
    dev = etf["close"].pct_change(fill_method=None) - index_close.pct_change()
    ex_div = pd.Series(index_close.index.month == 7, index=index_close.index) & (dev <= -tol)
    prev_ok = traded.shift(1, fill_value=False) & ((dev.shift(1).abs() < tol) | ex_div.shift(1, fill_value=False))
    by_gap = index_close.shift(1) * etf["open"] / etf["close"].shift(1)
    by_day = index_close * etf["open"] / etf["close"]
    out = by_gap.where(traded & prev_ok & ~ex_div)
    return out.fillna(by_day.where(traded)).fillna(index_close)
