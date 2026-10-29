from abc import ABC, abstractmethod

import pandas as pd


class IntradayStrategy(ABC):
    """5 分足のデイトレード戦略。

    generate_signals は各バーの「終値時点で判断した」値を持つ DataFrame を返す:
      - target: 目標ポジション（1 = 買い、-1 = 空売り、0 = ノーポジ）
      - stop:   損切り価格（NaN なら損切りなし。空売りなら今の値より上）。エンジンはエントリー時点の値で固定する。
    エンジンはこれを同じ日の次のバーの始値で執行し、大引けで必ず手仕舞う。

    bars には daily_context の列（prev_close, sma25 など。前日までの値）や、
    market_context の列（mkt_gap など。同じ時刻の相場全体の状態）が付いていることがある。
    """

    required_sma: tuple[int, ...] = ()  # 必要な日足移動平均の期間（daily_context に渡す）
    requires_market: bool = False       # True なら bars に market_context の列（mkt_gap など）が必要
    requires_regime: bool = False       # True なら bars に attach_regime の regime_down 列（日足の局面）が必要

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        ...


def trading_day(bars: pd.DataFrame) -> pd.Index:
    return bars.index.normalize()


def trend_filter(bars: pd.DataFrame, sma: int) -> pd.Series:
    """前日終値が前日までの SMA を上回っているか。日足の列がなければエラー。"""
    col = f"sma{sma}"
    if col not in bars or "prev_close" not in bars:
        raise ValueError(f"日足フィルタには daily_context の {col} / prev_close 列が必要です")
    return bars["prev_close"] > bars[col]
