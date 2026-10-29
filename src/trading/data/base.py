from abc import ABC, abstractmethod

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataSource(ABC):
    """日足データの取得元。

    すべての実装は DatetimeIndex（tz なし、昇順）と COLUMNS の列を持つ
    調整後株価の DataFrame を返す。
    """

    name: str

    @abstractmethod
    def get_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """証券コード code の日足を start〜end（両端含む）で返す。"""


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """取得結果を共通フォーマットに揃える。"""
    df = df.rename(columns=str.lower)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {missing}")
    df = df[COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "date"
    return df.dropna().sort_index()
