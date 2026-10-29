from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """売買戦略。

    generate_signals は各営業日の「終値時点で判断した目標ポジション」を返す
    （1 = ロング保有、-1 = 空売り、0 = ノーポジション）。エンジンはこれを翌営業日の寄付で執行する。
    target 列と reason 列（手仕舞いの理由。手仕舞いを決めた日に入れる）を持つ DataFrame を返してもよい。
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        ...
