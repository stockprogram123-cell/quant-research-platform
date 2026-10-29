import pandas as pd

from .base import DataSource


class JQuantsSource(DataSource):
    """J-Quants API（JPX 公式）から取得する。認証情報は .env で管理する。"""

    name = "jquants"

    def get_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("J-Quants データソースは未実装です（ステップ5で対応予定）")
