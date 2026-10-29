from pathlib import Path

import pandas as pd

from .base import DataSource


class CachedDataSource(DataSource):
    """取得結果をローカルの CSV にキャッシュするラッパー。"""

    def __init__(self, source: DataSource, cache_dir: str | Path = "data/cache"):
        self.source = source
        self.name = source.name
        self.cache_dir = Path(cache_dir)

    def get_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        path = self.cache_dir / f"{self.name}_{code}_{start}_{end}.csv"
        if path.exists():
            return pd.read_csv(path, index_col="date", parse_dates=True)
        df = self.source.get_daily(code, start, end)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        return df
