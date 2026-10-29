from .base import COLUMNS, DataSource, normalize
from .cache import CachedDataSource
from .intraday import (attach_daily, attach_market, attach_regime, daily_context, list_codes, load_5min,
                       load_daily_csv, market_context, session_vwap)
from .jquants_source import JQuantsSource
from .yfinance_source import YFinanceRawSource, YFinanceSource

SOURCES: dict[str, type[DataSource]] = {
    YFinanceSource.name: YFinanceSource,
    YFinanceRawSource.name: YFinanceRawSource,
    JQuantsSource.name: JQuantsSource,
}


def get_source(name: str, cache_dir: str | None = "data/cache") -> DataSource:
    """名前からデータソースを作る。cache_dir=None ならキャッシュしない。"""
    if name not in SOURCES:
        raise ValueError(f"未知のデータソース: {name}（選択肢: {list(SOURCES)}）")
    source = SOURCES[name]()
    return CachedDataSource(source, cache_dir) if cache_dir else source
