from pathlib import Path

import numpy as np
import pandas as pd

from .base import COLUMNS

SUFFIXES = ("_5min.csv", "_daily.csv")   # TradingView から書き出した 5 分足 / 日足（日足は以前 _5min.csv の名前だった）


def _csv_path(code: str, data_dir: str | Path) -> Path:
    for suffix in SUFFIXES:
        if (path := Path(data_dir) / f"{code}{suffix}").exists():
            return path
    raise FileNotFoundError(f"{code} の CSV が {data_dir} にありません（{' / '.join(SUFFIXES)}）")


def list_codes(data_dir: str | Path) -> list[str]:
    """data_dir 内の {code}_5min.csv / {code}_daily.csv から証券コード一覧を返す（"130A" のような英字入りも含む）。"""
    return sorted({p.name[: -len(suffix)] for suffix in SUFFIXES for p in Path(data_dir).glob(f"*{suffix}")})


def load_5min(code: str, data_dir: str | Path) -> pd.DataFrame:
    """5 分足 CSV（TradingView 形式）を読み込む。

    返り値は DatetimeIndex（JST・tz なし・バーの開始時刻）と COLUMNS の列を持つ。
    株価は未調整。
    """
    df = pd.read_csv(_csv_path(code, data_dir), parse_dates=["datetime"])
    df = df.set_index("datetime")[COLUMNS]
    df = df[~df.index.duplicated(keep="last")].dropna().sort_index()
    df.index.name = "datetime"
    return df


def load_daily_csv(code: str, data_dir: str | Path) -> pd.DataFrame:
    """TradingView 形式の日足 CSV（data/daily の {code}_daily.csv。列は 5 分足と同じ）を読み込む。

    index は日付。売買が成立しなかった日（出来高 0、四本値が同じ）は除く。
    """
    df = load_5min(code, data_dir)
    df.index = df.index.normalize()
    df.index.name = "date"
    return df[df["volume"] > 0]


def daily_context(daily: pd.DataFrame, sma_windows: tuple[int, ...] = (5, 25, 75)) -> pd.DataFrame:
    """日足から当日の判断に使える指標を作る。

    すべての列は 1 日シフト済みで、index の日付の寄付時点で既知の値だけを含む
    （prev_close = 前日終値、sma25 = 前日までの 25 日移動平均）。
    """
    ctx = pd.DataFrame(index=daily.index)
    ctx["prev_close"] = daily["close"]
    for n in sma_windows:
        ctx[f"sma{n}"] = daily["close"].rolling(n).mean()
    return ctx.shift(1)


def attach_daily(bars: pd.DataFrame, ctx: pd.DataFrame | None) -> pd.DataFrame:
    """5 分足の各バーに、その日の daily_context の値を付ける。"""
    if ctx is None:
        return bars
    joined = ctx.reindex(bars.index.normalize())
    joined.index = bars.index
    return bars.join(joined)


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """日ごとにリセットする VWAP（典型価格 (H+L+C)/3 で計算）。"""
    day = bars.index.normalize()
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_pv = (typical * bars["volume"]).groupby(day).cumsum()
    cum_v = bars["volume"].groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def market_context(bars_by_code: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """多銘柄の 5 分足から、各時点の相場全体の状態を作る（ETF の代わり）。

    - mkt_gap:     各銘柄の 終値 ÷ 当日 VWAP − 1 の平均（プラスなら相場全体が VWAP より上）
    - mkt_breadth: 終値が当日 VWAP より上の銘柄の割合
    どちらも各バーの終値時点で分かる値だけを使う。
    """
    gaps = pd.DataFrame({code: b["close"] / session_vwap(b) - 1 for code, b in bars_by_code.items()})
    return pd.DataFrame({"mkt_gap": gaps.mean(axis=1), "mkt_breadth": (gaps > 0).sum(axis=1) / gaps.notna().sum(axis=1)})


def attach_regime(bars: pd.DataFrame, regime: pd.Series | None) -> pd.DataFrame:
    """5 分足の各バーに、その日の局面（regime_down: 1 = 下げ相場 / 0 = 上げ相場）を付ける。"""
    if regime is None:
        return bars
    return bars.assign(regime_down=bars.index.normalize().map(regime))


def attach_market(bars: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    """5 分足の各バーに、同じ時刻の market_context の値を付ける。"""
    if market is None:
        return bars
    return bars.join(market.reindex(bars.index))
