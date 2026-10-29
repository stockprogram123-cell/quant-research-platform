"""値幅制限（ストップ高・ストップ安）。"""
import numpy as np
import pandas as pd

from .costs import tick_size

# 基準値段（前日終値）がこの値段未満なら、この幅まで（東証の制限値幅。2026 年時点の表）
LIMIT_TABLE = [
    (100, 30), (200, 50), (500, 80), (700, 100), (1_000, 150), (1_500, 300), (2_000, 400), (3_000, 500),
    (5_000, 700), (7_000, 1_000), (10_000, 1_500), (15_000, 3_000), (20_000, 4_000), (30_000, 5_000),
    (50_000, 7_000), (70_000, 10_000), (100_000, 15_000), (150_000, 30_000), (200_000, 40_000),
    (300_000, 50_000), (500_000, 70_000), (700_000, 100_000), (1_000_000, 150_000), (1_500_000, 300_000),
    (2_000_000, 400_000), (3_000_000, 500_000), (5_000_000, 700_000), (7_000_000, 1_000_000),
    (10_000_000, 1_500_000), (15_000_000, 3_000_000), (20_000_000, 4_000_000), (30_000_000, 5_000_000),
    (50_000_000, 7_000_000),
]

LIMIT_COLUMNS = ["up_close", "down_close", "up_locked", "down_locked"]


def limit_width(price: float) -> float:
    for upper, width in LIMIT_TABLE:
        if price < upper:
            return width
    return 10_000_000


def limit_flags(df: pd.DataFrame, factor: pd.Series | None = None) -> pd.DataFrame:
    """日足（分割調整済みでもよい）から、その日がストップ高・ストップ安だったかの印を作る。

    factor は調整済みの値段を実際の値段に戻す係数（trading.data.jpx.unadjust_factor）。
    制限値幅は実際の前日終値で決まるので、分割があった銘柄はこれがないと値段がずれる。
    - up_close / down_close: 大引けがストップ高 / ストップ安（引けで反対側の注文が約定しきらない）
    - up_locked / down_locked: 一日中その値段だけ（寄らずか張り付き。反対側の注文は一日約定しない）
    値段は呼値の半分までの誤差を許す。制度の細かい例外（連続ストップ時の値幅拡大など）は扱わない。
    この表が実データと合うのは 2014 年ごろ以降（それより前は値幅が狭い時期がある）なので、
    一日中同じ値段で前日から 10% 以上動いた日も張り付きとみなす（古い時期の大引けだけのストップ高は拾えない）。
    """
    f = pd.Series(1.0, index=df.index) if factor is None else factor.reindex(df.index).fillna(1.0)
    prev = (df["close"] * f).shift(1)
    width = prev.map(lambda p: np.nan if pd.isna(p) else limit_width(p))
    tol = prev.map(lambda p: np.nan if pd.isna(p) else tick_size(p) / 2) + 0.01 * f
    up, down = prev + width, prev - width
    c, h, lo = df["close"] * f, df["high"] * f, df["low"] * f
    one_price = (h - lo) <= tol
    out = pd.DataFrame({"up_close": c >= up - tol, "down_close": c <= down + tol}, index=df.index)
    chg = c / prev - 1
    out["up_locked"] = one_price & (out["up_close"] | (chg >= 0.10))
    out["down_locked"] = one_price & (out["down_close"] | (chg <= -0.10))
    out["up_close"] |= out["up_locked"]
    out["down_close"] |= out["down_locked"]
    return out
