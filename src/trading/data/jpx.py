"""JPX の公表資料（貸借銘柄の一覧）と、日足を分割前の値段に戻すための分割履歴。"""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

MARGIN_PAGE = "https://www.jpx.co.jp/listing/others/margin/index.html"
_UA = {"User-Agent": "Mozilla/5.0"}


def margin_list(cache_dir: str | Path = "data/cache/jpx", refresh: bool = False) -> pd.DataFrame:
    """JPX の「制度信用銘柄・貸借銘柄一覧」（最新の 1 回分）。列は code / name / market / margin。

    公表されているのは最新時点の区分だけなので、過去の検証に使うと「今の貸借銘柄」で絞ることになる
    （途中で貸借銘柄になった・外れた銘柄の扱いは正確でない）。一度取得したらキャッシュを使う。
    """
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("margin_*.xlsx"))
    if refresh or not files:
        html = requests.get(MARGIN_PAGE, headers=_UA, timeout=30).text
        href = re.search(r'href="([^"]+_list\.xlsx)"', html)
        if href is None:
            raise RuntimeError(f"{MARGIN_PAGE} に一覧の Excel が見つかりません")
        url = "https://www.jpx.co.jp" + href.group(1)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"margin_{Path(url).name.split('_')[0]}.xlsx"
        path.write_bytes(requests.get(url, headers=_UA, timeout=60).content)
        files = [path]
    raw = pd.read_excel(files[-1], header=None, dtype=str)
    start = raw.index[raw[0] == "銘柄コード"][0] + 1
    out = raw.iloc[start:, :4].set_axis(["code", "name", "market", "margin"], axis=1).dropna(subset=["code"])
    out["code"] = out["code"].str.strip()
    out.attrs["as_of"] = str(raw.iloc[0, 0])
    return out.reset_index(drop=True)


def margin_codes(cache_dir: str | Path = "data/cache/jpx") -> set[str]:
    """貸借銘柄（制度信用で空売りできる銘柄）のコード。"""
    m = margin_list(cache_dir)
    return set(m.loc[m["margin"] == "貸借銘柄", "code"])


def _fetch_splits(code: str) -> pd.Series:
    import yfinance as yf
    s = yf.Ticker(f"{code}.T").splits
    if len(s):
        s.index = s.index.tz_localize(None).normalize()
    return s.astype(float)


def split_history(codes: list[str], cache_dir: str | Path = "data/cache/splits", workers: int = 8) -> dict[str, pd.Series]:
    """銘柄ごとの株式分割（権利落ち日 → 分割比率。併合は 1 未満）。yfinance から取って銘柄ごとにキャッシュする。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in codes if not (cache_dir / f"{c}.csv").exists()]

    def fetch(code: str) -> None:
        try:
            s = _fetch_splits(code)
        except Exception:  # 取れなかった銘柄は次回また取りにいく
            return
        s.rename("ratio").rename_axis("date").to_csv(cache_dir / f"{code}.csv")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, missing))
    out = {}
    for c in codes:
        path = cache_dir / f"{c}.csv"
        if path.exists():
            out[c] = pd.read_csv(path, index_col=0, parse_dates=True)["ratio"]
    return out


def unadjust_factor(index: pd.DatetimeIndex, splits: pd.Series) -> pd.Series:
    """分割調整済みの値段に掛けると、その日の実際の値段に戻る係数（その日より後の分割比率の積）。"""
    f = pd.Series(1.0, index=index)
    for date, ratio in splits.items():
        if ratio > 0:
            f[index < date] *= ratio
    return f
