"""市場全体の信用取引残高（週次。2 市場＝東京・名古屋、制度信用と一般信用の合計）と信用評価損益率。

出典はトレーダーズ・ウェブの「信用残の推移」（2001 年 6 月から）。売り残・買い残は東京証券取引所の公表値、
評価損益率は日本経済新聞社が算出した値。JPX の過去推移表には評価損益率がないのでこちらを使う。
申込日（その週の最後の営業日）の残高が翌週の第 2 営業日の大引け後に公表されるので、売買の判断に使えるのは
申込日から 3 営業日目（ふつうは翌週の水曜）の判断からにする（available_from）。
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.traders.co.jp/margin_derivatives/margin_transition"
_UA = {"User-Agent": "Mozilla/5.0"}
LAG_DAYS = 3   # 申込日から何営業日目の判断から使えるか


def _page(n: int) -> pd.DataFrame:
    html = requests.get(URL if n == 1 else f"{URL}/{n}", headers=_UA, timeout=30).text
    tables = [t for t in pd.read_html(io.StringIO(html)) if t.shape[1] == 11]
    if not tables:
        return pd.DataFrame()
    t = tables[0]
    t.columns = ["date", "sell_shares", "sell_shares_chg", "sell_value", "sell_value_chg", "buy_shares", "buy_shares_chg",
                 "buy_value", "buy_value_chg", "pl_ratio", "credit_ratio"]
    return t


def market_margin(cache_dir: str | Path = "data/cache", refresh: bool = False, pause: float = 2.0) -> pd.DataFrame:
    """週次の信用残。index は申込日。列は売り残・買い残（株数は百万株、金額は億円）、評価損益率（%）、信用倍率。
    一度取得したらキャッシュ（data/cache/margin_weekly.csv）を使う。refresh=True で全ページを取り直す。"""
    path = Path(cache_dir) / "margin_weekly.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    pages, seen = [], set()
    for n in range(1, 100):   # 1 ページ 100 週。範囲の外のページは空か、前と同じ週が返る
        t = _page(n)
        t["date"] = pd.to_datetime(t["date"], format="%Y/%m/%d", errors="coerce") if not t.empty else t.get("date")
        t = t.dropna(subset=["date"]) if not t.empty else t
        new = set(t["date"]) - seen if not t.empty else set()
        if not new:
            break
        seen |= new
        pages.append(t)
        time.sleep(pause)
    df = pd.concat(pages, ignore_index=True)
    cols = ["sell_shares", "sell_value", "buy_shares", "buy_value", "pl_ratio", "credit_ratio"]
    df = df.set_index("date")[cols].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated()].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def available_from(weekly: pd.DataFrame, calendar: pd.DatetimeIndex, lag: int = LAG_DAYS) -> pd.DataFrame:
    """calendar の各営業日の大引けで判断するときに使える、最新の週の値（申込日から lag 営業日目以降）。列 week に申込日。"""
    pos = calendar.searchsorted(weekly.index)          # 申込日（休みなら次の営業日）の位置
    ok = pos + lag < len(calendar)
    avail = pd.DataFrame(weekly[ok].to_numpy(), columns=weekly.columns, index=calendar[pos[ok] + lag]).assign(week=weekly.index[ok])
    avail = avail[~avail.index.duplicated(keep="last")]
    return avail.reindex(calendar).ffill()
