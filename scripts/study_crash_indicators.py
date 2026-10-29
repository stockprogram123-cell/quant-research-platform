"""暴落の予兆として一般に使われる指標が、日経平均の暴落をどの程度前もって当てられるか、それを合図に空売りしていたら
資金がどうなっていたかを確かめる。

    python scripts/study_crash_indicators.py

暴落: 判断した日の翌営業日の大引けから 20 営業日以内に、終値が 10% 以上下がること（--crash-drop・--crash-days）。
  暴落の前の日が 20 営業日より近くに続くものは 1 回と数え、最初の日をその暴落の「始まり」とする。
判断と売買: その日の大引けまでに分かる値で判断し、翌営業日の大引けで持つ量を変える。米国の市場の値（yfinance）は前の日付の終値まで、
  FRED の値（日付の翌日以降に公表）はさらに 2 営業日遅らせ、サームルール（月次）は翌月 10 日から使う。
指標（しきい値は一般に言われる値で、この検証の結果を見て決めていない）:
  過熱: 騰落レシオ（25 日）120 以上 / 25 日線からの乖離 +5% 以上 / RSI（14 日）70 以上
  トレンド: 日経平均が 200 日線より下 / S&P 500 が 200 日線より下
  恐怖: VIX 20 以上 / VIX 12 以下（油断）/ VIX が 3 か月物の VIX3M 以上（期間構造の逆転。2006-07 から）
  景気・信用: 米国の逆イールド（10 年 − 3 か月 < 0）/ 逆イールドが終わってから 250 営業日 / Baa 社債と 10 年国債の利回りの差が
    6 か月平均より上（ハイイールド債のスプレッドは FRED に直近 3 年分しかないので代わりに使う）/ サームルール 0.5 以上（公表時点の値）
  為替: ドル円が 20 営業日で 5% 以上の円高
  市場の内部: ヒンデンブルグ・オーメン（新高値・新安値がともに値のついた銘柄の 2.2% 以上、日経平均が 50 営業日前より上、
    マクレラン・オシレーターが負、新高値が新安値の 2 倍以下。出た日から 30 営業日を警戒とする）
  騰落レシオと新高値・新安値は手元の日足の全銘柄から数える（上場廃止した銘柄が入っていないので、下げの日の数が少なめに出る）。
  手元の日足は 2003 年ごろまで銘柄が少ないので、値のついた銘柄が 1,000 以上の日が続いてから使う（騰落レシオは 2003 年、新高値・新安値は 2004 年から）。
  予想の力は指標のデータがある日だけで数え、運用ではデータがない間は警戒なしとして扱う。
予想の力: 警戒中の日のうち暴落の前の日の割合（的中率）とふだんの割合に対する倍率、暴落の前の日のうち警戒中だった割合（捕捉率）、
  警戒中の日の 20 営業日先の平均の騰落（空売りの損益の向きはこれで決まる）、
  暴落の始まりの日に警戒中だった回数、警戒の 1 回（5 営業日以内の途切れはつなぐ）のうち、その間に暴落が始まった割合、
  指標の値そのものの AUC（大きいほど危ないとみなす向きで。0.5 は当てずっぽう）。
運用（100 万円から）: A 警戒中だけ日経平均を空売り（ふだんは現金）/ B ふだんは日経平均を持ち、警戒中は空売り /
  C ふだんは持ち、警戒中は持たない / D 今の口座（DB #81）に、警戒中だけ資産の半分の空売りを足す。
  費用は study_index_margin_timing.run と同じ（持つ量の変化 1 単位あたり片道 0.05%・空売りは貸株料 年 1.1%・配当 年 1.8% を
  持つ量に比例して受け取り、空売りなら支払う。比べる日経平均の持ちっぱなしにも配当を入れる）。
結果は results/crash_indicators_*.csv。
"""
import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance

from simulate_portfolio import stats
from study_index_crash import IS_END, IS_START
from trading.data import get_source, list_codes, load_daily_csv
from trading.store import ResultStore

DATA_DIR = "data/daily"
CACHE = Path("data/cache")


def fred(series: str, refresh: bool) -> pd.Series:
    """FRED の系列（日付 → 値）。data/cache/fred に置き、なければ取ってくる。"""
    path = CACHE / "fred" / f"{series}.csv"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        path.write_bytes(subprocess.run(["curl", "-s", "-m", "120", url], check=True, capture_output=True).stdout)
    f = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(f, errors="coerce").dropna()


def usdjpy() -> pd.Series:
    d = yfinance.download("JPY=X", start="1996-01-01", progress=False, auto_adjust=False)
    return d["Close"].iloc[:, 0].dropna()


def known_on(s: pd.Series, cal: pd.DatetimeIndex, extra_days: int = 0) -> pd.Series:
    """日付 d の値を、d より後の最初の東京の営業日から使えるとみなし（さらに extra_days 営業日遅らせ）、cal に並べる。"""
    pos = cal.searchsorted(s.index, side="right") + extra_days
    ok = pos < len(cal)
    v = pd.Series(s.to_numpy()[ok], index=cal[pos[ok]])
    return v[~v.index.duplicated(keep="last")].reindex(cal).ffill()


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def breadth(cal: pd.DatetimeIndex, refresh: bool) -> pd.DataFrame:
    """手元の日足の全銘柄から、値上がり・値下がり・1 年の新高値・新安値の数（data/cache/breadth_daily.csv に置く）。"""
    path = CACHE / "breadth_daily.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).reindex(cal)
    cols = {}
    for code in list_codes(DATA_DIR):
        cols[code] = load_daily_csv(code, DATA_DIR)["close"].loc["1999-01-01":].astype("float32")
    c = pd.DataFrame(cols).sort_index()
    prev = c.shift(1)
    hi, lo = c.rolling(250, min_periods=250).max().shift(1), c.rolling(250, min_periods=250).min().shift(1)
    out = pd.DataFrame({"adv": (c > prev).sum(axis=1), "dec": (c < prev).sum(axis=1),
                        "issues": (c.notna() & prev.notna()).sum(axis=1),
                        "new_high": (c > hi).sum(axis=1), "new_low": (c < lo).sum(axis=1)})
    out.to_csv(path)
    return out.reindex(cal)


def spells(on: pd.Series, gap: int = 5) -> list[tuple[int, int]]:
    """True の続く区間（gap 営業日以内の途切れはつなぐ）の、始まりと終わりの位置。"""
    idx = np.flatnonzero(on.to_numpy())
    out = []
    for i in idx:
        if out and i - out[-1][1] <= gap + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [tuple(x) for x in out]


def auc(score: pd.Series, y: pd.Series) -> float:
    m = score.notna() & y.notna()
    s, t = score[m], y[m].astype(bool)
    n1, n0 = int(t.sum()), int((~t).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = s.rank()
    return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def indicators(nk: pd.Series, cal: pd.DatetimeIndex, br: pd.DataFrame, refresh: bool) -> dict:
    """名前 → (分類, 警戒（bool）, 指標の値（大きいほど危ない向き。なければ None）, 使える最初の日)。"""
    yf = get_source("yfinance")
    us = {s: yf.get_daily(s, "1985-01-01", cal[-1].strftime("%Y-%m-%d"))["close"] for s in ("^VIX", "^VIX3M", "^GSPC")}
    vix, vix3m = known_on(us["^VIX"], cal), known_on(us["^VIX3M"], cal)
    spx = us["^GSPC"]
    spx_gap = known_on(spx / spx.rolling(200).mean() - 1, cal)
    curve = known_on(fred("T10Y3M", refresh), cal, 2)
    baa = fred("BAA10Y", refresh)
    baa_gap = known_on(baa - baa.rolling(126).mean(), cal, 2)
    sahm = fred("SAHMREALTIME", refresh)
    sahm.index = sahm.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)   # 翌月 10 日から使う
    sahm = known_on(sahm, cal)
    fx = usdjpy()
    fx20 = known_on(fx / fx.shift(20) - 1, cal)

    full = (br["issues"] >= 1000).astype(float)   # 手元の日足は 2003 年ごろから銘柄がそろう。そろう前は「データなし」
    ok25, ok250 = full.rolling(25).min() == 1, full.rolling(250).min() == 1
    ad25 = (br["adv"].rolling(25).sum() / br["dec"].rolling(25).sum() * 100).where(ok25)
    gap25 = nk / nk.rolling(25).mean() - 1
    gap200 = nk / nk.rolling(200).mean() - 1
    r14 = rsi(nk)
    rana = (br["adv"] - br["dec"]) / (br["adv"] + br["dec"]) * 1000
    mcc = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    nh, nl = (br["new_high"] / br["issues"]).where(ok250), (br["new_low"] / br["issues"]).where(ok250)
    omen_day = (nh >= 0.022) & (nl >= 0.022) & (nk > nk.shift(50)) & (mcc < 0) & (br["new_high"] <= 2 * br["new_low"])
    omen = omen_day.astype(int).rolling(30, min_periods=1).max().astype(bool)
    inv = curve < 0
    after_inv = (inv.astype(int).rolling(250, min_periods=1).max().astype(bool)) & ~inv

    first = lambda s: s.first_valid_index()
    return {
        "騰落レシオ 120 以上": ("過熱", ad25 >= 120, ad25, first(ad25)),
        "25 日線から +5% 以上": ("過熱", gap25 >= 0.05, gap25, first(gap25)),
        "RSI 70 以上": ("過熱", r14 >= 70, r14, first(r14)),
        "日経平均が 200 日線の下": ("トレンド", gap200 < 0, -gap200, first(gap200)),
        "S&P 500 が 200 日線の下": ("トレンド", spx_gap < 0, -spx_gap, first(spx_gap)),
        "VIX 20 以上": ("恐怖", vix >= 20, vix, first(vix)),
        "VIX 12 以下（油断）": ("恐怖", vix <= 12, -vix, first(vix)),
        "VIX が VIX3M 以上": ("恐怖", vix >= vix3m, vix / vix3m, first(vix3m)),
        "逆イールド": ("景気・信用", inv, -curve, first(curve)),
        "逆イールドの終わりから 1 年": ("景気・信用", after_inv, None, first(curve)),
        "社債の利回り差が 6 か月平均より上": ("景気・信用", baa_gap > 0, baa_gap, first(baa_gap)),
        "サームルール 0.5 以上": ("景気・信用", sahm >= 0.5, sahm, first(sahm)),
        "20 日で 5% 以上の円高": ("為替", fx20 <= -0.05, -fx20, first(fx20)),
        "ヒンデンブルグ・オーメン（30 日）": ("市場の内部", omen, None, first(nh)),
    }


def simulate(c: pd.Series, ex: pd.Series, capital: float, args) -> pd.Series:
    """ex（その日の大引けで持つ量）での資産。費用は study_index_margin_timing.run と同じ。"""
    r = c.pct_change().fillna(0)
    held = ex.shift(1).fillna(0.0)
    ret = held * (r + args.div_yield / 245) - np.maximum(-held, 0) * args.short_fee / 245 \
        - ex.diff().fillna(ex.iloc[0]).abs() * args.switch_cost
    return capital * (1 + ret).cumprod()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash-drop", type=float, default=0.10)
    parser.add_argument("--crash-days", type=int, default=20)
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--switch-cost", type=float, default=0.0005)
    parser.add_argument("--short-fee", type=float, default=0.011)
    parser.add_argument("--div-yield", type=float, default=0.018)
    parser.add_argument("--account-run", type=int, default=81, help="D で空売りを足す口座（DB の実行）")
    parser.add_argument("--refresh", action="store_true", help="FRED と手元の全銘柄の集計を取り直す")
    args = parser.parse_args()
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    out = Path("results")
    out.mkdir(exist_ok=True)

    nk = get_source("yfinance").get_daily("^N225", "1985-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))["close"]
    cal = nk.index
    br = breadth(cal, args.refresh)
    ind = indicators(nk, cal, br, args.refresh)

    # 暴落の前の日: 翌営業日の大引け c[t+1] から、その後 crash_days 営業日の終値の最小が (1 - drop) 倍以下
    cv, n, h = nk.to_numpy(float), len(nk), args.crash_days
    fwd = np.full(n, np.nan)
    for t in range(n - h - 1):
        fwd[t] = cv[t + 2:t + 2 + h].min() / cv[t + 1] - 1
    fwd = pd.Series(fwd, index=cal)
    ret20 = pd.Series(np.r_[cv[1 + h:] / cv[1:-h] - 1, [np.nan] * (h + 1)], index=cal)   # 翌営業日の大引けから h 営業日後まで
    y = (fwd <= -args.crash_drop).where(fwd.notna())
    periods = {"IS": (IS_START, IS_END), "OOS": (IS_END + pd.Timedelta(days=1), cal[-1])}
    print(f"日経平均 {cal[0].date()}〜{cal[-1].date()}。暴落 = 翌営業日の大引けから {h} 営業日以内に {args.crash_drop:.0%} 以上の下げ")
    print(f"手元の日足から数えた銘柄数: " + " / ".join(f"{yr} 年 {int(br['issues'].loc[str(yr)].median())}" for yr in (2001, 2008, 2016, 2025)))

    # 暴落の回数（始まりの日）
    episodes = []
    for per, (s, e) in periods.items():
        yy = y[s:e]
        hit = np.flatnonzero(yy.fillna(False).to_numpy(bool))
        last = -10 ** 9
        for i in hit:
            if i - last > h:
                d = yy.index[i]
                j = cal.get_loc(d)
                episodes.append({"期間": per, "始まり": d, "下げ始めの日": cal[j + 1],
                                 "下げ": fwd[d:yy.index[min(i + h, len(yy) - 1)]].min()})
            last = i
    ep = pd.DataFrame(episodes)
    print("\n暴落（始まりの日 = その翌営業日の大引けから下げた）")
    for per in periods:
        e_ = ep[ep["期間"] == per]
        print(f"  {per} {len(e_)} 回: " + "、".join(f"{d.date()}（{v:.0%}）" for d, v in zip(e_["下げ始めの日"], e_["下げ"])))

    # 1. 予想の力
    rows, ep_flags = [], {}
    for name, (cat, alert, score, first) in ind.items():
        alert = alert.fillna(False).astype(bool)
        for per, (s, e) in periods.items():
            m = y[s:e].notna() & (y[s:e].index >= (first or s))   # 指標のデータがある日だけ
            a, yy = alert[s:e][m], y[s:e][m].astype(bool)
            e_ = ep[(ep["期間"] == per) & (ep["始まり"] >= (first or s))]
            at_start = sum(bool(alert.get(d, False)) for d in e_["始まり"])
            before = sum(bool(alert.iloc[max(cal.get_loc(d) - h, 0):cal.get_loc(d) + 1].any()) for d in e_["始まり"])
            sp = spells(a)
            sp_hit = sum(bool(yy.iloc[i:j + 1].any()) for i, j in sp)
            base = yy.mean()
            prec = yy[a].mean() if a.any() else np.nan
            rows.append({"指標": name, "分類": cat, "期間": per, "データの始まり": first.date() if first is not None else None,
                         "警戒の日の割合": a.mean(), "的中率": prec, "ふだん": base, "倍率": prec / base if base else np.nan,
                         "捕捉率": a[yy].mean() if yy.any() else np.nan,
                         "警戒中の20日先": ret20[s:e][m][a].mean() if a.any() else np.nan, "ふだんの20日先": ret20[s:e][m].mean(),
                         "始まりに警戒中": f"{at_start}/{len(e_)}", "20日前までに警戒": f"{before}/{len(e_)}",
                         "警戒の回数": len(sp), "その間に暴落": sp_hit,
                         "AUC": auc(score[s:e][m], yy) if score is not None else np.nan})
            for d in e_["始まり"]:
                if bool(alert.get(d, False)):
                    ep_flags.setdefault(d, []).append(name)
    ev = pd.DataFrame(rows)
    ev.to_csv(out / "crash_indicators_eval.csv", index=False)
    ep["始まりに警戒中だった指標"] = ep["始まり"].map(lambda d: "、".join(ep_flags.get(d, [])))
    ep.to_csv(out / "crash_indicators_episodes.csv", index=False)
    show = ["警戒の日の割合", "的中率", "ふだん", "倍率", "捕捉率", "警戒中の20日先", "ふだんの20日先", "始まりに警戒中", "20日前までに警戒", "警戒の回数", "その間に暴落", "AUC"]
    for per in periods:
        print(f"\n予想の力（{per}）")
        print(ev[ev["期間"] == per].set_index("指標")[show].round(3).to_string())

    # 2. 運用
    store = ResultStore()
    acct = store.equity(args.account_run)
    runs = []
    for name, (cat, alert, score, first) in list(ind.items()) + [("（比べる相手）", (None, pd.Series(False, index=cal), None, None))]:
        a = alert.fillna(False).astype(float)
        trade = a.shift(1).fillna(0.0)   # 翌営業日の大引けで持つ量を変える
        forms = {"A 警戒中だけ空売り": -trade, "B ふだん持ち・警戒中は空売り": 1 - 2 * trade, "C ふだん持ち・警戒中は持たない": 1 - trade}
        if name == "（比べる相手）":
            forms = {"現金のまま": trade * 0, "日経平均を持つだけ": trade * 0 + 1, "いつも空売り": trade * 0 - 1}
        for per, (s, e) in periods.items():
            c = nk[s:e]
            for form, ex in forms.items():
                eq = simulate(c, ex[s:e], args.capital, args)
                st = stats(eq)
                runs.append({"指標": name, "形": form, "期間": per, "最後の資産（万円）": eq.iloc[-1] / 1e4, "年率": st["年率"],
                             "最大DD": st["最大DD"], "シャープ": st["シャープ"], "最悪の年": st["最悪の年"],
                             "空売りの日の割合": (ex[s:e] < 0).mean(), "売買の回数": int((ex[s:e].diff().abs() > 0).sum())})
            # D: 今の口座に、警戒中だけ資産の半分の空売りを足す（日々の騰落に足し合わせる近似）
            q = acct[acct["period"] == per]
            q = q.set_index(pd.to_datetime(q["date"]))["equity"].astype(float)
            rb = q.pct_change().fillna(0)
            sh = -0.5 * trade.reindex(q.index).fillna(0.0)
            rn = nk.reindex(q.index).pct_change().fillna(0)
            held = sh.shift(1).fillna(0.0)
            rd = rb + held * (rn + args.div_yield / 245) - np.maximum(-held, 0) * args.short_fee / 245 - sh.diff().fillna(0).abs() * args.switch_cost
            eq = args.capital * (1 + rd).cumprod()
            st = stats(eq)
            runs.append({"指標": name, "形": f"D #{args.account_run} に空売りを足す" if name != "（比べる相手）" else f"#{args.account_run} のまま",
                         "期間": per, "最後の資産（万円）": eq.iloc[-1] / 1e4, "年率": st["年率"], "最大DD": st["最大DD"],
                         "シャープ": st["シャープ"], "最悪の年": st["最悪の年"], "空売りの日の割合": (sh < 0).mean(),
                         "売買の回数": int((sh.diff().abs() > 0).sum())})
    rr = pd.DataFrame(runs)
    rr.to_csv(out / "crash_indicators_runs.csv", index=False)
    for per in periods:
        print(f"\n運用（{per}・{args.capital / 1e4:,.0f} 万円から。最後の資産は万円）")
        w = rr[rr["期間"] == per].pivot_table(index="指標", columns="形", values="最後の資産（万円）", sort=False)
        print(w.round(0).to_string())
    for per in periods:
        print(f"\n運用の詳しい成績（{per}）")
        print(rr[rr["期間"] == per].set_index(["指標", "形"])[["年率", "最大DD", "シャープ", "最悪の年", "空売りの日の割合", "売買の回数"]]
              .round(3).to_string())
    print("\n暴落の始まりに警戒中だった指標")
    print(ep[["期間", "下げ始めの日", "下げ", "始まりに警戒中だった指標"]].assign(下げ=lambda x: x["下げ"].round(3)).to_string(index=False))


if __name__ == "__main__":
    main()
