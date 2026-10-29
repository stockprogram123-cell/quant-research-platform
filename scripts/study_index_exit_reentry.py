"""V 字の下げが多い相場（OOS の 2016 年以降）を前提に、日経平均を手放すタイミングと買い直すタイミングを探す。

    python scripts/study_index_exit_reentry.py

前提（相場の見立て）: 2026 年以降は OOS の期間のような急な下げと急な戻りが多い。そこで OOS を見て決まりを選ぶ。
  選びすぎを見分けるため、OOS を前半（2016〜2021）と後半（2022〜）に分け、両方で持ちっぱなしに勝つものを選ぶ
  （選ぶ目安は前半・後半の年率の上乗せの小さい方）。IS（2001〜2015）は「V 字でない相場が来たらどうなるか」の目安として出す。
手放す合図（その日の大引けで判断）:
  高値から d 下げた（直近 L 日の最高値。L = 20 / 60、d = 3〜8%。下回った最初の日）/ 5 営業日で x 下げた / 1 日で x 下げた /
  VIX が 50 日平均の k 倍を超えた（米国の値は前の日付まで）
買い直す合図（売った後、毎日の大引けで判断）:
  売ってからの最安値から r 戻した / 売値から x 下（下で拾う）/ 日経平均の終値が 25 日平均の −kσ 以下 /
  −3σ に触れた貸借銘柄が 10 以上（#60 の合図から局面の条件を除いたもの）/ VIX が売ってからの最高値から y 下がった。
  ほかに「終値が売値の b 上に戻った」（b なしも）か「売ってから W 営業日」でも買い直す。
売買: 合図の翌営業日の寄り（n225_daily の open_tradable = ETF の寄りから出した値段。2009 年より前などは大引け）か大引け。費用は片道 0.05%、配当 年 1.8% は持っている間だけ、
  手放している間の現金の利息は 0（study_index_margin_timing.run と同じ）。比べる相手は配当込みの持ちっぱなし。
結果は results/index_exit_reentry_*.csv。−3σ の銘柄数は data/cache/breadth_3sigma.csv に置く（--refresh で数え直し）。
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_stocks
from simulate_portfolio import n225_daily, stats
from study_crash_indicators import known_on
from study_index_crash import IS_END, IS_START, breadth_count, zscore
from trading.data import get_source
from trading.store import ResultStore

PERIODS = {"前半": ("2016-01-01", "2021-12-31"), "後半": ("2022-01-01", None), "OOS 全体": ("2016-01-01", None),
           "IS": (str(IS_START.date()), str(IS_END.date()))}
EXITS = ([("高値から", L, d) for L in (20, 60) for d in (0.03, 0.04, 0.05, 0.06, 0.08)]
         + [("5 日で", 5, x) for x in (0.03, 0.04, 0.05, 0.06)] + [("1 日で", 1, x) for x in (0.02, 0.03)]
         + [("VIX", 50, k) for k in (1.25, 1.5)])
BOTTOMS = ([("戻す", r) for r in (0.02, 0.03, 0.05, 0.07)] + [("下で拾う", x) for x in (0.05, 0.08, 0.12)]
           + [("σ", k) for k in (2.5, 3.0)] + [("銘柄", 10)] + [("VIX", y) for y in (0.15, 0.25)])
BREAKEVEN = (0.0, 0.02, None)
WAITS = (20, 40)


def exit_label(e: tuple) -> str:
    kind, n, x = e
    return {"高値から": f"{n} 日高値から {x:.0%} 下げ", "5 日で": f"5 日で {x:.0%} 下げ", "1 日で": f"1 日で {x:.0%} 下げ",
            "VIX": f"VIX が 50 日平均の {x} 倍"}[kind]


def bottom_label(b: tuple) -> str:
    kind, x = b
    return {"戻す": f"安値から {x:.0%} 戻す", "下で拾う": f"売値から {x:.0%} 下", "σ": f"−{x}σ 以下",
            "銘柄": "−3σ の銘柄 10 以上", "VIX": f"VIX が最高値から {x:.0%} 下がる"}[kind]


def exit_signal(e: tuple, c: pd.Series, vix: pd.Series) -> np.ndarray:
    kind, n, x = e
    if kind == "高値から":
        v, th = c / c.rolling(n, min_periods=1).max() - 1, -x
    elif kind in ("5 日で", "1 日で"):
        v, th = c / c.shift(n) - 1, -x
    else:
        v, th = -(vix / vix.rolling(n).mean()), -x
    return ((v <= th) & ~(v.shift(1) <= th)).to_numpy(bool)


def run(d: dict, sig: np.ndarray, mode: str, bottom: tuple, be: float | None, wait: int, args) -> tuple[np.ndarray, list[dict]]:
    """日ごとの騰落（持ちっぱなしと同じ計算）と手仕舞いの一覧。d は期間で切った配列（c, o, vix, z, br。寄りの売りは os、買いは ob の値段）。"""
    c, vix, z, br = d["c"], d["vix"], d["z"], d["br"]
    o = d["o"].copy()   # 寄りで売り買いした日だけ、その値段に置き換える
    n = len(c)
    night, day = np.ones(n), np.ones(n)
    trades, t = [], 0
    kind, x = bottom
    while t < n - 2:
        if not sig[t]:
            t += 1
            continue
        e = t + 1
        if mode == "open":
            o[e] = d["os"][e]
            day[e], px_e = 0.0, o[e]
        else:
            px_e = c[e]
        low, vmax, s = c[e], vix[e], e
        while True:
            if s >= n - 1:
                break
            low = min(low, c[s])
            vmax = np.nanmax([vmax, vix[s]])
            hit = {"戻す": c[s] >= low * (1 + x), "下で拾う": c[s] <= px_e * (1 - x), "σ": z[s] <= -x,
                   "銘柄": br[s] >= x, "VIX": vix[s] <= vmax * (1 - x)}[kind]
            if hit or (be is not None and c[s] >= px_e * (1 + be)) or s - e >= wait:
                break
            s += 1
        r = min(s + 1, n - 1)
        if mode == "open":
            night[e + 1:r + 1], day[e + 1:r] = 0.0, 0.0
            o[r] = d["ob"][r]
            px_r = o[r]
        else:
            night[e + 1:r + 1], day[e + 1:r + 1] = 0.0, 0.0
            px_r = c[r]
        trades.append({"合図": t, "売り": e, "売値": px_e, "買い直し": r, "買値": px_r, "差": px_e / px_r - 1})
        t = r
    g = np.r_[0.0, o[1:] / c[:-1] - 1]
    i = c / o - 1
    ret = (1 + night * g) * (1 + day * i) - 1 + day * args.div_yield / 245
    # 寄りで売り買いすれば昼の持ち方、大引けなら夜の持ち方が変わる（大引けの売買の費用は翌日に付く）
    changes = np.abs(np.diff(np.r_[1.0, day if mode == "open" else night]))
    ret -= changes * args.switch_cost
    return ret, trades


def summarize(ret: np.ndarray, idx: pd.DatetimeIndex, capital: float) -> dict:
    eq = pd.Series(capital * np.cumprod(1 + ret), index=idx)
    st = stats(eq)
    return {"最後の資産（万円）": eq.iloc[-1] / 1e4, "年率": st["年率"], "最大DD": st["最大DD"], "シャープ": st["シャープ"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--switch-cost", type=float, default=0.0005)
    parser.add_argument("--div-yield", type=float, default=0.018)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 40)
    out = Path("results")

    nkd = n225_daily(pd.Timestamp.today().strftime("%Y-%m-%d"), start="1999-01-01")
    cal, c = nkd.index, nkd["close"]
    vix = known_on(get_source("yfinance").get_daily("^VIX", "1990-01-01", cal[-1].strftime("%Y-%m-%d"))["close"], cal)
    path = Path("data/cache/breadth_3sigma.csv")
    if path.exists() and not args.refresh:
        br = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        br = breadth_count(load_stocks(ResultStore().run(73)["config"]))
        br.rename("n").to_csv(path)
    o = nkd["open_tradable"].to_numpy(float)
    arrays = {"c": c.to_numpy(float), "o": o, "ob": o, "os": o,
              "vix": vix.to_numpy(float),
              "z": zscore(c).to_numpy(float), "br": br.reindex(cal).fillna(0).to_numpy(float)}
    sl = {}
    for p, (s, e) in PERIODS.items():
        m = (cal >= pd.Timestamp(s)) & (cal <= (pd.Timestamp(e) if e else cal[-1]))
        sl[p] = (np.flatnonzero(m), cal[m])
    sigs = {ex: exit_signal(ex, c, vix) for ex in EXITS}

    base = {}
    for p, (ix, idx) in sl.items():
        d = {k: v[ix] for k, v in arrays.items()}
        ret, _ = run(d, np.zeros(len(ix), bool), "close", BOTTOMS[0], None, 1, args)
        base[p] = summarize(ret, idx, args.capital)
    print("持ちっぱなし（配当込み）: " + " / ".join(f"{p} {v['最後の資産（万円）']:.0f} 万円・年率 {v['年率']:+.1%}・DD {v['最大DD']:.0%}"
                                          for p, v in base.items()))

    rows, keep = [], {}
    for ex, mode, bottom, be, wait in itertools.product(EXITS, ("open", "close"), BOTTOMS, BREAKEVEN, WAITS):
        row = {"手放す": exit_label(ex), "売買": "翌日の寄り" if mode == "open" else "翌日の大引け", "買い直す": bottom_label(bottom),
               "売値の上": "なし" if be is None else f"{be:.0%}", "最長": wait}
        for p, (ix, idx) in sl.items():
            d = {k: v[ix] for k, v in arrays.items()}
            ret, tr = run(d, sigs[ex][ix], mode, bottom, be, wait, args)
            sm = summarize(ret, idx, args.capital)
            gains = np.array([t_["差"] for t_ in tr])
            row.update({f"{p} {k}": v for k, v in sm.items()})
            row[f"{p} 上乗せ"] = sm["年率"] - base[p]["年率"]
            row[f"{p} 回数"] = len(tr)
            row[f"{p} 下で買えた"] = (gains > 0).mean() if len(tr) else np.nan
            if p in ("OOS 全体", "IS"):
                keep[(tuple(row[k] for k in ("手放す", "売買", "買い直す", "売値の上", "最長")), p)] = (tr, idx, ret)
        rows.append(row)
    g = pd.DataFrame(rows)
    g["前半・後半の小さい方"] = g[["前半 上乗せ", "後半 上乗せ"]].min(axis=1)
    g.to_csv(out / "index_exit_reentry_grid.csv", index=False)

    n_rules = len(g)
    print(f"\n決まり {n_rules} 通り。年率で持ちっぱなしに勝った数: " + " / ".join(f"{p} {int((g[f'{p} 上乗せ'] > 0).sum())}" for p in PERIODS)
          + f"。前半と後半の両方 {int(((g['前半 上乗せ'] > 0) & (g['後半 上乗せ'] > 0)).sum())}")
    print(f"前半と後半の上乗せの順位相関 {g['前半 上乗せ'].corr(g['後半 上乗せ'], method='spearman'):.2f}、"
          f"OOS 全体と IS {g['OOS 全体 上乗せ'].corr(g['IS 上乗せ'], method='spearman'):.2f}")
    for a, b in (("前半", "後半"), ("後半", "前半")):
        top = g.nlargest(20, f"{a} 上乗せ")
        print(f"  {a}の上位 20 のうち {b}でも勝ったもの {int((top[f'{b} 上乗せ'] > 0).sum())}（{b}の上乗せの中央値 {top[f'{b} 上乗せ'].median():+.2%}）")

    key = ["手放す", "売買", "買い直す", "売値の上", "最長"]
    show = key + ["前半 上乗せ", "後半 上乗せ", "OOS 全体 最後の資産（万円）", "OOS 全体 年率", "OOS 全体 最大DD", "OOS 全体 シャープ",
                  "OOS 全体 回数", "OOS 全体 下で買えた", "IS 最後の資産（万円）", "IS 上乗せ", "IS 最大DD"]
    print("\n前半・後半の上乗せの小さい方の上位 25")
    print(g.nlargest(25, "前半・後半の小さい方")[show].round(3).to_string(index=False))

    print("\n分類ごとの最良（前半・後半の小さい方）")
    for col in ("手放す", "買い直す", "売買", "売値の上", "最長"):
        b = g.groupby(col)[["前半・後半の小さい方", "OOS 全体 上乗せ", "IS 上乗せ"]].agg(["max", "median"])
        print(b.round(4).sort_values(("前半・後半の小さい方", "max"), ascending=False).to_string())

    best = g.nlargest(1, "前半・後半の小さい方").iloc[0]
    bk = tuple(best[k] for k in key)
    print(f"\n選んだ決まり: {' / '.join(str(v) for v in bk)}")
    near = g[(g[key] == best[key]).sum(axis=1) >= 4]
    print("1 つだけ変えた決まり（まわりも良ければ偶然の当たりではない）")
    print(near[show].round(3).to_string(index=False))
    for p in ("OOS 全体", "IS"):
        tr, idx, _ = keep[(bk, p)]
        t = pd.DataFrame(tr)
        if len(t):
            for col in ("合図", "売り", "買い直し"):
                t[col] = idx[t[col].to_numpy()].date
            print(f"\n{p} の手仕舞い（差 = 売値 ÷ 買値 − 1。プラスなら下で買い直せた）")
            print(t.assign(差=t["差"].round(4), 売値=t["売値"].round(0), 買値=t["買値"].round(0)).to_string(index=False))
    eqs = {}
    for p in ("OOS 全体", "IS"):
        _, idx, ret = keep[(bk, p)]
        eqs[p] = pd.Series(args.capital * np.cumprod(1 + ret), index=idx)
    pd.concat(eqs, names=["期間", "date"]).rename("equity").to_csv(out / "index_exit_reentry_best_equity.csv")


if __name__ == "__main__":
    main()
