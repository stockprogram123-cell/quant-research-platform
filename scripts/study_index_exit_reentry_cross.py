"""日経平均で探した手放す・買い直すの決まり（scripts/study_index_exit_reentry.py の結果）が、同じ時期（2016〜）の S&P 500 と DAX でも
持ちっぱなしに勝つか。V 字の下げが多かったほかの市場で勝たなければ、日経平均での当たりは偶然とみなす。

    python scripts/study_index_exit_reentry.py          # 先に動かす（results/index_exit_reentry_grid.csv）
    python scripts/study_index_exit_reentry_cross.py

大引けで売買する決まりだけ（ほかの市場の寄りの値段は使わない）。−3σ の銘柄数の合図は日本の銘柄なので除く。VIX はその日付の値（米国の
大引けで分かる。DAX は前の日付の値）。配当は年 1.8% のまま、費用も同じ。"""
import argparse
import itertools

import numpy as np
import pandas as pd

import study_index_exit_reentry as m
from study_index_crash import zscore
from trading.data import get_source

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 30)
args = argparse.Namespace(switch_cost=0.0005, div_yield=0.018, capital=1_000_000)
yf = get_source("yfinance")
vix_raw = yf.get_daily("^VIX", "1990-01-01", "2026-10-26")["close"]
g = pd.read_csv("results/index_exit_reentry_grid.csv")
g["売値の上"] = g["売値の上"].astype(str)
key = ["手放す", "売買", "買い直す", "売値の上", "最長"]
rows = []
for sym, name, vlag in (("^GSPC", "S&P 500", 0), ("^GDAXI", "DAX", 1)):
    c = yf.get_daily(sym, "1999-01-01", "2026-10-26")["close"]
    cal = c.index
    vix = vix_raw.reindex(cal, method="ffill").shift(vlag)
    arr = {"c": c.to_numpy(float), "vix": vix.to_numpy(float), "z": zscore(c).to_numpy(float), "br": np.zeros(len(c))}
    arr["o"] = arr["ob"] = arr["os"] = arr["c"]
    for per, (s, e) in {"2016〜2021": ("2016-01-01", "2021-12-31"), "2022〜": ("2022-01-01", "2026-10-26"),
                        "2016〜": ("2016-01-01", "2026-10-26")}.items():
        ix = np.flatnonzero((cal >= s) & (cal <= e))
        d = {k: v[ix] for k, v in arr.items()}
        base, _ = m.run(d, np.zeros(len(ix), bool), "close", m.BOTTOMS[0], None, 1, args)
        bh = m.summarize(base, cal[ix], args.capital)
        sigs = {ex: m.exit_signal(ex, c, vix)[ix] for ex in m.EXITS}
        for ex, bottom, be, wait in itertools.product(m.EXITS, [b for b in m.BOTTOMS if b[0] != "銘柄"], m.BREAKEVEN, m.WAITS):
            ret, tr = m.run(d, sigs[ex], "close", bottom, be, wait, args)
            sm = m.summarize(ret, cal[ix], args.capital)
            rows.append({"市場": name, "期間": per, "手放す": m.exit_label(ex), "売買": "翌日の大引け", "買い直す": m.bottom_label(bottom),
                         "売値の上": "なし" if be is None else f"{be:.0%}", "最長": wait, "上乗せ": sm["年率"] - bh["年率"],
                         "最大DD": sm["最大DD"], "持ちっぱなしのDD": bh["最大DD"]})
        print(f"{name} {per}: 持ちっぱなし 年率 {bh['年率']:+.1%}・DD {bh['最大DD']:.0%}")
x = pd.DataFrame(rows)
w = x.pivot_table(index=key, columns=["市場", "期間"], values="上乗せ")
w.columns = [f"{a} {b}" for a, b in w.columns]
j = g[g["売買"] == "翌日の大引け"].set_index(key)[["前半 上乗せ", "後半 上乗せ", "OOS 全体 上乗せ", "IS 上乗せ"]].join(w, how="inner")
j.to_csv("results/index_exit_reentry_cross_market.csv")
print(f"\n大引けで売買する決まり {len(j)} 通り（−3σ の銘柄数を除く）")
for col in ("S&P 500 2016〜", "DAX 2016〜"):
    print(f"  日経平均の OOS 全体の上乗せと {col} の順位相関 {j['OOS 全体 上乗せ'].corr(j[col], method='spearman'):.2f}。"
          f"{col} で持ちっぱなしに勝った割合 {(j[col] > 0).mean():.0%}")
j["日経の前半・後半の小さい方"] = j[["前半 上乗せ", "後半 上乗せ"]].min(axis=1)
top = j.nlargest(30, "日経の前半・後半の小さい方")
print(f"\n日経平均の前半・後半の小さい方の上位 30 が、ほかの市場でも勝った割合: S&P 500 {(top['S&P 500 2016〜'] > 0).mean():.0%}・"
      f"DAX {(top['DAX 2016〜'] > 0).mean():.0%}（全体では {(j['S&P 500 2016〜'] > 0).mean():.0%}・{(j['DAX 2016〜'] > 0).mean():.0%}）")
fam = j[j.index.get_level_values("手放す") == "60 日高値から 8% 下げ"]
print(f"「60 日高値から 8% 下げ」の {len(fam)} 通り: 日経 前半・後半の両方で勝つ {int(((fam['前半 上乗せ'] > 0) & (fam['後半 上乗せ'] > 0)).sum())}・"
      f"S&P 500 で勝つ {int((fam['S&P 500 2016〜'] > 0).sum())}・DAX で勝つ {int((fam['DAX 2016〜'] > 0).sum())}")
print("\n手放す合図ごとの、ほかの市場での上乗せの中央値（2016〜）")
print(j.groupby(level="手放す")[["OOS 全体 上乗せ", "S&P 500 2016〜", "DAX 2016〜", "IS 上乗せ"]].median().round(4)
      .sort_values("OOS 全体 上乗せ", ascending=False).to_string())
print("\n3 つの市場（日経の前半・後半、S&P 500、DAX）すべてで勝った決まり")
allwin = j[(j["前半 上乗せ"] > 0) & (j["後半 上乗せ"] > 0) & (j["S&P 500 2016〜"] > 0) & (j["DAX 2016〜"] > 0)]
print(f"{len(allwin)} 通り（4 つが無関係なら {len(j) * (j['前半 上乗せ'] > 0).mean() * (j['後半 上乗せ'] > 0).mean() * (j['S&P 500 2016〜'] > 0).mean() * (j['DAX 2016〜'] > 0).mean():.1f}）")
print(allwin.round(3).to_string())
