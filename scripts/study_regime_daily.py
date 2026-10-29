"""日足で下げ相場の特徴を学び、局面ごとに空売り（と比較用の買い）の成績を調べる。

    python scripts/study_regime_daily.py

1. 日経平均の上げ・下げの局面（後から見た答え）を zigzag で作る
2. 各日の終値時点で分かる特徴が、下げ相場と上げ相場でどう違うか（AUC）を調べる
3. その特徴で「いま下げ相場である確率」のモデルを学習期間で作り、確認期間で当たり具合を見る
4. 売買代金上位 200 銘柄で、空売り・買いの形を局面ごとに比べる（次の日の寄りで建て、N 日後の大引けか損切りで決済）
結果は results/regime_*.csv、局面の確率は data/cache/regime_daily.csv（5 分足の検証で使う）に保存する。

--stocks-dir を指定すると、銘柄の日足はそのフォルダの CSV（data/daily）を使う。
銘柄数が多く、「今」売買代金が大きい銘柄に偏らないので、市場の広がり（breadth25 など）の特徴が正確になる。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trading.data import get_source, list_codes, load_daily_csv
from trading.regime import RegimeModel, legs, regime_features, zigzag_regime

START, END = "2000-01-01", "2026-10-23"
COST_BP = 5.0          # 片道のスプレッド（bp）。大型株の 1 呼値前後
SHORT_FEE = 0.011      # 空売りの貸株料（年率）
HOLDS = (5, 10, 20)    # 保有日数


def auc(x: pd.Series, y: pd.Series) -> float:
    """y = 1 の日の x が y = 0 の日の x より大きい確率（0.5 = 区別できない）。"""
    m = x.notna() & y.notna()
    r = x[m].rank()
    n1, n0 = (y[m] == 1).sum(), (y[m] == 0).sum()
    return (r[y[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def signals_of(d: pd.DataFrame) -> dict[str, tuple[int, pd.Series]]:
    """各銘柄の売買の形（向き, その日の終値で条件を満たしたか）。空売りは -1。"""
    c, o, h, lo = d["close"], d["open"], d["high"], d["low"]
    s25, s75 = c.rolling(25).mean(), c.rolling(75).mean()
    slope = s25 / s25.shift(5) - 1
    trend_dn = (c < s25) & (s25 < s75) & (slope < 0)
    trend_up = (c > s25) & (s25 > s75) & (slope > 0)
    low20, high20 = lo.rolling(20).min().shift(1), h.rolling(20).max().shift(1)
    return {
        # 空売り
        "売:下げトレンド入り": (-1, trend_dn & ~trend_dn.shift(1, fill_value=False)),
        "売:戻り売り": (-1, (slope < 0) & (c < s75) & (h.shift(1) >= s25.shift(1) * 0.995) & (c < o) & (c < s25)),
        "売:20日安値割れ": (-1, (c < low20) & ~(c.shift(1) < low20.shift(1))),
        # 買い（比較用）
        "買:上げトレンド入り": (1, trend_up & ~trend_up.shift(1, fill_value=False)),
        "買:押し目買い": (1, (slope > 0) & (c > s75) & (lo.shift(1) <= s25.shift(1) * 1.005) & (c > o) & (c > s25)),
        "買:20日高値抜け": (1, (c > high20) & ~(c.shift(1) > high20.shift(1))),
        "買:売られすぎの反発": (1, (c / s25 - 1 < -0.08) & (c > o) & (c > c.shift(1))),
    }


def outcomes(d: pd.DataFrame, side: int, stop_pct: float) -> pd.DataFrame:
    """シグナルの翌日の寄りで建て、H 日後の大引けか損切り（建値の stop_pct 逆行）で決済した損益（bp、コスト込み）。"""
    entry = d["open"].shift(-1)
    stop = entry * (1 - side * stop_pct)
    adverse = d["low"] if side > 0 else d["high"]
    hit = pd.Series(np.nan, index=d.index)
    hit_px = pd.Series(np.nan, index=d.index)
    out = {}
    for k in range(1, max(HOLDS) + 1):
        a, o = adverse.shift(-k), d["open"].shift(-k)
        newly = hit.isna() & (side * (a - stop) <= 0)
        hit[newly] = k
        hit_px[newly] = (np.minimum(o, stop) if side > 0 else np.maximum(o, stop))[newly]  # 窓を開けたら始値
        if k in HOLDS:
            px = d["close"].shift(-k).where(hit.isna(), hit_px)
            days = hit.fillna(k)
            fee = SHORT_FEE * days * 1.45 / 365 * 1e4 if side < 0 else 0  # 営業日 → 暦日でおよそ 1.45 倍
            out[f"bp_{k}"] = side * (px / entry - 1) * 1e4 - 2 * COST_BP - fee
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.10, help="局面の転換とみなす値動き")
    parser.add_argument("--train-end", default="2015-12-31", help="ここまでで局面モデルを学習")
    parser.add_argument("--stop-pct", type=float, default=0.07)
    parser.add_argument("--stocks-dir", default=None, help="銘柄の日足 CSV のフォルダ（省略時は yfinance の 200 銘柄）")
    parser.add_argument("--cache-out", default="data/cache/regime_daily.csv")
    args = parser.parse_args()
    pd.set_option("display.width", 250)

    src = get_source("yfinance")
    nk = src.get_daily("^N225", START, END)
    if args.stocks_dir:
        stocks = {c: d for c in list_codes(args.stocks_dir) if len(d := load_daily_csv(c, args.stocks_dir)) >= 300}
    else:
        u = pd.read_csv("data/cache/intraday_universe_2026-08-31.csv", dtype={"code": str})
        u = u[(u["turnover"] >= 1e9) & (u["price"] >= 300)].sort_values("turnover", ascending=False).head(200)
        stocks = {c: src.get_daily(c, START, END) for c in u["code"]}
    print(f"銘柄 {len(stocks)}")

    label = zigzag_regime(nk["close"], args.threshold)
    feats = regime_features(nk, stocks)
    train_end = pd.Timestamp(args.train_end)
    is_train = feats.index <= train_end

    # 2. 下げ相場と上げ相場で特徴がどう違うか
    rows = []
    for col in feats:
        rows.append({"feature": col, "下げ相場の中央値": feats.loc[label == 1, col].median(),
                     "上げ相場の中央値": feats.loc[label == 0, col].median(),
                     "AUC(学習)": auc(feats.loc[is_train, col], label[is_train]),
                     "AUC(確認)": auc(feats.loc[~is_train, col], label[~is_train])})
    diff = pd.DataFrame(rows).sort_values("AUC(学習)")
    print("===== 下げ相場と上げ相場の特徴の違い（AUC: 1 に近いほど『大きいと下げ相場』、0 に近いほど『小さいと下げ相場』） =====")
    print(diff.round(3).to_string(index=False))

    # 3. 局面モデル
    usable = feats.dropna().index.intersection(label.dropna().index)
    train_idx = usable[usable <= train_end]
    model = RegimeModel(list(feats.columns), l2=5.0).fit(feats.loc[train_idx], label[train_idx])
    p_down = model.predict(feats.dropna())
    print("\n===== 局面モデルの重み（標準化した特徴 1 単位あたり。プラス = 大きいほど下げ相場らしい） =====")
    print(model.coefficients().round(2).to_string())
    pred = (p_down > 0.5).astype(int)
    for name, idx in (("学習", train_idx), ("確認", usable[usable > train_end])):
        y, p = label[idx], pred[idx]
        print(f"{name}: 正解率 {(y == p).mean():.0%}、下げ相場の日のうち当てた割合 {p[y == 1].mean():.0%}、"
              f"下げと判定した日のうち本当に下げだった割合 {y[p == 1].mean():.0%}")

    # 下げ相場（確認期間）が始まってから、モデルが何日目に気づいたか
    lg = legs(label, nk["close"])
    late = []
    for _, r in lg[(lg["down"] == 1) & (lg["start"] > train_end)].iterrows():
        seg = pred.loc[r["start"]:r["end"]]
        first = seg[seg == 1].index.min()
        late.append({"開始": r["start"].date(), "日数": r["days"], "下落率": r["change"],
                     "気づいた日": first.date() if pd.notna(first) else None,
                     "何日目": seg.index.get_loc(first) + 1 if pd.notna(first) else None,
                     "気づくまでの下落": nk["close"].loc[first] / nk["close"].loc[r["start"]] - 1 if pd.notna(first) else None,
                     "そこからの下落": nk["close"].loc[r["end"]] / nk["close"].loc[first] - 1 if pd.notna(first) else None})
    print("\n===== 確認期間の下げ相場に、モデルが何日目で気づいたか =====")
    print(pd.DataFrame(late).round(3).to_string(index=False))

    regime = pd.DataFrame({"p_down": p_down, "down_label": label}).join(feats)
    regime.to_csv(args.cache_out)

    # 4. 局面ごとの売買の成績
    recs = []
    for code, d in stocks.items():
        d = d[["open", "high", "low", "close"]].dropna()
        for name, (side, sig) in signals_of(d).items():
            res = outcomes(d, side, args.stop_pct)[sig.reindex(d.index, fill_value=False)]
            if res.empty:
                continue
            recs.append(res.assign(tactic=name, code=code))
    ev = pd.concat(recs).dropna()
    ev["p_down"] = ev.index.map(p_down)
    ev["label"] = ev.index.map(label)
    ev = ev.dropna(subset=["p_down"])
    ev["model"] = np.where(ev["p_down"] > 0.5, "下げ", "上げ")
    ev["period"] = np.where(ev.index <= train_end, "学習", "確認")
    ev["leg"] = ev.index.map((label != label.shift()).cumsum())

    rows = []
    for (period, tactic, model_regime), g in ev.groupby(["period", "tactic", "model"]):
        for h in HOLDS:
            bp = g[f"bp_{h}"]
            by_leg = g.groupby("leg")[f"bp_{h}"].mean()
            rows.append({"期間": period, "形": tactic, "モデルの局面": model_regime, "保有": h, "回数": len(bp),
                         "勝率": (bp > 0).mean(), "平均bp": bp.mean(),
                         "局面区間ごとのプラス率": (by_leg > 0).mean(), "区間数": len(by_leg)})
    table = pd.DataFrame(rows)
    table.to_csv("results/regime_daily_tactics.csv", index=False)
    print("\n===== 局面ごとの売買の成績（保有 10 日、コスト込み） =====")
    view = table[table["保有"] == 10].pivot_table(index="形", columns=["モデルの局面", "期間"],
                                                 values=["平均bp", "勝率", "回数"])
    print(view.round(2).to_string())
    print("\n（保有 5 / 20 日も含む全体は results/regime_daily_tactics.csv）")
    diff.to_csv("results/regime_features.csv", index=False)


if __name__ == "__main__":
    main()
