"""-2σ 割れの買い（BB だけ）を、手仕舞い方・局面で比べ、他の指標と組み合わせる研究（study_bb2_filters.py）用の表を作る。

    python scripts/study_bb2_base.py --config config/daily_bb3_long_down.yaml

#35（-3σ 割れの買い）は角度が高いが回数が少ない。-2σ なら回数は何倍にもなるので、まず BB だけでどのくらいの成績かを見る。
建て方は #35 と同じ（寄りで -2σ より下なら寄りで、場中に割れたらその値段で買う。貸借銘柄・売買代金 1 億円以上・張り付き日は取り消し）。
手仕舞い方（利益確定のバンド EXIT_K・損切り STOPS・最長の保有日数 HOLDS）を総当たりし、局面（前日の p_down）で分けて、
IS / OOS ごとに 1 取引の平均（コスト込み）・勝率・PF・日ごとの平均とその t 値・プラスの月の割合を出す。
選ぶのは IS の日ごとの t 値（毎日コツコツ勝てるか）。結果は results/bb2_base_grid.csv。

あわせて、-2σ に触れた日（注文が約定しうる日）ごとに、建てる時点で分かる値（FEATURES。前日の大引けまでの値と、寄りの値）を計算し、
IS で選んだ手仕舞い方の取引に付けて results/bb2_cands.pkl に保存する。
"""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_backtest import load_stocks
from trading.backtest import CostModel
from trading.backtest.touch_engine import strategy_trades
from trading.data import get_source, list_codes
from trading.data.jpx import margin_codes
from trading.strategy import get_strategy

EXIT_K = (1.0, 0.0, None)     # -1σ / 真ん中（25 日平均）/ 指値なし（期限の大引け）
STOPS = (0.10, 0.20)
HOLDS = (3, 5, 10)
REGIMES = {"全部": (-1.0, 1.01), "下げ（p>0.5）": (0.5, 1.01), "上げ（p≦0.5）": (-1.0, 0.5)}

FEATURES = {
    "z_prev": "前日終値のバンド内の位置（σ）",
    "dist": "前日終値から -2σ の注文値段までの距離",
    "gap": "寄りの騰落（前日終値から）",
    "open_below": "寄りが注文値段より何 % 下か（場中に約定した取引はプラス）",
    "at_open": "寄りで約定した（1）か場中か（0）",
    "ret1": "前日の騰落", "ret5": "直近 5 日の騰落", "ret20": "直近 20 日の騰落",
    "sma75": "75 日平均からの乖離", "sma200": "200 日平均からの乖離", "high250": "1 年高値からの下落率",
    "vol20": "20 日の日次リターンの標準偏差", "bbw": "バンドの幅（σ ÷ 25 日平均）", "slope25": "25 日平均の 5 日の傾き",
    "rsi14": "RSI（14 日）", "vratio": "前日の出来高 ÷ 20 日平均", "turnover": "20 日平均売買代金（対数）",
    "price": "株価（対数）", "down_streak": "前日までの連続安の日数", "touch20": "直近 20 日に -2σ を下回って引けた日数",
    "p_down": "局面モデルの前日の下げ相場の確率", "regime_day": "下げ局面の何日目か（局面でなければ 0）",
    "breadth": "前日に -2σ を下回って引けた銘柄の割合（売買代金 1 億円以上）", "breadth_chg": "その割合の前日からの変化",
    "mkt_ret1": "市場平均の前日の騰落", "mkt_ret5": "市場平均の直近 5 日の騰落",
    "nk_ret1": "日経平均の前日の騰落", "nk_dev25": "日経平均の 25 日平均からの乖離", "nk_gap": "日経平均の寄りの騰落",
    "weekday": "曜日（0 = 月）",
}
_G: dict = {}


def stock_features(df: pd.DataFrame, entry: pd.Series) -> pd.DataFrame:
    """1 銘柄の、注文を出す日ごとの値（entry はその日の注文値段 -2σ）。寄りの値以外は前日の大引けまで。"""
    c, v = df["close"], df["volume"]
    pc = c.shift(1)
    r = c.pct_change(fill_method=None)
    m25, s25 = c.rolling(25).mean(), c.rolling(25).std(ddof=0)
    z = (c - m25) / s25
    up, dn = r.clip(lower=0).rolling(14).mean(), (-r).clip(lower=0).rolling(14).mean()
    down = (r < 0).astype(int)
    streak = down.groupby((down == 0).cumsum()).cumsum()
    out = pd.DataFrame({
        "z_prev": z.shift(1), "dist": pc / entry - 1, "gap": df["open"] / pc - 1, "open_below": df["open"] / entry - 1,
        "ret1": r.shift(1), "ret5": pc / c.shift(6) - 1, "ret20": pc / c.shift(21) - 1,
        "sma75": (c / c.rolling(75).mean() - 1).shift(1), "sma200": (c / c.rolling(200).mean() - 1).shift(1),
        "high250": (c / c.rolling(250).max() - 1).shift(1), "vol20": r.rolling(20).std().shift(1),
        "bbw": (s25 / m25).shift(1), "slope25": (m25 / m25.shift(5) - 1).shift(1),
        "rsi14": (100 * up / (up + dn)).shift(1), "vratio": (v / v.rolling(20).mean()).shift(1),
        "turnover": np.log10((c * v).rolling(20).mean().shift(1)), "price": np.log10(pc),
        "down_streak": streak.shift(1), "touch20": (z < -2).astype(float).rolling(20).sum().shift(1),
    }, index=df.index)
    return out


def _init(cfg: dict, params: dict) -> None:
    _G.update(cfg=cfg, params=params)


def _work(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """codes の銘柄だけ読み、手仕舞い方ごとの取引・触れた日の値・日ごとの集計（市場の値用）を返す。"""
    cfg, base = _G["cfg"], _G["params"]
    stocks = load_stocks({**cfg, "data": {**cfg["data"], "codes": codes}, "universe": {}})
    trades, feats, daily = [], [], []
    for code, df in stocks.items():
        for ek, stop, hold in itertools.product(EXIT_K, STOPS, HOLDS):
            s = get_strategy("bb_touch_fade", **{**base, "exit_k": ek, "stop_pct": stop, "max_hold": hold})
            o, idx = df["open"].to_numpy(float), df.index
            for t in strategy_trades(df, s):
                trades.append((code, idx[t.entry_i], idx[t.exit_i], t.entry_price, t.exit_price, t.reason,
                               t.entry_price == o[t.entry_i], ek, stop, hold))
        od = s.orders(df)
        touch = od["long_entry"].notna() & od["can_enter"] & (df["low"] <= od["long_entry"])
        f = stock_features(df, od["long_entry"])[touch]
        feats.append(f.assign(code=code, p_down=df["p_down_prev"][touch]).rename_axis("entry_date").reset_index())
        c = df["close"]
        liquid = ((c * df["volume"]).rolling(20).mean() >= base["min_turnover"]).shift(1, fill_value=False)
        z = ((c - c.rolling(25).mean()) / c.rolling(25).std(ddof=0)).shift(1)
        daily.append(pd.DataFrame({"n": liquid.astype(int), "below": (liquid & (z < -2)).astype(int),
                                   "ret": (c.pct_change(fill_method=None)).where(liquid).fillna(0),
                                   "n_ret": (liquid & c.pct_change(fill_method=None).notna()).astype(int)}))
    cols = ["code", "entry_date", "exit_date", "entry_price", "exit_price", "reason", "at_open", "exit_k", "stop", "hold"]
    agg = pd.concat(daily).groupby(level=0).sum() if daily else pd.DataFrame()
    return pd.DataFrame(trades, columns=cols), pd.concat(feats, ignore_index=True) if feats else pd.DataFrame(), agg


def summarize(g: pd.DataFrame) -> dict:
    """候補の表（bp 列 = コスト込みの 1 取引の損益 bp）の成績。日ごとの平均は、同じ日に建てた取引を等金額で持った場合。"""
    if g.empty:
        return {}
    day = g.groupby("entry_date")["bp"].mean()
    month = day.groupby(day.index.to_period("M")).sum()
    year = day.groupby(day.index.year).sum()
    win, loss = g.loc[g["bp"] > 0, "bp"].sum(), -g.loc[g["bp"] < 0, "bp"].sum()
    return {"取引数": len(g), "日数": len(day), "平均bp": g["bp"].mean(), "勝率": (g["bp"] > 0).mean(),
            "PF": win / loss if loss > 0 else np.inf, "日ごとbp": day.mean(),
            "日ごとt": day.mean() / day.std() * np.sqrt(len(day)) if len(day) > 1 else np.nan,
            "上位1%除くbp": g.loc[g["bp"] < g["bp"].quantile(0.99), "bp"].mean(),
            "プラスの月": (month > 0).mean(), "プラスの年": (year > 0).mean()}


def add_bp(t: pd.DataFrame, costs: CostModel) -> pd.DataFrame:
    buy = np.array([costs.fill_price(p, "buy") for p in t["entry_price"]])
    sell = np.array([costs.fill_price(p, "sell") for p in t["exit_price"]])
    return t.assign(bp=(sell / buy - 1) * 1e4)


def market_features(daily: pd.DataFrame, regime_file: str, threshold: float, end: pd.Timestamp) -> pd.DataFrame:
    """日ごとの市場の値（すべて寄りの前に分かる。nk_gap だけ寄りの値）。"""
    breadth = (daily["below"] / daily["n"].where(daily["n"] > 0))   # 既に前日の z を使っている
    mret = (daily["ret"] / daily["n_ret"].where(daily["n_ret"] > 0)).fillna(0)
    nk = get_source("yfinance").get_daily("^N225", "2000-01-01", str(end.date())).reindex(daily.index).ffill()
    on = pd.read_csv(regime_file, index_col=0, parse_dates=True)["p_down"].shift(1) > threshold
    rday = on.groupby((~on).cumsum()).cumsum().reindex(daily.index).fillna(0)
    return pd.DataFrame({
        "breadth": breadth, "breadth_chg": breadth - breadth.shift(1),
        "mkt_ret1": mret.shift(1), "mkt_ret5": (1 + mret).rolling(5).apply(np.prod, raw=True).shift(1) - 1,
        "nk_ret1": nk["close"].pct_change().shift(1), "nk_dev25": (nk["close"] / nk["close"].rolling(25).mean() - 1).shift(1),
        "nk_gap": nk["open"] / nk["close"].shift(1) - 1, "regime_day": rday, "weekday": daily.index.weekday,
    }, index=daily.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/daily_bb3_long_down.yaml")
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="試しに先頭の何銘柄だけで動かす")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    params = {**cfg["strategy"]["params"], "k": args.k, "long_min_p_down": None}
    costs = CostModel(**cfg["backtest"].get("costs", {}))
    margin = margin_codes()
    codes = [c for c in list_codes(cfg["data"]["dir"]) if c in margin][:args.limit]
    print(f"貸借銘柄 {len(codes)} / -{args.k:g}σ 割れの買い（寄り＋場中）")
    chunks = [codes[i::args.workers * 4] for i in range(args.workers * 4)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg, params)) as ex:
        parts = list(ex.map(_work, chunks))
    trades = add_bp(pd.concat([p[0] for p in parts], ignore_index=True), costs)
    feats = pd.concat([p[1] for p in parts], ignore_index=True)
    daily = pd.concat([p[2] for p in parts]).groupby(level=0).sum().sort_index()
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    trades["期間"] = np.where(trades["entry_date"] <= is_end, "IS", "OOS")
    trades = trades.merge(feats[["code", "entry_date", "p_down"]], on=["code", "entry_date"], how="left")

    rows = []
    for (ek, stop, hold), g in trades.groupby(["exit_k", "stop", "hold"], dropna=False):
        for reg, (lo, hi) in REGIMES.items():
            gg = g[(g["p_down"].fillna(-0.5) > lo) & (g["p_down"].fillna(-0.5) <= hi)]
            for per, x in gg.groupby("期間"):
                rows.append({"利益確定": "なし" if pd.isna(ek) else f"-{ek:g}σ" if ek else "25日平均", "損切り": stop,
                             "最長": hold, "局面": reg, "期間": per, **summarize(x)})
    grid = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    grid.to_csv("results/bb2_base_grid.csv", index=False)
    keys = ["利益確定", "損切り", "最長", "局面"]
    wide = grid.pivot_table(index=keys, columns="期間", values=["取引数", "平均bp", "勝率", "PF", "日ごとbp", "日ごとt", "プラスの月"])
    wide = wide.sort_values(("日ごとt", "IS"), ascending=False)
    print("\nBB だけ（コスト込み）。IS の日ごとの t 値順")
    print(wide.round(2).to_string())

    best = grid[(grid["期間"] == "IS") & (grid["局面"] == "全部")].sort_values("日ごとt", ascending=False).iloc[0]
    label = best["利益確定"]
    ek = None if label == "なし" else 0.0 if label == "25日平均" else float(label.strip("-σ"))
    print(f"\n局面で絞らない中で IS の t 値が最も高い手仕舞い方: 利益確定 {best['利益確定']} / 損切り {best['損切り']:.0%} / 最長 {best['最長']} 日")
    sel = trades[(trades["exit_k"].fillna(-1) == (-1 if ek is None else ek)) & (trades["stop"] == best["損切り"])
                 & (trades["hold"] == best["最長"])].drop(columns=["p_down"])
    mkt = market_features(daily, cfg["data"]["regime_file"], cfg["strategy"]["params"]["long_min_p_down"], daily.index[-1])
    cands = (sel.merge(feats, on=["code", "entry_date"], how="left")
             .merge(mkt.rename_axis("entry_date").reset_index(), on="entry_date", how="left"))
    cands["at_open"] = cands["at_open"].astype(int)
    cands.attrs["exit"] = {"exit_k": ek, "stop_pct": float(best["損切り"]), "max_hold": int(best["最長"])}
    cands.to_pickle("results/bb2_cands.pkl")
    print(f"取引 {len(cands):,} 件に {len(FEATURES)} 個の値を付けて results/bb2_cands.pkl に保存しました")


if __name__ == "__main__":
    main()
