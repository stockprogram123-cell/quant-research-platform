"""-2σ 割れの押し目買いを「大引けで判断して、引けか翌日の寄りで買う」形で調べ直す。

    python scripts/study_bb2_close.py            # 候補表（保存してあれば使う）→ 指標の効き方・条件選び → 資金制約ありの運用 → DB 保存
    python scripts/study_bb2_close.py --rebuild  # 候補表を作り直す

場中に -2σ の指値で買う形（config/daily_bb2_long_pullback.yaml、DB #53）には 3 つの弱みがあった:
指値に触れただけで約定する扱いが楽観的・全候補に指値を置く買付余力がない・日経平均との乗り換えの費用。
大引けの値で判断して引け（15:25 の値で判断して引けの成行。終値で約定とみなす）か翌日の寄りで買えば約定は確実で、
その日の候補を全部見てから、空き枠の分だけ選んで買える。

合図: 終値が 25 日ボリンジャーバンドの -k σ 以下（当日の終値を含むバンド）・前日までの 20 日平均売買代金 1 億円以上・
      ストップ安で引けていない（翌日の寄りで買うときは、その日が一日中張り付いていない）
手仕舞い（EXITS）: 終値が -1σ 以上に戻った日の大引け（最長 5 / 10 日）、決めた日数後の大引け、翌日の寄り（引けで買った場合だけ）。
      手仕舞う日にストップ安で引けたら翌日の寄りで売る。
評価: 1 取引の損益（コスト込み bp）と、同じ期間に日経平均を持っていた場合に対する上乗せ（乗り換え費用 片道 --switch-cost を引く）。
      待機中の資金は日経平均で持つので、勝つべき相手は日経平均。
1. 建て方 × 手仕舞い方 × 最初の割れだけか × 前の押し目ルール（#53）を当てはめるかの成績
2. IS の上乗せの日ごとの t 値が最も高い建て方・手仕舞い方で、指標（FEATURES。その日の大引けまでに分かる値）の 5 分位
3. 上乗せの日ごとの t 値を最も上げる条件を IS で 1 つずつ足していく（境目は IS の分位点）。OOS は選択に使わない
4. 資金制約ありの運用（日経平均で待機）。条件・枠・その日の候補から買う順を IS のシャープで選び、OOS と、
   同じ日に無作為な銘柄を買う「偽の手法」と比べる。選んだ設定を DB に保存する
結果は results/bb2c_*.csv。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_backtest import load_stocks
from simulate_portfolio import data_start, index_returns, n225_daily, stats
from study_bb2_base import summarize
from trading.backtest import CostModel, summarize as summarize_run
from trading.backtest.costs import TICK_TABLE_TOPIX500
from trading.backtest.slot_sim import simulate_slots
from trading.data import list_codes
from trading.data.jpx import margin_codes
from trading.store import DEFAULT_DB, ResultStore, describe_run, universe_label

ENTRIES = {"引け": "close", "翌日の寄り": "next_open"}
EXITS = {"−1σで引け・最長5日": ("z", 5), "−1σで引け・最長10日": ("z", 10), "1日後の引け": ("hold", 1),
         "3日後の引け": ("hold", 3), "5日後の引け": ("hold", 5), "翌日の寄り": ("open", 1)}
FEATURES = {
    "z": "その日の終値のバンド内の位置（σ）", "z_prev": "前日の終値のバンド内の位置", "ret1": "その日の騰落",
    "ret5": "5 日の騰落", "ret20": "20 日の騰落", "sma75": "75 日平均からの乖離", "sma200": "200 日平均からの乖離",
    "high250": "1 年高値からの下落率", "vol20": "20 日の値動きの大きさ", "bbw": "バンドの幅", "slope25": "25 日平均の 5 日の傾き",
    "rsi14": "RSI（14 日）", "vratio": "その日の出来高 ÷ 前日までの 20 日平均", "turnover": "20 日平均売買代金（対数）",
    "price": "株価（対数）", "down_streak": "連続安の日数", "touch20": "前日までの 20 日に -2σ 以下で引けた日数",
    "breaks5": "前日までの 5 日に -kσ 以下で引けた日数", "gap": "寄りの窓", "intraday": "寄りから引けまでの騰落",
    "lower_shadow": "下ひげ（安値からの戻り ÷ 終値）", "close_pos": "その日の値幅の中の終値の位置（0 = 安値）",
    "p_down": "局面モデルの前日の下げ相場の確率", "breadth": "-2σ 以下で引けた銘柄の割合", "breadth_chg": "その割合の前日からの変化",
    "mkt_ret1": "市場平均のその日の騰落", "mkt_ret5": "市場平均の 5 日の騰落", "nk_ret1": "日経平均のその日の騰落",
    "nk_dev25": "日経平均の 25 日平均からの乖離", "regime_day": "下げ局面の何日目か", "weekday": "曜日（0 = 月）", "month": "月",
    "idio1": "その日の騰落 − 市場平均（個別の下げ）", "idio5": "5 日の騰落 − 市場平均", "mday": "月の何営業日目か",
    "mday_end": "月末まで何営業日か（0 = 月末）",
}
LABELS = {"ret5": "5日の騰落", "ret20": "20日の騰落", "high250": "1年高値から", "sma200": "200日線から", "sma75": "75日線から",
          "z": "終値のσ", "z_prev": "前日のσ", "ret1": "その日の騰落", "vol20": "値動きの大きさ", "bbw": "バンドの幅",
          "slope25": "25日線の傾き", "rsi14": "RSI", "vratio": "出来高の倍率", "turnover": "売買代金（対数）", "price": "株価（対数）",
          "down_streak": "連続安", "touch20": "20日の割れ回数", "breaks5": "5日の割れ回数", "gap": "寄りの窓", "intraday": "寄りから引け",
          "lower_shadow": "下ひげ", "close_pos": "終値の位置", "p_down": "下げ確率", "breadth": "割れた銘柄の割合",
          "breadth_chg": "割合の変化", "mkt_ret1": "市場のその日", "mkt_ret5": "市場の5日", "nk_ret1": "日経平均のその日",
          "nk_dev25": "日経平均の25日線から", "regime_day": "下げ局面の日数", "weekday": "曜日", "month": "月",
          "idio1": "個別のその日", "idio5": "個別の5日", "mday": "月の営業日", "mday_end": "月末まで",
          "dist": "前日終値から注文値段まで", "open_below": "寄りが注文値段より", "nk_gap": "日経平均の寄りの窓"}
PCT_FEATS = {"ret1", "ret5", "ret20", "sma75", "sma200", "high250", "vol20", "bbw", "slope25", "gap", "intraday", "lower_shadow",
             "breadth", "breadth_chg", "mkt_ret1", "mkt_ret5", "nk_ret1", "nk_dev25", "idio1", "idio5", "dist", "open_below",
             "nk_gap"}
DISCRETE = {"down_streak", "touch20", "breaks5", "regime_day", "weekday", "month", "mday", "mday_end"}


def add_extra_features(c: pd.DataFrame, calendar: pd.DatetimeIndex, date_col: str = "date") -> pd.DataFrame:
    """候補表から作れる指標: 個別の下げ（市場平均を引いた騰落）と、月の何営業日目か・月末まで何営業日か。"""
    cal = pd.Series(calendar, index=calendar)
    month = cal.dt.to_period("M")
    mday = cal.groupby(month).cumcount() + 1
    mday_end = cal.groupby(month).cumcount(ascending=False)
    d = c[date_col]
    return c.assign(idio1=(c["ret1"] - c["mkt_ret1"]).astype("float32"), idio5=(c["ret5"] - c["mkt_ret5"]).astype("float32"),
                    mday=mday.reindex(d).to_numpy(), mday_end=mday_end.reindex(d).to_numpy())
OLD_RULE = {"high250": -0.1516, "ret5": -0.02279, "sma200": -0.04539}   # #53 の押し目ルール（向きは ≧ / ≦ / ≧）
PICKS = {"乱数の順": None, "深く下げた順（終値のσが低い順）": ("z", True), "5日で大きく下げた順": ("ret5", True),
         "1年高値に近い順": ("high250", False)}
_G: dict = {}


def close_features(df: pd.DataFrame) -> pd.DataFrame:
    """1 銘柄の、その日の大引けまでに分かる値。"""
    c, v, o, h, lo = (df[k] for k in ("close", "volume", "open", "high", "low"))
    r = c.pct_change(fill_method=None)
    m25, s25 = c.rolling(25).mean(), c.rolling(25).std(ddof=0)
    z = (c - m25) / s25
    up, dn = r.clip(lower=0).rolling(14).mean(), (-r).clip(lower=0).rolling(14).mean()
    down = (r < 0).astype(int)
    below = (z <= -2).astype(float)
    return pd.DataFrame({
        "z": z, "z_prev": z.shift(1), "ret1": r, "ret5": c / c.shift(5) - 1, "ret20": c / c.shift(20) - 1,
        "sma75": c / c.rolling(75).mean() - 1, "sma200": c / c.rolling(200).mean() - 1, "high250": c / c.rolling(250).max() - 1,
        "vol20": r.rolling(20).std(), "bbw": s25 / m25, "slope25": m25 / m25.shift(5) - 1, "rsi14": 100 * up / (up + dn),
        "vratio": v / v.rolling(20).mean().shift(1), "turnover": np.log10((c * v).rolling(20).mean().shift(1)),
        "price": np.log10(c), "down_streak": down.groupby((down == 0).cumsum()).cumsum(),
        "touch20": below.shift(1).rolling(20).sum(), "gap": o / c.shift(1) - 1, "intraday": c / o - 1,
        "lower_shadow": (np.minimum(o, c) - lo) / c, "close_pos": ((c - lo) / (h - lo)).where(h > lo),
        "p_down": df["p_down_prev"] if "p_down_prev" in df else np.nan,
    }, index=df.index)


def _exit(kind: str, h: int, i: int, z, o, c, dclose, locked, n: int):
    """合図の日 i から数えた手仕舞い（行番号, 値段, 寄りか引けか, 理由）。データの終わりまでに手仕舞えなければ None。"""
    if kind == "open":
        j = i + 1
        while j < n and locked[j]:
            j += 1
        return (j, o[j], "open", "time") if j < n else None
    j, reason = None, "time"
    for jj in range(i + 1, min(i + h, n - 1) + 1):
        if kind == "z" and z[jj] >= -1.0:
            j, reason = jj, "target"
            break
    if j is None:
        j = i + h
        if j >= n:
            return None
    if dclose[j]:   # ストップ安で引けた日は大引けで売れないので、次に値段がついた日の寄りで売る
        jj = j + 1
        while jj < n and locked[jj]:
            jj += 1
        return (jj, o[jj], "open", "limit") if jj < n else None
    return j, c[j], "close", reason


def _init(cfg: dict, k: float) -> None:
    _G.update(cfg=cfg, k=k)


def _work(codes: list[str]):
    cfg, k = _G["cfg"], _G["k"]
    stocks = load_stocks({**cfg, "data": {**cfg["data"], "codes": codes}, "universe": {}})
    feats, outs, daily = [], [], []
    for code, df in stocks.items():
        f = close_features(df)
        n = len(df)
        c, o, z = df["close"].to_numpy(float), df["open"].to_numpy(float), f["z"].to_numpy()
        zero = np.zeros(n, bool)
        dclose = df["down_close"].to_numpy(bool) if "down_close" in df else zero
        locked = (df["up_locked"].to_numpy(bool) | df["down_locked"].to_numpy(bool)) if "up_locked" in df else zero
        liquid = ((df["close"] * df["volume"]).rolling(20).mean().shift(1) >= 1e8).to_numpy()
        below_k = z <= -k
        breaks5 = pd.Series(below_k.astype(float), index=df.index).shift(1).rolling(5, min_periods=1).sum()
        sig = below_k & liquid & ~dclose
        idx = df.index
        for i in np.flatnonzero(sig):
            for ename, entry in ENTRIES.items():
                e = i if entry == "close" else i + 1
                if e >= n or (entry == "next_open" and locked[e]):
                    continue
                for xname, (kind, h) in EXITS.items():
                    if kind == "open" and entry != "close":
                        continue
                    res = _exit(kind, h, i, z, o, c, dclose, locked, n)
                    if res is None:
                        continue
                    j, xp, x_at, reason = res
                    outs.append((code, idx[i], ename, xname, idx[e], idx[j], c[i] if entry == "close" else o[e], xp,
                                 x_at, reason))
        if sig.any():
            feats.append(f[sig].assign(code=code, breaks5=breaks5[sig]).rename_axis("date").reset_index())
        cl = df["close"]
        liq = pd.Series(liquid, index=idx)
        r = cl.pct_change(fill_method=None)
        daily.append(pd.DataFrame({"n": liq.astype(int), "below": (liq & (f["z"] <= -2)).astype(int),
                                   "ret": r.where(liq).fillna(0), "n_ret": (liq & r.notna()).astype(int)}))
    cols = ["code", "date", "entry", "exit", "entry_date", "exit_date", "entry_price", "exit_price", "exit_at", "reason"]
    return (pd.concat(feats, ignore_index=True) if feats else pd.DataFrame(), pd.DataFrame(outs, columns=cols),
            pd.concat(daily).groupby(level=0).sum() if daily else pd.DataFrame())


def market_close_features(daily: pd.DataFrame, nk: pd.DataFrame) -> pd.DataFrame:
    """日ごとの市場の値（その日の大引けまでに分かる値）。"""
    breadth = daily["below"] / daily["n"].where(daily["n"] > 0)
    mret = (daily["ret"] / daily["n_ret"].where(daily["n_ret"] > 0)).fillna(0)
    nkc = nk["close"].reindex(daily.index).ffill()
    return pd.DataFrame({"breadth": breadth, "breadth_chg": breadth - breadth.shift(1), "mkt_ret1": mret,
                         "mkt_ret5": (1 + mret).rolling(5).apply(np.prod, raw=True) - 1, "nk_ret1": nkc.pct_change(),
                         "nk_dev25": nkc / nkc.rolling(25).mean() - 1}, index=daily.index)


def half_spread(p: np.ndarray, costs: CostModel) -> np.ndarray:
    """CostModel.fill_price と同じ気配の差の半分（TOPIX500 の呼値で最低 1 呼値）を、まとめて計算する。"""
    tick = np.full(len(p), 5_000.0)
    for upper, t in reversed(TICK_TABLE_TOPIX500):
        tick[p <= upper] = t
    spread = np.maximum(costs.spread_yen, tick) if costs.spread_min_tick else np.full(len(p), costs.spread_yen)
    return spread / 2 + p * costs.slippage_bps / 1e4


def build(cfg: dict, k: float, workers: int, nk: pd.DataFrame, regime_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    margin = margin_codes()
    codes = [c for c in list_codes(cfg["data"]["dir"]) if c in margin]
    chunks = [codes[i::workers * 4] for i in range(workers * 4)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg, k)) as ex:
        parts = list(ex.map(_work, chunks))
    feats = pd.concat([p[0] for p in parts], ignore_index=True)
    outs = pd.concat([p[1] for p in parts], ignore_index=True)
    daily = pd.concat([p[2] for p in parts]).groupby(level=0).sum().sort_index()
    mkt = market_close_features(daily, nk)
    on = pd.read_csv(regime_file, index_col=0, parse_dates=True)["p_down"].shift(1) > 0.5
    rday = on.groupby((~on).cumsum()).cumsum()
    feats = feats.merge(mkt.rename_axis("date").reset_index(), on="date", how="left")
    feats["regime_day"] = rday.reindex(feats["date"]).fillna(0).to_numpy()
    feats["weekday"], feats["month"] = feats["date"].dt.weekday, feats["date"].dt.month
    for col in feats.columns:
        if feats[col].dtype == "float64":
            feats[col] = feats[col].astype("float32")
    return feats, outs


def add_returns(outs: pd.DataFrame, nk: pd.DataFrame, costs: CostModel, switch: float) -> pd.DataFrame:
    """コスト込みの損益（bp）・同じ期間の日経平均（bp）・日経平均に対する上乗せ（乗り換え費用込み、bp）。
    nk は n225_daily の日足（寄りの値段は実際に売買できる目安 open_tradable を使う）。"""
    e, x = outs["entry_price"].to_numpy(float), outs["exit_price"].to_numpy(float)
    bp = ((x - half_spread(x, costs)) / (e + half_spread(e, costs)) - 1) * 1e4
    n_in = np.where(outs["entry"] == "引け", nk["close"].reindex(outs["entry_date"]).to_numpy(),
                    nk["open_tradable"].reindex(outs["entry_date"]).to_numpy())
    n_out = np.where(outs["exit_at"] == "close", nk["close"].reindex(outs["exit_date"]).to_numpy(),
                     nk["open_tradable"].reindex(outs["exit_date"]).to_numpy())
    nk_bp = (n_out / n_in - 1) * 1e4
    return outs.assign(bp=bp.astype("float32"), nk_bp=nk_bp.astype("float32"),
                       ex=(bp - nk_bp - 2 * switch * 1e4).astype("float32"))


def table_rows(g: pd.DataFrame, label: dict) -> list[dict]:
    rows = []
    for per, x in g.groupby("期間"):
        s, sx = summarize(x.assign(entry_date=x["date"])), summarize(x.assign(entry_date=x["date"], bp=x["ex"]))
        rows.append({**label, "期間": per, "取引数": s["取引数"], "平均bp": s["平均bp"], "日ごとbp": s["日ごとbp"],
                     "日ごとt": s["日ごとt"], "上乗せbp": sx["平均bp"], "上乗せ 日ごとbp": sx["日ごとbp"], "上乗せ 日ごとt": sx["日ごとt"]})
    return rows


def old_rule(c: pd.DataFrame) -> pd.Series:
    return (c["high250"] >= OLD_RULE["high250"]) & (c["ret5"] <= OLD_RULE["ret5"]) & (c["sma200"] >= OLD_RULE["sma200"])


def conditions(c: pd.DataFrame, feats=None) -> dict[str, tuple[str, str, float, pd.Series]]:
    """「指標が IS の下位 / 上位 p% の範囲」という条件（境目は IS の分位点）。名前 → (指標, 向き, 境目, 当てはまるか)。"""
    is_c = c[c["期間"] == "IS"]
    out = {}
    for f in feats or FEATURES:
        if f in ("weekday", "month", "at_open"):
            continue
        for p in (0.2, 0.4, 0.6, 0.8):
            lo, hi = float(is_c[f].quantile(p)), float(is_c[f].quantile(1 - p))
            out[f"{f} ≦ {lo:.4g}"] = (f, "≦", lo, c[f] <= lo)
            out[f"{f} ≧ {hi:.4g}"] = (f, "≧", hi, c[f] >= hi)
    return out


def greedy(c: pd.DataFrame, max_rules: int, min_trades: int, feats=None) -> tuple[list[tuple[str, str, float]], pd.DataFrame]:
    """上乗せの IS の日ごとの t 値を最も上げる条件を 1 つずつ足す。"""
    conds = conditions(c, feats)
    is_mask = (c["期間"] == "IS").to_numpy()
    ex, day = c["ex"].to_numpy(), c["date"].to_numpy()
    keep = np.ones(len(c), bool)

    def tval(m: np.ndarray) -> float:
        d = pd.Series(ex[m]).groupby(day[m]).mean()
        return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 1 else -np.inf

    steps = [{"条件": "（-2σ で引けただけ）", **{f"{r['期間']} {k}": v for r in table_rows(c, {}) for k, v in r.items()
                                         if k not in ("期間",)}}]
    cur, chosen = tval(keep & is_mask), []
    for _ in range(max_rules):
        best, best_t = None, cur
        for name, (f, op, v, m) in conds.items():
            mm = keep & m.to_numpy() & is_mask
            if mm.sum() < min_trades:
                continue
            t = tval(mm)
            if t > best_t:
                best, best_t = name, t
        if best is None:
            break
        f, op, v, m = conds[best]
        keep &= m.to_numpy()
        cur = best_t
        chosen.append((f, op, v))
        steps.append({"条件": f"＋ {rule_text([(f, op, v)])}", **{f"{r['期間']} {k}": v2 for r in table_rows(c[keep], {})
                                                                  for k, v2 in r.items() if k not in ("期間",)}})
    return chosen, pd.DataFrame(steps)


def rule_text(rules: list[tuple[str, str, float]]) -> str:
    def val(f, v):
        return f"{v * 100:+.3g}%" if f in PCT_FEATS else f"{v:.3g}"
    return "・".join(f"{LABELS.get(f, f)}{op}{val(f, v)}" for f, op, v in rules)


def apply_rules(c: pd.DataFrame, rules: list[tuple[str, str, float]]) -> pd.Series:
    m = pd.Series(True, index=c.index)
    for f, op, v in rules:
        m &= (c[f] <= v) if op == "≦" else (c[f] >= v)
    return m


def _init_sim(closes, idx_ret, costs, lot, cands):
    _G.update(closes=closes, idx_ret=idx_ret, costs=costs, lot=lot, cands=cands)


def _sim(job):
    label, pick, slots, per, s, e, switch, seed, entry, keep = job
    c = _G["cands"][(label, pick)]
    c = c[(c["entry_date"] >= s) & (c["entry_date"] <= e)]
    ir = _G["idx_ret"][s:e]
    gap = ir["ret"] if entry == "引け" else ir["gap"]   # 引けで買う日は、売った指数にもその日一日の騰落が付く
    r = simulate_slots(c, _G["closes"][s:e], 10_000_000, slots, _G["costs"], _G["lot"], np.random.default_rng(seed),
                       cash_return=ir["ret"], cash_gap=gap, switch_cost=switch, skip_held=True)
    row = {"候補": label, "買う順": pick, "枠": slots, "期間": per, "乗り換え費用": switch, "seed": seed,
           **stats(r.equity, r.trades), "取引数": len(r.trades), "平均bp": r.trades["ret"].mean() * 1e4,
           "投資割合": r.exposure.mean()}
    return row, ((r.trades, r.equity) if keep else None)


def slot_cands(t: pd.DataFrame, pick) -> pd.DataFrame:
    """simulate_slots の候補。買う順は priority（同じ日の中の順位。乱数の順なら全部 0）。"""
    out = t.assign(side=1, priority=0)
    if pick is not None:
        f, asc = pick
        out["priority"] = out.groupby("date")[f].rank(ascending=asc, method="first")
    return out[["code", "date", "entry_date", "exit_date", "side", "entry_price", "exit_price", "reason", "priority"]]


def common_args(parser: argparse.ArgumentParser, config: str) -> None:
    parser.add_argument("--config", default=config)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--switch-cost", type=float, default=0.0005, help="日経平均の売り買いの片道の費用（条件選び・運用の既定）")
    parser.add_argument("--max-rules", type=int, default=4)
    parser.add_argument("--min-trades", type=int, default=1000, help="条件を足した後に IS に残る取引の下限")
    parser.add_argument("--slots", default="5,10,20")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--is-start", default=None,
                        help="IS の始まり（省略時は data_start。2002〜2003 年は 4 本値が同じ日・寄りが安値の日が多く、始値の質が落ちる）")


def market_data(cfg: dict):
    """資金制約ありの運用に使う終値・日経平均の騰落・データの範囲と、IS の始まり（data_start）。"""
    stocks = load_stocks(cfg)
    closes = pd.DataFrame({k: d["close"] for k, d in stocks.items()}).sort_index()
    idx_ret = index_returns("n225", stocks, 1e8, closes.index)
    return closes, idx_ret, universe_label(cfg, len(stocks)), data_start(closes.index)


def run_study(cand: pd.DataFrame, feat_names: dict, *, cfg: dict, args, costs: CostModel, is_end: pd.Timestamp,
              start: pd.Timestamp, closes: pd.DataFrame, idx_ret: pd.DataFrame, universe: str, prefix: str,
              strategy: str, script: str, exits: dict | None = None, picks: dict | None = None) -> None:
    """候補表（cand: 1 行 = 合図 × 建て方 × 手仕舞い方。date / entry / exit / bp / ex と指標の列）から、
    1. 建て方 × 手仕舞い方の成績 2. 指標の 5 分位 3. IS で条件選び 4. 資金制約ありの運用と DB 保存 を行う。
    exits は手仕舞い方の名前 → (種類, 日数)、picks はその日の候補から買う順（指標, 小さい順か）。省略時は引けで買う形のもの。"""
    exits, picks = exits or EXITS, picks or PICKS
    out = Path("results")
    rows = []
    first = cand["breaks5"] == 0
    rule = old_rule(cand)
    for (en, xn), g in cand.groupby(["entry", "exit"], sort=False):
        for f_lab, fm in (("全部", None), ("最初の割れだけ", first)):
            for r_lab, rm in (("ルールなし", None), ("#53 の押し目ルール", rule)):
                m = pd.Series(True, index=g.index)
                if fm is not None:
                    m &= fm[g.index]
                if rm is not None:
                    m &= rm[g.index]
                rows += table_rows(g[m], {"建て": en, "手仕舞い": xn, "割れ": f_lab, "ルール": r_lab})
    base_t = pd.DataFrame(rows)
    base_t.to_csv(out / f"{prefix}_base.csv", index=False)
    print("\n1. 建て方 × 手仕舞い方（コスト込み。上乗せは日経平均に対して）")
    show = base_t.pivot_table(index=["建て", "手仕舞い", "割れ", "ルール"], columns="期間",
                              values=["取引数", "平均bp", "上乗せbp", "上乗せ 日ごとbp", "上乗せ 日ごとt"], sort=False)
    print(show.round(1).to_string())

    sel = base_t[(base_t["期間"] == "IS") & (base_t["割れ"] == "最初の割れだけ") & (base_t["ルール"] == "ルールなし")]
    best = sel.sort_values("上乗せ 日ごとt", ascending=False).iloc[0]
    en, xn = best["建て"], best["手仕舞い"]
    print(f"\nIS の上乗せの日ごとの t 値が最も高い組み合わせ（最初の割れだけ・ルールなしの中で）: 建て {en} / 手仕舞い {xn}")
    c = cand[(cand["entry"] == en) & (cand["exit"] == xn) & first].reset_index(drop=True)

    # 2. 指標の 5 分位
    qrows = []
    is_c = c[c["期間"] == "IS"]
    for f in feat_names:
        if f in DISCRETE and c[f].nunique() <= 25:
            q = c[f]
        else:
            edges = np.unique(is_c[f].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())
            q = pd.Series(np.searchsorted(edges, c[f].to_numpy(), side="right") + 1, index=c.index).where(c[f].notna())
        for (per, qq), g in c.assign(q=q).dropna(subset=["q"]).groupby(["期間", "q"]):
            qrows.append({"指標": f, "期間": per, "区分": qq, "件数": len(g), "上乗せbp": g["ex"].mean(),
                          "上乗せ 日ごとbp": g.groupby("date")["ex"].mean().mean(), "中央値": g[f].median()})
    qt = pd.DataFrame(qrows)
    qt.to_csv(out / f"{prefix}_quintiles.csv", index=False)
    spread = qt.pivot_table(index="指標", columns=["期間", "区分"], values="上乗せbp")
    rng = pd.DataFrame({per: spread[per].max(axis=1) - spread[per].min(axis=1) for per in ("IS", "OOS")})
    corr = {}
    for f in spread.index:
        a, b = spread.loc[f, "IS"], spread.loc[f, "OOS"]
        ok = a.notna() & b.notna()
        corr[f] = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan
    rng["IS と OOS の形の相関"] = pd.Series(corr)
    print("\n2. 指標ごとの 5 分位（離散の指標は値ごと）の上乗せ bp の幅（最大 − 最小）と、IS と OOS で形が似ているか（相関）")
    print(rng.sort_values("IS", ascending=False).round(2).to_string())

    # 3. 条件を IS で選ぶ（1 つずつ足すので、最初の n 個が「条件 n 個」の答え）
    rules, steps = greedy(c, args.max_rules, args.min_trades, feat_names)
    steps.to_csv(out / f"{prefix}_rules.csv", index=False)
    print(f"\n3. 上乗せの IS の日ごとの t 値を最も上げる条件を足していく（IS に {args.min_trades} 件以上残るもの）")
    cols = ["条件"] + [f"{p} {k}" for p in ("IS", "OOS") for k in ("取引数", "平均bp", "上乗せbp", "上乗せ 日ごとbp", "上乗せ 日ごとt")]
    print(steps[cols].round(2).to_string(index=False))

    # 4. 資金制約ありの運用
    old = [(k, "≦" if k == "ret5" else "≧", v) for k, v in OLD_RULE.items()]
    rule_sets = {"最初の割れだけ": [], "＋ #53 の押し目ルール": old}
    for n in sorted({1, 2, len(rules)}):
        if 0 < n <= len(rules):
            rule_sets[f"＋ IS で選んだ条件 {n} 個（{rule_text(rules[:n])}）"] = rules[:n]
    sets = {label: c[apply_rules(c, r)] for label, r in rule_sets.items()}
    periods = {"IS": (start, is_end), "OOS": (is_end + pd.Timedelta(days=1), closes.index[-1])}
    slot_list = [int(s) for s in args.slots.split(",")]
    jobs, scs = [], {}
    for label, t in sets.items():
        for pick_name, pick in picks.items():
            scs[(label, pick_name)] = slot_cands(t, pick)
            for slots in slot_list:
                for per, (s, e) in periods.items():
                    for sw in (args.switch_cost, 0.0001):
                        for seed in range(args.seeds if pick is None else 3):
                            jobs.append((label, pick_name, slots, per, s, e, sw, seed, en, seed == 0))
    print(f"\n4. 資金制約ありの運用 {len(jobs)} 通りを計算しています", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init_sim,
                             initargs=(closes, idx_ret, costs, cfg["backtest"]["lot_size"], scs)) as ex:
        res = list(ex.map(_sim, jobs, chunksize=4))
    grid = pd.DataFrame([r[0] for r in res])
    firsts = {(j[0], j[1], j[2], j[3], j[6]): r[1] for j, r in zip(jobs, res) if r[1] is not None}
    grid.to_csv(out / f"{prefix}_portfolio.csv", index=False)
    nkr = 1 + idx_ret["ret"]
    bench = {per: stats(nkr[s:e].cumprod())["年率"] for per, (s, e) in periods.items()}
    g = grid.groupby(["候補", "買う順", "枠", "期間", "乗り換え費用"], sort=False)
    med = g[["年率", "最大DD", "シャープ", "取引数", "平均bp", "投資割合"]].median().reset_index()
    med["日経平均との差"] = med["年率"] - med["期間"].map(bench)
    print("日経平均の年率: " + " / ".join(f"{p} {v:+.1%}" for p, v in bench.items()))
    wide = med.pivot_table(index=["候補", "買う順", "枠"], columns=["期間", "乗り換え費用"],
                           values=["日経平均との差", "シャープ"], sort=False)
    print(wide.round(3).to_string())

    is_m = med[(med["期間"] == "IS") & (med["乗り換え費用"] == args.switch_cost)].sort_values("シャープ", ascending=False)
    b = is_m.iloc[0]
    pick_sel = (b["候補"], b["買う順"], int(b["枠"]))
    print(f"\nIS のシャープで選んだ設定: {pick_sel[0]} / {pick_sel[1]} / {pick_sel[2]} 枠（乗り換え片道 {args.switch_cost:.2%}）")
    print(med[(med["候補"] == pick_sel[0]) & (med["買う順"] == pick_sel[1]) & (med["枠"] == pick_sel[2])]
          .round(3).to_string(index=False))

    if args.no_save:
        return
    saved = {}
    for per in periods:
        tr, eq = firsts[(pick_sel[0], pick_sel[1], pick_sel[2], per, args.switch_cost)]
        saved[per] = (summarize_run(tr, eq), tr, eq)
    chosen = rule_sets[pick_sel[0]]
    params = {"window": 25, "k": args.k, "exit_k": 1.0 if exits[xn][0] == "z" else None, "entry": en, "exit": xn,
              "first_break": True, "rules": [list(r) for r in chosen], "rules_text": rule_text(chosen),
              "pick": pick_sel[1], "slots": pick_sel[2], "capital": 10_000_000, "idle_in": "n225",
              "switch_cost": args.switch_cost}
    run_cfg = {"data": cfg["data"], "universe": cfg["universe"], "split": {"is_end": str(is_end.date())},
               "backtest": {**cfg["backtest"], "initial_cash": 10_000_000}}
    desc = describe_run("daily", strategy, params, {"universe": universe})
    run_id = ResultStore(args.db).save_run("daily", strategy, params, run_cfg, saved, **desc,
                                           note=f"{script}。建て方・手仕舞い方・条件・枠・買う順は IS で選択。IS は {start.date()} から")
    print(f"\nDB に保存しました（run_id={run_id}）: {desc['name']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    common_args(parser, "config/daily_bb2_long_pullback.yaml")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    out = Path("results")
    out.mkdir(exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    costs = CostModel(**cfg["backtest"].get("costs", {}))
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    nk = n225_daily(pd.Timestamp.today().strftime("%Y-%m-%d"))   # 寄りの値段は実際に売買できる目安（open_tradable）
    closes, idx_ret, universe, start = market_data(cfg)
    start = max(start, pd.Timestamp(args.is_start)) if args.is_start else start
    fpath, opath = out / "bb2c_signals.pkl", out / "bb2c_outcomes.pkl"
    if args.rebuild or not fpath.exists():
        print("候補表を作っています（全貸借銘柄）", flush=True)
        feats, outs = build(cfg, args.k, args.workers, nk, cfg["data"]["regime_file"])
        feats.to_pickle(fpath)
        outs.to_pickle(opath)
    feats, outs = pd.read_pickle(fpath), pd.read_pickle(opath)
    outs = add_returns(outs, nk, costs, args.switch_cost)
    cand = outs.merge(feats, on=["code", "date"], how="left")
    cand = add_extra_features(cand[cand["date"] >= start].reset_index(drop=True), closes.index)
    cand["期間"] = np.where(cand["date"] <= is_end, "IS", "OOS")
    print(f"合図 {cand.drop_duplicates(['code', 'date']).shape[0]:,} 回（{start.date()}〜）/ 取引の組み合わせ {len(cand):,} 件。"
          f"上乗せは日経平均に対して、乗り換え片道 {args.switch_cost:.2%} を引いた値", flush=True)
    run_study(cand, FEATURES, cfg=cfg, args=args, costs=costs, is_end=is_end, start=start, closes=closes, idx_ret=idx_ret,
              universe=universe, prefix="bb2c", strategy="bb_close", script="scripts/study_bb2_close.py")


if __name__ == "__main__":
    main()
