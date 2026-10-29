"""2023 年 4 月以降を「取り置き」（ペーパートレードと同じく、手法を選ぶのに使わない期間）にして、今運用できる 2 つの手法を学習し直す。

    python scripts/study_holdout.py
    python scripts/study_holdout.py --holdout-start 2023-04-01 --capitals 1000000,3000000

期間: IS 2001-07-18〜2015-12-31（学習。局面モデルもここまでで学習済み）/ OOS 2016-01-01〜取り置きの前日（確かめ）/
      取り置き --holdout-start〜（最後に 1 回だけ見る）
手法（少ない資金で実際に注文を出せるもの）:
  個別株: 終値が −2σ 以下で引けた銘柄（最初の割れだけ）を翌日の寄りに成行で買い、大引けで売る（#73 の形。候補表は study_bb2_close.py）
  日経平均: 合図の翌日の寄りで買い増し、終値が −1σ に戻った日か最長の日数目の大引けで売る（#60 の形）。
    合図は B（日経平均の終値が −kσ 以下）と C（−3σ に触れた貸借銘柄が K 以上）、前日の下げ相場の確率で絞るかどうか
1. 資金の上限なし: 合図をすべて取ったとみなした 1 取引の成績で学習する。
   個別株: 手仕舞い方を IS の上乗せ（日経平均に対する、乗り換え費用込み）の日ごとの t 値で選び、条件を IS で 1 つずつ足す
   （study_bb2_close.greedy）。条件の数は、IS だけで決めたもの（足せるだけ）と、OOS の t 値が最も高いものの 2 通りを出す。
   日経平均: 合図の日すべての買い増し分の 1 取引（費用込み。日経平均を 1 倍持つのに対する上乗せ）の IS の日ごとの t 値で選ぶ。
2. 資金の上限あり: 1 で学習した決まりを、資金（--capitals）・実際の 100 株単位・枠で動かす。
   まず個別株だけ（待機中は日経平均）で、条件の数・買う順・枠を資金ごとに IS のシャープで選ぶ。
   次に日経平均の買い増しを足した口座を、待機中の資金のうち日経平均で持つ割合 50 / 75 / 100% で動かす（買い増しも資産 × 割合）。
   比べる相手は日経平均の持ちっぱなしと、今の決まり（DB #83 と同じ設定をこの期間で動かしたもの）。
結果は results/holdout_*.csv。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_stocks
from simulate_portfolio import data_start, index_returns, n225_daily, stats
from simulate_small_account import run_close, split_lot
from study_bb2_close import FEATURES, PICKS, add_extra_features, add_returns, apply_rules, greedy, rule_text, slot_cands, table_rows
from study_bb2_signal_t import episodes, t_of
from study_index_crash import breadth_count, excess, load_n225, overlay, trades_close, zscore
from study_index_signal_t import all_signal_trades
from trading.backtest import CostModel
from trading.store import DEFAULT_DB, ResultStore

ENTRY = "翌日の寄り"
PERIOD_NAMES = ("IS", "OOS", "取り置き")
_G: dict = {}


def index_signals(nk: pd.DataFrame, br: pd.Series) -> dict:
    """日経平均の買い増しの合図の候補。(合図, 値, 最長, 局面) → 合図の日の真偽。"""
    p_on = nk["p_down_prev"] > 0.5
    z = zscore(nk["close"])
    out = {}
    for hold in (5, 10):
        for regime in ("局面で絞らない", "下げ局面だけ"):
            for k in (2.0, 2.5, 3.0):
                sig = z < -k
                out[("B", f"k={k:g}", hold, regime)] = sig & p_on if regime == "下げ局面だけ" else sig
            for kk in (10, 30, 100):
                sig = br.reindex(nk.index).fillna(0) >= kk
                out[("C", f"K={kk}", hold, regime)] = sig & p_on if regime == "下げ局面だけ" else sig
    return out


def _init(closes, idx_ret, costs, switch, cands):
    _G.update(closes=closes, idx_ret=idx_ret, costs=costs, switch=switch, cands=cands)


def _stock_job(job):
    """個別株だけの口座（待機中は日経平均）。"""
    n_rules, pick, slots, capital, per, s, e = job
    r = run_close(_G["cands"][(n_rules, pick)], _G["closes"], _G["idx_ret"], capital, slots, _G["costs"], _G["switch"], s, e)
    return {"条件の数": n_rules, "買う順": pick, "枠": slots, "資金": capital, "期間": per, **stats(r.equity, r.trades),
            "取引数": len(r.trades), "見送り（買えない）": r.skipped["price"], "最後の資産の倍率": r.equity.iloc[-1] / capital}


def account(cands, closes, idx_ret, costs, switch, nk_e, t_idx, capital, slots, w, s, e, rate):
    """個別株 ＋ 日経平均の買い増しの口座（simulate_small_account.py と同じ形）。t_idx が None なら買い増しなし。"""
    r = run_close(cands, closes, idx_ret, capital, slots, costs, switch, s, e, 0, w, 0.0)
    if t_idx is None:
        return r.equity, r.trades, r
    base = r.equity.pct_change().fillna(0.0)
    cash_frac = (1 - w) * (1 - r.exposure.reindex(base.index).ffill().fillna(0.0))
    tt = t_idx[(t_idx["entry_date"] >= s) & (t_idx["entry_date"] <= e)]
    eq, tr = overlay(nk_e, tt, s, e, capital, w, rate, switch, base_ret=base, cash_frac=cash_frac)
    return eq, pd.concat([r.trades, tr]), r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-start", default="2023-04-01")
    parser.add_argument("--capitals", default="1000000,3000000")
    parser.add_argument("--slots", default="2,3,4,5")
    parser.add_argument("--max-rules", type=int, default=4)
    parser.add_argument("--min-trades", type=int, default=1000, help="条件を足した後に IS に残る取引の下限")
    parser.add_argument("--min-signals", type=int, default=20, help="日経平均の合図で、IS に要る合図の日の数")
    parser.add_argument("--rate", type=float, default=0.028, help="信用買いの金利（年）")
    parser.add_argument("--switch-cost", type=float, default=0.0005)
    parser.add_argument("--current-run", type=int, default=83, help="比べる今の決まり（small_account）")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    out = Path("results")
    out.mkdir(exist_ok=True)

    store = ResultStore(args.db)
    cur = store.run(args.current_run)["params"]
    cur73, cur60 = cur["parts"]
    cfg = store.run(cur73["run"])["config"]
    costs = CostModel(**cfg["backtest"]["costs"])
    is_end, hold_start = pd.Timestamp(cfg["split"]["is_end"]), pd.Timestamp(args.holdout_start)
    stocks = load_stocks(cfg)
    closes = pd.DataFrame({k: d["close"] for k, d in stocks.items()}).sort_index()
    start, end = data_start(closes.index), closes.index[-1]
    idx_ret = index_returns("n225", stocks, 1e8, closes.index)
    nk_open = n225_daily(str(end.date()))
    periods = {"IS": (start, is_end), "OOS": (is_end + pd.Timedelta(days=1), hold_start - pd.Timedelta(days=1)),
               "取り置き": (hold_start, end)}

    def label(d: pd.Series) -> np.ndarray:
        return np.select([d <= is_end, d < hold_start], ["IS", "OOS"], "取り置き")

    print("期間: " + " / ".join(f"{p} {s.date()}〜{e.date()}" for p, (s, e) in periods.items()), flush=True)
    nk = load_n225(cfg["data"]["regime_file"])
    bench = {}
    for per, (s, e) in periods.items():
        n = nk["close"][s:e]
        bench[per] = {**stats(n / n.iloc[0]), "最後の資産の倍率": n.iloc[-1] / n.iloc[0]}
    print("日経平均の持ちっぱなし（配当なし）: " + " / ".join(
        f"{p} 年率 {b['年率']:+.1%}・最大の下げ {b['最大DD']:.0%}・シャープ {b['シャープ']:.2f}" for p, b in bench.items()))

    # ---- 1. 資金の上限なし: 個別株 ----
    feats, outs = pd.read_pickle("results/bb2c_signals.pkl"), pd.read_pickle("results/bb2c_outcomes.pkl")
    outs = add_returns(outs[outs["entry"] == ENTRY], nk_open, costs, args.switch_cost)
    cand = outs.merge(feats, on=["code", "date"], how="left")
    cand = add_extra_features(cand[cand["date"] >= start].reset_index(drop=True), closes.index)
    cand = cand[cand["breaks5"] == 0].reset_index(drop=True)
    cand["期間"] = label(cand["date"])
    rows = []
    for xn, g in cand.groupby("exit", sort=False):
        rows += table_rows(g, {"手仕舞い": xn})
    ex_t = pd.DataFrame(rows)
    ex_t.to_csv(out / "holdout_exits.csv", index=False)
    print("\n1. 資金の上限なし（合図をすべて取る）: 個別株の手仕舞い方（翌日の寄りで買う・最初の割れだけ。上乗せは日経平均に対して）")
    print(ex_t.pivot_table(index="手仕舞い", columns="期間", values=["取引数", "上乗せbp", "上乗せ 日ごとt"], sort=False)
          .reindex(columns=PERIOD_NAMES, level=1).round(2).to_string())
    xn = ex_t[ex_t["期間"] == "IS"].sort_values("上乗せ 日ごとt", ascending=False).iloc[0]["手仕舞い"]
    c = cand[cand["exit"] == xn].reset_index(drop=True)
    print(f"IS の t 値で選んだ手仕舞い方: {xn}", flush=True)

    rules, steps = greedy(c, args.max_rules, args.min_trades, FEATURES)
    steps.insert(0, "条件の数", range(len(steps)))
    steps.to_csv(out / "holdout_rules.csv", index=False)
    print(f"\n条件を IS で 1 つずつ足す（上乗せの IS の日ごとの t 値を最も上げるもの。IS に {args.min_trades} 件以上残る）")
    cols = ["条件の数", "条件"] + [f"{p} {k}" for p in PERIOD_NAMES for k in ("取引数", "上乗せbp", "上乗せ 日ごとt")]
    print(steps[cols].round(2).to_string(index=False))
    n_oos = int(steps.loc[steps["OOS 上乗せ 日ごとt"].idxmax(), "条件の数"])
    print(f"IS で選んだ条件: {rule_text(rules)}（{len(rules)} 個）。OOS の t 値が最も高いのは {n_oos} 個", flush=True)

    # ---- 1. 資金の上限なし: 日経平均の買い増し ----
    br = breadth_count(stocks)
    tradable = nk_open["open_tradable"].reindex(nk.index)
    nk_e = nk.assign(open=tradable.fillna(nk["close"]))
    drift = nk["close"].pct_change()
    drift = {p: drift[s:e].mean() for p, (s, e) in periods.items()}
    sigs = index_signals(nk, br)
    irows = []
    for key, sig in sigs.items():
        sig = sig[sig.index >= start]
        t = all_signal_trades(nk, nk_e["open"], sig, "next_open", key[2])
        if t.empty:
            continue
        days = (t["exit_date"] - t["entry_date"]).dt.days
        t["net"] = ((t["exit_price"] / t["entry_price"] - 1) - 2 * args.switch_cost - args.rate * days / 365) * 1e4
        t["期間"] = label(t["date"])
        t["adj"] = t["net"] - t["期間"].map(drift) * np.maximum(t["hold"], 1) * 1e4
        t["episode"] = episodes(t["date"], nk.index)
        for per in PERIOD_NAMES:
            g = t[t["期間"] == per]
            ep = g.groupby("episode")["net"].mean()
            irows.append({"合図": key[0], "値": key[1], "最長": key[2], "局面": key[3], "期間": per, "合図の日": len(g),
                          "まとまり": len(ep), "1取引bp": g["net"].mean(), "日ごとt": t_of(g["net"]), "まとまりごとt": t_of(ep),
                          "ふだんとの差bp": g["adj"].mean(), "勝率": (g["net"] > 0).mean()})
    it = pd.DataFrame(irows)
    it.to_csv(out / "holdout_index_signals.csv", index=False)
    wide = it.pivot_table(index=["合図", "値", "最長", "局面"], columns="期間", values=["合図の日", "1取引bp", "日ごとt"], sort=False)
    is_rank = it[(it["期間"] == "IS") & (it["合図の日"] >= args.min_signals)].sort_values("日ごとt", ascending=False)
    print("\n1. 資金の上限なし: 日経平均の買い増し（合図の日すべてで翌日の寄りに買い増したとみなす。費用・金利込み）。IS の t 値の高い順")
    order = [tuple(r) for r in is_rank[["合図", "値", "最長", "局面"]].to_numpy()]
    print(wide.reindex(order).reindex(columns=PERIOD_NAMES, level=1).round(2).to_string())
    best = is_rank.iloc[0]
    ikey = (best["合図"], best["値"], int(best["最長"]), best["局面"])
    print(f"IS の t 値で選んだ合図: {ikey}", flush=True)
    t_idx = trades_close(nk_e, sigs[ikey][sigs[ikey].index >= start], "next_open", ikey[2])
    cur_key = ("C", f"K={cur60['min_stocks']}", cur60["max_hold"], "下げ局面だけ" if cur60.get("long_min_p_down") else "局面で絞らない")
    t_cur = trades_close(nk_e, sigs[cur_key][sigs[cur_key].index >= start], "next_open", cur_key[2])

    # ---- 2. 資金の上限あり: 個別株だけ ----
    c["lot"] = split_lot(c)
    scs = {}
    for n in range(len(rules) + 1):
        sub = c[apply_rules(c, rules[:n])]
        for pick_name, pick in PICKS.items():
            sc = slot_cands(sub, pick)
            sc["lot"] = sub["lot"].to_numpy()
            scs[(n, pick_name)] = sc
    c_cur = c if cur73["exit"] == xn else cand[cand["exit"] == cur73["exit"]].reset_index(drop=True)
    if "lot" not in c_cur:
        c_cur["lot"] = split_lot(c_cur)
    cur_sub = c_cur[apply_rules(c_cur, [tuple(r) for r in cur73["rules"]])]
    sc_cur = slot_cands(cur_sub, PICKS[cur73["pick"]])
    sc_cur["lot"] = cur_sub["lot"].to_numpy()
    capitals = [float(x) for x in args.capitals.split(",")]
    codes = sorted(c["code"].unique())
    closes = closes[[k for k in codes if k in closes.columns]]
    jobs = [(n, pick, slots, cap, per, s, e) for (n, pick) in scs for slots in [int(x) for x in args.slots.split(",")]
            for cap in capitals for per, (s, e) in periods.items()]
    print(f"\n2. 資金の上限あり: 個別株だけの口座 {len(jobs)} 通りを計算しています", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(closes, idx_ret, costs, args.switch_cost, scs)) as ex:
        grid = pd.DataFrame(list(ex.map(_stock_job, jobs, chunksize=4)))
    grid.to_csv(out / "holdout_stock_accounts.csv", index=False)
    chosen = {}
    for cap in capitals:
        g = grid[grid["資金"] == cap]
        wide_g = g.pivot_table(index=["条件の数", "買う順", "枠"], columns="期間", values=["シャープ", "年率", "最大DD"], sort=False)
        wide_g = wide_g.sort_values(("シャープ", "IS"), ascending=False)
        print(f"\n資金 {cap / 1e4:,.0f} 万円: IS のシャープの上位 10（取り置きは参考）")
        print(wide_g.head(10).reindex(columns=PERIOD_NAMES, level=1).round(3).to_string())
        top = wide_g.index[0]
        chosen[cap] = top
        print(f"IS のシャープで選んだ設定: 条件 {top[0]} 個（{rule_text(rules[:top[0]]) or 'なし'}）・{top[1]}・{top[2]} 枠")

    # ---- 2. 資金の上限あり: 日経平均の買い増しを足した口座 ----
    rows = []

    def add(label_, cap, eq, tr, per, extra=None):
        st = stats(eq, tr)
        rows.append({"口座": label_, "資金": cap, "期間": per, "最後の資産の倍率": eq.iloc[-1] / eq.iloc[0], "年率": st["年率"],
                     "最大DD": st["最大DD"], "シャープ": st["シャープ"], "最悪の年": st["最悪の年"],
                     "日経平均との差 t": excess(eq, nk)["差の t 値"], **(extra or {})})

    for cap in capitals:
        n, pick, slots = chosen[cap]
        sc = scs[(n, pick)]
        for per, (s, e) in periods.items():
            for w in (0.5, 0.75, 1.0):
                eq, tr, r = account(sc, closes, idx_ret, costs, args.switch_cost, nk_e, t_idx, cap, slots, w, s, e, args.rate)
                add(f"学習し直した決まり・日経平均 {w:.0%}", cap, eq, tr, per, {"個別株の取引": len(r.trades), "見送り": r.skipped["price"]})
            eq, tr, r = account(sc, closes, idx_ret, costs, args.switch_cost, nk_e, None, cap, slots, 1.0, s, e, args.rate)
            add("学習し直した個別株だけ（待機中は日経平均）", cap, eq, tr, per, {"個別株の取引": len(r.trades), "見送り": r.skipped["price"]})
            n_ = nk_e["close"][s:e]
            eq, tr = overlay(nk_e, t_idx[(t_idx["entry_date"] >= s) & (t_idx["entry_date"] <= e)], s, e, cap, 1.0, args.rate,
                             args.switch_cost, base_ret=n_.pct_change().fillna(0.0))
            add("学習し直した日経平均の買い増しだけ（いつも日経平均）", cap, eq, tr, per)
            eq, tr, r = account(sc_cur, closes, idx_ret, costs, args.switch_cost, nk_e, t_cur, cap, cur["slots"],
                                cur["index_weight"], s, e, args.rate)
            add(f"今の決まり（#{args.current_run} と同じ設定）", cap, eq, tr, per, {"個別株の取引": len(r.trades), "見送り": r.skipped["price"]})
            add("日経平均の持ちっぱなし", cap, n_ / n_.iloc[0] * cap, None, per)
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "holdout_accounts.csv", index=False)
    for cap in capitals:
        print(f"\n資金 {cap / 1e4:,.0f} 万円の口座（取り置きは {hold_start.date()} に {cap / 1e4:,.0f} 万円で始めた場合）")
        a = acc[acc["資金"] == cap]
        print(a.pivot_table(index="口座", columns="期間", values=["最後の資産の倍率", "年率", "最大DD", "シャープ", "日経平均との差 t"], sort=False)
              .reindex(columns=PERIOD_NAMES, level=1).round(3).to_string())


if __name__ == "__main__":
    main()
