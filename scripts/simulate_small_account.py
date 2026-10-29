"""少額（既定 200 万円）の 1 つの口座で、DB #73（終値の -2σ 割れを翌日の寄りで買う）と DB #60（-3σ 割れの銘柄が多い日に
日経平均を買い増す）を一緒に運用したらどうなるかを調べる。

    python scripts/simulate_small_account.py                 # 200 万円。結果を DB に保存
    python scripts/simulate_small_account.py --capital 3000000 --no-save
    python scripts/simulate_small_account.py --capital 1000000 --slots 4 --index-weight 0.75 --n225-entry next_open --index-k 2.5
                                                              # 買い増しの合図を「日経平均の終値が −2.5σ より下」に替える
    python scripts/simulate_small_account.py ...（同じ決まり） --oos-end 2023-03-31 --replace 85
                                                              # OOS を 2023-03 までにして入れ替える（2023-04 以降は #85 PT）

口座の形: 資金はいつも日経平均（ETF）で持ち、#73 の買いの日だけ寄りで売って個別株に乗り換える（#73 と同じ）。
  #60 の合図の間は、口座の資産と同じ額の日経平均を信用で買い増す（#60 と同じ。金利 年 rate、売り買い片道 switch_cost）。
  --index-weight w を 1 より小さくすると、個別株に入っていない資金のうち w だけを日経平均、残りを現金（年 --cash-rate）で持ち、
  #60 の買い増しも資産 × w にする（まず現金で買い、足りない分だけ信用で借りる。金利は借りた分だけ）。#73 の買いは日経平均と現金を
  同じ割合で使う（日経平均を売る費用は w 倍）。現金の割合は日々 w に保つとみなす（ペーパートレードでは月に 1 回そろえる）。
  建て・手仕舞いの値段は DB の実行と同じ。#73 の手仕舞い「終値が -1σ 以上で引けたら大引けで売る」は、その日の -1σ になる値段を
  前日までの終値から計算して引け指値（売り）を出せば、そのとおりに約定する。
  #60 の「合図の日の大引けで買い増す」は大引けの直前に場中の値で合図を数える必要があるので、
  大引けの後に判断して翌日の寄りで買い増す形（ETF の寄りの値段）も比べる。
売買単位: 個別株は実際の 100 株単位。検証の値段は分割調整済みなので、後の株式分割の比率で実際の値段に戻し、
  100 株が 1 枠の金額に収まるかを判定する（分割調整済みの値段のままだと、昔の値がさの銘柄も買えたことになる）。
  2018 年 10 月より前は 1,000 株単位の銘柄も多かったが、ここでは今の 100 株単位で数える（併合はその前後で 1 単元の金額が
  変わらないので、併合の比率は戻さない）。日経平均の ETF（1321 は 1 口 650 円前後）の端数は無視する。
比べる相手: 日経平均を持ちっぱなしにした場合（配当なし）。月ごとの差とその t 値も出す。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_backtest import load_stocks
from simulate_portfolio import data_start, index_returns, n225_daily, stats
from study_bb2_close import PICKS, add_extra_features, add_returns, apply_rules, slot_cands
from study_index_crash import breadth_count, excess, load_n225, overlay, trades_close, zscore
from trading.backtest import CostModel, summarize as summarize_run
from trading.backtest.slot_sim import simulate_slots
from trading.data.jpx import split_history
from trading.store import DEFAULT_DB, ResultStore, describe_run, universe_label


def split_lot(cands: pd.DataFrame, lot: int = 100) -> pd.Series:
    """実際の lot 株が、分割調整済みの値段で何株にあたるか（建てた日より後の株式分割の比率の積 × lot）。"""
    splits = split_history(sorted(cands["code"].unique()))
    out = np.full(len(cands), float(lot))
    for i, (code, d) in enumerate(zip(cands["code"], cands["entry_date"])):
        s = splits.get(code)
        if s is not None and len(s):
            ratios = s[(s.index > d) & (s > 1)]
            out[i] *= float(ratios.prod()) if len(ratios) else 1.0
    return pd.Series(out, index=cands.index)


def close_candidates(p: dict, cfg: dict, nk: pd.DataFrame, costs: CostModel, start, calendar) -> pd.DataFrame:
    """DB #73 と同じ取引の候補（scripts/study_bb2_close.py の候補表から、建て方・手仕舞い方・最初の割れ・条件で絞る）。"""
    outs = pd.read_pickle("results/bb2c_outcomes.pkl")
    outs = outs[(outs["entry"] == p["entry"]) & (outs["exit"] == p["exit"])]
    outs = add_returns(outs, nk, costs, p["switch_cost"])
    cand = outs.merge(pd.read_pickle("results/bb2c_signals.pkl"), on=["code", "date"], how="left")
    cand = add_extra_features(cand[cand["date"] >= start].reset_index(drop=True), calendar)
    cand = cand[cand["breaks5"] == 0]
    cand = cand[apply_rules(cand, [tuple(r) for r in p["rules"]])]
    return slot_cands(cand, PICKS[p["pick"]]).reset_index(drop=True)


def index_trades(p: dict, stocks: dict, regime_file: str, entry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DB #60 と同じ合図（C: -3σ に触れた貸借銘柄が min_stocks 以上）か、日経平均の終値が -kσ より下の合図（B）で、
    前日の p_down が long_min_p_down を超えた日の日経平均の取引。
    entry = close なら合図の日の大引け、next_open なら翌日の寄り（ETF の寄りの値段 open_tradable）で買う。"""
    nk = load_n225(regime_file)
    if p.get("signal", "C") == "B":
        hit = zscore(nk["close"]) < -p["k"]
    else:
        hit = breadth_count(stocks).reindex(nk.index).fillna(0) >= p["min_stocks"]
    sig = hit & (nk["p_down_prev"] > p["long_min_p_down"])
    if entry == "next_open":
        tradable = n225_daily(str(nk.index[-1].date()))["open_tradable"].reindex(nk.index)
        nk = nk.assign(open=tradable.fillna(nk["close"]))
    return nk, trades_close(nk, sig, entry, p["max_hold"])


def run_close(cands, closes, idx_ret, capital, slots, costs, switch, s, e, seed=0, weight=1.0, cash_rate=0.0):
    """個別株の手法。待機中の資金は weight だけ日経平均、残りは現金（年 cash_rate）。"""
    c = cands[(cands["entry_date"] >= s) & (cands["entry_date"] <= e)]
    ir = idx_ret[s:e]
    return simulate_slots(c, closes[s:e], capital, slots, costs, 100, np.random.default_rng(seed),
                          cash_return=weight * ir["ret"] + (1 - weight) * cash_rate / 245, cash_gap=weight * ir["gap"],
                          switch_cost=weight * switch, skip_held=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=2_000_000)
    parser.add_argument("--close-run", type=int, default=73, help="個別株の手法（bb_close）の DB の実行")
    parser.add_argument("--index-run", type=int, default=60, help="日経平均の買い増し（index_overlay）の DB の実行")
    parser.add_argument("--slots", default="3,4,5", help="比べる枠の数（保存するのは IS のシャープが最も高いもの）")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n225-entry", choices=["close", "next_open"], default=None,
                        help="#60 の買い増しの建て方（省略時は両方を比べ、DB には大引けの後に判断できる next_open を保存）")
    parser.add_argument("--index-weight", type=float, default=1.0,
                        help="個別株に入っていない資金のうち日経平均で持つ割合（残りは現金。#60 の買い増しもこの割合に比例）")
    parser.add_argument("--cash-rate", type=float, default=0.0, help="現金の利息（年）")
    parser.add_argument("--index-k", type=float, default=None,
                        help="#60 の合図を「日経平均の終値が −kσ より下」（合図 B。scripts/study_holdout.py で IS から選んだ形）に替える")
    parser.add_argument("--oos-end", default=None,
                        help="OOS の終わり（省略時はデータの最後の日。ペーパートレードを動かす決まりは 2023-03-31 で、その後は PT）")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--replace", type=int, default=None,
                        help="新しく保存せず、この DB の実行の成績を入れ替える（同じ決まりで計算し直すとき。--slots は 1 つだけ）")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    w = args.index_weight

    store = ResultStore(args.db)
    r73, r60 = store.run(args.close_run), store.run(args.index_run)
    p73, p60 = r73["params"], r60["params"]
    idx_label, idx_name = f"#{args.index_run}", r60["name"]
    if args.index_k is not None:   # 合図だけ B に替える（最長の日数・局面・金利・費用は --index-run のまま）
        p60 = {**{k: v for k, v in p60.items() if k != "min_stocks"}, "signal": "B", "k": args.index_k,
               "selected_by": "scripts/study_holdout.py"}
        idx_label = f"日経平均の−{args.index_k:g}σ"
        idx_name = f"日経平均の終値が−{args.index_k:g}σより下の日に買い増し（#{args.index_run} の合図を替えたもの）"
    cfg = r73["config"]
    costs = CostModel(**cfg["backtest"]["costs"])
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    stocks = load_stocks(cfg)
    closes = pd.DataFrame({k: d["close"] for k, d in stocks.items()}).sort_index()
    start = data_start(closes.index)
    idx_ret = index_returns("n225", stocks, 1e8, closes.index)
    nk_open = n225_daily(str(closes.index[-1].date()))
    oos_end = min(pd.Timestamp(args.oos_end), closes.index[-1]) if args.oos_end else closes.index[-1]
    periods = {"IS": (start, is_end), "OOS": (is_end + pd.Timedelta(days=1), oos_end)}
    print(f"#{args.close_run} {r73['name']}\n{idx_label} {idx_name}\n資金 {args.capital:,.0f} 円 / "
          f"IS {start.date()}〜{is_end.date()} / OOS 〜{oos_end.date()}" +
          (f"。待機中の資金の {w:.0%} を日経平均・残りは現金（年 {args.cash_rate:.1%}）" if w < 1 else ""), flush=True)

    cands = close_candidates(p73, cfg, nk_open, costs, start, closes.index)
    cands["lot"] = split_lot(cands)
    raw_price = cands["entry_price"] * cands["lot"]   # 実際の 100 株の金額
    print(f"#{args.close_run} の候補 {len(cands):,} 件。実際の 100 株の金額: 中央値 {raw_price.median():,.0f} 円・"
          f"40 万円超 {(raw_price > 4e5).mean():.0%}・67 万円超 {(raw_price > 6.67e5).mean():.0%}", flush=True)

    entries = [args.n225_entry] if args.n225_entry else ["close", "next_open"]
    idx = {en: index_trades(p60, stocks, cfg["data"]["regime_file"], en) for en in entries}
    nk = idx[entries[0]][0]
    bench = {per: stats(nk["close"][s:e] / nk["close"][s:e].iloc[0])["年率"] for per, (s, e) in periods.items()}
    print("日経平均の年率: " + " / ".join(f"{per} {v:+.1%}" for per, v in bench.items()))

    def row(label, per, eq, tr, extra=None):
        return {"運用": label, "期間": per, **stats(eq, tr), **excess(eq, nk), "取引数": len(tr), **(extra or {})}

    rows, keep = [], {}
    slot_list = [int(s) for s in args.slots.split(",")]
    for per, (s, e) in periods.items():
        for slots in slot_list:
            for lot_label, cc in (("実際の100株", cands), ("分割調整済みの100株", cands.drop(columns="lot"))):
                for seed in range(args.seeds):
                    r = run_close(cc, closes, idx_ret, args.capital, slots, costs, p73["switch_cost"], s, e, seed, w, args.cash_rate)
                    rows.append(row(f"#{args.close_run} だけ・{slots}枠・{lot_label}", per, r.equity, r.trades,
                                    {"見送り（買えない）": r.skipped["price"], "見送り（枠）": r.skipped["full"],
                                     "投資割合": r.exposure.mean(), "seed": seed}))
                    if lot_label == "実際の100株":
                        keep[("close", slots, per, seed)] = r
        for en, (nk_e, t60) in idx.items():
            t = t60[(t60["entry_date"] >= s) & (t60["entry_date"] <= e)]
            only = None if w == 1 else (w * idx_ret["ret"] + (1 - w) * args.cash_rate / 245)
            eq, tr = overlay(nk_e, t, s, e, args.capital, p60["extra"] * w, p60["rate"], p60["switch_cost"], base_ret=only,
                             cash_frac=None if w == 1 else pd.Series(1 - w, index=idx_ret.index))
            rows.append(row(f"{idx_label} だけ・{'大引け' if en == 'close' else '翌日の寄り'}で買い増す", per, eq, tr))
            for slots in slot_list:
                for seed in range(args.seeds):
                    r = keep[("close", slots, per, seed)]
                    base = r.equity.pct_change().fillna(0.0)
                    cash_frac = None if w == 1 else (1 - w) * (1 - r.exposure.reindex(base.index).ffill().fillna(0.0))
                    eq, tr = overlay(nk_e, t, s, e, args.capital, p60["extra"] * w, p60["rate"], p60["switch_cost"], base_ret=base,
                                     cash_frac=cash_frac)
                    label = f"両方・{slots}枠・{idx_label} は{'大引け' if en == 'close' else '翌日の寄り'}"
                    both = pd.concat([r.trades, tr], ignore_index=True).sort_values("entry_date").reset_index(drop=True)
                    rows.append(row(label, per, eq, both, {"seed": seed, "見送り（買えない）": r.skipped["price"],
                                                           "見送り（枠）": r.skipped["full"]}))
                    keep[("both", en, slots, per, seed)] = (eq, both)
    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/small_account.csv", index=False)
    med = table.groupby(["運用", "期間"], sort=False).median(numeric_only=True).reset_index()
    med["日経平均との差"] = med["年率"] - med["期間"].map(bench)
    cols = ["年率", "日経平均との差", "月ごとの差（年換算）", "差の t 値", "最大DD", "シャープ", "最悪の年", "取引数",
            "見送り（買えない）", "見送り（枠）", "投資割合"]
    wide = med.pivot_table(index="運用", columns="期間", values=cols, sort=False)
    print(f"\n乱数 {args.seeds} 回の中央値（#{args.close_run} は同じ日の候補を 5 日の騰落の小さい順に建てるので、乱数は同順位のときだけ効く）")
    print(wide.reindex(columns=pd.MultiIndex.from_product([cols, ["IS", "OOS"]])).round(3).to_string())

    if args.no_save:
        return
    en = args.n225_entry or "next_open"
    is_both = med[(med["期間"] == "IS") & med["運用"].str.startswith("両方") &
                  med["運用"].str.endswith("大引け" if en == "close" else "翌日の寄り")]
    best = is_both.sort_values("シャープ", ascending=False).iloc[0]
    slots = int(best["運用"].split("・")[1].removesuffix("枠"))
    print(f"\nIS のシャープで選んだ枠: {slots}（{idx_label} は{'大引け' if en == 'close' else '翌日の寄り'}で買い増す）")
    saved = {}
    for per in periods:
        eq, tr = keep[("both", en, slots, per, 0)]
        saved[per] = (summarize_run(tr, eq), tr, eq)
    close_part = {k: v for k, v in p73.items() if k not in ("capital", "slots")}
    index_part = {k: v for k, v in p60.items() if k != "capital"}
    params = {"window": 25, "k": p73["k"], "exit_k": 1.0, "capital": args.capital, "slots": slots, "lot": "実際の100株",
              **({"index_weight": w, "cash_rate": args.cash_rate} if w < 1 else {}),
              "parts": [{"strategy": "bb_close", "run": args.close_run, **close_part},
                        {"strategy": "index_overlay", "run": args.index_run, "entry": en, **index_part}]}
    split = {"is_end": str(is_end.date()), **({"oos_end": str(oos_end.date())} if args.oos_end else {})}
    run_cfg = {"data": cfg["data"], "universe": cfg["universe"], "split": split,
               "backtest": {**cfg["backtest"], "initial_cash": args.capital}, "regime_file": cfg["data"]["regime_file"]}
    if args.replace is not None:
        old = store.run(args.replace)
        if old["strategy"] != "small_account" or old["params"] != json.loads(json.dumps(params)):
            raise SystemExit(f"DB #{args.replace} は決まりが違うので入れ替えません（資金・枠・日経平均の割合・{idx_label} の建て方をそろえる）")
        store.replace_periods(args.replace, saved)
        if args.oos_end:   # OOS の期間だけを変えた（決まりは同じ）
            store.update_config(args.replace, {**old["config"], "split": {**old["config"]["split"], **split}})
            note = f"。2026-10-29 に OOS を {oos_end.date()} までにした（その後はペーパートレード #{args.replace} PT の期間）"
        else:
            note = "。2026-10-26 に寄りの値段（trading.data.index_open）を直して計算し直した（決まりは同じ）"
        if note not in old["note"]:
            store.update_note(args.replace, old["note"] + note)
        print(f"DB #{args.replace} の成績を入れ替えました: {old['name']}")
        return
    desc = describe_run("daily", "small_account", params, {"universe": universe_label(cfg, len(stocks))})
    run_id = store.save_run("daily", "small_account", params, run_cfg, saved, **desc,
                            note=f"scripts/simulate_small_account.py。#{args.close_run} と {idx_label} を 1 つの口座で運用。"
                                 "枠は IS のシャープで選択。個別株は実際の 100 株単位（分割前の値段）で買えるかを判定")
    print(f"DB に保存しました（run_id={run_id}）: {desc['name']}")


if __name__ == "__main__":
    main()
