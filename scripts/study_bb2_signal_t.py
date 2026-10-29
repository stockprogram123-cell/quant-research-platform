"""#73（終値 -2σ 割れを翌日の寄りで買う）の合図そのものに中身があるかを、口座で買えなかった候補も含めた全候補で測り直す。

    python scripts/study_bb2_signal_t.py            # DB #73 の条件で測る
    python scripts/study_bb2_signal_t.py --run 73

口座の t 値（月末の資産の騰落と日経平均の差）は「この口座が日経平均に勝つか」で、OOS では 124 取引しか使っていない。
ここでは条件を新しく選ばず、#73 の条件を満たした全候補（最初の割れ・翌日の寄りで買い・-1σ で引け・最長 10 日）について、
1 取引の上乗せ（コスト込み、乗り換え片道 --switch-cost を 2 回引く）を次の 2 通りで測る。
  日経平均との差: 同じ期間に日経平均を持っていた場合との差（今の測り方。β=1）
  β で調整した差: 日経平均 × β との差。β は合図の前日までの 250 日の日次の騰落から（120 日未満なら 1）
t 値は、同じ日の候補を等金額で 1 回と数える「日ごと」と、間が 10 営業日以内の合図の日をひとまとまりにする「まとまりごと」
（暴落は何日も続き、持つ期間も重なるので、日ごとでは独立な回数を多く数えすぎる）の 2 通り。
同じ日の候補の中の順位（5 日の騰落の小さい順。#73 の買う順）ごとの成績と、口座で実際に買った取引（DB の取引）の成績も出す。
候補表は study_bb2_close.py が作った results/bb2c_signals.pkl / bb2c_outcomes.pkl を使う。結果は results/bb2s_*.csv。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulate_portfolio import n225_daily
from study_bb2_close import add_returns, apply_rules, market_data
from trading.backtest import CostModel
from trading.store import DEFAULT_DB, ResultStore

EPISODE_GAP = 10   # この営業日数より間が空いたら別のまとまり（最長の保有日数と同じ）


def betas(closes: pd.DataFrame, nk_close: pd.Series, window: int = 250, min_periods: int = 120) -> pd.DataFrame:
    """日ごと・銘柄ごとの日経平均に対する β（その日の前日までの window 日の日次の騰落から）。"""
    r = closes.pct_change(fill_method=None)
    m = nk_close.reindex(closes.index).pct_change()
    mm = pd.DataFrame(np.where(r.notna(), m.to_numpy()[:, None], np.nan), index=r.index, columns=r.columns)
    roll = dict(window=window, min_periods=min_periods)
    cov = (r * mm).rolling(**roll).mean() - r.rolling(**roll).mean() * mm.rolling(**roll).mean()
    var = (mm ** 2).rolling(**roll).mean() - mm.rolling(**roll).mean() ** 2
    return (cov / var).shift(1)


def episodes(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """合図の日ごとのまとまりの番号。前の合図の日から EPISODE_GAP 営業日より空いたら新しいまとまり。"""
    days = pd.Index(sorted(dates.unique()))
    pos = calendar.get_indexer(days)
    ep = np.cumsum(np.r_[1, np.diff(pos) > EPISODE_GAP])
    return dates.map(pd.Series(ep, index=days))


def t_of(x: pd.Series) -> float:
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan


def measure(g: pd.DataFrame, col: str) -> dict:
    """1 取引の平均、日ごと（同じ日の候補を等金額）の平均と t 値、まとまりごとの平均と t 値。"""
    day = g.groupby("date")[col].mean()
    ep = day.groupby(g.groupby("date")["episode"].first()).mean()
    top = day.nlargest(5).index
    return {"取引数": len(g), "日数": len(day), "まとまり": len(ep), "1取引bp": g[col].mean(), "日ごとbp": day.mean(),
            "日ごとt": t_of(day), "まとまりごとbp": ep.mean(), "まとまりごとt": t_of(ep),
            "上位5日を除く日ごとt": t_of(day.drop(top))}


def table(groups: dict[str, pd.DataFrame], cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, g in groups.items():
        for per, gg in g.groupby("期間"):
            for cname, col in cols.items():
                rows.append({"対象": name, "期間": per, "測り方": cname, **measure(gg, col)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=73, help="条件・建て方・手仕舞い方を写す DB の実行")
    parser.add_argument("--config", default="config/daily_bb2_long_pullback.yaml")
    parser.add_argument("--switch-cost", type=float, default=0.0005)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    out = Path("results")

    run = ResultStore(args.db).run(args.run)
    p = run["params"]
    rules = [tuple(r) for r in p["rules"]]
    print(f"#{args.run} {run['name']}\n条件: {p['rules_text']} / 建て: {p['entry']} / 手仕舞い: {p['exit']}")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    costs = CostModel(**cfg["backtest"].get("costs", {}))
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    nk = n225_daily(pd.Timestamp.today().strftime("%Y-%m-%d"))
    closes, _, universe, start = market_data(cfg)
    print(f"{universe} / IS {start.date()}〜{is_end.date()} / OOS それより後", flush=True)

    outs = add_returns(pd.read_pickle(out / "bb2c_outcomes.pkl"), nk, costs, args.switch_cost)
    outs = outs[(outs["entry"] == p["entry"]) & (outs["exit"] == p["exit"])]
    cand = outs.merge(pd.read_pickle(out / "bb2c_signals.pkl"), on=["code", "date"], how="left")
    cand = cand[(cand["date"] >= start) & (cand["breaks5"] == 0)].reset_index(drop=True)
    cand["期間"] = np.where(cand["date"] <= is_end, "IS", "OOS")

    b = betas(closes, nk["close"])
    ri, ci = b.index.get_indexer(cand["date"]), b.columns.get_indexer(cand["code"])
    beta = np.where((ri >= 0) & (ci >= 0), b.to_numpy()[ri, np.maximum(ci, 0)], np.nan)
    cand["beta"] = np.where(np.isfinite(beta), beta, 1.0)
    cand["ex_beta"] = cand["bp"] - cand["beta"] * cand["nk_bp"] - 2 * args.switch_cost * 1e4

    sig = cand[apply_rules(cand, rules)].copy()
    sig["episode"] = episodes(sig["date"], closes.index)
    sig["順位"] = sig.groupby("date")["ret5"].rank(method="first")
    print(f"全候補（最初の割れ）{len(cand):,} 件 → #{args.run} の条件を満たす {len(sig):,} 件。"
          f"β の中央値 IS {sig.loc[sig['期間'] == 'IS', 'beta'].median():.2f} / OOS {sig.loc[sig['期間'] == 'OOS', 'beta'].median():.2f}")

    trades = ResultStore(args.db).trades(args.run)
    held = sig.merge(trades[["code", "entry_date"]].drop_duplicates(), on=["code", "entry_date"])
    print(f"口座の取引 {len(trades)} 件のうち候補と一致 {len(held)} 件")

    traded_day = sig["date"].isin(held["date"])
    first_day = sig["date"] == sig.groupby("episode")["date"].transform("min")
    cols = {"日経平均との差": "ex", "β で調整した差": "ex_beta", "損益そのもの": "bp"}
    groups = {"条件を満たす全候補": sig, "口座で買った取引": held,
              "口座が買った日の全候補": sig[traded_day], "口座が買わなかった日の全候補": sig[~traded_day],
              "まとまりの最初の日": sig[first_day], "まとまりの 2 日目以降": sig[~first_day],
              "同じ日の 1 番目（5 日で最も下げた銘柄）": sig[sig["順位"] == 1],
              "同じ日の 2〜5 番目": sig[(sig["順位"] >= 2) & (sig["順位"] <= 5)],
              "同じ日の 6 番目以降": sig[sig["順位"] >= 6],
              "（参考）条件なしの全候補": cand.assign(episode=episodes(cand["date"], closes.index))}
    res = table(groups, cols)
    res.to_csv(out / "bb2s_signal_t.csv", index=False)
    show = res.set_index(["対象", "測り方", "期間"])[["取引数", "日数", "まとまり", "1取引bp", "日ごとbp", "日ごとt",
                                                   "まとまりごとbp", "まとまりごとt", "上位5日を除く日ごとt"]]
    print("\n1 取引の上乗せ（bp、コスト込み）と t 値")
    print(show.round(2).to_string())

    yr = sig.groupby(["期間", sig["date"].dt.year]).apply(
        lambda g: pd.Series({"取引数": len(g), "日数": g["date"].nunique(), "日ごとbp（日経平均との差）": g.groupby("date")["ex"].mean().mean(),
                             "日ごとbp（β 調整）": g.groupby("date")["ex_beta"].mean().mean()}), include_groups=False)
    yr.to_csv(out / "bb2s_by_year.csv")
    print("\n年ごと（条件を満たす全候補）")
    print(yr.round(1).to_string())


if __name__ == "__main__":
    main()
