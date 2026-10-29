"""#60（暴落の日に日経平均を買い増す）の合図そのものに中身があるかを、口座が使わなかった合図も含めた全部の合図の日で測り直す。

    python scripts/study_index_signal_t.py            # DB #60 の決まりで測る
    python scripts/study_index_signal_t.py --run 60

口座は買い増している間に出た合図を使わないので、OOS の取引は 59 回しかない。ここでは決まりを新しく選ばず、
合図の出た日すべてについて「その日に買い増した」とみなした取引を作り（重なってもよい）、次の 2 通りで測る。
  損益そのもの: 買い増し分の 1 取引の損益（費用込み。片道 --switch-cost の 2 回分と金利）。日経平均を 1 倍で持った場合との差と同じ
  ふだんとの差: 損益そのものから、同じ営業日数だけ持ったときの期間の平均（いつ買い増しても得られる分）を引いたもの
建て方は #60 の「合図の日の大引け」と、#75 の「翌日の寄り（ETF の寄りの窓で求めた売買できる値段）」の 2 通り。
t 値は 1 回 1 回を独立とみなす「日ごと」と、間が 10 営業日以内の合図の日をひとまとまりにする「まとまりごと」の 2 通り。
口座の実際の取引（DB #60 と、#75 の日経平均の買い増し）と、まとまりの最初の日かどうかでも分ける。結果は results/index_signal_t*.csv。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulate_portfolio import n225_daily
from study_bb2_signal_t import episodes, measure
from study_index_crash import IS_END, IS_START, breadth, load_n225, trades_close, zscore
from trading.store import DEFAULT_DB, ResultStore


def all_signal_trades(nk: pd.DataFrame, open_px: pd.Series, signal: pd.Series, entry: str, max_hold: int) -> pd.DataFrame:
    """合図の日ごとに 1 取引（持っている間の合図も使う）。手仕舞いは trades_close と同じ（終値が -1σ 以上か max_hold 営業日目の大引け）。"""
    z, c, o = zscore(nk["close"]).to_numpy(), nk["close"].to_numpy(float), open_px.reindex(nk.index).to_numpy(float)
    rows, n = [], len(nk)
    for i in np.flatnonzero(signal.reindex(nk.index).fillna(False).to_numpy(bool)):
        e = i + 1 if entry == "next_open" else i
        j = e
        while j < n - 1:
            j += 1
            if z[j] >= -1.0 or j - e >= max_hold:
                break
        else:
            continue
        rows.append({"date": nk.index[i], "entry_date": nk.index[e], "exit_date": nk.index[j], "hold": j - e,
                     "entry_price": o[e] if entry == "next_open" else c[e], "exit_price": c[j]})
    return pd.DataFrame(rows)


def add_measures(t: pd.DataFrame, nk: pd.DataFrame, rate: float, switch: float) -> pd.DataFrame:
    """損益そのもの（net）と、期間の平均の騰落 × 保有営業日数を引いたふだんとの差（adj）。bp。"""
    days = (t["exit_date"] - t["entry_date"]).dt.days
    net = ((t["exit_price"] / t["entry_price"] - 1) - 2 * switch - rate * days / 365) * 1e4
    r = nk["close"].pct_change()
    drift = {per: r[s:e].mean() for per, (s, e) in {"IS": (IS_START, IS_END), "OOS": (IS_END + pd.Timedelta(days=1), r.index[-1])}.items()}
    per = np.where(t["date"] <= IS_END, "IS", "OOS")
    base = pd.Series(per).map(drift).to_numpy() * np.maximum(t["hold"].to_numpy(), 1) * 1e4
    return t.assign(期間=per, net=net, adj=net - base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=60, help="決まりを写す DB の実行（index_overlay の合図 C）")
    parser.add_argument("--overlay-run", type=int, default=75, help="翌日の寄りで買い増す形の口座の取引を持つ DB の実行")
    parser.add_argument("--config", default="config/daily_bb3_long_down.yaml", help="局面モデルと貸借銘柄（合図 C の数え方）")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    out = Path("results")

    store = ResultStore(args.db)
    run = store.run(args.run)
    p = run["params"]
    print(f"#{args.run} {run['name']}\n合図: -3σ に触れた貸借銘柄 {p['min_stocks']} 以上・前日の下げ相場の確率 > {p['long_min_p_down']} / "
          f"最長 {p['max_hold']} 日 / 金利 年 {p['rate']:.1%}・片道 {p['switch_cost']:.2%}")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    nk = load_n225(cfg["data"]["regime_file"])
    nk = nk[nk.index >= IS_START - pd.Timedelta(days=60)]
    open_px = n225_daily(str(nk.index[-1].date()))["open_tradable"]
    signal = (breadth(cfg).reindex(nk.index).fillna(0) >= p["min_stocks"]) & (nk["p_down_prev"] > p["long_min_p_down"])
    signal = signal[signal.index >= IS_START]

    # 口座の取引を作り直して DB と一致するか確かめる（大引けで建てる #60）
    acct = trades_close(nk, signal, "close", p["max_hold"])
    db60 = store.trades(args.run)
    same = acct.merge(db60[["entry_date", "exit_date"]], on=["entry_date", "exit_date"])
    print(f"合図の日 {int(signal.sum())} 日。口座の取引を作り直すと {len(acct)} 件（DB #{args.run} は {len(db60)} 件、一致 {len(same)} 件）")
    db75 = store.trades(args.overlay_run)
    db75 = db75[db75["code"] == "^N225"]

    cols = {"損益そのもの": "net", "ふだんとの差": "adj"}
    rows, by_year = [], []
    for entry, label, held_db in (("close", "合図の日の大引け（#60）", db60), ("next_open", "翌日の寄り（#75）", db75)):
        t = add_measures(all_signal_trades(nk, open_px, signal, entry, p["max_hold"]), nk, p["rate"], p["switch_cost"])
        t["episode"] = episodes(t["date"], nk.index)
        first = t["date"] == t.groupby("episode")["date"].transform("min")
        held = t.merge(held_db[["entry_date", "exit_date"]], on=["entry_date", "exit_date"])
        groups = {"全部の合図の日": t, "口座の取引": held, "まとまりの最初の日": t[first], "まとまりの 2 日目以降": t[~first]}
        print(f"\n{label}: 合図の日の取引 {len(t)} 件 / 口座の取引 {len(held_db)} 件のうち一致 {len(held)} 件")
        for name, g in groups.items():
            for per, gg in g.groupby("期間"):
                for cname, col in cols.items():
                    rows.append({"建て方": label, "対象": name, "期間": per, "測り方": cname, **measure(gg, col),
                                 "平均の保有日数": gg["hold"].mean()})
        by_year.append(t.groupby(["期間", t["date"].dt.year]).agg(合図の日=("net", "size"), 損益そのものbp=("net", "mean"),
                                                                ふだんとの差bp=("adj", "mean")).assign(建て方=label))
    res = pd.DataFrame(rows)
    res.to_csv(out / "index_signal_t.csv", index=False)
    show = res.set_index(["建て方", "対象", "測り方", "期間"])[["取引数", "まとまり", "1取引bp", "日ごとt", "まとまりごとbp",
                                                         "まとまりごとt", "上位5日を除く日ごとt", "平均の保有日数"]]
    print("\n買い増し分の 1 取引（bp、費用込み）と t 値。日ごと = 合図の日 1 つずつ（同じ日に 2 つはないので 1 取引と同じ）")
    print(show.round(2).to_string())
    yr = pd.concat(by_year)
    yr.to_csv(out / "index_signal_t_by_year.csv")
    print("\n年ごと（全部の合図の日）")
    print(yr.round(1).to_string())


if __name__ == "__main__":
    main()
