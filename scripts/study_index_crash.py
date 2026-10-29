"""-3σ 割れ買いの「暴落の日に買う」部分を、個別株ではなく日経平均で確かめる。

    python scripts/study_index_crash.py

個別株では生存者バイアス（今上場している銘柄だけ）、指値の約定、買付余力（全候補に指値を置けない）が問題になる。
日経平均ならどれもないので、暴落の日に買うこと自体に効き目があるかを調べられる。

合図（3 種類。どれも前日の大引けまでの値か、その日に分かる値だけを使う）:
  A 日経平均が 25 日ボリンジャーバンドの -kσ に場中で触れたら買う（寄りで既に下なら寄りで）。-1σ で利益確定の指値、
    建値から 20% 下で損切り、最長 max_hold 日で大引け（個別株の -3σ 割れ買いと同じ。trading.backtest.touch_engine）。
    1989 年から（それより前の日経平均は終値しかない）
  B 日経平均の終値が -kσ を下回ったら翌日の寄りで買い、終値が -1σ 以上に戻った日か max_hold 日目の大引けで売る（1965 年から）
  C その日に -3σ に触れた貸借銘柄（前日までの 20 日平均売買代金 1 億円以上）が K 銘柄以上なら、日経平均を大引けで買い、
    B と同じく手仕舞う（2001-07 から）
  局面: 前日の p_down が 0.5 を超える日だけにするもの（2000 年から。2015 年までは局面モデルの学習期間なので当てはめた値）としないもの
運用の形: 資金はいつも日経平均で持ち（#55 などの待機と同じ）、合図の間だけ同じ額をもう 1 つ買い増す（信用買いで 2 倍。
  金利 年 --rate、売り買い片道 --switch-cost）。日経平均を持ちっぱなしにした場合と比べる。
期間: 古い期間（1965 年〜2001-07。選択にも IS にも使っていない）/ IS 2001-07-18〜2015 / OOS 2016〜。
設定は IS のシャープ（取引 --min-trades 回以上）で選び、OOS と古い期間で評価して DB に保存する。結果は results/index_crash_*.csv。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_backtest import load_stocks
from simulate_portfolio import stats
from trading.backtest import summarize as summarize_run
from trading.backtest.touch_engine import touch_trades
from trading.data import get_source
from trading.store import DEFAULT_DB, ResultStore, describe_run
from trading.strategy import BandTouchFade
from trading.strategy.band_touch import band_price

IS_START, IS_END = pd.Timestamp("2001-07-18"), pd.Timestamp("2015-12-31")   # 個別株の検証と同じ IS
OLD_START = {"A": pd.Timestamp("1989-01-04"), "B": pd.Timestamp("1965-01-05"), "C": None}
SIGNAL_NAMES = {"A": "日経平均の−kσに場中で触れたら", "B": "日経平均の終値が−kσを割ったら翌日の寄りで",
                "C": "−3σ割れの貸借銘柄がK銘柄以上の日の大引けで"}


def load_n225(regime_file: str) -> pd.DataFrame:
    nk = get_source("yfinance").get_daily("^N225", "1965-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    p_down = pd.read_csv(regime_file, index_col=0, parse_dates=True)["p_down"].shift(1)
    return nk.assign(p_down_prev=p_down.reindex(nk.index))


def zscore(c: pd.Series, window: int = 25) -> pd.Series:
    """その日の終値を含む window 日のボリンジャーバンドでの位置（σ）。"""
    return (c - c.rolling(window).mean()) / c.rolling(window).std(ddof=0)


def breadth(cfg: dict) -> pd.Series:
    """その日に -3σ の値段（band_price）に触れた、流動性のある貸借銘柄の数。"""
    return breadth_count(load_stocks(cfg))


def breadth_count(stocks: dict[str, pd.DataFrame]) -> pd.Series:
    """breadth の、読み込み済みの銘柄から数える版。"""
    counts = []
    for df in stocks.values():
        c = df["close"]
        liquid = (c * df["volume"]).rolling(20).mean().shift(1) >= 1e8
        counts.append((liquid & (df["low"] <= band_price(c, 25, -3.0))).astype(int))
    return pd.concat(counts, axis=1, sort=True).sum(axis=1)


def trades_touch(nk: pd.DataFrame, k: float, p_min: float | None, max_hold: int) -> pd.DataFrame:
    """A: 場中に -kσ に触れたら買う（個別株と同じ約定の扱い。損切りが先・建てた日は利益確定しない）。"""
    s = BandTouchFade(window=25, k=k, exit_k=1.0, stop_pct=0.20, max_hold=max_hold, sides="long",
                      long_min_p_down=p_min)
    df = nk.dropna(subset=["p_down_prev"]) if p_min is not None else nk
    od = s.orders(df)
    o, h, lo, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    ts = touch_trades(o, h, lo, c, od["short_entry"].to_numpy(float), od["long_entry"].to_numpy(float),
                      od["short_target"].to_numpy(float), od["long_target"].to_numpy(float),
                      od["can_enter"].to_numpy(bool), s.stop_pct, s.max_hold, s.fill)
    return pd.DataFrame([{"entry_date": df.index[t.entry_i], "exit_date": df.index[t.exit_i], "entry_price": t.entry_price,
                          "exit_price": t.exit_price, "reason": t.reason} for t in ts])


def trades_close(nk: pd.DataFrame, signal: pd.Series, entry: str, max_hold: int) -> pd.DataFrame:
    """B / C: signal の日に買い（entry = next_open なら翌日の寄り、close ならその日の大引け）、終値が -1σ 以上に戻った日か
    max_hold 営業日目の大引けで売る。持っている間の合図は使わない。"""
    z, c, o = zscore(nk["close"]).to_numpy(), nk["close"].to_numpy(float), nk["open"].to_numpy(float)
    sig = signal.reindex(nk.index).fillna(False).to_numpy(bool)
    rows, i, n = [], 0, len(nk)
    while i < n:
        if not sig[i]:
            i += 1
            continue
        e = i + 1 if entry == "next_open" else i
        if e >= n:
            break
        price = o[e] if entry == "next_open" else c[e]
        j, reason = e, None
        while j < n - 1:
            j += 1
            if z[j] >= -1.0:
                reason = "target"
                break
            if j - e >= max_hold:
                reason = "time"
                break
        if reason is None:
            break
        rows.append({"entry_date": nk.index[e], "exit_date": nk.index[j], "entry_price": price,
                     "exit_price": c[j], "reason": reason})
        i = j + 1
    return pd.DataFrame(rows)


def overlay(nk: pd.DataFrame, trades: pd.DataFrame, start, end, capital: float, extra: float, rate: float,
            switch: float, base_ret: pd.Series | None = None, cash_frac: pd.Series | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """いつも資産の 100% を日経平均で持ち、取引の間だけ資産 × extra を買い増す。買い増し分は片道 switch の費用と年 rate の金利。
    base_ret（日付 → その日の騰落率）を渡すと、いつも持っている分を日経平均ではなくその運用（例: 個別株の手法）にする。
    cash_frac（日付 → 資産に対する現金の割合）を渡すと、建てた日の現金でまかなえる分は現金で買い、足りない分だけ信用で借りる
    （金利は借りた分だけ）。"""
    c = nk["close"][start:end]
    t = trades[(trades["entry_date"] >= c.index[0]) & (trades["exit_date"] <= c.index[-1])]
    by_entry = {r.entry_date: r for r in t.itertuples()}
    equity, rows = {}, []
    eq, prev_c, prev_d, pos = capital, None, None, None
    for d, px in c.items():
        if prev_c is None:   # いつも持っている分
            pnl = 0.0
        else:
            pnl = eq * (px / prev_c - 1 if base_ret is None else base_ret.get(d, 0.0))
        if pos is not None:   # 前日から持ち越した買い増し分
            pnl -= pos["debt"] * rate * (d - prev_d).days / 365
            last = pos["exit"] if d == pos["exit_date"] else px
            pnl += pos["shares"] * (last - prev_c)
        if pos is None and d in by_entry and extra > 0:
            r = by_entry[d]
            shares = eq * extra / r.entry_price
            debt = eq * extra if cash_frac is None else max(0.0, eq * extra - eq * max(float(cash_frac.get(d, 0.0)), 0.0))
            pos = {"shares": shares, "entry": r.entry_price, "entry_date": d, "exit_date": r.exit_date,
                   "exit": r.exit_price, "reason": r.reason, "cost": shares * r.entry_price * switch, "debt": debt}
            last = r.exit_price if d == r.exit_date else px
            pnl += shares * (last - r.entry_price) - pos["cost"]
        if pos is not None and d == pos["exit_date"]:
            sell_cost = pos["shares"] * pos["exit"] * switch
            pnl -= sell_cost
            days = max((d - pos["entry_date"]).days, 0)
            p = pos["shares"] * (pos["exit"] - pos["entry"]) - pos["cost"] - sell_cost - pos["debt"] * rate * days / 365
            rows.append({"code": "^N225", "entry_date": pos["entry_date"], "exit_date": d, "shares": pos["shares"],
                         "entry_price": pos["entry"], "exit_price": pos["exit"], "pnl": p,
                         "ret": p / (pos["shares"] * pos["entry"]), "exit_reason": pos["reason"], "side": "long"})
            pos = None
        eq += pnl
        equity[d] = eq
        prev_c, prev_d = px, d
    cols = ["code", "entry_date", "exit_date", "shares", "entry_price", "exit_price", "pnl", "ret", "exit_reason", "side"]
    return pd.Series(equity, name="equity"), pd.DataFrame(rows, columns=cols)


def trade_stats(t: pd.DataFrame, nk: pd.DataFrame, rate: float, switch: float) -> dict:
    """1 取引の成績（買い増し分。費用込み）と、同じ日数だけ持ったときの平均（いつでも買った場合）との差。"""
    if t.empty:
        return {"取引数": 0}
    days = np.array([nk.index.get_loc(x) - nk.index.get_loc(e) for e, x in zip(t["entry_date"], t["exit_date"])])
    gross = (t["exit_price"] / t["entry_price"] - 1).to_numpy() * 1e4
    net = gross - 2 * switch * 1e4 - rate * (t["exit_date"] - t["entry_date"]).dt.days.to_numpy() / 365 * 1e4
    r = nk["close"][t["entry_date"].min():t["exit_date"].max()].pct_change().dropna()
    base = r.mean() * np.maximum(days, 1) * 1e4
    top = np.argsort(net)[::-1][:5]
    rest = np.delete(net, top)
    return {"取引数": len(t), "平均bp": net.mean(), "勝率": (net > 0).mean(), "t値": net.mean() / net.std(ddof=1) * np.sqrt(len(net))
            if len(net) > 1 else np.nan, "上位5回を除く平均bp": rest.mean() if len(rest) else np.nan,
            "いつでも買った場合との差bp": (gross - base).mean(), "平均の保有日数": days.mean()}


def excess(eq: pd.Series, nk: pd.DataFrame) -> dict:
    """日経平均を持ちっぱなしにした場合との差（月ごと）。"""
    n = nk["close"].reindex(eq.index)
    ms, mn = eq.resample("ME").last().pct_change().dropna(), n.resample("ME").last().pct_change().dropna()
    d = (ms - mn).dropna()
    return {"日経平均の年率": stats(n / n.iloc[0])["年率"], "月ごとの差（年換算）": d.mean() * 12,
            "差の t 値": d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 1 else np.nan}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/daily_bb3_long_down.yaml", help="局面モデルと貸借銘柄（C の合図）の設定")
    parser.add_argument("--capital", type=float, default=10_000_000)
    parser.add_argument("--extra", type=float, default=1.0, help="合図の間に買い増す額（資産に対する割合）")
    parser.add_argument("--rate", type=float, default=0.028, help="信用買いの金利（年）")
    parser.add_argument("--switch-cost", type=float, default=0.0005, help="日経平均の売り買いの片道の費用")
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    nk = load_n225(cfg["data"]["regime_file"])
    print(f"日経平均 {nk.index[0].date()}〜{nk.index[-1].date()}（高値・安値は 1988 年途中から）", flush=True)
    br = breadth(cfg)
    p_on = nk["p_down_prev"] > 0.5
    variants = {}
    for k in (2.0, 2.5, 3.0):
        for hold in (5, 10):
            for p in (None, 0.5):
                tag = "下げ局面だけ" if p else "局面で絞らない"
                df = nk[nk.index >= OLD_START["A"]]
                variants[("A", f"k={k:g}", hold, tag)] = trades_touch(df, k, p, hold)
                sig = zscore(nk["close"]) < -k
                variants[("B", f"k={k:g}", hold, tag)] = trades_close(nk, sig & p_on if p else sig, "next_open", hold)
    for kk in (10, 30, 100):
        for hold in (5, 10):
            for p in (None, 0.5):
                tag = "下げ局面だけ" if p else "局面で絞らない"
                sig = br.reindex(nk.index).fillna(0) >= kk
                variants[("C", f"K={kk}", hold, tag)] = trades_close(nk, sig & p_on if p else sig, "close", hold)

    periods = {"古い期間": (nk.index[0], IS_START - pd.Timedelta(days=1)), "IS": (IS_START, IS_END),
               "OOS": (IS_END + pd.Timedelta(days=1), nk.index[-1])}
    rows, keep = [], {}
    for key, t in variants.items():
        for per, (s, e) in periods.items():
            if t.empty:
                continue
            tt = t[(t["entry_date"] >= s) & (t["entry_date"] <= e)]
            if per == "古い期間" and (key[0] == "C" or key[3] == "下げ局面だけ"):
                continue
            s0 = OLD_START[key[0]] if per == "古い期間" else s
            eq, tr = overlay(nk, tt, s0, e, args.capital, args.extra, args.rate, args.switch_cost)
            keep[(key, per)] = (eq, tr)
            rows.append({"合図": key[0], "値": key[1], "最長": key[2], "局面": key[3], "期間": per,
                         "開始": eq.index[0].date(), **trade_stats(tt, nk, args.rate, args.switch_cost),
                         **{f"運用の{k}": v for k, v in stats(eq, tr).items() if k in ("年率", "最大DD", "シャープ")},
                         **excess(eq, nk)})
    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/index_crash_grid.csv", index=False)
    for per in periods:
        print(f"\n{per}（買い増し分の 1 取引は費用込み。運用はいつも日経平均を持ち、合図の間は {1 + args.extra:g} 倍）")
        print(table[table["期間"] == per].drop(columns=["期間"]).round(3).to_string(index=False))

    is_t = table[(table["期間"] == "IS") & (table["取引数"] >= args.min_trades)].sort_values("運用のシャープ", ascending=False)
    best = is_t.iloc[0]
    key = (best["合図"], best["値"], int(best["最長"]), best["局面"])
    print(f"\nIS のシャープで選んだ設定: 合図 {key[0]}（{SIGNAL_NAMES[key[0]]}）・{key[1]}・最長 {key[2]} 日・{key[3]}")
    print(table[(table["合図"] == key[0]) & (table["値"] == key[1]) & (table["最長"] == key[2]) & (table["局面"] == key[3])]
          .drop(columns=["合図", "値", "最長", "局面"]).round(3).to_string(index=False))
    print("\nIS のシャープの上位 10")
    print(is_t.head(10)[["合図", "値", "最長", "局面", "取引数", "平均bp", "運用の年率", "運用のシャープ", "月ごとの差（年換算）", "差の t 値"]]
          .round(3).to_string(index=False))

    if args.no_save:
        return
    params = {"signal": key[0], "window": 25, "max_hold": key[2], "extra": args.extra, "rate": args.rate,
              "switch_cost": args.switch_cost, "capital": args.capital,
              "long_min_p_down": 0.5 if key[3] == "下げ局面だけ" else None}
    if key[0] == "C":
        params["min_stocks"] = int(key[1].split("=")[1])
    else:
        params["k"] = float(key[1].split("=")[1])
    params["exit_k"] = 1.0   # A は -1σ の指値、B / C は終値が -1σ に戻った日の大引け（確認画面のチャートの利益確定の線にも使う）
    if key[0] == "A":
        params.update(stop_pct=0.20, sides="long")
    saved = {per: (summarize_run(keep[(key, per)][1], keep[(key, per)][0]), keep[(key, per)][1], keep[(key, per)][0])
             for per in ("IS", "OOS") if (key, per) in keep}
    run_cfg = {"data": {"source": "yfinance", "start": "1965-01-01", "end": str(nk.index[-1].date())},
               "split": {"is_start": str(IS_START.date()), "is_end": str(IS_END.date())},
               "regime_file": cfg["data"]["regime_file"]}
    desc = describe_run("daily", "index_overlay", params, {"universe": "日経平均（yfinance・配当なし）"})
    run_id = ResultStore(args.db).save_run("daily", "index_overlay", params, run_cfg, saved, **desc,
                                           note="scripts/study_index_crash.py。合図の種類・値・最長の日数・局面は IS のシャープで選択。"
                                                "古い期間の成績は results/index_crash_grid.csv")
    print(f"\nDB に保存しました（run_id={run_id}）: {desc['name']}")


if __name__ == "__main__":
    main()
