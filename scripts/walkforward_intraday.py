"""デイトレード戦略をウォークフォワードで検証する。

    python scripts/walkforward_intraday.py --config config/intraday_opening_gap.yaml \
        --param min_gap=0.005,0.01,0.02 --param stop_pct=0.01,0.015,0.02

検証する月ごとに「その月より前のデータだけ」で
  1. 銘柄を選ぶ（売買代金上位。毎月選び直す）
  2. パラメータを選ぶ（設定ファイルの params を土台に --param の全組み合わせを試し、損益合計が最大のもの）
そのうえで翌月 1 か月を検証し、各月の検証結果をつなげて評価する。調整に使う期間は
--train-months を省くとデータの最初から（拡大）、指定すると直前 N か月（移動）。
比較のため、設定ファイルの params のまま固定した場合の成績も同じ月・同じ銘柄で出す。
"""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from run_intraday import make_config
from trading.backtest import round_trip_table, run_intraday_backtest, summarize
from trading.data import attach_market, list_codes, load_5min, market_context
from trading.store import DEFAULT_DB, ResultStore, describe_run
from trading.strategy import get_intraday_strategy


def _daily_turnover_of(args):
    code, data_dir = args
    bars = load_5min(code, data_dir)
    day = bars.index.normalize()
    return pd.DataFrame({"turnover": (bars["close"] * bars["volume"]).groupby(day).sum(),
                         "price": bars["close"].groupby(day).mean()}).assign(code=code)


def daily_turnover(data_dir: str, cache_dir: Path, workers: int) -> pd.DataFrame:
    """全銘柄の日ごとの売買代金・平均終値（月ごとの銘柄選びに使う）。"""
    path = cache_dir / "intraday_daily_turnover.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
    codes = list_codes(data_dir)
    print(f"{len(codes)} 銘柄の日ごとの売買代金を集計中...")
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_daily_turnover_of, [(c, data_dir) for c in codes], chunksize=20))
    table = pd.concat(parts).rename_axis("date").reset_index()
    cache_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def select_universe(table: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ucfg: dict) -> list[str]:
    """start〜end（end を含まない）の売買代金・株価だけで銘柄を選ぶ。"""
    t = table[(table["date"] >= start) & (table["date"] < end)]
    stats = t.groupby("code").agg(turnover=("turnover", "mean"), price=("price", "mean"))
    stats = stats[(stats["turnover"] >= float(ucfg["min_turnover"])) & (stats["price"] >= float(ucfg["min_price"]))]
    return stats.sort_values("turnover", ascending=False).head(ucfg["top_n"]).index.tolist()


def run_stock(args):
    """1 銘柄で全候補を試す（並列処理の 1 単位）。取引は日内で完結するので、
    期間を通して 1 回動かしてから取引日で調整・検証に振り分ける（指標の計算に必要な助走も自然に取れる）。"""
    fold, code, bars, name, candidates, config = args
    return [(fold, i, code, round_trip_table(run_intraday_backtest(bars, get_intraday_strategy(name, **p), config),
                                             code))
            for i, p in enumerate(candidates)]


def equity_of(trades: pd.DataFrame, days: pd.DatetimeIndex, n_stocks: int, capital: float) -> pd.Series:
    """取引一覧から全銘柄合算の日次資産を作る（取引のない日は損益 0）。
    初日の損益が日次リターンに入るよう、期間の前日に元手だけの点を置く。"""
    pnl = (trades.groupby(pd.to_datetime(trades["exit_date"]).dt.normalize())["pnl"].sum() if len(trades)
           else pd.Series(dtype=float))
    daily = pd.concat([pd.Series([0.0], index=[days[0] - pd.Timedelta(days=1)]), pnl.reindex(days, fill_value=0.0)])
    return (n_stocks * capital + daily.cumsum()).rename("equity")


def evaluate(trades: pd.DataFrame, days: pd.DatetimeIndex, n_stocks: int, capital: float) -> dict:
    return summarize(trades, equity_of(trades, days, n_stocks, capital))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--param", action="append", default=[], help="name=v1,v2,...（複数指定可）")
    parser.add_argument("--first-test", default="2026-08", help="最初に検証する月（それより前は調整専用。5 分足はほぼ 6 月末から）")
    parser.add_argument("--train-months", type=int, default=0, help="調整に使う直前の月数（0 = データの最初から）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name, base = cfg["strategy"]["name"], cfg["strategy"].get("params", {})
    sweep = {k: [yaml.safe_load(x) for x in v.split(",")] for k, v in (p.split("=", 1) for p in args.param)}
    candidates = [{**base, **dict(zip(sweep, values))} for values in itertools.product(*sweep.values())]
    if base not in candidates:
        candidates.append(dict(base))
    fixed = candidates.index(base)
    config = make_config(cfg["backtest"])
    capital = cfg["backtest"]["capital_per_stock"]
    data_dir = cfg["data"]["dir"]

    table = daily_turnover(data_dir, Path("data/cache"), args.workers)
    # 一部の銘柄は何年も前のデータを持つので、大半の銘柄がそろった日をデータの始まりとする
    per_day = table.groupby("date")["code"].size()
    data_start, data_end = per_day[per_day >= per_day.max() / 2].index.min(), table["date"].max()
    print(f"5 分足のデータ: {data_start.date()}〜{data_end.date()}")
    months = pd.period_range(pd.Period(args.first_test, "M"), data_end.to_period("M"), freq="M")
    folds = []
    for m in months:
        test_start, test_end = m.start_time, (m + 1).start_time
        train_start = max(data_start, (m - args.train_months).start_time) if args.train_months else data_start
        folds.append({"month": str(m), "train": (train_start, test_start), "test": (test_start, test_end),
                      "codes": select_universe(table, train_start, test_start, cfg["universe"])})
    print(f"{name}: {len(candidates)} 通り × {len(folds)} か月")

    # 各月の銘柄で相場全体の列を作り直し、検証月の終わりまでのデータで全候補を動かす
    cache: dict[str, pd.DataFrame] = {}
    tasks = []
    for k, f in enumerate(folds):
        bars_by_code = {c: cache.setdefault(c, load_5min(c, data_dir)) for c in f["codes"]}
        bars_by_code = {c: b[b.index < f["test"][1]] for c, b in bars_by_code.items()}
        market = market_context(bars_by_code) if any(get_intraday_strategy(name, **p).requires_market
                                                     for p in candidates) else None
        tasks += [(k, c, attach_market(b, market), name, candidates, config) for c, b in bars_by_code.items()]
    rts: dict[tuple[int, int], list[pd.DataFrame]] = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for n, out in enumerate(pool.map(run_stock, tasks), 1):
            for k, i, code, r in out:
                rts.setdefault((k, i), []).append(r)
            if n % 100 == 0:
                print(f"  {n}/{len(tasks)}")

    all_days = pd.DatetimeIndex(sorted(table["date"].unique()))

    def window(k: int, i: int, span: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
        parts = [r for r in rts.get((k, i), []) if not r.empty]
        r = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if r.empty:
            return r
        entry = pd.to_datetime(r["entry_date"])
        return r[(entry >= span[0]) & (entry < span[1])].reset_index(drop=True)

    rows, wf_trades, fixed_trades = [], [], []
    for k, f in enumerate(folds):
        train_days = all_days[(all_days >= f["train"][0]) & (all_days < f["train"][1])]
        test_days = all_days[(all_days >= f["test"][0]) & (all_days < f["test"][1])]
        n = len(f["codes"])
        scores = [evaluate(window(k, i, f["train"]), train_days, n, capital) for i in range(len(candidates))]
        if all(s["num_trades"] == 0 for s in scores):
            raise SystemExit(f"{f['month']} の調整期間に取引がありません。--first-test を後ろにずらしてください")
        best = max(range(len(candidates)), key=lambda i: scores[i]["total_pnl"])
        for label, i in (("wf", best), ("fixed", fixed)):
            t = window(k, i, f["test"])
            s = evaluate(t, test_days, n, capital)
            rows.append({"month": f["month"], "method": label, **{p: candidates[i][p] for p in sweep},
                         "train_pnl": scores[i]["total_pnl"], "train_bp": scores[i]["avg_ret_bp"],
                         "trades": s["num_trades"], "win_rate": s.get("win_rate"), "avg_bp": s["avg_ret_bp"],
                         "pf": s.get("profit_factor"), "pnl": s["total_pnl"]})
            (wf_trades if label == "wf" else fixed_trades).append(t)

    table_out = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"walkforward_{name}.csv"
    table_out.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print("\n===== 月ごとの検証結果（wf = 毎月選び直し / fixed = 設定ファイルの params のまま） =====")
    print(table_out.round(3).to_string(index=False))

    test_days = all_days[all_days >= folds[0]["test"][0]]
    n_avg = round(sum(len(f["codes"]) for f in folds) / len(folds))
    periods = {}
    for label, parts in (("wf", wf_trades), ("fixed", fixed_trades)):
        parts = [t for t in parts if not t.empty]
        t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        equity = equity_of(t, test_days, n_avg, capital)
        s = summarize(t, equity)
        periods[label] = (s, t, equity)
        print(f"\n--- {label} 通算（{folds[0]['month']}〜{folds[-1]['month']}） ---")
        print(f"取引 {s['num_trades']} 回, 勝率 {s.get('win_rate', float('nan')):.1%}, "
              f"平均 {s['avg_ret_bp']:+.1f} bp, PF {s.get('profit_factor', float('nan')):.2f}, "
              f"損益 {s['total_pnl']:+,.0f} 円, シャープ {s.get('sharpe', float('nan')):.2f}")
    print(f"\n{path} に保存しました")

    if not args.no_save:
        store = ResultStore(args.db)
        span = f"{folds[0]['month']}〜{folds[-1]['month']}"
        train = f"直前 {args.train_months} か月" if args.train_months else "データの最初から"
        for label, note in (("wf", f"ウォークフォワード（毎月 銘柄・パラメータを選び直し、調整は{train}）{span}"),
                            ("fixed", f"ウォークフォワード（毎月 銘柄を選び直し、パラメータは固定）{span}")):
            summary, t, equity = periods[label]
            params = base if label == "fixed" else {"sweep": sweep}
            run_config = {**cfg, "walkforward": table_out[table_out["method"] == label].to_dict("records")}
            ctx = {"regime_mode": cfg.get("regime", {}).get("mode"),
                   "focus": "ウォークフォワード（" + ("毎月パラメータも選び直し" if label == "wf" else "パラメータ固定") + "）"}
            run_id = store.save_run("intraday", name, params, run_config, {"OOS": (summary, t, equity)}, note=note,
                                    **describe_run("intraday", name, base, ctx))
            print(f"DB に保存しました（run_id={run_id}）: {note}")


if __name__ == "__main__":
    main()
