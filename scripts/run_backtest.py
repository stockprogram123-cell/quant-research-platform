"""設定ファイルを読み込んでバックテストを実行する。

    python scripts/run_backtest.py --config config/daily_bb3_long_down.yaml   # 多銘柄（data.dir の日足 CSV）

data.source / data.code の設定なら 1 銘柄（yfinance など）で検証する。設定ファイルの name は確認画面に出す名前。

data.dir があれば、そのフォルダの全銘柄（または data.codes）で銘柄ごとに独立に検証し、
split.is_end までを調整期間（IS）、それより後を検証期間（OOS）として集計する（取引は建てた日で分ける）。
universe.margin_only: true なら JPX の貸借銘柄一覧（最新）にある銘柄だけにする。
backtest.price_limits: true ならストップ高・ストップ安で手仕舞えない日を考える（trading.backtest.touch_engine.touch_trades を参照）。
data.regime_file（例: data/cache/regime_daily.csv）を指定すると、前日の下げ相場の確率を p_down_prev 列として付ける。
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from trading.backtest import BacktestConfig, CostModel, round_trip_table, run_backtest, summarize
from trading.backtest.limits import limit_flags
from trading.data import get_source, list_codes, load_daily_csv
from trading.data.jpx import margin_codes, split_history, unadjust_factor
from trading.report import plot_equity
from trading.store import DEFAULT_DB, ResultStore, describe_run, universe_label
from trading.strategy import get_strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/daily_bb3_long_down.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--db", default=DEFAULT_DB, help="結果を保存する SQLite ファイル")
    parser.add_argument("--no-save", action="store_true", help="DB に保存しない")
    parser.add_argument("--note", default="", help="DB に残すメモ")
    parser.add_argument("--focus", default="", help="確認画面のタイトルに出す、この実行で重視したこと（省略時は設定から）")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, strat_cfg, bt_cfg = cfg["data"], cfg["strategy"], cfg["backtest"]
    if data_cfg.get("dir"):
        run_many(cfg, args)
        return

    source = get_source(data_cfg["source"], cache_dir=None if args.no_cache else "data/cache")
    df = source.get_daily(data_cfg["code"], data_cfg["start"], data_cfg["end"])
    strategy = get_strategy(strat_cfg["name"], **strat_cfg.get("params", {}))
    config = BacktestConfig(
        initial_cash=bt_cfg["initial_cash"],
        lot_size=bt_cfg["lot_size"],
        position_size=bt_cfg["position_size"],
        costs=CostModel(**bt_cfg.get("costs", {})),
    )

    result = run_backtest(df, strategy, config)
    benchmark = df["close"] / df["close"].iloc[0] * config.initial_cash
    bh_return = benchmark.iloc[-1] / config.initial_cash - 1

    m = result.metrics
    code = data_cfg["code"]
    print(f"=== {code} / {strat_cfg['name']} {strat_cfg.get('params', {})} ===")
    print(f"期間          : {df.index[0].date()} 〜 {df.index[-1].date()} ({len(df)} 営業日)")
    print(f"最終資産      : {m['final_equity']:,.0f} 円")
    print(f"総リターン    : {m['total_return']:+.2%}  (Buy&Hold: {bh_return:+.2%})")
    print(f"CAGR          : {m['cagr']:+.2%}")
    print(f"最大DD        : {m['max_drawdown']:.2%}")
    print(f"シャープレシオ: {m['sharpe']:.2f}")
    print(f"取引回数      : {m['num_trades']}")
    print(f"勝率          : {m['win_rate']:.1%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{code}_{strat_cfg['name']}"
    result.trades.to_csv(out / f"{stem}_trades.csv", index=False)
    result.equity.to_csv(out / f"{stem}_equity.csv")
    plot_equity(result.equity, out / f"{stem}.png", title=f"{code} {strat_cfg['name']}",
                benchmark=benchmark)
    print(f"\n結果を {out}/ に保存しました")

    if not args.no_save:
        rts = round_trip_table(result, code)
        periods = {"ALL": (summarize(rts, result.equity), rts, result.equity)}
        run_id = ResultStore(args.db).save_run("daily", strat_cfg["name"], strat_cfg.get("params", {}),
                                               cfg, periods, note=args.note,
                                               **describe_run("daily", strat_cfg["name"], strat_cfg.get("params", {}),
                                                              {"universe": f"{data_cfg['code']} 1銘柄", "focus": args.focus}))
        print(f"DB に保存しました: {args.db}（run_id={run_id}）")


def load_stocks(cfg: dict) -> dict[str, pd.DataFrame]:
    """設定の data / universe / backtest.price_limits に従って、多銘柄の日足を読む（期間で切った後 100 日未満の銘柄は除く）。"""
    data_cfg, bt_cfg = cfg["data"], cfg["backtest"]
    start, end = pd.Timestamp(data_cfg.get("start", "1900-01-01")), pd.Timestamp(data_cfg.get("end", "2100-01-01"))
    codes = [str(c) for c in data_cfg.get("codes") or list_codes(data_cfg["dir"])]
    if cfg.get("universe", {}).get("margin_only"):
        n_all, margin = len(codes), margin_codes()
        codes = [c for c in codes if c in margin]
        print(f"貸借銘柄に絞りました: {n_all} → {len(codes)} 銘柄")
    splits = split_history(codes) if bt_cfg.get("price_limits") else {}
    p_down_prev = None
    if data_cfg.get("regime_file"):  # 寄りの前に分かるのは前日の大引け時点の確率
        p_down_prev = pd.read_csv(data_cfg["regime_file"], index_col=0, parse_dates=True)["p_down"].shift(1)

    stocks = {}
    for code in codes:
        df = load_daily_csv(code, data_cfg["dir"])
        if bt_cfg.get("price_limits"):  # 前日終値から値幅を出すので、期間で切る前に作る
            df = df.join(limit_flags(df, unadjust_factor(df.index, splits.get(code, pd.Series(dtype=float)))))
        if p_down_prev is not None:
            df = df.assign(p_down_prev=p_down_prev.reindex(df.index))
        df = df.loc[start:end]
        if len(df) >= 100:
            stocks[code] = df
    return stocks


def run_many(cfg: dict, args) -> None:
    """data.dir の日足 CSV の全銘柄で、銘柄ごとに同じ資金で検証して合算する。"""
    data_cfg, strat_cfg, bt_cfg = cfg["data"], cfg["strategy"], cfg["backtest"]
    name, params = strat_cfg["name"], strat_cfg.get("params", {})
    strategy = get_strategy(name, **params)
    config = BacktestConfig(initial_cash=bt_cfg["initial_cash"], lot_size=bt_cfg["lot_size"],
                            position_size=bt_cfg["position_size"], costs=CostModel(**bt_cfg.get("costs", {})))
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    rts, equities = [], {}
    for code, df in load_stocks(cfg).items():
        result = run_backtest(df, strategy, config)
        rts.append(round_trip_table(result, code))
        equities[code] = result.equity
    rts = pd.concat([r for r in rts if not r.empty], ignore_index=True)
    capital = config.initial_cash * len(equities)
    equity = pd.DataFrame(equities).sort_index().ffill().fillna(config.initial_cash).sum(axis=1).rename("equity")

    print(f"=== {name} {params} / {len(equities)} 銘柄 / IS 〜{is_end.date()} / OOS それより後 ===")
    periods = {}
    for label, in_period, eq in [("IS", rts["entry_date"] <= is_end, equity[:is_end]),
                                 ("OOS", rts["entry_date"] > is_end, equity[equity.index > is_end])]:
        if label == "OOS" and len(eq):  # 検証期間の損益は検証期間の始めを 0 として数える
            eq = eq - (equity[:is_end].iloc[-1] - capital)
        r = rts[in_period].reset_index(drop=True)
        s = summarize(r, eq)
        periods[label] = (s, r, eq)
        if s["num_trades"]:
            print(f"{label}: 取引 {s['num_trades']:,} 回, 勝率 {s['win_rate']:.1%}, 平均 {s['avg_ret_bp']:+.1f} bp, "
                  f"PF {s['profit_factor']:.2f}, 損益 {s['total_pnl']:+,.0f} 円, 決済 {s['exit_reasons']}")
            for side, g in r.groupby("side"):
                print(f"    {side:5s}: {len(g):,} 回, 勝率 {(g['pnl'] > 0).mean():.1%}, "
                      f"平均 {g['ret'].mean() * 1e4:+.1f} bp, 損益 {g['pnl'].sum():+,.0f} 円")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rts.to_csv(out / f"daily_{name}_trades.csv", index=False)
    plot_equity(equity, out / f"daily_{name}.png", title=f"{name} ({len(equities)} stocks)")
    print(f"\n結果を {out}/ に保存しました")
    if not args.no_save:
        run_config = {**cfg, "backtest": {**bt_cfg, "initial_cash": capital, "capital_per_stock": config.initial_cash}}
        run_id = ResultStore(args.db).save_run("daily", name, params, run_config, periods, note=args.note,
                                                 **describe_run("daily", name, params,
                                                                {"universe": universe_label(cfg, len(equities)),
                                                                 "focus": args.focus}))
        print(f"DB に保存しました: {args.db}（run_id={run_id}）")


if __name__ == "__main__":
    main()
