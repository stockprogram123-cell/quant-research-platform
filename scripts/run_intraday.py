"""5 分足のデイトレード戦略を多銘柄で検証する（インサンプル/アウトオブサンプル）。

    python scripts/run_intraday.py --config config/intraday_opening_gap.yaml
"""
import argparse
import itertools
import warnings
from pathlib import Path

import pandas as pd
import yaml

from trading.backtest import (CostModel, IntradayConfig, round_trip_table, run_intraday_backtest,
                              summarize)
from trading.data import (attach_market, attach_regime, daily_context, get_source, list_codes, load_5min,
                          market_context)
from trading.regime import market_regime
from trading.report import plot_equity
from trading.store import DEFAULT_DB, ResultStore, describe_run
from trading.strategy import get_intraday_strategy


def build_universe(data_dir: str, is_end: pd.Timestamp, ucfg: dict, cache_dir: Path) -> pd.DataFrame:
    """インサンプル期間の売買代金・株価だけで銘柄を選ぶ（将来の情報を使わない）。"""
    path = cache_dir / f"intraday_universe_{is_end.date()}.csv"
    if path.exists():
        stats = pd.read_csv(path, dtype={"code": str})
    else:
        rows = []
        codes = list_codes(data_dir)
        print(f"{len(codes)} 銘柄の売買代金を集計中...")
        for code in codes:
            bars = load_5min(code, data_dir)
            bars = bars[bars.index < is_end + pd.Timedelta(days=1)]
            if bars.empty:
                continue
            turnover = (bars["close"] * bars["volume"]).groupby(bars.index.normalize()).sum()
            rows.append({"code": code, "turnover": turnover.mean(),
                         "price": bars["close"].mean(), "days": len(turnover)})
        stats = pd.DataFrame(rows)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stats.to_csv(path, index=False)
    stats = stats[(stats["turnover"] >= float(ucfg["min_turnover"]))
                  & (stats["price"] >= float(ucfg["min_price"]))]
    return stats.sort_values("turnover", ascending=False).head(ucfg["top_n"]).reset_index(drop=True)


def load_daily_ctx(source, code: str, start: pd.Timestamp, end: pd.Timestamp,
                   sma: tuple[int, ...]) -> pd.DataFrame | None:
    if not sma:
        return None
    try:
        daily = source.get_daily(code, str(start.date()), str(end.date()))
    except Exception as e:  # noqa: BLE001  取得できない銘柄は除外して続行する
        warnings.warn(f"{code}: 日足を取得できないため除外します ({e})")
        return None
    return daily_context(daily, sma_windows=sma)


def run_period(data: dict, strategy_name: str, params: dict, config: IntradayConfig):
    """全銘柄で独立にバックテストし、取引一覧と銘柄ごとの日次資産を返す。"""
    strategy = get_intraday_strategy(strategy_name, **params)
    rts, equities = [], {}
    for code, (bars, ctx) in data.items():
        if strategy.required_sma and ctx is None:
            continue
        result = run_intraday_backtest(bars, strategy, config, ctx)
        rts.append(round_trip_table(result, code))
        equities[code] = result.equity
    rts = [r for r in rts if not r.empty]
    return (pd.concat(rts, ignore_index=True) if rts else pd.DataFrame()), equities


def combine_equity(equities: dict[str, pd.Series], capital: float) -> pd.Series:
    """銘柄ごとの日次資産を合算する（データがない日は前日の値を使う）。"""
    df = pd.DataFrame(equities).sort_index().ffill().fillna(capital)
    return df.sum(axis=1).rename("equity")


def make_config(bt_cfg: dict) -> IntradayConfig:
    return IntradayConfig(
        initial_cash=bt_cfg["capital_per_stock"],
        lot_size=bt_cfg["lot_size"],
        position_size=bt_cfg.get("position_size", 1.0),
        costs=CostModel(**bt_cfg.get("costs", {})),
        max_entries_per_day=bt_cfg.get("max_entries_per_day", 1),
        last_entry_time=bt_cfg.get("last_entry_time", "14:30"),
    )


def candidates_of(strat_cfg: dict) -> list[dict]:
    """パラメータ候補（grid がなければ params のみ）。"""
    base = strat_cfg.get("params", {})
    grid = strat_cfg.get("grid") or {}
    return [{**base, **dict(zip(grid, values))} for values in itertools.product(*grid.values())]


def prepare_data(cfg: dict, name: str, candidates: list[dict]):
    """ユニバースを決め、各銘柄の 5 分足（必要なら日足・相場全体の列付き）を IS / OOS に分けて返す。"""
    data_cfg = cfg["data"]
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    oos_start = is_end + pd.Timedelta(days=1)
    universe = build_universe(data_cfg["dir"], is_end, cfg["universe"], Path("data/cache"))
    print(f"ユニバース: {len(universe)} 銘柄（IS 期間の売買代金上位）")

    strategies = [get_intraday_strategy(name, **p) for p in candidates]
    sma = tuple(sorted({n for s in strategies for n in s.required_sma}))
    bars_by_code = {code: load_5min(code, data_cfg["dir"]) for code in universe["code"]}
    market = None
    if any(s.requires_market for s in strategies):
        # 相場全体の状態はユニバース全銘柄から作る（各時刻の終値時点で分かる値だけ）
        market = market_context(bars_by_code)

    source = get_source(data_cfg.get("daily_source", "yfinance"))
    regime = None
    if any(s.requires_regime for s in strategies):
        # 日経平均の日足から各日の局面を作る（mode と threshold は設定ファイルの regime）
        rcfg = cfg.get("regime", {})
        nk = source.get_daily("^N225", "2000-01-01", str(max(b.index[-1] for b in bars_by_code.values()).date()))
        regime = market_regime(nk["close"], rcfg.get("mode", "confirmed"), rcfg.get("threshold", 0.05))
        print(f"局面: {rcfg.get('mode', 'confirmed')}（{rcfg.get('threshold', 0.05):.0%}）")
    is_data, oos_data = {}, {}
    for code, bars in bars_by_code.items():
        ctx = load_daily_ctx(source, code, bars.index[0] - pd.Timedelta(days=200), bars.index[-1], sma)
        bars = attach_regime(attach_market(bars, market), regime)
        is_data[code] = (bars[bars.index < oos_start], ctx)
        if (bars.index >= oos_start).any():
            oos_data[code] = (bars[bars.index >= oos_start], ctx)
    return universe, is_data, oos_data


def print_summary(label: str, s: dict) -> None:
    print(f"--- {label} ---")
    if s["num_trades"] == 0:
        print("取引なし")
        return
    print(f"取引回数        : {s['num_trades']}")
    print(f"損益合計        : {s['total_pnl']:+,.0f} 円")
    print(f"勝率            : {s['win_rate']:.1%}")
    print(f"平均リターン/回 : {s['avg_ret_bp']:+.1f} bp")
    print(f"プロフィットF   : {s['profit_factor']:.2f}")
    print(f"シャープ(合算)  : {s['sharpe']:.2f}")
    print(f"最大DD(合算)    : {s['max_drawdown']:.2%}")
    print(f"決済理由        : {s['exit_reasons']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/intraday_opening_gap.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--db", default=DEFAULT_DB, help="結果を保存する SQLite ファイル")
    parser.add_argument("--no-save", action="store_true", help="DB に保存しない")
    parser.add_argument("--note", default="", help="DB に残すメモ")
    parser.add_argument("--focus", default="", help="確認画面のタイトルに出す、この実行で重視したこと（省略時は設定から）")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    strat_cfg, bt_cfg = cfg["strategy"], cfg["backtest"]
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    oos_start = is_end + pd.Timedelta(days=1)
    capital = bt_cfg["capital_per_stock"]
    config = make_config(bt_cfg)

    name = strat_cfg["name"]
    candidates = candidates_of(strat_cfg)
    universe, is_data, oos_data = prepare_data(cfg, name, candidates)

    print(f"\n=== {name} / IS 〜{is_end.date()} / OOS {oos_start.date()}〜 ===")
    best = candidates[0]
    scores, grid_results = [], []
    if len(candidates) > 1:
        print("\nインサンプルでのパラメータ探索（損益合計で選択）:")
        for p in candidates:
            rts, eqs = run_period(is_data, name, p, config)
            s = summarize(rts, combine_equity(eqs, capital))
            scores.append((s["total_pnl"], p))
            grid_results.append({"params": p, "num_trades": s["num_trades"],
                                 "total_pnl": s["total_pnl"], "avg_ret_bp": s["avg_ret_bp"]})
            print(f"  {p}: 取引 {s['num_trades']} 回, 損益 {s['total_pnl']:+,.0f} 円, "
                  f"平均 {s['avg_ret_bp']:+.1f} bp")
        best = max(scores, key=lambda x: x[0])[1]
    print(f"\n採用パラメータ: {best}\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"intraday_{name}"
    all_rts, periods = [], {}
    for label, data in [("IS", is_data), ("OOS", oos_data)]:
        rts, eqs = run_period(data, name, best, config)
        equity = combine_equity(eqs, capital)
        summary = summarize(rts, equity)
        print_summary(f"{label}（{len(eqs)} 銘柄）", summary)
        periods[label] = (summary, rts, equity)
        if not rts.empty:
            all_rts.append(rts.assign(period=label))
            plot_equity(equity, out / f"{stem}_{label}.png", title=f"{name} {label} ({len(eqs)} stocks)")

    if all_rts:
        trades = pd.concat(all_rts, ignore_index=True)
        trades.to_csv(out / f"{stem}_trades.csv", index=False)
        by_stock = trades.groupby(["period", "code"]).agg(
            trades=("pnl", "size"), pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: (x > 0).mean()), avg_ret_bp=("ret", lambda x: x.mean() * 1e4))
        by_stock.to_csv(out / f"{stem}_by_stock.csv")
    print(f"\n結果を {out}/ に保存しました")

    if not args.no_save:
        run_config = {**cfg, "universe_codes": universe["code"].tolist(), "grid_results": grid_results}
        ctx = {"regime_mode": cfg.get("regime", {}).get("mode"), "universe": f"売買代金上位{len(universe)}銘柄",
               "focus": args.focus}
        run_id = ResultStore(args.db).save_run("intraday", name, best, run_config, periods, note=args.note,
                                                 **describe_run("intraday", name, best, ctx))
        print(f"DB に保存しました: {args.db}（run_id={run_id}）")


if __name__ == "__main__":
    main()
