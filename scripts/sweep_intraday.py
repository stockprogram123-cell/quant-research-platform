"""デイトレード戦略のパラメータを総当たりで試し、IS / OOS の成績を表にする。

    python scripts/sweep_intraday.py --config config/intraday_opening_gap.yaml \
        --param min_gap=0.005,0.01,0.02 --param stop_pct=0.01,0.015,0.02

設定ファイルの params を土台に、--param で指定した値の全組み合わせを試す（設定ファイルの grid は使わない）。
パラメータを選ぶのは IS の成績で行い、OOS は「選んだ後の確認用」として見ること。
"""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from run_intraday import combine_equity, make_config, prepare_data
from trading.backtest import round_trip_table, run_intraday_backtest, summarize
from trading.strategy import get_intraday_strategy


def run_stock(args):
    """1 銘柄で全候補を試す（並列処理の 1 単位）。"""
    code, periods, name, candidates, config = args
    out = []
    for i, params in enumerate(candidates):
        strategy = get_intraday_strategy(name, **params)
        for label, (bars, ctx) in periods.items():
            result = run_intraday_backtest(bars, strategy, config, ctx)
            out.append((i, label, code, round_trip_table(result, code), result.equity))
    return out


def extra_metrics(rts: pd.DataFrame) -> dict:
    """損切り幅と損益のトレードオフを見るための指標（bp）。"""
    if rts.empty:
        return {}
    bp = rts["ret"] * 1e4
    return {"avg_win_bp": bp[bp > 0].mean(), "avg_loss_bp": bp[bp <= 0].mean(), "worst_bp": bp.min(),
            "stop_share": (rts["exit_reason"] == "stop").mean()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--param", action="append", default=[], help="name=v1,v2,...（複数指定可）")
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name, base = cfg["strategy"]["name"], cfg["strategy"].get("params", {})
    sweep = {k: [yaml.safe_load(x) for x in v.split(",")] for k, v in (p.split("=", 1) for p in args.param)}
    candidates = [{**base, **dict(zip(sweep, values))} for values in itertools.product(*sweep.values())]
    config = make_config(cfg["backtest"])
    capital = cfg["backtest"]["capital_per_stock"]
    print(f"{name}: {len(candidates)} 通りを試します")

    _, is_data, oos_data = prepare_data(cfg, name, candidates)
    tasks = [(code, {"IS": is_data[code], **({"OOS": oos_data[code]} if code in oos_data else {})},
              name, candidates, config) for code in is_data]
    rts = {}
    eqs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for n, out in enumerate(pool.map(run_stock, tasks), 1):
            for i, label, code, r, e in out:
                rts.setdefault((i, label), []).append(r)
                eqs.setdefault((i, label), {})[code] = e
            if n % 50 == 0:
                print(f"  {n}/{len(tasks)} 銘柄")

    rows = []
    for (i, label), parts in rts.items():
        parts = [r for r in parts if not r.empty]
        r = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        s = summarize(r, combine_equity(eqs[(i, label)], capital))
        rows.append({**{k: candidates[i][k] for k in sweep}, "period": label,
                     **{k: s[k] for k in ("num_trades", "win_rate", "avg_ret_bp", "profit_factor", "total_pnl",
                                          "max_drawdown")}, **extra_metrics(r)})
    table = pd.DataFrame(rows).sort_values([*sweep, "period"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sweep_{name}.csv"
    table.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    keys = list(sweep)
    for label in ("IS", "OOS"):
        t = table[table["period"] == label]
        print(f"\n===== {label} =====")
        print(t.set_index(keys)[["num_trades", "win_rate", "avg_ret_bp", "profit_factor", "total_pnl",
                                 "avg_win_bp", "avg_loss_bp", "worst_bp", "stop_share"]].round(3).to_string())
    print(f"\n{path} に保存しました")


if __name__ == "__main__":
    main()
