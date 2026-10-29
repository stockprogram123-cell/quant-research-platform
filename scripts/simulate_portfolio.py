"""多銘柄の戦略を、1 つの資金で同時に持てる銘柄数（枠）を決めて運用したらどうなるかを調べる。

    python scripts/simulate_portfolio.py --config config/daily_bb3_long_down.yaml

run_backtest.py の多銘柄検証は銘柄ごとに 100 万円ずつ持つ想定なので、暴落日に何百銘柄も同時に建てられる。
ここでは資金（--capital）を枠の数で等分し、空いている枠の分だけ建てる（trading.backtest.slot_sim）。
同じ日の候補が空き枠より多いとき、寄りで約定する候補（9:00 に決まる）を先に、場中の候補はどれが先に約定するか
日足では分からないので乱数の順に建てる。乱数を --seeds 回変えて、成績のばらつきも出す。
調整期間（IS）と検証期間（OOS）は別々に、それぞれ --capital から始める。IS は日付が途切れずに続く最初の日（data_start）から。
比べる相手は、売買代金が --min-turnover 以上の銘柄を毎日等金額で持った場合（市場平均）。
--max-per-day で 1 日に新しく建てる数を制限できる（下げが続く局面の初日に資金を使い切らないため）。
--wait-days で下げ局面の始まりからその日数は建てない（初日の候補は成績が悪い。scripts/study_bb_spread.py）。
--rank-by 点数[:asc|desc] と --orders-per-slot で、寄りの前に注文を出す銘柄を点数の上位（空き枠 × 倍率）に絞れる。
点数は前日の大引けまでの値で計算する（order_scores、点数の一覧は FEATURES）。
--idle-in market|n225 で、使っていない現金を市場平均（上の等金額）か日経平均で運用し、建てる日だけ寄りで売って乗り換える
（指数の売り買いは片道 --switch-cost）。このときの比べる相手は同じ指数を持ちっぱなしにした場合。
--save-slots の枠の数・最初の乱数の結果を DB に保存する。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_backtest import load_stocks
from trading.backtest import CostModel, summarize
from trading.backtest.metrics import compute_metrics, max_drawdown
from trading.backtest.slot_sim import simulate_slots
from trading.backtest.touch_engine import strategy_trades
from trading.data import get_source
from trading.data.index_open import tradable_open
from trading.store import DEFAULT_DB, ResultStore, describe_run, universe_label
from trading.strategy import get_strategy


def candidates(stocks: dict[str, pd.DataFrame], strategy) -> pd.DataFrame:
    """全銘柄の取引の候補（銘柄ごとに独立に作ったもの）。"""
    rows = []
    for code, df in stocks.items():
        o, idx = df["open"].to_numpy(float), df.index
        for t in strategy_trades(df, strategy):
            rows.append({"code": code, "entry_date": idx[t.entry_i], "exit_date": idx[t.exit_i], "side": t.side,
                         "entry_price": t.entry_price, "exit_price": t.exit_price, "reason": t.reason,
                         "priority": 0 if t.entry_price == o[t.entry_i] else 1})
    return pd.DataFrame(rows)


FEATURES = {
    "dist": "前日終値から -3σ の値段までの距離（前日終値に対する割合）",
    "dist_z": "その距離を 20 日の値動きの大きさで割ったもの",
    "vol20": "20 日の日次リターンの標準偏差",
    "ret5": "直近 5 日のリターン",
    "ret20": "直近 20 日のリターン",
    "sma75": "75 日移動平均からの乖離",
    "high250": "1 年（250 日）高値からの下落率",
    "turnover": "20 日平均売買代金（円）",
    "price": "前日終値",
}


def order_scores(df: pd.DataFrame, strategy) -> pd.DataFrame:
    """買いの注文を出せる日（-3σ の値段があり can_enter）の点数（FEATURES）。すべて前日の大引けまでの値。"""
    od = strategy.orders(df)
    c = df["close"]
    pc, vol = c.shift(1), c.pct_change(fill_method=None).rolling(20).std().shift(1)
    dist = pc / od["long_entry"] - 1
    out = pd.DataFrame({
        "dist": dist, "dist_z": dist / vol, "vol20": vol,
        "ret5": pc / c.shift(6) - 1, "ret20": pc / c.shift(21) - 1,
        "sma75": pc / c.rolling(75).mean().shift(1) - 1, "high250": pc / c.rolling(250).max().shift(1) - 1,
        "turnover": (c * df["volume"]).rolling(20).mean().shift(1), "price": pc,
    }, index=df.index)
    return out[od["long_entry"].notna() & od["can_enter"]]


def all_scores(stocks: dict[str, pd.DataFrame], strategy) -> pd.DataFrame:
    """全銘柄の order_scores を縦に並べたもの（列 date / code / 点数）。"""
    return pd.concat([order_scores(df, strategy).rename_axis("date").reset_index().assign(code=code)
                      for code, df in stocks.items()], ignore_index=True)


def add_rank(cands: pd.DataFrame, scores: pd.DataFrame, feature: str, ascending: bool) -> pd.DataFrame:
    """その日に注文を出せる銘柄の中での順位（1 が最良）を cands の rank 列に付ける。点数がない銘柄は最下位。"""
    rank = scores.groupby("date")[feature].rank(ascending=ascending, method="first", na_option="bottom")
    key = scores[["date", "code"]].assign(rank=rank)
    out = cands.merge(key, left_on=["entry_date", "code"], right_on=["date", "code"], how="left").drop(columns="date")
    return out.assign(rank=out["rank"].fillna(np.inf))


def regime_day(regime_file: str, threshold: float) -> pd.Series:
    """各日が下げ局面の何日目か（前日の大引け時点の確率で判定。局面でない日は 0）。"""
    on = pd.read_csv(regime_file, index_col=0, parse_dates=True)["p_down"].shift(1) > threshold
    return on.groupby((~on).cumsum()).cumsum()


MAX_GAP_DAYS = 14   # 日付がこれより長く空いたらデータの穴とみなす（年末年始・大型連休は 11 日まで）


def data_start(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """日付が途切れずに続く最初の日（最後のデータの穴の直後）。IS はここから始める。
    data_daily は 1 銘柄 6,000 行までなので、2001 年 7 月より前は数十銘柄しかなく、日付に 1 年以上の穴がある。"""
    gap = dates.to_series().diff() > pd.Timedelta(days=MAX_GAP_DAYS)
    return dates[gap.to_numpy()][-1] if gap.any() else dates[0]


def drop_gap_returns(r: pd.DataFrame) -> pd.DataFrame:
    """データの穴をまたぐ日の騰落率を 0 にする（穴の間の値動きを 1 日で計上しないため）。"""
    gap = (r.index.to_series().diff() > pd.Timedelta(days=MAX_GAP_DAYS)).to_numpy()
    out = r.copy()
    out.loc[gap] = 0.0
    return out


def market_returns(stocks: dict[str, pd.DataFrame], min_turnover: float) -> pd.DataFrame:
    """前日までの 20 日平均売買代金が min_turnover 以上の銘柄を毎日等金額で持った場合の、日々の騰落率（ret: 前日の大引け
    から大引けまで / gap: 前日の大引けから寄りまで）。"""
    closes = pd.DataFrame({c: d["close"] for c, d in stocks.items()}).sort_index()
    opens = pd.DataFrame({c: d["open"] for c, d in stocks.items()}).reindex(closes.index)
    liquid = pd.DataFrame({c: (d["close"] * d["volume"]).rolling(20).mean() >= min_turnover
                           for c, d in stocks.items()}).reindex(closes.index).fillna(False).shift(1, fill_value=False)
    prev = closes.shift(1)
    return drop_gap_returns(pd.DataFrame({"ret": (closes / prev - 1).where(liquid).mean(axis=1).fillna(0),
                                          "gap": (opens / prev - 1).where(liquid).mean(axis=1).fillna(0)}))


def market_index(stocks: dict[str, pd.DataFrame], min_turnover: float) -> pd.Series:
    """market_returns の市場平均の値動き（最初の日を 1 とした値）。"""
    return (1 + market_returns(stocks, min_turnover)["ret"]).cumprod()


N225_ETF = "1330"   # 日興 上場投信 日経225（調整しない値段で読む）


def n225_daily(end: str, cache_dir: str | None = "data/cache", start: str = "2000-01-01") -> pd.DataFrame:
    """日経平均の日足に、寄りで実際に売買できる値段の目安 open_tradable を足したもの。

    指数の始値は 9:00 にまだ寄っていない銘柄を前日の終値のまま計算するので、大きく下に（上に）窓を開ける日ほど、
    実際に売買できる値段より前日の終値に近く出る（2009 年以降、指数が寄りで -0.84% 以下の日は、指数の窓 -129bp に対して
    ETF の窓 -162bp。寄り→引けは指数 -40bp・ETF -1bp）。そこで寄りの値段は ETF の寄りから求める（trading.data.index_open.tradable_open。
    配当落ちの日や前日の ETF の終値が指数からずれていた日は、その日の ETF の寄り ÷ 引けの比を使う）。ETF の値がない日（2009 年より前・
    出来高のない日）は大引けで売買したとみなす。
    cache_dir=None なら取得結果をキャッシュしない（同じ日のうちに取り直すと場中の値が残るため）。"""
    nk = get_source("yfinance", cache_dir).get_daily("^N225", start, end)
    etf = get_source("yfinance_raw", cache_dir).get_daily(N225_ETF, max(start, "2008-01-01"), end)
    return nk.assign(open_tradable=tradable_open(nk["close"], etf))


def index_returns(kind: str, stocks: dict[str, pd.DataFrame], min_turnover: float, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """使っていない現金を運用する指数の日々の騰落率（ret / gap）。kind は market（市場平均）か n225（日経平均、配当なし）。
    n225 の gap は寄りで実際に売買できる値段（n225_daily の open_tradable）まで。"""
    if kind == "market":
        return market_returns(stocks, min_turnover).reindex(dates).fillna(0)
    nk = n225_daily(str(dates[-1].date())).reindex(dates).ffill()
    prev = nk["close"].shift(1)
    return drop_gap_returns(pd.DataFrame({"ret": nk["close"] / prev - 1, "gap": nk["open_tradable"] / prev - 1}).fillna(0))


def stats(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    m = compute_metrics(equity, trades if trades is not None else pd.DataFrame({"pnl": []}))
    years = equity.groupby(equity.index.year).last() / equity.groupby(equity.index.year).first() - 1
    return {"年率": m["cagr"], "最大DD": m["max_drawdown"], "シャープ": m["sharpe"],
            "プラスの年": (years > 0).mean(), "最悪の年": years.min()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/daily_bb3_long_down.yaml")
    parser.add_argument("--capital", type=float, default=10_000_000, help="運用資金（円）")
    parser.add_argument("--slots", default="5,10,20,40", help="同時に持てる銘柄数（カンマ区切りで複数）")
    parser.add_argument("--seeds", type=int, default=20, help="同じ日の場中の候補の順番を変えて試す回数")
    parser.add_argument("--min-turnover", type=float, default=1e8, help="市場平均に入れる銘柄の売買代金の下限")
    parser.add_argument("--max-per-day", type=int, default=None, help="1 日に新しく建てる銘柄数の上限")
    parser.add_argument("--rank-by", default=None, help="注文を出す銘柄を選ぶ点数（例: dist_z:asc）")
    parser.add_argument("--orders-per-slot", type=float, default=None, help="空き枠 1 つあたり何銘柄に注文を出すか")
    parser.add_argument("--wait-days", type=int, default=0,
                        help="下げ局面（前日の p_down > long_min_p_down）の始まりからこの日数は建てない（scripts/study_bb_spread.py）")
    parser.add_argument("--idle-in", choices=["market", "n225"], default=None,
                        help="使っていない現金を運用する指数（market = 市場平均、n225 = 日経平均）")
    parser.add_argument("--switch-cost", type=float, default=0.0005, help="指数の売り買いの片道の費用（割合）")
    parser.add_argument("--save-slots", type=int, default=10, help="DB に保存する枠の数（0 なら保存しない）")
    parser.add_argument("--focus", default="", help="確認画面のタイトルに出す、この実行で重視したこと（省略時は設定から）")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    pd.set_option("display.width", 250)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    strat_cfg, bt_cfg = cfg["strategy"], cfg["backtest"]
    strategy = get_strategy(strat_cfg["name"], **strat_cfg.get("params", {}))
    costs = CostModel(**bt_cfg.get("costs", {}))
    stocks = load_stocks(cfg)
    cands = candidates(stocks, strategy)
    rank_note = ""
    if args.wait_days:
        rday = regime_day(cfg["data"]["regime_file"], strat_cfg["params"]["long_min_p_down"])
        cands = cands[rday.reindex(cands["entry_date"]).to_numpy() > args.wait_days].reset_index(drop=True)
        rank_note += f" / 下げ局面の {args.wait_days} 日目までは建てない"
    if args.rank_by:
        feature, _, direction = args.rank_by.partition(":")
        cands = add_rank(cands, all_scores(stocks, strategy), feature, direction != "desc")
        rank_note = f" / {FEATURES[feature]}の{'大きい' if direction == 'desc' else '小さい'}順に空き枠×{args.orders_per_slot:g}銘柄へ注文"
    closes = pd.DataFrame({c: d["close"] for c, d in stocks.items()}).sort_index()
    if args.idle_in:
        idx_ret = index_returns(args.idle_in, stocks, args.min_turnover, closes.index)
        market = (1 + idx_ret["ret"]).cumprod()
        rank_note += f" / 使っていない現金は{'市場平均' if args.idle_in == 'market' else '日経平均'}で運用"
    else:
        idx_ret, market = None, market_index(stocks, args.min_turnover)
    bench = "市場平均" if args.idle_in != "n225" else "日経平均"
    is_end = pd.Timestamp(cfg["split"]["is_end"])
    periods = {"IS": (data_start(closes.index), is_end), "OOS": (is_end + pd.Timedelta(days=1), closes.index[-1])}
    slot_list = [int(s) for s in args.slots.split(",")]

    limit = (f" / 1 日 {args.max_per_day} 銘柄まで" if args.max_per_day else "") + rank_note
    print(f"{cfg.get('name', strat_cfg['name'])} / {len(stocks)} 銘柄 / 資金 {args.capital:,.0f} 円{limit}")
    per_day = cands.groupby("entry_date").size()
    print(f"取引の候補 {len(cands):,} 件、候補のあった日 {len(per_day):,} 日（1 日平均 {per_day.mean():.1f} 件、"
          f"最多 {per_day.max()} 件: {per_day.idxmax().date()}）")

    rows, yearly, saved = [], {}, None
    for per, (start, end) in periods.items():
        mk = market[start:end]
        rows.append({"期間": per, "枠": bench, **stats(mk / mk.iloc[0])})
        yearly[(per, bench)] = mk.groupby(mk.index.year).last() / mk.groupby(mk.index.year).first() - 1
        c = cands[(cands["entry_date"] >= start) & (cands["entry_date"] <= end)]
        for slots in slot_list:
            runs = []
            for seed in range(args.seeds):
                r = simulate_slots(c, closes[start:end], args.capital, slots, costs, bt_cfg["lot_size"],
                                   np.random.default_rng(seed), max_per_day=args.max_per_day,
                                   orders_per_slot=args.orders_per_slot if args.rank_by else None,
                                   cash_return=None if idx_ret is None else idx_ret["ret"][start:end],
                                   cash_gap=None if idx_ret is None else idx_ret["gap"][start:end],
                                   switch_cost=args.switch_cost)
                runs.append({**stats(r.equity, r.trades), "取引数": len(r.trades),
                             "見送り": (r.skipped["full"] + r.skipped["day"] + r.skipped["rank"]) / max(len(c), 1),
                             "平均の投資割合": r.exposure.mean(), "勝率": (r.trades["pnl"] > 0).mean(),
                             "平均bp": r.trades["ret"].mean() * 1e4})
                if seed == 0:
                    eq = r.equity
                    yearly[(per, f"{slots}枠")] = eq.groupby(eq.index.year).last() / eq.groupby(eq.index.year).first() - 1
                    if slots == args.save_slots:
                        saved = saved or {}
                        saved[per] = (summarize(r.trades, r.equity), r.trades, r.equity)
            g = pd.DataFrame(runs)
            rows.append({"期間": per, "枠": f"{slots}枠", **g.median().to_dict(),
                         "年率（下位10%）": g["年率"].quantile(0.1), "年率（上位10%）": g["年率"].quantile(0.9),
                         "最大DD（悪い方10%）": g["最大DD"].quantile(0.1)})
    table = pd.DataFrame(rows)
    pct = ["年率", "年率（下位10%）", "年率（上位10%）", "最大DD", "最大DD（悪い方10%）", "最悪の年", "プラスの年",
           "見送り", "平均の投資割合", "勝率"]
    show = table.copy()
    for col in pct:
        show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if "年率" in col or "DD" in col
                                  or col == "最悪の年" else f"{v:.0%}")
    show["シャープ"] = show["シャープ"].round(2)
    show["取引数"] = show["取引数"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
    show["平均bp"] = show["平均bp"].map(lambda v: "" if pd.isna(v) else f"{v:+.0f}")
    print(f"\n乱数 {args.seeds} 回の中央値（年率は下位・上位 10% も）。IS {periods['IS'][0].date()}〜{is_end.date()} / "
          "OOS それより後、それぞれ資金を元に戻して開始")
    print(show[["期間", "枠", "年率", "年率（下位10%）", "年率（上位10%）", "最大DD", "最大DD（悪い方10%）", "シャープ",
                "プラスの年", "最悪の年", "取引数", "勝率", "平均bp", "見送り", "平均の投資割合"]]
          .to_string(index=False))

    y = pd.DataFrame(yearly)
    y.columns = [f"{p} {s}" for p, s in y.columns]
    y = y.T.groupby(lambda k: k.split(" ")[1]).first().T   # IS と OOS の年は重ならないので 1 列にまとめる
    print("\n年ごとの損益率（乱数 1 回目）")
    print(y[[bench] + [f"{s}枠" for s in slot_list]].map(lambda v: f"{v:+.0%}" if pd.notna(v) else "")
          .to_string())

    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/slot_sim.csv", index=False)
    if saved:
        params = {**strat_cfg.get("params", {}), "slots": args.save_slots, "capital": args.capital,
                  "max_per_day": args.max_per_day, "rank_by": args.rank_by, "orders_per_slot": args.orders_per_slot,
                  "wait_days": args.wait_days or None,
                  "idle_in": args.idle_in, "switch_cost": args.switch_cost if args.idle_in else None}
        run_config = {**cfg, "backtest": {**bt_cfg, "initial_cash": args.capital, "slots": args.save_slots}}
        desc = describe_run("daily", strat_cfg["name"], params,
                            {"universe": universe_label(cfg, len(stocks)), "focus": args.focus})
        run_id = ResultStore(args.db).save_run("daily", strat_cfg["name"], params, run_config, saved,
                                               note="資金制約あり（scripts/simulate_portfolio.py）。IS と OOS は別々に資金を戻して開始",
                                               **desc)
        print(f"\nDB に保存しました: {args.db}（run_id={run_id}）: {desc['name']}")


if __name__ == "__main__":
    main()
