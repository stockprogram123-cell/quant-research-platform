"""ペーパートレード（PT）。実際の注文は出さない。DB の実行（--run）と同じ決まりの口座を 1 日ずつ動かし、毎日の大引けの後に記録する。
決まりの違う口座をいくつも並べて動かせる。DB #85 の決まりの口座を「#85 PT」と呼び、状態と報告を data/paper/85/ に置き、
DB #85 の期間 PT として記録する（ペーパートレードに別の番号は付けない）。

    python scripts/paper_trade.py --init --run 85 --start-date 2023-03-31   # #85 PT を始める（局面モデル・銘柄・決まりを固定する）
    python scripts/paper_trade.py                                   # 毎営業日の 16:30 以降に 1 回。全部の PT の約定・時価評価・翌日の注文
    python scripts/paper_trade.py --run 87                          # #87 PT だけ進める
    python scripts/paper_trade.py --replay 2025-01-06 --local --run 85   # 手元の日足で過去の日から動かし、DB の実行の取引と比べる（確かめ用）

--start-date が手元の日足の最後の日（--local-end）より前なら、そこまでは手元の日足で後から計算し、その後は yfinance の日足で続ける
（#85 PT・#87 PT は 2023-03-31 の大引けの後から。2023 年 4 月以降を、手法を選ぶのに使わない取り置きの期間としたため）。

決まり（trading.paper を参照）: 個別株に入っていない資金は、DB の実行の index_weight（#75 は 1、#77・#79・#81 は 0.5、#83・#85 は 0.75）を日経平均（ETF）、残りを現金で持ち、
- DB #73: 終値が -2σ 以下で引けた貸借銘柄（最初の割れ・IS で選んだ 4 条件）を翌日の寄りで買い、-1σ に戻った日の大引けか 10 営業日目に売る
- DB #60: -3σ に触れた貸借銘柄が 10 以上（#85 からは日経平均の終値が −2.5σ より下）で前日の下げ相場の確率が 0.5 超の日の翌日の寄りで、日経平均を資産 × index_weight だけ買い増す
  （まず現金を使い、足りない分を信用で借りる）
--init のときに固定するもの（data/paper/<番号>/setup.json）: 局面モデル（2015 年までで学習した重み。data/cache/regime_daily.csv と同じ値）、
  局面の特徴に使う 1,191 銘柄、売買する貸借銘柄（今の一覧）、DB の実行の決まり。あとから変えると前向きの検証にならない。
データ: yfinance の調整しない値段（分割だけ調整済み。手元の data_daily と一致することを確かめてある）を毎回 2 年分取り直す
  （全部の PT の銘柄をまとめて 1 回取る）。16:30 より前に動かすと、その日の足は場中の値かもしれないので使わない（--cutoff で変えられる）。
口座の状態は data/paper/<番号>/account.json、毎日の報告は data/paper/<番号>/reports/日付.md、
確認画面には DB の実行（#85 など）の期間「ペーパートレード（PT）」として出る。
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from simulate_portfolio import n225_daily
from trading.backtest import CostModel, summarize
from trading.data import get_source, list_codes, load_daily_csv
from trading.data.base import normalize
from trading.data.jpx import margin_codes, margin_list, split_history, unadjust_factor
from trading.paper import Account, compute_signals, equity_frame, exit_limit, next_orders, step
from trading.regime import RegimeModel, regime_features, zigzag_regime
from trading.store import DEFAULT_DB, ResultStore

DATA_DIR = "data/daily"
PAPER = Path("data/paper")
WINDOW_DAYS = 730          # 毎回取り直す日足の長さ（暦日）。特徴に要るのは 1 年ほど
JST = ZoneInfo("Asia/Tokyo")


# ---- 開始時に固定するもの ----
def freeze_regime(nk_end: str = "2026-10-23") -> dict:
    """局面モデルを scripts/study_regime_daily.py --stocks-dir と同じ形で学習し直す（data_daily のうちコードが 7〜9 で始まる、
    300 日以上ある 1,191 銘柄。data/cache/regime_daily.csv を作ったときの銘柄）。"""
    nk = get_source("yfinance").get_daily("^N225", "2000-01-01", nk_end)
    stocks = {c: d for c in list_codes(DATA_DIR) if c[0] in "789" and len(d := load_daily_csv(c, DATA_DIR)) >= 300}
    feats = regime_features(nk, stocks)
    label = zigzag_regime(nk["close"], 0.10)
    usable = feats.dropna().index.intersection(label.dropna().index)
    train = usable[usable <= "2015-12-31"]
    model = RegimeModel(list(feats.columns), l2=5.0).fit(feats.loc[train], label[train])
    ref = pd.read_csv("data/cache/regime_daily.csv", index_col=0, parse_dates=True)["p_down"]
    diff = (model.predict(feats.dropna()) - ref).abs().max()
    print(f"局面モデル: {len(stocks):,} 銘柄で学習。data/cache/regime_daily.csv との差 最大 {diff:.2e}")
    return {"model": model.to_dict(), "codes": sorted(stocks)}


def setup_from_run(run_id: int, db: str) -> dict:
    run = ResultStore(db).run(run_id)
    close_p, index_p = run["params"]["parts"]
    return {"run": run_id, "name": run["name"], "slots": run["params"]["slots"], "rules": close_p["rules"],
            "max_hold": int(close_p["exit"].split("最長")[1].removesuffix("日")), "min_stocks": index_p.get("min_stocks"),
            "index_signal": index_p.get("signal", "C"), "index_k": index_p.get("k"),
            "index_p_down": index_p["long_min_p_down"], "index_max_hold": index_p["max_hold"],
            "extra": index_p["extra"], "rate": index_p["rate"], "switch_cost": index_p["switch_cost"],
            "index_weight": run["params"].get("index_weight", 1.0), "costs": run["config"]["backtest"]["costs"],
            "params": run["params"]}


# ---- データ ----
def fetch_yf(codes: list[str], start: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """yfinance の調整しない日足（分割だけ調整済み）と、分割前の値段に戻す係数。出来高 0 の日は除く（手元のデータと同じ）。"""
    import yfinance as yf
    warnings.filterwarnings("ignore")
    stocks, factors = {}, {}
    todo = list(codes)
    for attempt in range(3):
        missing = []
        for i in range(0, len(todo), 200):
            chunk = todo[i:i + 200]
            d = yf.download([c + ".T" for c in chunk], start=start, auto_adjust=False, actions=True,
                            group_by="ticker", threads=True, progress=False)
            for c in chunk:
                try:
                    x = d[c + ".T"].dropna(subset=["Close"])
                except KeyError:
                    missing.append(c)
                    continue
                x = x[x["Volume"] > 0]
                if x.empty:
                    missing.append(c)
                    continue
                df = normalize(x)
                stocks[c] = df
                sp = x.get("Stock Splits", pd.Series(dtype=float))
                sp = sp[sp > 0]
                if len(sp):
                    sp.index = pd.to_datetime(sp.index).tz_localize(None).normalize()
                    factors[c] = unadjust_factor(df.index, sp)
        todo = missing
        if not todo:
            break
        time.sleep(5)
    if todo:
        print(f"取れなかった銘柄 {len(todo)}: {', '.join(todo[:20])}{' …' if len(todo) > 20 else ''}")
    return stocks, factors


def fetch_local(codes: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """手元の data_daily（--replay --local）。"""
    splits = split_history(codes)
    stocks, factors = {}, {}
    for c in codes:
        try:
            df = load_daily_csv(c, DATA_DIR)
        except FileNotFoundError:
            continue
        f = unadjust_factor(df.index, splits.get(c, pd.Series(dtype=float)))
        df, f = df.loc[start:end], f.loc[start:end]
        if len(df):
            stocks[c], factors[c] = df, f
    return stocks, factors


def setup_codes(setups: list[dict]) -> list[str]:
    """PT の売買する銘柄と局面の特徴に使う銘柄を、全部の PT について合わせたもの。"""
    return sorted(set().union(*(set(s["universe"]) | set(s["regime"]["codes"]) for s in setups)))


def fetch(codes: list[str], start: str, end: str | None, local: bool) -> tuple:
    """銘柄の日足・分割の係数・日経平均（寄りで売買できる値段つき）。"""
    if local:
        stocks, factors = fetch_local(codes, start, end)
        nk = n225_daily(end, start=start)
    else:
        print(f"日足を取っています（{len(codes):,} 銘柄・{start} から）", flush=True)
        stocks, factors = fetch_yf(codes, start)
        nk = n225_daily(datetime.now(JST).strftime("%Y-%m-%d"), cache_dir=None, start=start)
    nk = nk[nk.index >= start]
    if end:
        nk = nk[nk.index <= end]
    return stocks, factors, nk


def prepare(setup: dict, stocks: dict, nk: pd.DataFrame) -> tuple:
    """PT の売買する銘柄と、その日の下げ相場の確率（PT に固定した局面モデルで計算する）。"""
    model = RegimeModel.from_dict(setup["regime"]["model"])
    feats = regime_features(nk, {c: stocks[c] for c in setup["regime"]["codes"] if c in stocks})
    trade = {c: stocks[c] for c in setup["universe"] if c in stocks and len(stocks[c]) >= 100}
    return trade, model.predict(feats.dropna())


def market(setup: dict, start: str, end: str | None, local: bool) -> tuple:
    """売買する銘柄の日足・分割の係数・日経平均・その日の下げ相場の確率（1 つの PT の分だけ取る）。"""
    stocks, factors, nk = fetch(setup_codes([setup]), start, end, local)
    trade, p_down = prepare(setup, stocks, nk)
    return trade, factors, nk, p_down


def model_label(setup: dict) -> str:
    if setup.get("index_signal", "C") == "B":
        return f"買い増しは日経平均 −{setup['index_k']:g}σ"
    return f"買い増しは −3σ に触れた銘柄 {setup['min_stocks']} 以上"


def complete_days(nk: pd.DataFrame, cutoff: str) -> pd.DatetimeIndex:
    """大引けの値が確定している日（今日の足は cutoff の時刻より前なら場中の値かもしれないので除く）。"""
    now = datetime.now(JST)
    days = nk.index
    if days[-1].date() == now.date() and now.strftime("%H:%M") < cutoff:
        days = days[:-1]
    return days


# ---- 報告 ----
def names() -> dict[str, str]:
    try:
        m = margin_list()
        return dict(zip(m["code"], m["name"]))
    except Exception:
        return {}


def report(acct: Account, setup: dict, sig, day: pd.Timestamp) -> str:
    nm = names()
    e = equity_frame(acct)
    m = sig.market.loc[day]
    ds = str(day.date())
    eq = e["equity"].iloc[-1]
    nk0 = e["nk_close"].iloc[0]
    live = setup["live_from"]
    lines = [f"# #{setup['run']} PT（{model_label(setup)}） {ds}"
             f"（{setup['start']}に開始・資金 {acct.capital / 1e4:,.0f} 万円）", "",
             "実際の注文は出していません。DB #{}（{}）と同じ決まりの口座です。".format(setup["run"], setup["name"])
             + (f"{live} までは手元の日足で後から計算し、その後は毎日の大引けの後に記録しています。" if live > setup["start_date"] else
                "毎日の大引けの後に記録しています。"), "",
             "## 口座", "",
             f"- 資産 {eq:,.0f} 円（開始から {eq / acct.capital - 1:+.2%}）。同じ期間の日経平均 {m['nk_close'] / nk0 - 1:+.2%}"]
    if live > setup["start_date"] and pd.Timestamp(live) in e.index:
        e0 = e.loc[pd.Timestamp(live)]
        lines.append(f"- 毎日の記録を始めた {live} の大引けから: 資産 {eq / e0['equity'] - 1:+.2%}・日経平均 {m['nk_close'] / e0['nk_close'] - 1:+.2%}")
    lines.append(f"- 日経平均（ETF）で持っている額 {acct.idle:,.0f} 円")
    o = acct.overlay
    used = o["units"] * o["entry_price"] - o.get("debt", o["units"] * o["entry_price"]) if o else 0.0
    if acct.index_weight < 1:
        lines.append(f"- 現金 {acct.cash - used:,.0f} 円" + (f"（ほかに {used:,.0f} 円を買い増しにあてている）" if used else "")
                     + f"。待機中の資金のうち日経平均は {acct.index_weight:.0%} にそろえる")
    if o:
        lines.append(f"- 日経平均の買い増し: {o['units'] * m['nk_close']:,.0f} 円分（{o['entry_date']} に {o['entry_price']:,.0f} で建て、"
                     f"信用の借入 {o.get('debt', o['units'] * o['entry_price']):,.0f} 円、"
                     f"損益 {o['units'] * (m['nk_close'] - o['entry_price']) - o['fee'] - o['interest']:+,.0f} 円）")
    else:
        lines.append("- 日経平均の買い増し: なし")
    lines += ["", "## 持っている銘柄", ""]
    if acct.positions:
        lines += ["| コード | 名前 | 株数 | 買った日 | 買値 | 終値 | 損益 | 合図から |", "|---|---|--:|---|--:|--:|--:|--:|"]
        for p in acct.positions:
            c = sig.feats[p["code"]]["close"][:day].iloc[-1]
            lines.append(f"| {p['code']} | {nm.get(p['code'], '')} | {p['shares']:,} | {p['entry_date']} | {p['entry_price']:,.1f} | "
                         f"{c:,.1f} | {p['shares'] * (c - p['entry_price']):+,.0f} | {p['days']} 日 |")
    else:
        lines.append("なし")
    today = [t for d, t in acct.events if d == ds]
    lines += ["", "## 今日の約定・見送り", ""] + ([f"- {t}" for t in today] or ["なし"])
    lines += ["", "## 次の営業日の注文", ""]
    orders = []
    if not acct.in_index:
        orders.append(f"- 寄りで日経平均（ETF）を {acct.idle:,.0f} 円分買う（待機中の資金の {acct.index_weight:.0%}。最初の日だけ）"
                      + (f"。残りの {acct.cash:,.0f} 円は現金で持つ" if acct.cash > 0 else ""))
    cost = 0.0
    for b in acct.orders["buy"]:
        orders.append(f"- 寄りの成行で買う: {b['code']} {nm.get(b['code'], '')} {b['shares']:,} 株"
                      f"（目安 {b['shares'] * b['est_price']:,.0f} 円）")
        cost += b["shares"] * b["est_price"]
    if cost:
        share = acct.nk_share()
        orders.append(f"- 寄りで日経平均（ETF）を {cost * share:,.0f} 円分売って、上の買いにあてる" +
                      (f"（残りの {cost * (1 - share):,.0f} 円は現金から）" if share < 1 else ""))
    for p in acct.positions:
        if p.get("exit_next_open"):
            orders.append(f"- 寄りの成行で売る: {p['code']} {nm.get(p['code'], '')} {p['shares']:,} 株（前の日がストップ安で引けたため）")
        elif p["days"] + 1 >= acct.max_hold:
            orders.append(f"- 大引けの成行で売る: {p['code']} {nm.get(p['code'], '')} {p['shares']:,} 株（最長の日数）")
        else:
            lim = exit_limit(sig.feats[p["code"]]["close"][:day])
            orders.append(f"- 引け指値で売る: {p['code']} {nm.get(p['code'], '')} {p['shares']:,} 株・{lim:,.1f} 円"
                          "（終値がこれ以上なら −1σ に戻ったことになる）")
    if acct.orders.get("overlay_buy"):
        value = eq * acct.extra * acct.index_weight
        debt = max(0.0, value - max(acct.cash - cost * (1 - acct.nk_share()), 0.0))
        how = "信用" if debt >= value else f"現金 {value - debt:,.0f} 円" + (f"・信用 {debt:,.0f} 円" if debt > 0 else "")
        orders.append(f"- 寄りで日経平均（ETF）を {value:,.0f} 円分買い増す（{how}）")
    if o:
        if o["days"] + 1 >= acct.index_max_hold:
            orders.append("- 日経平均の買い増し分を大引けで売る（最長の日数）")
        else:
            lim = exit_limit(sig.market["nk_close"][:day])
            orders.append(f"- 日経平均の終値が {lim:,.0f} 以上なら、買い増し分を大引けで売る（−1σ に戻ったことになる）")
    lines += orders or ["なし"]
    lines += ["", "## 合図の状況（今日の大引け）", "",
              f"- 下げ相場の確率 {m['p_down']:.3f}（次の営業日の判定に使う。"
              f"個別株は {dict((f, v) for f, _, v in setup['rules'])['p_down']:.3f} 以上、買い増しは {setup['index_p_down']} 超）",
              f"- −2σ 以下で引けた銘柄の割合 {m['breadth']:.1%}（前日から {m['breadth_chg']:+.1%}）",
              f"- 日経平均の終値の位置 {m['nk_z']:+.2f}σ（−{setup['index_k']:g}σ より下で買い増しの合図。前日の確率 {m['p_down_prev']:.3f}）"
              if setup.get("index_signal", "C") == "B" else
              f"- −3σ に触れた銘柄 {m['count3']:.0f}（{setup['min_stocks']} 以上で買い増しの合図。前日の確率 {m['p_down_prev']:.3f}）"]
    todays = sig.buy[sig.buy["date"] == day]
    if len(todays):
        lines.append("- 条件を満たした −2σ 割れ: " + "、".join(f"{c} {nm.get(c, '')}" for c in todays["code"]))
    return "\n".join(lines) + "\n"


# ---- DB ----
def sync_db(acct: Account, setup: dict, db: str) -> None:
    """DB の実行（setup["run"]）の期間 PT を、口座の取引と資産で入れ替える。PT の始め方は実行の設定の paper に書く（確認画面が使う）。"""
    store = ResultStore(db)
    trades = pd.DataFrame(acct.trades, columns=["code", "entry_date", "exit_date", "shares", "entry_price", "exit_price",
                                                "pnl", "ret", "exit_reason", "side"])
    for col in ("entry_date", "exit_date"):
        trades[col] = pd.to_datetime(trades[col])
    eq = equity_frame(acct)["equity"]
    store.replace_periods(setup["run"], {"PT": (summarize(trades, eq), trades, eq)})
    config = store.run(setup["run"])["config"]
    paper = {"dir": f"{PAPER.as_posix()}/{setup['run']}", "start": setup["start_date"], "live_from": setup["live_from"],
             "capital": acct.capital, "universe": f"貸借{len(setup['universe']):,}銘柄"}
    if config.get("paper") != paper:
        store.update_config(setup["run"], {**config, "paper": paper})


def save(acct: Account, path: Path) -> None:
    path.write_text(json.dumps(acct.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


def run_days(acct: Account, sig, days, costs) -> None:
    for d in days:
        if acct.last_date is None or d > pd.Timestamp(acct.last_date):
            step(acct, d, sig, costs)


def new_account(setup: dict, capital: float) -> Account:
    w = setup.get("index_weight", 1.0)
    return Account(capital=capital, slots=setup["slots"], max_hold=setup["max_hold"],
                   index_max_hold=setup["index_max_hold"], extra=setup["extra"], rate=setup["rate"],
                   switch_cost=setup["switch_cost"], index_weight=w, idle=capital * w, cash=capital * (1 - w))


def start_account(setup: dict, sig, d0: pd.Timestamp, capital: float, costs: CostModel) -> Account:
    """d0 の大引けの後に始める口座（次の営業日の寄りで日経平均を買う）。"""
    acct = new_account(setup, capital)
    acct.equity.append([str(d0.date()), acct.capital, float(sig.market.loc[d0, "nk_close"])])
    acct.orders, acct.last_date = next_orders(acct, d0, sig, costs, acct.capital), str(d0.date())
    acct.events.append([str(d0.date()), f"開始。次の営業日の寄りで {acct.idle:,.0f} 円分の日経平均（ETF）を買う"
                        + (f"（残りの {acct.cash:,.0f} 円は現金）" if acct.cash > 0 else "")])
    return acct


def signals(setup: dict, stocks, factors, nk, p_down):
    return compute_signals(stocks, nk, p_down, setup["rules"], setup["min_stocks"], setup["index_p_down"], factors,
                           setup.get("index_signal", "C"), setup.get("index_k"))


def model_dirs() -> list[Path]:
    return [d for d in sorted(PAPER.iterdir()) if (d / "setup.json").exists()] if PAPER.exists() else []


def advance(mdir: Path, setup: dict, acct: Account, sig, day: pd.Timestamp, db: str) -> None:
    """口座と決まりを保存し、DB の実行の期間 PT と報告を書く。"""
    save(acct, mdir / "account.json")
    sync_db(acct, setup, db)
    (mdir / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=1), encoding="utf-8")
    text = report(acct, setup, sig, day)
    (mdir / "reports").mkdir(exist_ok=True)
    (mdir / "reports" / f"{day.date()}.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"DB #{setup['run']} の PT に記録しました。報告: {mdir / 'reports' / f'{day.date()}.md'}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="--run の決まりの PT を始める")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--run", type=int, default=None,
                        help="決まりを写す DB の実行（scripts/simulate_small_account.py）。毎日の実行では、その番号の PT だけ進める")
    parser.add_argument("--start-date", default=None, help="--init のとき、この営業日の大引けの後から始める（その後の営業日も続けて処理する）")
    parser.add_argument("--replay", default=None, help="この日から手元のデータで動かして DB の実行と比べる（確かめ用。状態は保存しない）")
    parser.add_argument("--local-end", "--replay-end", dest="local_end", default="2026-10-23", help="手元の日足の最後の日")
    parser.add_argument("--local", action="store_true", help="手元の data_daily を使う（--replay のとき）")
    parser.add_argument("--cutoff", default="16:30", help="この時刻より前は今日の足を使わない")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    if (PAPER / "setup.json").exists():
        raise SystemExit(f"{PAPER}/setup.json は前の形（PT 1 つ）です。data/paper/<番号>/ に移してください")

    if args.init or args.replay:
        if args.run is None:
            raise SystemExit("--run（決まりを写す DB の実行）を指定してください")
        mdir = PAPER / str(args.run)
        if args.init and (mdir / "setup.json").exists():
            raise SystemExit(f"{mdir} が既にあります。やり直すときは退避してから --init してください")
        setup = setup_from_run(args.run, args.db)
        setup["regime"] = freeze_regime()
        setup["universe"] = sorted(margin_codes() & set(list_codes(DATA_DIR)))
        costs = CostModel(**setup["costs"])
        if args.replay:
            start = (pd.Timestamp(args.replay) - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
            stocks, factors, nk, p_down = market(setup, start, args.local_end, local=args.local)
            sig = signals(setup, stocks, factors, nk, p_down)
            days = nk.index[nk.index >= args.replay]
            acct = new_account(setup, args.capital)
            d0 = nk.index[nk.index < args.replay][-1]
            acct.equity.append([str(d0.date()), acct.capital, float(nk.loc[d0, "close"])])
            acct.orders, acct.last_date = next_orders(acct, d0, sig, costs, acct.capital), str(d0.date())
            acct.in_index = True   # 検証（DB）と同じく、始めから日経平均で持っていたとする
            run_days(acct, sig, days, costs)
            compare(acct, args.run, args.db, args.replay)
            return
        acct = None
        if args.start_date and pd.Timestamp(args.start_date) < pd.Timestamp(args.local_end):
            # 手元の日足の最後の日までは、手元の日足で後から計算する
            start = (pd.Timestamp(args.start_date) - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
            stocks, factors, nk, p_down = market(setup, start, args.local_end, local=True)
            sig = signals(setup, stocks, factors, nk, p_down)
            d0 = nk.index[nk.index <= args.start_date][-1]
            acct = start_account(setup, sig, d0, args.capital, costs)
            run_days(acct, sig, nk.index[nk.index > d0], costs)
            setup["live_from"] = acct.last_date
        start = (pd.Timestamp.today() - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
        stocks, factors, nk, p_down = market(setup, start, None, local=False)
        sig = signals(setup, stocks, factors, nk, p_down)
        days = complete_days(nk, args.cutoff)
        if acct is None:
            d0 = days[days <= args.start_date][-1] if args.start_date else days[-1]
            acct = start_account(setup, sig, d0, args.capital, costs)
            setup["live_from"] = str(d0.date())
        d0 = pd.Timestamp(acct.equity[0][0])
        setup["start"], setup["start_date"] = f"{d0.date()} の大引けの後", str(d0.date())
        run_days(acct, sig, days[days > pd.Timestamp(acct.last_date)], costs)   # 始めた日より後の営業日も続けて処理する
        mdir.mkdir(parents=True, exist_ok=True)
        advance(mdir, setup, acct, sig, pd.Timestamp(acct.last_date), args.db)
        return

    dirs = [PAPER / str(args.run)] if args.run else model_dirs()
    if not dirs or not all((d / "setup.json").exists() for d in dirs):
        raise SystemExit("動かす PT がありません。--init --run <DB の実行> --start-date <日付> で始めてください")
    setups = {d: json.loads((d / "setup.json").read_text(encoding="utf-8")) for d in dirs}
    start = (pd.Timestamp.today() - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    stocks, factors, nk = fetch(setup_codes(list(setups.values())), start, None, local=False)   # 全部の PT の分を 1 回で取る
    days = complete_days(nk, args.cutoff)
    for mdir, setup in setups.items():
        acct = Account.from_dict(json.loads((mdir / "account.json").read_text(encoding="utf-8")))
        new = days[days > pd.Timestamp(acct.last_date)]
        if new.empty:
            print(f"{mdir.name}: 新しい営業日の大引けはまだありません（最後に記録した日 {acct.last_date}）")
            continue
        trade, p_down = prepare(setup, stocks, nk)
        sig = signals(setup, trade, factors, nk, p_down)
        run_days(acct, sig, new, CostModel(**setup["costs"]))
        advance(mdir, setup, acct, sig, new[-1], args.db)


def compare(acct: Account, run_id: int, db: str, start: str) -> None:
    """--replay の取引と、DB の実行（OOS）の同じ期間の取引を比べる（DB の OOS の終わりまで。PT を動かす決まりは 2023-03 まで）。"""
    store = ResultStore(db)
    db_eq = store.equity(run_id)
    db_eq = db_eq[db_eq["period"] == "OOS"].set_index("date")["equity"]
    end = db_eq.index[-1]
    if pd.Timestamp(start) > end:
        print(f"\nDB #{run_id} の OOS は {end.date()} までなので比べられません。OOS を切る前（--oos-end なし）の実行で比べてください")
        return
    t = store.trades(run_id)
    t = t[(t["period"] == "OOS") & (t["entry_date"] >= start)]
    mine = pd.DataFrame(acct.trades)
    mine = mine.assign(entry_date=pd.to_datetime(mine["entry_date"]), exit_date=pd.to_datetime(mine["exit_date"]))
    mine = mine[mine["entry_date"] <= end]
    key = ["code", "entry_date", "exit_date"]
    both = mine.merge(t, on=key, how="outer", indicator=True, suffixes=("", "_db"))
    print(f"\n{start}〜{end.date()} に建てた取引: ペーパートレード {len(mine)} 件 / DB #{run_id} {len(t)} 件 / 一致 {(both['_merge'] == 'both').sum()} 件")
    diff = both[both["_merge"] != "both"].sort_values("entry_date")
    if len(diff):
        print("一致しない取引（left_only = ペーパートレードだけ、right_only = DB だけ）")
        print(diff[key + ["_merge", "exit_reason", "exit_reason_db"]].to_string(index=False))
    e = equity_frame(acct)["equity"][:end]
    db_eq = db_eq[db_eq.index >= e.index[0]]
    r_mine = e.iloc[-1] / e.iloc[0] - 1
    r_db = db_eq.iloc[-1] / db_eq.iloc[0] - 1
    print(f"資産の騰落 {e.index[0].date()}〜{e.index[-1].date()}: ペーパートレード {r_mine:+.2%} / DB #{run_id} {r_db:+.2%}")


if __name__ == "__main__":
    main()
