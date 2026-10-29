"""ペーパートレード（前向きの検証）。実際の注文は出さない。

DB #75 / #77（scripts/simulate_small_account.py）と同じ決まりで、1 つの口座を 1 日ずつ動かす。
- 個別株に入っていない資金（待機中の資金）は、index_weight の割合を日経平均（ETF）、残りを現金で持つ（#75 は 1、#77 は 0.5）。
  個別株を買うときは日経平均と現金を今の割合で使い、売った代金は index_weight の割合で日経平均と現金に戻す。
  月の初めの営業日の大引けで、割合が 2 ポイントより大きくずれていれば index_weight にそろえる（買い増しの間はしない）
- 個別株（DB #73）: 終値が -2σ 以下で引けた日（最初の割れだけ）に、IS で選んだ条件を満たした貸借銘柄を、5 日の騰落の
  小さい順に空き枠の数だけ翌日の寄りの成行で買う（その分の日経平均は寄りで売る）。終値が -1σ 以上に戻った日の大引けか、
  合図の日から max_hold 営業日目の大引けで売り、代金は大引けで日経平均に戻す（その日の -1σ になる値段で引け指値の売りを
  出しておけば、そのとおりに約定する）。ストップ安で引けた日は売れないので、次に値段がついた日の寄りで売る。
- 日経平均の買い増し（DB #60）: -3σ に触れた貸借銘柄（売買代金 1 億円以上）が min_stocks 以上（合図 C。DB #85 からは
  日経平均の終値が -2.5σ より下の合図 B）で、前日の下げ相場の確率が
  long_min_p_down を超えた日の翌日の寄りで、資産 × extra × index_weight の日経平均を買い増す（まず現金で買い、足りない分だけ
  信用で借りる。金利は借りた分だけ）。日経平均の終値が -1σ 以上に戻った日か、建てた日から max_hold 営業日目の大引けで売る。
  持っている間と売った日の合図は使わない。
毎日の手順（step）: その日の寄りの約定 → 大引けの約定と時価評価 → 翌日の注文。
検証（simulate_slots と study_index_crash.overlay）と同じ計算で、違うのは次の 2 つだけ:
- 寄りの成行は株数を決めて出すので、株数は前日の終値で決める（寄りの値段が上なら現金の範囲に減らす）
- 買い増しの損益はその日のうちに待機中の資金に足す（検証では資産全体に個別株の手法の騰落を掛けている）
- 待機中の資金の日経平均と現金の割合は月に 1 回そろえる（検証では毎日 index_weight に保つとみなしている）
どちらも、買い増しの間に個別株を買うと買い増しに使った現金をもう一度使う形になる（実際はその分だけ信用の借入が増えるが、
金利は買い増しを建てた日の借入で計算する。数日分なので小さい）。
"""
import math
from dataclasses import asdict, dataclass, field

import pandas as pd

from .backtest.costs import CostModel
from .backtest.limits import limit_flags

OOS_END = "2023-03-31"   # PT を動かす決まりの OOS の終わり。2023 年 4 月以降は手法を選ぶのに使わず、ペーパートレード（PT）の期間にする
from .strategy.band_touch import band_price

LOT = 100


def stock_features(df: pd.DataFrame, factor: pd.Series | None = None) -> pd.DataFrame:
    """1 銘柄の、その日の大引けまでに分かる値（scripts/study_bb2_close.py の _work と close_features と同じ計算）。
    factor は分割調整済みの値段を実際の値段に戻す係数（値幅制限の判定に使う）。"""
    c, v, lo = df["close"], df["volume"], df["low"]
    z = (c - c.rolling(25).mean()) / c.rolling(25).std(ddof=0)
    liquid = ((c * v).rolling(20).mean().shift(1) >= 1e8)
    below = z <= -2
    lim = limit_flags(df, factor)
    return pd.DataFrame({
        "open": df["open"], "close": c, "z": z, "ret5": c / c.shift(5) - 1, "vratio": v / v.rolling(20).mean().shift(1),
        "liquid": liquid, "below": below & liquid,
        "first": below.astype(float).shift(1).rolling(5, min_periods=1).sum() == 0,
        "touch3": liquid & (lo <= band_price(c, 25, -3.0)),
        "down_close": lim["down_close"], "locked": lim["up_locked"] | lim["down_locked"],
    }, index=df.index)


@dataclass
class Signals:
    feats: dict[str, pd.DataFrame]   # 銘柄 → stock_features
    market: pd.DataFrame             # 日付 → breadth / breadth_chg / count3 / p_down / p_down_prev / nk_close / nk_open / nk_z / index_signal
    buy: pd.DataFrame                # 個別株の合図（列 date / code / ret5 / z / vratio。ret5 の小さい順）


def compute_signals(stocks: dict[str, pd.DataFrame], nk: pd.DataFrame, p_down: pd.Series, rules: list,
                    min_stocks: int | None, index_p_down: float, factors: dict[str, pd.Series] | None = None,
                    index_signal: str = "C", index_k: float | None = None) -> Signals:
    """nk は日経平均の日足（close と、寄りで売買できる値段 open_tradable）、p_down はその日の大引け時点の下げ相場の確率。
    rules は DB #73 の条件 [(指標, "≧" | "≦", 境目)]（指標は z / vratio / breadth_chg / p_down）。
    日経平均の買い増しの合図は、index_signal = "C" なら -3σ に触れた貸借銘柄が min_stocks 以上、"B" なら日経平均の終値が
    -index_k σ より下（どちらも前日の下げ相場の確率が index_p_down を超えた日だけ）。"""
    factors = factors or {}
    feats = {code: stock_features(df, factors.get(code)) for code, df in stocks.items()}
    liq = pd.concat({c: f["liquid"] for c, f in feats.items()}, axis=1, sort=True).fillna(False)
    below = pd.concat({c: f["below"] for c, f in feats.items()}, axis=1, sort=True).fillna(False)
    touch = pd.concat({c: f["touch3"] for c, f in feats.items()}, axis=1, sort=True).fillna(False)
    n = liq.sum(axis=1)
    breadth = below.sum(axis=1) / n.where(n > 0)
    nkc = nk["close"]
    p_prev = p_down.shift(1)
    market = pd.DataFrame({"breadth": breadth, "breadth_chg": breadth - breadth.shift(1),
                           "count3": touch.sum(axis=1)}).reindex(nk.index)
    market["count3"] = market["count3"].fillna(0)
    market = market.assign(p_down=p_down.reindex(nk.index), p_down_prev=p_prev.reindex(nk.index), nk_close=nkc,
                           nk_open=nk["open_tradable"],
                           nk_z=(nkc - nkc.rolling(25).mean()) / nkc.rolling(25).std(ddof=0))
    hit = market["nk_z"] < -index_k if index_signal == "B" else market["count3"] >= min_stocks
    market["index_signal"] = hit & (market["p_down_prev"] > index_p_down)

    rows = []
    for code, f in feats.items():
        s = f[f["below"] & f["first"] & ~f["down_close"]]
        if s.empty:
            continue
        rows.append(s[["z", "ret5", "vratio"]].assign(code=code).rename_axis("date").reset_index())
    buy = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "z", "ret5", "vratio", "code"])
    buy["breadth_chg"] = market["breadth_chg"].reindex(buy["date"]).to_numpy()
    buy["p_down"] = p_prev.reindex(buy["date"]).to_numpy()
    keep = pd.Series(True, index=buy.index)
    for f, op, v in rules:
        keep &= (buy[f] <= v) if op == "≦" else (buy[f] >= v)
    buy = buy[keep].sort_values(["date", "ret5"]).reset_index(drop=True)
    return Signals(feats, market, buy)


@dataclass
class Account:
    capital: float
    slots: int
    max_hold: int = 10                 # 個別株: 合図の日から数えた最長の営業日数
    index_max_hold: int = 10           # 日経平均の買い増し: 建てた日から数えた最長の営業日数
    extra: float = 1.0                 # 買い増す額（資産に対する割合）
    rate: float = 0.028                # 信用買いの金利（年）
    switch_cost: float = 0.0005        # 日経平均の売り買いの片道の費用
    index_weight: float = 1.0          # 待機中の資金のうち日経平均で持つ割合（残りは現金）
    idle: float = 0.0                  # 日経平均（ETF）で持っている額（直近の大引けの時価）
    cash: float = 0.0                  # 現金（買い増しに使った分も含む。買い増しの損益も index_weight の割合で入る）
    in_index: bool = False             # False なら idle はまだ現金（最初の日の寄りで日経平均を買う）
    positions: list = field(default_factory=list)
    overlay: dict | None = None
    orders: dict = field(default_factory=lambda: {"buy": [], "overlay_buy": False})
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)     # [日付, 資産, 日経平均の終値]
    events: list = field(default_factory=list)     # [日付, 内容]（約定・取り消し・見送り）
    last_date: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        return cls(**d)

    def value(self, day: pd.Timestamp, sig: Signals) -> float:
        held = 0.0
        for p in self.positions:
            f = sig.feats.get(p["code"])
            px = f["close"][:day].iloc[-1] if f is not None and len(f[:day]) else p["entry_price"]
            held += p["shares"] * px
        return self.idle + self.cash + held

    def nk_share(self) -> float:
        """待機中の資金のうち日経平均の割合（個別株を買うときに日経平均を売る割合）。"""
        total = self.idle + self.cash
        return self.idle / total if total > 0 else 1.0


def _trade(code, entry_date, exit_date, shares, entry, exit_, pnl, reason) -> dict:
    return {"code": code, "entry_date": str(entry_date)[:10], "exit_date": str(exit_date)[:10], "shares": shares,
            "entry_price": entry, "exit_price": exit_, "pnl": pnl, "ret": pnl / (shares * entry), "exit_reason": reason,
            "side": "long"}


def step(acct: Account, day: pd.Timestamp, sig: Signals, costs: CostModel) -> None:
    """day の寄り → 大引け → 翌日の注文 を 1 日分進める（day は日経平均のある営業日）。"""
    m = sig.market.loc[day]
    prev = sig.market.loc[:day].iloc[:-1]
    prev_close = prev["nk_close"].iloc[-1]
    prev_day = prev.index[-1]
    eq_prev = acct.equity[-1][1] if acct.equity else acct.capital
    ds = str(day.date())

    def row(code):
        f = sig.feats.get(code)
        return f.loc[day] if f is not None and day in f.index else None

    # ---- 寄り ----
    w, share = acct.index_weight, acct.nk_share()
    nk_open, cash_open = acct.idle, acct.cash
    spent_n = spent_c = 0.0   # 個別株の買いにあてた日経平均（前日の大引けの時価、売る費用込み）と現金
    sold_open = []
    for p in [p for p in acct.positions if p.get("exit_next_open")]:
        r = row(p["code"])
        if r is None or r["locked"]:
            continue
        price = costs.fill_price(r["open"], "sell")
        sold_open.append((p, price))
    for o in acct.orders["buy"]:
        r = row(o["code"])
        if r is None or r["locked"]:
            acct.events.append([ds, f"{o['code']} 寄りの買いは約定せず（売買なし・張り付き）"])
            continue
        price = costs.fill_price(r["open"], "buy")
        unit = price * (1 + acct.switch_cost * share)
        shares = o["shares"]
        while shares > 0 and shares * unit > nk_open + cash_open - spent_n - spent_c:
            shares -= LOT
        if shares <= 0:
            acct.events.append([ds, f"{o['code']} 寄りの値段が高く、待機中の資金で買えず見送り"])
            continue
        spent_n += shares * price * share * (1 + acct.switch_cost)
        spent_c += shares * price * (1 - share)
        acct.positions.append({"code": o["code"], "shares": shares, "entry_date": ds, "entry_price": price,
                               "signal_date": o["signal_date"], "days": 0, "exit_next_open": False})
        acct.events.append([ds, f"{o['code']} を寄りで {shares:,} 株買い（{price:,.1f} 円）"])
    if acct.orders.get("overlay_buy"):
        value = eq_prev * acct.extra * w
        units = value / m["nk_open"]
        debt = max(0.0, value - max(cash_open - spent_c, 0.0))   # まず現金で買い、足りない分だけ信用で借りる
        acct.overlay = {"units": units, "entry_date": ds, "entry_price": float(m["nk_open"]), "days": 0,
                        "fee": value * acct.switch_cost, "interest": 0.0, "debt": debt}
        how = "信用" if debt >= value else f"現金 {value - debt:,.0f} 円・信用 {debt:,.0f} 円"
        acct.events.append([ds, f"日経平均を寄りで {value:,.0f} 円分買い増し（{m['nk_open']:,.0f}。{how}）"])
    if acct.in_index:   # 一日中持っていた分にはその日の騰落、寄りで売った分には寄りまでの騰落が付く
        ret, gap = m["nk_close"] / prev_close - 1, m["nk_open"] / prev_close - 1
        acct.idle = (nk_open - spent_n) * (1 + ret) + spent_n * gap
    else:               # 最初の日: 現金で寄りに日経平均を買う
        acct.idle = (nk_open - spent_n) * (1 - acct.switch_cost) * m["nk_close"] / m["nk_open"]
        acct.in_index = True
    acct.cash = cash_open - spent_c

    # ---- 大引け ----
    for p, price in sold_open:
        pnl = p["shares"] * (price - p["entry_price"])
        _back(acct, p["shares"] * price)
        acct.trades.append(_trade(p["code"], p["entry_date"], ds, p["shares"], p["entry_price"], price, pnl, "limit"))
        acct.positions.remove(p)
        acct.events.append([ds, f"{p['code']} を寄りで売り（前の日がストップ安で引けたため）"])
    for p in [p for p in acct.positions if not p.get("exit_next_open")]:
        r = row(p["code"])
        if r is None:
            continue
        p["days"] += 1
        reason = "target" if r["z"] >= -1.0 else "time" if p["days"] >= acct.max_hold else None
        if reason is None:
            continue
        if r["down_close"]:
            p["exit_next_open"], p["reason"] = True, reason
            acct.events.append([ds, f"{p['code']} はストップ安で引けたので、次の寄りで売る"])
            continue
        price = costs.fill_price(r["close"], "sell")
        pnl = p["shares"] * (price - p["entry_price"])
        _back(acct, p["shares"] * price)
        acct.trades.append(_trade(p["code"], p["entry_date"], ds, p["shares"], p["entry_price"], price, pnl, reason))
        acct.positions.remove(p)
        acct.events.append([ds, f"{p['code']} を大引けで売り（{'−1σに戻った' if reason == 'target' else '最長の日数'}）"])
    overlay_closed = False
    if acct.overlay is not None:
        o = acct.overlay
        if o["entry_date"] == ds:
            pnl = o["units"] * (m["nk_close"] - o["entry_price"]) - o["fee"]
        else:
            interest = o.get("debt", o["units"] * o["entry_price"]) * acct.rate * (day - prev_day).days / 365
            o["interest"] += interest
            pnl = o["units"] * (m["nk_close"] - prev_close) - interest
            o["days"] += 1
        reason = None
        if o["entry_date"] != ds:
            reason = "target" if m["nk_z"] >= -1.0 else "time" if o["days"] >= acct.index_max_hold else None
        if reason:
            sell_fee = o["units"] * m["nk_close"] * acct.switch_cost
            pnl -= sell_fee
            total = o["units"] * (m["nk_close"] - o["entry_price"]) - o["fee"] - sell_fee - o["interest"]
            acct.trades.append(_trade("^N225", o["entry_date"], ds, o["units"], o["entry_price"], float(m["nk_close"]),
                                      total, reason))
            acct.overlay, overlay_closed = None, True
            acct.events.append([ds, "日経平均の買い増しを大引けで売り（" +
                                ("−1σに戻った" if reason == "target" else "最長の日数") + "）"])
        acct.idle += pnl * w
        acct.cash += pnl * (1 - w)
    if w < 1 and acct.overlay is None and prev_day.month != day.month:   # 月の初めの営業日に割合をそろえる
        total = acct.idle + acct.cash
        if total > 0 and abs(acct.idle / total - w) > 0.02:
            diff = w * total - acct.idle
            acct.idle += diff
            acct.cash -= diff + abs(diff) * acct.switch_cost
            acct.events.append([ds, f"日経平均と現金の割合を {w:.0%} にそろえた（日経平均を大引けで {diff:+,.0f} 円）"])
    equity = acct.value(day, sig)
    acct.equity.append([ds, equity, float(m["nk_close"])])

    acct.orders = next_orders(acct, day, sig, costs, equity, allow_overlay=not overlay_closed)
    acct.last_date = ds


def _back(acct: Account, proceeds: float) -> None:
    """個別株を売った代金を、index_weight の割合で日経平均（大引けで買う。費用込み）と現金に戻す。"""
    acct.idle += proceeds * acct.index_weight * (1 - acct.switch_cost)
    acct.cash += proceeds * (1 - acct.index_weight)


def next_orders(acct: Account, day: pd.Timestamp, sig: Signals, costs: CostModel, equity: float,
                allow_overlay: bool = True) -> dict:
    """day の大引けの後に出す、翌日の注文（寄りの成行の買いと、日経平均の買い増し）。
    個別株は合図の銘柄を 5 日の騰落の小さい順に、空き枠の数だけ。株数は day の終値で決める。"""
    ds = str(day.date())
    m = sig.market.loc[day]
    orders = {"buy": [], "overlay_buy": False}
    free = acct.slots - len(acct.positions)
    held = {p["code"] for p in acct.positions}
    avail, share = acct.idle + acct.cash, acct.nk_share()
    for c in sig.buy[sig.buy["date"] == day].itertuples():
        if c.code in held:
            continue
        if free <= 0:
            acct.events.append([ds, f"{c.code} は合図が出たが枠が埋まっていて見送り"])
            continue
        price = costs.fill_price(sig.feats[c.code].loc[day, "close"], "buy")
        budget = min(equity / acct.slots, avail)
        shares = math.floor(budget / (price * LOT * (1 + acct.switch_cost * share))) * LOT
        if shares == 0:
            acct.events.append([ds, f"{c.code} は 100 株が 1 枠（{equity / acct.slots:,.0f} 円）に収まらず見送り"])
            continue
        orders["buy"].append({"code": c.code, "shares": shares, "signal_date": ds, "est_price": float(price)})
        avail -= shares * price * (1 + acct.switch_cost * share)
        free -= 1
    if acct.overlay is None and allow_overlay and bool(m["index_signal"]):
        orders["overlay_buy"] = True
    return orders


def exit_limit(close: pd.Series) -> float:
    """翌日の大引けで終値が -1σ 以上になる値段（引け指値の売りの値段）。close は今日までの終値。"""
    prev = close.iloc[-24:]
    return float(prev.mean() - 1.0 * prev.std(ddof=0) * math.sqrt(25 / (25 - 1 - 1)))


def equity_frame(acct: Account) -> pd.DataFrame:
    return pd.DataFrame(acct.equity, columns=["date", "equity", "nk_close"]).assign(
        date=lambda d: pd.to_datetime(d["date"])).set_index("date")
