"""多銘柄の取引候補を、1 つの資金で同時に何銘柄まで持てるかの制約付きで運用するシミュレーション。

銘柄ごとに独立に検証すると、暴落日に何百銘柄も同時に建てられることになる。ここでは資金を slots 等分し、
空いている枠の分だけ建てる。
- 1 枠の金額は前日の大引け時点の資産 ÷ slots（複利）。現金が足りなければ現金の範囲で、100 株単位
  （cands に lot 列があればその候補はその株数単位。分割調整済みの値段で、実際の 100 株が何株にあたるかを渡すときに使う）
- 同じ日の候補が空き枠より多いときは priority の小さい順（同じなら乱数の順）に建てる。
  max_per_day を指定すると 1 日に新しく建てるのはその数まで（下げが続く局面で初日に資金を使い切らないため）。
  day_fraction を指定すると 1 日に新しく建てるのは ceil(その日の初めの空き枠 × day_fraction) まで（下げが続くほど 1 日に使う枠が減り、
  資金を何日かに分けて使う。max_per_day もあれば小さい方）
- orders_per_slot を指定すると、寄りの前に注文を出すのは順位（cands の rank 列。その日に注文を出せる銘柄の中の順位、
  1 が最良）が ceil(その日の初めの空き枠（max_per_day があればそれと小さい方）× orders_per_slot) 以内の銘柄だけ。
  約定した後で良い方を選ぶことはできないので、注文を出す銘柄を寄りの前に絞る形にしている
- skip_held=True なら、既に持っている銘柄の候補は建てない（複数の手法の候補を合わせたとき、同じ銘柄を二重に持たないため）
- 手仕舞った枠と現金は翌営業日から使う（同じ日のうちに手仕舞いと建てが前後する順番は日足では分からないため）
- 資産は毎日の大引けで時価評価する（データがない日は直近の終値）
- cash_return（日付 → 前日の大引けからその日の大引けまでの騰落率）を渡すと、使っていない現金を指数などで運用したことにする
  （買いだけを想定）。建てる日は寄りで指数を売って資金にし、売った分には cash_gap（前日の大引けからその日の寄りまでの騰落率。
  省略時は 0）だけが付く。手仕舞った代金は大引けで指数に戻す。指数の売り買いのたびに金額 × switch_cost の費用がかかる
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .engine import _affordable_shares


@dataclass
class SlotResult:
    equity: pd.Series          # 日々の資産（大引け時点）
    trades: pd.DataFrame       # 建てた取引（round_trip_table と同じ列）
    exposure: pd.Series        # 日々の建玉の金額 ÷ 資産
    skipped: dict[str, int]    # 建てなかった候補の数（"full" = 枠が埋まっていた / "price" = 1 単元も買えない /
                               # "day" = その日に建てる数の上限 / "rank" = 順位が低く注文を出していなかった /
                               # "held" = 既に持っている銘柄）


def simulate_slots(cands: pd.DataFrame, closes: pd.DataFrame, capital: float, slots: int, costs: CostModel,
                   lot_size: int = 100, rng: np.random.Generator | None = None,
                   max_per_day: int | None = None, orders_per_slot: float | None = None,
                   day_fraction: float | None = None, skip_held: bool = False,
                   cash_return: pd.Series | None = None, cash_gap: pd.Series | None = None,
                   switch_cost: float = 0.0) -> SlotResult:
    """cands は 1 行 = 1 取引の候補（code / entry_date / exit_date / side（1 買い, -1 空売り）/ entry_price /
    exit_price / reason / priority）。値段はコストを含まない約定値。closes は日付 × 銘柄の終値（運用する期間の分）。
    """
    rng = rng or np.random.default_rng(0)
    if orders_per_slot is not None and "rank" not in cands:
        raise ValueError("orders_per_slot を使うには cands に rank 列が要ります")
    if "rank" not in cands:
        cands = cands.assign(rank=1)
    if "lot" not in cands:
        cands = cands.assign(lot=lot_size)
    cands = cands.assign(_r=rng.random(len(cands))).sort_values(["entry_date", "priority", "_r"])
    by_day = {d: g for d, g in cands.groupby("entry_date")}
    closes = closes.ffill()
    cash_ret = None if cash_return is None else cash_return.reindex(closes.index).fillna(0.0)
    gap = pd.Series(0.0, index=closes.index) if cash_gap is None else cash_gap.reindex(closes.index).fillna(0.0)
    cash, prev_equity = float(capital), float(capital)
    positions: list[dict] = []
    rows, equity, exposure = [], {}, {}
    skipped = {"full": 0, "price": 0, "day": 0, "rank": 0, "held": 0}
    for date in closes.index:
        opened, cash_open = 0, cash
        day_cap = math.inf if max_per_day is None else max_per_day
        if day_fraction is not None:
            day_cap = min(day_cap, math.ceil((slots - len(positions)) * day_fraction))
        n_orders = math.inf
        if orders_per_slot is not None:
            n_orders = math.ceil(min(slots - len(positions), day_cap) * orders_per_slot)
        for c in by_day.get(date, pd.DataFrame()).itertuples():
            if c.rank > n_orders:
                skipped["rank"] += 1
                continue
            if skip_held and any(p["code"] == c.code for p in positions):
                skipped["held"] += 1
                continue
            if len(positions) >= slots:
                skipped["full"] += 1
                continue
            if opened >= day_cap:
                skipped["day"] += 1
                continue
            price = costs.fill_price(c.entry_price, "buy" if c.side > 0 else "sell")
            budget = prev_equity / slots if c.side < 0 else min(prev_equity / slots, cash)
            shares = _affordable_shares(budget, price, c.lot, costs)
            if shares == 0:
                skipped["price"] += 1
                continue
            fee = costs.commission(shares * price)
            cash -= c.side * shares * price + fee
            if cash_ret is not None:
                cash -= shares * price * switch_cost
            opened += 1
            positions.append({"code": c.code, "side": c.side, "shares": shares, "entry_date": date, "entry": price,
                              "fee": fee, "exit_date": c.exit_date, "exit_price": c.exit_price, "reason": c.reason})
        if cash_ret is not None:  # 一日中指数で持っていた分にはその日の騰落、寄りで売った分には寄りまでの騰落が付く
            kept = max(min(cash_open, cash), 0.0)
            cash += kept * cash_ret.loc[date] + max(cash_open - kept, 0.0) * gap.loc[date]
        for p in [p for p in positions if p["exit_date"] == date]:
            price = costs.fill_price(p["exit_price"], "sell" if p["side"] > 0 else "buy")
            notional = p["shares"] * price
            fee = costs.commission(notional)
            if p["side"] < 0:
                fee += costs.short_fee(notional, max((date - p["entry_date"]).days, 1))
            cash += p["side"] * notional - fee
            if cash_ret is not None:
                cash -= notional * switch_cost
            pnl = p["side"] * p["shares"] * (price - p["entry"]) - p["fee"] - fee
            rows.append({"code": p["code"], "entry_date": p["entry_date"], "exit_date": date, "shares": p["shares"],
                         "entry_price": p["entry"], "exit_price": price, "pnl": pnl,
                         "ret": pnl / (p["shares"] * p["entry"]), "exit_reason": p["reason"],
                         "side": "long" if p["side"] > 0 else "short"})
            positions.remove(p)
        row = closes.loc[date]
        held = sum(p["side"] * p["shares"] * row[p["code"]] for p in positions)
        prev_equity = cash + held
        equity[date] = prev_equity
        exposure[date] = sum(p["shares"] * row[p["code"]] for p in positions) / prev_equity
    trades = pd.DataFrame(rows, columns=["code", "entry_date", "exit_date", "shares", "entry_price", "exit_price",
                                         "pnl", "ret", "exit_reason", "side"])
    return SlotResult(pd.Series(equity, name="equity"), trades, pd.Series(exposure, name="exposure"), skipped)
