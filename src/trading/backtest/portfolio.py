from dataclasses import dataclass, field, fields

import pandas as pd


@dataclass
class Trade:
    date: pd.Timestamp
    side: str
    shares: int
    price: float
    commission: float


@dataclass
class RoundTrip:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    shares: int
    pnl: float  # 手数料込みの損益
    exit_reason: str = ""  # デイトレード用: "signal" / "stop" / "eod"
    side: str = "long"     # "long"（買い→売り） / "short"（空売り→買い戻し）
    entry_price: float = float("nan")  # 平均約定価格（スプレッド込み）
    exit_price: float = float("nan")


ROUND_TRIP_COLUMNS = [f.name for f in fields(RoundTrip)]


@dataclass
class Portfolio:
    """単一銘柄のポジション管理。shares が負なら空売り中。

    空売りは売った代金を現金に入れ、買い戻しで払う（equity = 現金 + 株数×株価 がそのまま使える）。
    """

    cash: float
    shares: int = 0
    trades: list[Trade] = field(default_factory=list)
    round_trips: list[RoundTrip] = field(default_factory=list)
    _entry_date: pd.Timestamp | None = None
    _entry_cost: float = 0.0  # 手数料込みの取得総額
    _entry_value: float = 0.0  # 約定価格×株数の合計（平均約定価格の計算用）

    def buy(self, date: pd.Timestamp, shares: int, price: float, commission: float) -> None:
        cost = shares * price + commission
        if cost > self.cash + 1e-9:
            raise ValueError("現金が不足しています")
        if self.shares == 0:
            self._entry_date = date
        self.cash -= cost
        self.shares += shares
        self._entry_cost += cost
        self._entry_value += shares * price
        self.trades.append(Trade(date, "buy", shares, price, commission))

    def sell_all(self, date: pd.Timestamp, price: float, commission: float,
                 reason: str = "") -> None:
        proceeds = self.shares * price - commission
        self.cash += proceeds
        self.trades.append(Trade(date, "sell", self.shares, price, commission))
        self.round_trips.append(
            RoundTrip(self._entry_date, date, self.shares, proceeds - self._entry_cost, reason,
                      "long", self._entry_value / self.shares, price)
        )
        self._reset()

    def short(self, date: pd.Timestamp, shares: int, price: float, commission: float) -> None:
        """空売りする（ノーポジのときだけ）。"""
        if self.shares != 0:
            raise ValueError("ポジションを持ったまま空売りはできません")
        proceeds = shares * price - commission
        self._entry_date = date
        self.cash += proceeds
        self.shares = -shares
        self._entry_cost = proceeds  # 空売りでは受け取った代金（手数料差し引き後）
        self._entry_value = shares * price
        self.trades.append(Trade(date, "short", shares, price, commission))

    def cover_all(self, date: pd.Timestamp, price: float, commission: float, fee: float = 0.0,
                  reason: str = "") -> None:
        """空売りを全部買い戻す。fee は貸株料など保有期間に応じた費用（円）。"""
        shares = -self.shares
        cost = shares * price + commission + fee
        self.cash -= cost
        self.trades.append(Trade(date, "cover", shares, price, commission + fee))
        self.round_trips.append(
            RoundTrip(self._entry_date, date, shares, self._entry_cost - cost, reason,
                      "short", self._entry_value / shares, price)
        )
        self._reset()

    def _reset(self) -> None:
        self.shares = 0
        self._entry_date = None
        self._entry_cost = 0.0
        self._entry_value = 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price
