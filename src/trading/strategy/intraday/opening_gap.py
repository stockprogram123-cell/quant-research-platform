import pandas as pd

from .base import IntradayStrategy, trading_day


class OpeningGap(IntradayStrategy):
    """寄り付きの窓の方向についていき、前場のうちに手仕舞う。下げ相場の立ち回り用。

    side="short": 前日終値より min_gap 以上安く寄った銘柄を、最初の足（09:00〜09:05）の終値を見て
      次の足の始値で空売りし、exit_time の足の終値で手仕舞う（実際の約定はその次の足の始値）。
      損切りはシグナル時の終値の stop_pct 上。
    side="long": 上下を逆にしたもの（min_gap 以上高く寄った銘柄を買う）。
    regime を "down" / "up" にすると、bars の regime_down 列（1 = 下げ相場の日、attach_regime で付ける）が
    合う日だけ取引する。None なら局面を見ない。
    7〜10 月の 5 分足では、下げ相場の日は寄り〜前場に下げ続け、窓を開けて下げた銘柄ほど下げが続いた
    （scripts/study_regime_intraday.py）。
    """

    def __init__(self, side: str = "short", min_gap: float = 0.01, exit_time: str = "10:55",
                 stop_pct: float = 0.015, regime: str | None = "down"):
        if side not in ("long", "short"):
            raise ValueError("side は 'long' か 'short' にしてください")
        if regime not in ("down", "up", None):
            raise ValueError("regime は 'down' / 'up' / None にしてください")
        self.side = side
        self.min_gap = min_gap
        self.exit_time = exit_time
        self.stop_pct = stop_pct
        self.regime = regime
        self.requires_regime = regime is not None

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if self.requires_regime and "regime_down" not in bars:
            raise ValueError("局面の条件には attach_regime の regime_down 列が必要です")
        s = 1 if self.side == "long" else -1
        day = trading_day(bars)
        close = bars["close"]
        first_open = bars["open"].groupby(day).transform("first")
        prev_close = close.groupby(day).last().shift(1)
        gap = first_open / pd.Series(day, index=bars.index).map(prev_close) - 1
        is_first = ~day.duplicated()
        entry = pd.Series(is_first, index=bars.index) & (s * gap >= self.min_gap)
        if self.requires_regime:
            entry &= bars["regime_down"] == (1 if self.regime == "down" else 0)
        before_exit = pd.Series(bars.index.strftime("%H:%M") < self.exit_time, index=bars.index)
        holding = entry.groupby(day).cummax() & before_exit
        return pd.DataFrame({"target": s * holding.astype(int), "stop": close * (1 - s * self.stop_pct)})
