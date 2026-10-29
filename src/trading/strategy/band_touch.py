import math

import pandas as pd


def band_price(close: pd.Series, window: int, z: float) -> pd.Series:
    """その日の値段がいくらになったら、その値段を含む window 日のボリンジャーバンドで z σ の位置に来るか。

    前日までの window-1 日の終値（平均 m、標準偏差 s）から寄りの前に計算できる:
    値段 P を含めた z は P - m の増加関数で、z σ になるのは P = m + z·s·√(window / (window-1-z²))。
    チャートのバンドは日中の値段で動くので、「+3σ に触れた」はこの値段に触れたことと同じ。
    |z| は √(window-1) より大きくなれない（window=25 なら 4.9）。
    """
    if z * z >= window - 1:
        raise ValueError(f"window={window} では |z| は {math.sqrt(window - 1):.2f} 未満です")
    prev = close.shift(1).rolling(window - 1)
    return prev.mean() + z * prev.std(ddof=0) * math.sqrt(window / (window - 1 - z * z))


class BandTouchFade:
    """日足のボリンジャーバンド（window 日、±k σ）に日中に触れたら、その場で逆張りする。

    +kσ に触れたら空売り、-kσ に触れたら買い（寄りで既に越えていたら寄りで建てる）。
    利益確定は ±exit_k σ（その日のバンドの値段）に戻ったところの指値、
    損切りは建値から stop_pct 逆行したところの逆指値、max_hold 営業日たったら大引けで手仕舞う。
    日足からは日中の値動きの順番が分からないので、fill（conservative / optimistic）で扱いを選ぶ
    （trading.backtest.touch_engine を参照）。min_turnover は前日までの 20 日平均売買代金（円）の下限。

    gap_only=True なら、寄りの時点で既にバンドの外にある（窓を開けて越えた）日だけ寄りで建てる。
    exit_k=None なら利益確定の指値を置かない（max_hold=0 と組み合わせると「寄りで建てて大引けで手仕舞い」）。
    この 2 つを使うと、建て・手仕舞いの順番に迷う場面がなくなり、日足で正確に計算できる。

    long_exit_k / long_stop_pct / long_max_hold を指定すると、買いだけ手仕舞い方を変えられる
    （省略時は空売りと同じ。long_exit_k=None は買いの利益確定の指値なし）。

    long_min_p_down / short_max_p_down を指定すると、局面で建てる日を絞る（df に p_down_prev 列が要る。
    局面モデルの前日の大引け時点の下げ相場の確率。scripts/run_backtest.py の data.regime_file で付く）。
    買いは p_down_prev がこの値より大きい日だけ、空売りはこの値以下の日だけ建てる。確率がない日は建てない。

    long_min_high250 / long_max_ret5 / long_min_sma200 を指定すると、買いを「上昇トレンドの押し目」に絞る（すべて前日の大引けまでの値）:
    1 年（250 日）高値からの下落率がこの値以上、直近 5 日の騰落率がこの値以下、200 日平均からの乖離がこの値以上の日だけ建てる。
    long_cooldown を指定すると、前日までの long_cooldown 日に一度でも買いの値段（-kσ）を割っていたら建てない
    （下げの途中の 2 回目以降の割れではなく、最初の割れだけを買う）。
    """

    def __init__(self, window: int = 25, k: float = 3.0, exit_k: float | None = 2.0, stop_pct: float = 0.10,
                 max_hold: int = 10, sides: str = "both", fill: str = "conservative", min_turnover: float = 0.0,
                 gap_only: bool = False, long_exit_k: float | None | str = "same",
                 long_stop_pct: float | None = None, long_max_hold: int | None = None,
                 long_min_p_down: float | None = None, short_max_p_down: float | None = None,
                 long_min_high250: float | None = None, long_max_ret5: float | None = None,
                 long_min_sma200: float | None = None, long_cooldown: int | None = None):
        if sides not in ("both", "long", "short"):
            raise ValueError("sides は both / long / short です")
        if fill not in ("conservative", "optimistic"):
            raise ValueError("fill は conservative か optimistic です")
        if long_exit_k == "same":
            long_exit_k = exit_k
        for ek in (exit_k, long_exit_k):
            if ek is not None and not ek < k:
                raise ValueError("exit_k は k より小さくしてください")
        self.window, self.k, self.exit_k = window, float(k), None if exit_k is None else float(exit_k)
        self.stop_pct, self.max_hold, self.sides, self.fill = float(stop_pct), int(max_hold), sides, fill
        self.min_turnover, self.gap_only = float(min_turnover), bool(gap_only)
        self.long_exit_k = None if long_exit_k is None else float(long_exit_k)
        self.long_stop_pct = None if long_stop_pct is None else float(long_stop_pct)
        self.long_max_hold = None if long_max_hold is None else int(long_max_hold)
        self.long_min_p_down, self.short_max_p_down = long_min_p_down, short_max_p_down
        self.long_min_high250, self.long_max_ret5, self.long_min_sma200 = long_min_high250, long_max_ret5, long_min_sma200
        self.long_cooldown = long_cooldown

    def orders(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        no_target = pd.Series(math.nan, index=df.index)
        out = pd.DataFrame({
            "short_entry": band_price(c, self.window, self.k),
            "long_entry": band_price(c, self.window, -self.k),
            "short_target": no_target if self.exit_k is None else band_price(c, self.window, self.exit_k),
            "long_target": no_target if self.long_exit_k is None else band_price(c, self.window, -self.long_exit_k),
        }, index=df.index)
        if self.gap_only:  # 寄りで越えていない日は注文を出さない
            out["short_entry"] = out["short_entry"].where(df["open"] >= out["short_entry"])
            out["long_entry"] = out["long_entry"].where(df["open"] <= out["long_entry"])
        if self.long_min_p_down is not None or self.short_max_p_down is not None:
            if "p_down_prev" not in df:
                raise ValueError("局面で絞るには df に p_down_prev 列が要ります（設定の data.regime_file）")
            p = df["p_down_prev"]
            if self.long_min_p_down is not None:
                out["long_entry"] = out["long_entry"].where(p > self.long_min_p_down)
            if self.short_max_p_down is not None:
                out["short_entry"] = out["short_entry"].where(p <= self.short_max_p_down)
        if self.long_cooldown:  # 絞り込む前の値段で、前日までに割っていたか
            touched = (df["low"] <= band_price(c, self.window, -self.k)).astype(float)
            out["long_entry"] = out["long_entry"].where(touched.shift(1).rolling(self.long_cooldown, min_periods=1).max() == 0)
        if self.long_min_high250 is not None:
            out["long_entry"] = out["long_entry"].where((c / c.rolling(250).max() - 1).shift(1) >= self.long_min_high250)
        if self.long_max_ret5 is not None:
            out["long_entry"] = out["long_entry"].where((c.shift(1) / c.shift(6) - 1) <= self.long_max_ret5)
        if self.long_min_sma200 is not None:
            out["long_entry"] = out["long_entry"].where((c / c.rolling(200).mean() - 1).shift(1) >= self.long_min_sma200)
        if self.sides == "long":
            out["short_entry"] = math.nan
        elif self.sides == "short":
            out["long_entry"] = math.nan
        turnover = (c * df["volume"]).rolling(20).mean().shift(1)
        out["can_enter"] = out["short_entry"].notna() | out["long_entry"].notna()
        if self.min_turnover > 0:
            out["can_enter"] &= turnover >= self.min_turnover
        return out

    def bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """チャート用: 終値時点のバンドと、利益確定のバンド。"""
        c = df["close"]
        mid, sd = c.rolling(self.window).mean(), c.rolling(self.window).std(ddof=0)
        out = pd.DataFrame({"mid": mid, "upper": mid + self.k * sd, "lower": mid - self.k * sd})
        if self.exit_k is not None:
            out["exit_upper"] = mid + self.exit_k * sd
        if self.long_exit_k is not None:
            out["exit_lower"] = mid - self.long_exit_k * sd
        return out
