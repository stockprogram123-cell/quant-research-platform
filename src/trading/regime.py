"""相場の局面（上げ相場 / 下げ相場）を日足から判定する。

- zigzag_regime: 日経平均の終値が直近の高値（安値）から threshold 以上動いたら局面が変わったとみなし、
  高値→安値を「下げ」、安値→高値を「上げ」とする。後から見た答え（学習のラベル）で、その日には分からない。
- regime_features: 各日の終値時点で分かる値だけで作る特徴（指数の位置・勢い・変動率、銘柄の広がり）。
- RegimeModel: 特徴から「いま下げ相場である確率」を出すロジスティック回帰（numpy のみ）。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _zigzag(values: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    """転換点の一覧と、各日の終値時点で確定している向き（1 = 下げ、0 = 上げ、NaN = 不明）。"""
    pivots = [(0, 0)]   # (位置, 1 = そこが高値 / -1 = そこが安値 / 0 = 開始)
    state = np.full(len(values), np.nan)
    hi_i = lo_i = 0
    trend = 0           # 1 = 上げの途中、-1 = 下げの途中、0 = 不明
    for i in range(1, len(values)):
        c = values[i]
        if trend >= 0 and c > values[hi_i]:
            hi_i = i
        if trend <= 0 and c < values[lo_i]:
            lo_i = i
        if trend >= 0 and c <= values[hi_i] * (1 - threshold):
            pivots.append((hi_i, 1))
            trend, lo_i = -1, i
        elif trend <= 0 and c >= values[lo_i] * (1 + threshold):
            pivots.append((lo_i, -1))
            trend, hi_i = 1, i
        if trend != 0:
            state[i] = 1 if trend == -1 else 0
    return pivots, state


def zigzag_regime(close: pd.Series, threshold: float = 0.10) -> pd.Series:
    """各日の局面（1 = 下げ、0 = 上げ、最初の転換点より前は NaN）。後から見た答え。

    高値から threshold 下げた時点で「その高値から下げ相場」、安値から threshold 上げた時点で
    「その安値から上げ相場」と確定する。最後の区間は終わりが未確定のまま直近の向きを入れる。
    """
    pivots, _ = _zigzag(close.to_numpy(dtype=float), threshold)
    labels = np.full(len(close), np.nan)
    for (start, kind), (end, _) in zip(pivots, pivots[1:] + [(len(close), 0)]):
        if kind != 0:
            labels[start:end] = 1 if kind == 1 else 0
    return pd.Series(labels, index=close.index, name="down")


def confirmed_regime(close: pd.Series, threshold: float = 0.05) -> pd.Series:
    """その日の終値時点で確定している局面（先読みなし）。直近の高値から threshold 下げたら下げ相場、
    直近の安値から threshold 戻したら上げ相場に切り替わる。翌日の判断に使うこと。"""
    _, state = _zigzag(close.to_numpy(dtype=float), threshold)
    return pd.Series(state, index=close.index, name="down")


def market_regime(close: pd.Series, mode: str, threshold: float) -> pd.Series:
    """5 分足の各日に使う局面（日付 → 1 = 下げ / 0 = 上げ）。

    mode="hindsight": 後から見た局面（zigzag_regime）。「いまは下げ相場だ」という判断が正しい前提で
      その期間の立ち回りを検証するためのもの。実際には当日に分からない（先読み）。
    mode="confirmed": 前日の終値までで確定している局面（confirmed_regime を 1 日ずらす）。先読みなし。
    """
    if mode == "hindsight":
        return zigzag_regime(close, threshold)
    if mode == "confirmed":
        return confirmed_regime(close, threshold).shift(1)
    raise ValueError("mode は 'hindsight' か 'confirmed' にしてください")


def legs(regime: pd.Series, close: pd.Series) -> pd.DataFrame:
    """局面の区間の一覧（開始日・終了日・日数・騰落率）。"""
    r = regime.dropna()
    block = (r != r.shift()).cumsum()
    rows = []
    for _, seg in r.groupby(block):
        start, end = seg.index[0], seg.index[-1]
        rows.append({"down": int(seg.iloc[0]), "start": start, "end": end, "days": len(seg),
                     "change": close.loc[end] / close.loc[start] - 1})
    return pd.DataFrame(rows)


def regime_features(index_daily: pd.DataFrame, stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """各日の終値時点で分かる局面の特徴。index_daily は指数の日足、stocks は銘柄ごとの日足。"""
    c, o = index_daily["close"], index_daily["open"]
    ret = c.pct_change()
    sma = {n: c.rolling(n).mean() for n in (25, 75, 200)}
    f = pd.DataFrame(index=c.index)
    for n, s in sma.items():
        f[f"px_sma{n}"] = c / s - 1                                  # 移動平均からの乖離
    f["sma25_slope"] = sma[25] / sma[25].shift(5) - 1                # 移動平均の向き
    f["sma75_slope"] = sma[75] / sma[75].shift(10) - 1
    f["ret20"] = c / c.shift(20) - 1
    f["ret60"] = c / c.shift(60) - 1
    f["dd252"] = c / c.rolling(252).max() - 1                        # 1 年の高値からの下落率
    f["vol20"] = ret.rolling(20).std() * np.sqrt(245)                # 変動率（年率）
    f["vol_ratio"] = ret.rolling(20).std() / ret.rolling(120).std()  # 変動率がふだんより高いか
    f["up_days20"] = (ret > 0).rolling(20).mean()
    f["gap_down20"] = (o / c.shift(1) - 1 < -0.01).rolling(20).mean()  # 1% 超の窓を開けた下げの頻度

    closes = pd.DataFrame({k: v["close"] for k, v in stocks.items()}).reindex(c.index)
    lows = pd.DataFrame({k: v["low"] for k, v in stocks.items()}).reindex(c.index)
    valid = closes.notna().sum(axis=1)
    f["breadth25"] = (closes > closes.rolling(25).mean()).sum(axis=1) / valid    # 25 日線より上の銘柄の割合
    f["breadth75"] = (closes > closes.rolling(75).mean()).sum(axis=1) / valid
    f["new_low20"] = (lows <= lows.rolling(20).min()).sum(axis=1) / valid        # 20 日安値を付けた銘柄の割合
    f["breadth25_chg"] = f["breadth25"] - f["breadth25"].shift(5)
    return f


@dataclass
class RegimeModel:
    """ロジスティック回帰（標準化 + L2）。predict は「下げ相場である確率」。"""

    features: list[str]
    l2: float = 1.0
    mean_: pd.Series | None = None
    std_: pd.Series | None = None
    coef_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeModel":
        X = X[self.features]
        self.mean_, self.std_ = X.mean(), X.std()
        Z = np.column_stack([np.ones(len(X)), ((X - self.mean_) / self.std_).to_numpy()])
        w = np.zeros(Z.shape[1])
        penalty = np.diag([0.0] + [self.l2] * (Z.shape[1] - 1))
        for _ in range(50):  # ニュートン法
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y.to_numpy()) + penalty @ w
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + penalty
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef_ = w
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        Z = ((X[self.features] - self.mean_) / self.std_).to_numpy()
        return pd.Series(1 / (1 + np.exp(-(self.coef_[0] + Z @ self.coef_[1:]))), index=X.index, name="p_down")

    def to_dict(self) -> dict:
        """学習した値（JSON に保存できる形）。"""
        return {"features": self.features, "l2": self.l2, "mean": self.mean_.tolist(), "std": self.std_.tolist(),
                "coef": self.coef_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeModel":
        m = cls(d["features"], d["l2"])
        m.mean_, m.std_ = pd.Series(d["mean"], index=d["features"]), pd.Series(d["std"], index=d["features"])
        m.coef_ = np.array(d["coef"])
        return m

    def coefficients(self) -> pd.Series:
        """標準化した特徴 1 単位あたりの重み（プラスなら大きいほど下げ相場らしい）。"""
        return pd.Series(self.coef_[1:], index=self.features).sort_values()
