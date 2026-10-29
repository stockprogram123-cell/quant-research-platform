import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 245  # 東証の年間営業日数の目安


def max_drawdown(equity: pd.Series) -> float:
    """最大ドローダウン（負の値。-0.2 = -20%）。"""
    return float((equity / equity.cummax() - 1).min())


def compute_metrics(equity: pd.Series, round_trips: pd.DataFrame) -> dict[str, float]:
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    returns = equity.pct_change().dropna()
    std = returns.std()
    sharpe = returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else np.nan

    n_trades = len(round_trips)
    win_rate = (round_trips["pnl"] > 0).mean() if n_trades else np.nan

    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown(equity),
        "sharpe": float(sharpe),
        "num_trades": n_trades,
        "win_rate": float(win_rate),
    }


def summarize(round_trips: pd.DataFrame, equity: pd.Series) -> dict:
    """往復取引（ret 列付き）と資産推移から結果の要約を作る。DB 保存・画面表示に使う。"""
    if round_trips.empty:
        return {"num_trades": 0, "total_pnl": 0.0, "avg_ret_bp": float("nan")}
    m = compute_metrics(equity, round_trips)
    pnl = round_trips["pnl"]
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "num_trades": len(round_trips),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "avg_ret_bp": float(round_trips["ret"].mean() * 1e4),
        "profit_factor": float(gains / losses) if losses > 0 else float("nan"),
        "total_return": m["total_return"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "exit_reasons": round_trips["exit_reason"].value_counts().to_dict(),
    }
