from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_equity(equity: pd.Series, path: str | Path, title: str = "",
                benchmark: pd.Series | None = None) -> None:
    """資産推移とドローダウンを PNG に保存する。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity, label="Strategy")
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark, label="Buy & Hold", alpha=0.6)
    ax1.set_title(title)
    ax1.set_ylabel("Equity (JPY)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    drawdown = equity / equity.cummax() - 1
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.4, color="tab:red")
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
