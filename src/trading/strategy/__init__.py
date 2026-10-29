from .band_touch import BandTouchFade, band_price
from .base import Strategy
from .intraday import IntradayStrategy, OpeningGap

STRATEGIES: dict[str, type[Strategy]] = {
    "bb_touch_fade": BandTouchFade,
}

INTRADAY_STRATEGIES: dict[str, type[IntradayStrategy]] = {
    "opening_gap": OpeningGap,
}


def get_strategy(name: str, **params) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"未知の戦略: {name}（選択肢: {list(STRATEGIES)}）")
    return STRATEGIES[name](**params)


def get_intraday_strategy(name: str, **params) -> IntradayStrategy:
    if name not in INTRADAY_STRATEGIES:
        raise ValueError(f"未知のデイトレード戦略: {name}（選択肢: {list(INTRADAY_STRATEGIES)}）")
    return INTRADAY_STRATEGIES[name](**params)
