from dataclasses import dataclass

# TOPIX500 採用銘柄の呼値（値段の刻み）。(この価格以下, 呼値) の順
TICK_TABLE_TOPIX500 = [(1_000, 0.1), (3_000, 0.5), (10_000, 1), (30_000, 5), (100_000, 10), (300_000, 50),
                       (1_000_000, 100), (3_000_000, 500), (10_000_000, 1_000)]


def tick_size(price: float) -> float:
    """TOPIX500 採用銘柄の呼値。"""
    for upper, tick in TICK_TABLE_TOPIX500:
        if price <= upper:
            return tick
    return 5_000


@dataclass
class CostModel:
    commission_rate: float = 0.0   # 約定代金に対する定率
    commission_fixed: float = 0.0  # 1 約定あたりの定額（円）
    commission_min: float = 0.0    # 最低手数料（円）
    slippage_bps: float = 0.0      # 約定価格の不利方向へのずれ（bp）
    spread_yen: float = 0.0        # 売り買いの気配の差（円）。片道でその半分だけ不利に約定する
    spread_min_tick: bool = False  # True ならスプレッドを最低 1 呼値とする（株価が高い銘柄は 1 円で収まらない）
    short_fee_rate: float = 0.0    # 空売りの貸株料（年率）。日計りでも 1 日分かかる

    def commission(self, notional: float) -> float:
        return max(notional * self.commission_rate + self.commission_fixed, self.commission_min)

    def short_fee(self, notional: float, days: int = 1) -> float:
        return notional * self.short_fee_rate * days / 365

    def fill_price(self, price: float, side: str) -> float:
        spread = max(self.spread_yen, tick_size(price)) if self.spread_min_tick else self.spread_yen
        slip = price * self.slippage_bps / 10_000 + spread / 2
        return price + slip if side == "buy" else price - slip
