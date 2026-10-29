import pandas as pd

from trading.data.margin_balance import available_from


def test_weekly_margin_is_used_from_the_third_business_day():
    # 申込日 1/5（金）の残高は、翌週の第 2 営業日（月曜が休みなので 1/10 水）の大引け後に公表され、1/11 の判断から使える
    cal = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
                            "2024-01-15", "2024-01-16", "2024-01-17"])
    weekly = pd.DataFrame({"pl_ratio": [-8.0, -9.0]}, index=pd.DatetimeIndex(["2024-01-05", "2024-01-12"]))
    a = available_from(weekly, cal)
    assert a.loc[:"2024-01-10", "pl_ratio"].isna().all()
    assert a.loc["2024-01-11", "pl_ratio"] == -8.0 and a.loc["2024-01-16", "pl_ratio"] == -8.0
    assert a.loc["2024-01-17", "pl_ratio"] == -9.0 and a.loc["2024-01-17", "week"] == pd.Timestamp("2024-01-12")
