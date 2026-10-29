import pandas as pd
import yfinance as yf

from .base import DataSource, normalize


class YFinanceSource(DataSource):
    """Yahoo Finance（非公式 API）から取得する。東証銘柄は "7203.T" 形式、指数は "^N225" のようにそのまま。"""

    name = "yfinance"
    adjust = True   # 配当・分割で過去の値段を調整する

    def get_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        ticker = code if "." in code or code.startswith("^") else f"{code}.T"
        # yfinance の end は排他的なので 1 日足す
        end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start, end=end_exclusive, auto_adjust=self.adjust)
        if df.empty:
            raise ValueError(f"データを取得できませんでした: {ticker}")
        return normalize(df)


class YFinanceRawSource(YFinanceSource):
    """調整しない値段。日経平均の ETF（1330 など）は、配当で調整した値段が壊れていることがある（マイナスの値段など）。"""

    name = "yfinance_raw"
    adjust = False
