"""ダッシュボード用のグラフ（plotly）。"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 色。日本の証券アプリと同じく利益=朱・損失=藍。ここだけ変えれば全体に反映される
WIN = "#cf3a2b"
LOSS = "#1f5fae"
INK = "#1c2430"
MUTED = "#6b7686"
GRID = "#e6e9ee"
OOS_COLOR = "#b7791f"
FONT = "BIZ UDPGothic, 'Hiragino Sans', 'Yu Gothic UI', sans-serif"
PERIOD_COLORS = {"ALL": INK, "IS": "#8a94a6", "OOS": OOS_COLOR, "PT": "#2f7d5b"}
PERIOD_NAMES = {"ALL": "全期間", "IS": "IS（調整用）", "OOS": "OOS（検証用）", "PT": "ペーパートレード（PT）"}
PERIODS = ("ALL", "IS", "OOS", "PT")   # 画面に並べる順
REASON_NAMES = {"": "シグナル", "signal": "シグナル", "stop": "損切り", "eod": "大引け", "target": "目標",
                "time": "期限", "limit": "ストップで持ち越し"}
REASON_COLORS = {"シグナル": "#3d4a5c", "損切り": LOSS, "大引け": "#b8c0cc", "目標": WIN, "期限": "#8a94a6", "ストップで持ち越し": "#7a3e9d"}
WEEKDAYS = ["月", "火", "水", "木", "金"]


def sign_colors(values) -> list[str]:
    return [WIN if v >= 0 else LOSS for v in values]


def _layout(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=72, r=16, t=70 if title else 24, b=44),  # 軸の数字が切れないよう余白は固定
        title=dict(text=title, y=0.98, yanchor="top", yref="container", x=0, xref="paper",
                   font=dict(size=15, color=INK)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), bargap=0.15,
        font=dict(family=FONT, size=12, color=INK), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=GRID),
    )
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont_color=MUTED, automargin=True)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_annotations(font=dict(family=FONT, size=13))
    return fig


def pnl_curve(kind: str, initial_cash: float, trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """期間ごとの累積損益（円）を日次で返す。OOS は IS の、ペーパートレード（PT）は OOS の最終値から続けてつなぐ。

    日足は含み損益込みの資産推移（各期間を initial_cash から始めた資産）から、デイトレードは毎日手仕舞うので確定損益の累計から作る。
    """
    curves = []
    offset = 0.0
    for period in [p for p in PERIODS if p in set(equity["period"]) | set(trades["period"])]:
        if kind == "daily":
            eq = equity[equity["period"] == period].set_index("date")["equity"]
            cum = eq - initial_cash
        else:
            t = trades[trades["period"] == period]
            cum = t.groupby(t["exit_date"].dt.normalize())["pnl"].sum().cumsum()
            days = equity.loc[equity["period"] == period, "date"]
            if len(days):
                cum = cum.reindex(pd.DatetimeIndex(days)).ffill().fillna(0.0)
        if cum.empty:
            continue
        c = pd.DataFrame({"period": period, "date": cum.index, "cum": cum.values + offset})
        if curves:  # 前の期間の最後の点から線をつなぐ
            c = pd.concat([curves[-1].tail(1).assign(period=period), c], ignore_index=True)
        curves.append(c)
        offset += cum.iloc[-1]
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["period", "date", "cum"])


def overview(curve: pd.DataFrame, trades: pd.DataFrame, live_from: str | None = None) -> go.Figure:
    """上段: 累積損益、中段: 日ごとの確定損益、下段: ドローダウン（累積損益の山からの下落幅）。
    live_from はペーパートレードで毎日の記録を始めた日（それより前は後から計算した分なので、そこに線を引く）。"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.2], vertical_spacing=0.05,
                        subplot_titles=("累積損益（円）", "日ごとの確定損益（円）", "ドローダウン（円）"))
    for period, c in curve.groupby("period", sort=False):
        fig.add_scatter(x=c["date"], y=c["cum"], name=PERIOD_NAMES.get(period, period), mode="lines",
                        line=dict(color=PERIOD_COLORS.get(period), width=2.5),
                        hovertemplate="%{x|%Y-%m-%d}<br>累積 %{y:+,.0f} 円<extra></extra>", row=1, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=1, col=1)

    daily = trades.groupby(trades["exit_date"].dt.normalize())["pnl"].sum()
    fig.add_bar(x=daily.index, y=daily.values, marker_color=sign_colors(daily.values), showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:+,.0f} 円<extra></extra>", row=2, col=1)

    if not curve.empty:
        cum = curve.drop_duplicates("date", keep="last").set_index("date")["cum"]
        dd = cum - cum.cummax().clip(lower=0)
        fig.add_scatter(x=dd.index, y=dd.values, fill="tozeroy", mode="lines", showlegend=False,
                        line=dict(color=LOSS, width=1), fillcolor="rgba(31,95,174,0.18)",
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} 円<extra></extra>", row=3, col=1)
        for period, text in (("OOS", " 検証期間（OOS）"), ("PT", " ペーパートレード（PT）")):
            span = curve[curve["period"] == period]
            if not span.empty and curve["period"].nunique() > 1:
                fig.add_vrect(x0=span["date"].min(), x1=span["date"].max(), fillcolor=PERIOD_COLORS[period],
                              opacity=0.08, line_width=0)
                fig.add_annotation(x=span["date"].min(), y=1, xref="x", yref="y domain", text=text,
                                   showarrow=False, xanchor="left", yanchor="top", font_color=PERIOD_COLORS[period])
        if live_from and curve["date"].min() < pd.Timestamp(live_from):
            fig.add_vline(x=pd.Timestamp(live_from), line=dict(color=PERIOD_COLORS["PT"], width=1.5, dash="dot"))
            fig.add_annotation(x=pd.Timestamp(live_from), y=0.9, xref="x", yref="y domain", text="毎日の記録を開始 ",
                               showarrow=False, xanchor="right", yanchor="top", font_color=PERIOD_COLORS["PT"])   # 期間の名前の下
    fig.update_yaxes(tickformat=",.0f")
    span = curve["date"].max() - curve["date"].min() if not curve.empty else pd.Timedelta(0)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])],
                     tickformat="%Y/%m" if span > pd.Timedelta(days=370) else "%m/%d")
    return _layout(fig, height=620)


def return_histogram(trades: pd.DataFrame) -> go.Figure:
    """1 回ごとのリターン（bp）の分布。勝ちと負けで色分けし、平均を縦線で示す。"""
    bp = trades["ret"] * 1e4
    lo, hi = bp.quantile(0.01), bp.quantile(0.99)
    size = max((hi - lo) / 40, 1.0)
    fig = go.Figure()
    for label, mask, color in [("勝ち", bp > 0, WIN), ("負け", bp <= 0, LOSS)]:
        fig.add_histogram(x=bp[mask].clip(lo, hi), name=label, marker_color=color, opacity=0.8,
                          xbins=dict(start=lo, end=hi + size, size=size),
                          hovertemplate="%{x:.0f} bp 付近: %{y} 回<extra></extra>")
    fig.add_vline(x=bp.mean(), line=dict(color=INK, dash="dash"),
                  annotation_text=f"平均 {bp.mean():+.1f} bp", annotation_position="top right")
    fig.update_layout(barmode="overlay", xaxis_title="1 回のリターン（bp, 手数料・スリッページ込み）",
                      yaxis_title="回数")
    _layout(fig, title="リターンの分布")
    return fig.update_layout(legend=dict(orientation="v", x=1, xanchor="right", y=1, yanchor="top"))


def exit_reasons(trades: pd.DataFrame) -> go.Figure:
    """決済理由ごとの回数（ドーナツ）と損益合計（棒）。"""
    by = (trades.assign(reason=trades["exit_reason"].map(lambda r: REASON_NAMES.get(r, r)))
          .groupby("reason")["pnl"].agg(["size", "sum"]).sort_values("size", ascending=False))
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]],
                        subplot_titles=("回数", "損益合計（円）"), horizontal_spacing=0.2)
    fig.add_pie(labels=by.index, values=by["size"], hole=0.55, sort=False, showlegend=False,
                marker=dict(colors=[REASON_COLORS.get(r, "#64748b") for r in by.index]),
                textinfo="label+percent", row=1, col=1)
    fig.add_bar(x=by.index, y=by["sum"], marker_color=sign_colors(by["sum"]), showlegend=False,
                text=[f"{v:+,.0f}" for v in by["sum"]], textposition="outside",
                hovertemplate="%{x}: %{y:+,.0f} 円<extra></extra>", row=1, col=2)
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, title="決済理由")


def monthly_pnl(trades: pd.DataFrame) -> go.Figure:
    m = trades.groupby(trades["exit_date"].dt.to_period("M"))["pnl"].sum()
    fig = go.Figure(go.Bar(x=m.index.astype(str), y=m.values, marker_color=sign_colors(m.values),
                           text=[f"{v / 1e4:+,.1f}万" for v in m.values], textposition="outside",
                           hovertemplate="%{x}: %{y:+,.0f} 円<extra></extra>"))
    fig.update_yaxes(tickformat=",.0f", title="円")
    fig.update_xaxes(type="category")
    return _layout(fig, title="月別の損益")


def stock_ranking(by_stock: pd.DataFrame, n: int = 15) -> go.Figure:
    """損益の上位・下位 n 銘柄。"""
    top = by_stock.nlargest(n, "損益")
    bottom = by_stock.nsmallest(n, "損益")
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"損益 上位 {len(top)}", f"損益 下位 {len(bottom)}"),
                        horizontal_spacing=0.12)
    for col, d in [(1, top.iloc[::-1]), (2, bottom.iloc[::-1])]:
        fig.add_bar(y=d["code"], x=d["損益"], orientation="h", marker_color=sign_colors(d["損益"]),
                    showlegend=False, customdata=d[["取引数", "勝率"]],
                    hovertemplate="%{y}: %{x:+,.0f} 円<br>取引 %{customdata[0]} 回 / 勝率 %{customdata[1]:.0%}"
                                  "<extra></extra>", row=1, col=col)
    fig.update_yaxes(type="category")
    fig.update_xaxes(tickformat=",.0f")
    return _layout(fig, height=max(360, 22 * n + 80))


def stock_scatter(by_stock: pd.DataFrame) -> go.Figure:
    """銘柄ごとの取引数と 1 回あたり平均リターン。右上ほど「回数も多く儲かる」銘柄。"""
    fig = go.Figure(go.Scatter(
        x=by_stock["取引数"], y=by_stock["平均bp"], mode="markers", text=by_stock["code"],
        marker=dict(color=sign_colors(by_stock["損益"]), size=9, opacity=0.7, line=dict(width=0.5, color="white")),
        customdata=by_stock[["損益", "勝率"]],
        hovertemplate="%{text}<br>取引 %{x} 回 / 平均 %{y:+.1f} bp<br>損益 %{customdata[0]:+,.0f} 円"
                      " / 勝率 %{customdata[1]:.0%}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(xaxis_title="取引数", yaxis_title="1 回あたり平均（bp）")
    return _layout(fig, title="銘柄ごとの取引数と平均リターン（緑=損益プラス）")


def by_entry_time(trades: pd.DataFrame) -> go.Figure:
    """エントリー時刻ごとの平均リターン（棒）と勝率（線）。"""
    g = trades.groupby(trades["entry_date"].dt.strftime("%H:%M"))
    d = pd.DataFrame({"avg": g["ret"].mean() * 1e4, "win": g["pnl"].apply(lambda x: (x > 0).mean()),
                      "n": g.size()})
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d.index, y=d["avg"], name="平均リターン（bp）", marker_color=sign_colors(d["avg"]),
                customdata=d["n"], hovertemplate="%{x} 平均 %{y:+.1f} bp（%{customdata} 回）<extra></extra>")
    fig.add_scatter(x=d.index, y=d["win"], name="勝率", mode="lines+markers", line=dict(color=INK),
                    hovertemplate="%{x} 勝率 %{y:.0%}<extra></extra>", secondary_y=True)
    fig.update_yaxes(title_text="平均リターン（bp）", secondary_y=False)
    fig.update_yaxes(title_text="勝率", tickformat=".0%", range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1],
                     secondary_y=True, showgrid=False)
    fig.update_xaxes(type="category", title="エントリー時刻（約定したバー）")
    _layout(fig, title="エントリー時刻ごとの成績")
    return fig.update_layout(margin_r=60)


def by_weekday(trades: pd.DataFrame) -> go.Figure:
    g = trades.groupby(trades["entry_date"].dt.weekday)
    avg = (g["ret"].mean() * 1e4).reindex(range(5))
    n = g.size().reindex(range(5)).fillna(0).astype(int)
    fig = go.Figure(go.Bar(x=WEEKDAYS, y=avg.values, marker_color=sign_colors(avg.fillna(0)),
                           text=[f"{v:+.1f}bp<br>{c}回" for v, c in zip(avg.fillna(0), n)], textposition="outside",
                           hovertemplate="%{x}曜 平均 %{y:+.1f} bp<extra></extra>"))
    fig.update_yaxes(title="平均リターン（bp）")
    return _layout(fig, title="曜日ごとの平均リターン")


def _markers(fig: go.Figure, x_entry, y_entry, x_exit, y_exit, row: int | None = 1) -> None:
    rc = dict(row=row, col=1) if row else {}
    # 売買の印は損益の色と混ざらないよう墨色（買い=塗り▲、売り=白抜き▼）
    fig.add_scatter(x=x_entry, y=y_entry, mode="markers", name="買い", **rc,
                    marker=dict(symbol="triangle-up", size=15, color=INK, line=dict(width=1.5, color="white")),
                    hovertemplate="買い %{y:,.1f}<extra></extra>")
    fig.add_scatter(x=x_exit, y=y_exit, mode="markers", name="売り", **rc,
                    marker=dict(symbol="triangle-down", size=15, color="white", line=dict(width=2, color=INK)),
                    hovertemplate="売り %{y:,.1f}<extra></extra>")


def intraday_trade(bars: pd.DataFrame, vwap: pd.Series, trade: pd.Series) -> go.Figure:
    """1 日分の 5 分足に、VWAP と売買の位置を重ねる。昼休みを詰めるため横軸は時刻の文字列。"""
    x = bars.index.strftime("%H:%M")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_candlestick(x=x, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
                        name="5分足", increasing_line_color=WIN, decreasing_line_color=LOSS, row=1, col=1)
    fig.add_scatter(x=x, y=vwap, name="VWAP", mode="lines", line=dict(color=OOS_COLOR, width=2), row=1, col=1)
    fig.add_hrect(y0=min(trade["entry_price"], trade["exit_price"]), y1=max(trade["entry_price"], trade["exit_price"]),
                  fillcolor=WIN if trade["pnl"] > 0 else LOSS, opacity=0.1, line_width=0, row=1, col=1)
    _markers(fig, [trade["entry_date"].strftime("%H:%M")], [trade["entry_price"]],
             [trade["exit_date"].strftime("%H:%M")], [trade["exit_price"]])
    fig.add_bar(x=x, y=bars["volume"], marker_color="#c3cad4", name="出来高", showlegend=False, row=2, col=1)
    fig.update_xaxes(type="category", rangeslider_visible=False, nticks=12)
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, height=520)


def daily_trades(prices: pd.DataFrame, trades: pd.DataFrame, selected: pd.Series | None,
                 sma: tuple[int, ...] = (), bands: pd.DataFrame | None = None) -> go.Figure:
    """日足のローソク足に、全取引の売買位置（と移動平均・ボリンジャーバンド）を重ねる。選んだ取引は帯で強調する。"""
    fig = go.Figure()
    fig.add_candlestick(x=prices.index, open=prices["open"], high=prices["high"], low=prices["low"],
                        close=prices["close"], name="日足", increasing_line_color=WIN, decreasing_line_color=LOSS)
    for n, color in zip(sma, [OOS_COLOR, "#8a94a6", "#3d4a5c"]):
        fig.add_scatter(x=prices.index, y=prices["close"].rolling(n).mean(), name=f"SMA{n}", mode="lines",
                        line=dict(color=color, width=1.5))
    if bands is not None:
        for col, name, dash in [("upper", "上のバンド", "dot"), ("mid", "中心線", "solid"),
                                ("lower", "下のバンド", "dot"), ("exit_upper", "利益確定", "dash"),
                                ("exit_lower", "利益確定", "dash"), ("lower_2", "下のバンド（2 つ目の手法）", "dashdot")]:
            if col not in bands:
                continue
            fig.add_scatter(x=bands.index, y=bands[col], name=name, mode="lines",
                            line=dict(color=OOS_COLOR, width=1.3, dash=dash))
    shorts = trades["side"] == "short" if "side" in trades else pd.Series(False, index=trades.index)
    longs = trades[~shorts]
    _markers(fig, longs["entry_date"], longs["entry_price"], longs["exit_date"], longs["exit_price"], row=None)
    if shorts.any():
        s = trades[shorts]
        fig.add_scatter(x=s["entry_date"], y=s["entry_price"], mode="markers", name="空売り",
                        marker=dict(symbol="triangle-down", size=15, color=INK, line=dict(width=1.5, color="white")),
                        hovertemplate="空売り %{y:,.1f}<extra></extra>")
        fig.add_scatter(x=s["exit_date"], y=s["exit_price"], mode="markers", name="買い戻し",
                        marker=dict(symbol="triangle-up", size=15, color="white", line=dict(width=2, color=INK)),
                        hovertemplate="買い戻し %{y:,.1f}<extra></extra>")
    if selected is not None:
        fig.add_vrect(x0=selected["entry_date"], x1=selected["exit_date"], line_width=0, opacity=0.12,
                      fillcolor=WIN if selected["pnl"] > 0 else LOSS)
        pad = pd.Timedelta(days=90)
        lo, hi = selected["entry_date"] - pad, selected["exit_date"] + pad
        window = prices.loc[lo:hi]
        margin = (window["high"].max() - window["low"].min()) * 0.05
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[window["low"].min() - margin, window["high"].max() + margin])
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])], tickformat="%Y/%m")
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, height=480)


def compare_runs(results: pd.DataFrame) -> go.Figure:
    """実行・期間ごとのプロフィットファクターと勝率（保有期間が違っても比べやすい指標）。"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("プロフィットファクター（1 超で利益）", "勝率"))
    for period, d in results.groupby("period", sort=False):
        common = dict(x=d["label"], name=PERIOD_NAMES.get(period, period), marker_color=PERIOD_COLORS.get(period),
                      legendgroup=period, textposition="outside")
        fig.add_bar(y=d["profit_factor"], text=[f"{v:.2f}" for v in d["profit_factor"]], **common, row=1, col=1)
        fig.add_bar(y=d["win_rate"], text=[f"{v:.0%}" for v in d["win_rate"]], showlegend=False, **common,
                    row=1, col=2)
    fig.add_hline(y=1, line=dict(color=INK, dash="dash", width=1), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    _layout(fig, height=420)
    return fig.update_layout(legend=dict(y=-0.15, yanchor="top"), margin=dict(t=40))
