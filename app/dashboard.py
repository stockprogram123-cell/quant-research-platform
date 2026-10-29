"""バックテスト結果（SQLite）を確認する画面。

    streamlit run app/dashboard.py
"""
import html
import inspect
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
from trading.data import get_source, load_5min, load_daily_csv
from trading.strategy import STRATEGIES
from trading.store import DEFAULT_DB, ResultStore
from trading.data.intraday import session_vwap

st.set_page_config(page_title="売買の検証", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

MODES = {"intraday": "5分足デイトレ", "daily": "日足トレード", "compare": "実行を比べる"}
MODE_ICONS = {"intraday": ":material/schedule:", "daily": ":material/calendar_month:", "compare": ":material/compare_arrows:"}
STRATEGY_NAMES = {"opening_gap": "寄りの窓についていく（前場）", "bb_touch_fade": "BB±3σタッチで逆張り（日足）"}
PERIOD_TITLES = {"ALL": ("全期間", ""), "IS": ("調整期間（IS）", "パラメータを決めるのに使った期間"),
                 "OOS": ("検証期間（OOS）", "調整に使っていない期間。本番に近い成績です"),
                 "PT": ("ペーパートレード（PT）", "実際の注文は出さず、同じ決まりの口座を 1 日ずつ動かした成績")}
WEEKDAYS = "月火水木金土日"
# チャートのバンドを描くのに使う戦略（日経平均の買い増し・大引けで判断して買う形も同じバンド）
BAND_STRATEGY = {"index_overlay": "bb_touch_fade", "bb_close": "bb_touch_fade", "bb_open": "bb_touch_fade",
                 "small_account": "bb_touch_fade"}
RULE_SPECS = ("手法", "絞り込み", "局面", "手仕舞い", "パラメータ")   # 設定のうち売買のルール（残りは資金とデータ）

CSS = f"""
<style>
:root {{ --ink: {charts.INK}; --muted: {charts.MUTED}; --line: #dde2e8; --soft: #f6f7f9;
        --win: {charts.WIN}; --loss: {charts.LOSS}; --oos: {charts.OOS_COLOR}; --pt: {charts.PERIOD_COLORS['PT']}; }}
.block-container {{ padding-top: 2.4rem; max-width: 1500px; }}
.win {{ color: var(--win); }} .loss {{ color: var(--loss); }} .flat {{ color: var(--muted); }}
* {{ font-variant-numeric: tabular-nums; }}
button:focus-visible {{ outline: 2px solid var(--oos) !important; outline-offset: 2px; }}

/* 上部の墨の帯。選んだ画面のタブだけ紙の色になり、下の画面とつながる */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stHeader"] button, [data-testid="stSidebarCollapsedControl"] button {{ color: #fff; }}
.st-key-topbar {{ background: var(--ink); box-shadow: 0 0 0 100vmax var(--ink); clip-path: inset(-100vmax -100vmax 0);
                 padding-top: .2rem; padding-bottom: 0; margin-bottom: 1.4rem; gap: 0; }}
.brand {{ font-size: 1.15rem; font-weight: 700; color: #fff; letter-spacing: .04em; margin: 0 0 .7rem; }}
.brand small {{ display: block; font-size: .78rem; font-weight: 400; color: #9aa5b4; letter-spacing: 0; margin-top: .1rem; }}
.st-key-modebar [role="radiogroup"] {{ gap: .3rem; flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; }}
.st-key-modebar button, .st-key-glossary button {{ white-space: nowrap; flex-shrink: 0; }}
.st-key-modebar button {{ background: transparent; border: none !important; border-radius: 10px 10px 0 0 !important;
                         padding: .7rem 1.4rem .8rem; min-height: 3rem; color: #c3cbd6; }}
.st-key-modebar button p {{ font-size: 1.02rem; font-weight: 700; }}
.st-key-modebar button:hover {{ background: rgba(255,255,255,.08); color: #fff; }}
.st-key-modebar button[data-selected="true"], .st-key-modebar button[data-selected="true"]:hover {{ background: #fff !important; }}
.st-key-modebar button[data-selected="true"] * {{ color: var(--ink) !important; }}
.st-key-topbar [data-testid="stHorizontalBlock"] {{ align-items: flex-end; }}
.st-key-topbar .stElementContainer {{ margin-bottom: 0; }}
.st-key-glossary button {{ background: transparent; color: #fff; border-color: #4a5566; margin-bottom: .6rem; }}
.st-key-glossary button:hover {{ border-color: #fff; color: #fff; }}

/* 実行のカード。上の線の色が検証期間の成績（朱＝PF 1 以上、藍＝1 未満） */
.st-key-runcards {{ flex-wrap: nowrap !important; overflow-x: auto; align-items: stretch; padding: .1rem .1rem .6rem; gap: .6rem; }}
[class*="st-key-rc_"] button {{ display: block; width: 17rem; min-height: 7.4rem; height: 100%; white-space: normal; text-align: left; padding: .55rem .85rem .6rem;
                               background: #fff; border: 1px solid var(--line); border-top: 4px solid #b8c0cc;
                               border-radius: 4px 4px 8px 8px; color: var(--ink); }}
[class*="st-key-rc_"] button:hover {{ border-color: var(--ink); border-top-width: 4px; }}
[class*="st-key-rc_"][class*="_win"] button {{ border-top-color: var(--win); }}
[class*="st-key-rc_"][class*="_loss"] button {{ border-top-color: var(--loss); }}
[class*="st-key-rc_"] button > div, [class*="st-key-rc_"] button span,
[class*="st-key-rc_"] button [data-testid="stMarkdownContainer"] {{ display: block !important; width: 100%; text-align: left; }}
[class*="st-key-rc_"] button p {{ display: block; width: 100%; overflow: visible !important; padding-bottom: 1px; margin: 0; font-size: .8rem; color: var(--muted); line-height: 1.45; text-align: left; }}
[class*="st-key-rc_"] button p strong {{ display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                                         overflow: hidden; font-size: .95rem; line-height: 1.35; color: var(--ink);
                                         min-height: 2.7em; margin: .1rem 0 .3rem; white-space: normal; }}
[class*="st-key-rc_"] button p:nth-child(2) {{ white-space: normal; }}
/* タイトルの下の細かい設定（資金・データ）は 1 行ずつ、はみ出す分は省略 */
[class*="st-key-rc_"] button p:not(:first-child):not(:nth-child(2)):not(:last-child) {{
    white-space: nowrap; overflow: hidden !important; text-overflow: ellipsis; line-height: 1.5; }}
[class*="st-key-rc_"] button p:last-child {{ margin-top: .3rem; }}
[class*="st-key-rc_"] button p em {{ font-style: normal; font-weight: 700; font-size: 1.05rem; }}
[class*="st-key-rc_"][class*="_win"] button p em {{ color: var(--win); }}
[class*="st-key-rc_"][class*="_loss"] button p em {{ color: var(--loss); }}
[class*="st-key-rc_"][class*="_sel"] button {{ background: var(--soft); border-color: var(--ink);
                                              box-shadow: inset 0 0 0 1px var(--ink); }}

/* 切り替えのボタン。選んだものは墨で塗る（勝ち・負けと期間は意味の色で塗る） */
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{ padding: .4rem 1rem; }}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p {{ font-weight: 700; }}
[data-testid="stMain"] button[data-variant="segmented_control"][data-selected="true"] {{ background: var(--ink); border-color: var(--ink); }}
[data-testid="stMain"] button[data-variant="segmented_control"][data-selected="true"] p {{ color: #fff; }}
[class*="st-key-outcome_"] button:nth-child(2)[data-selected="true"] {{ background: var(--win) !important; border-color: var(--win) !important; }}
[class*="st-key-outcome_"] button:nth-child(3)[data-selected="true"] {{ background: var(--loss) !important; border-color: var(--loss) !important; }}
[class*="st-key-period_"] button:last-child[data-selected="true"] {{ background: var(--oos) !important; border-color: var(--oos) !important; }}
[class*="st-key-reason_"] button {{ padding-left: .7rem; }}
[class*="st-key-reason_"] button p::before {{ content: ""; display: inline-block; width: .6em; height: .6em; border-radius: 50%;
                                            margin-right: .45em; background: var(--dot, var(--muted)); }}
[class*="st-key-reason_"] button[data-selected="true"] {{ background: #fff; border-color: var(--ink); color: var(--ink); }}
[class*="st-key-reason_"] button[data-selected="false"] {{ color: #9aa5b4; text-decoration: line-through; }}
[class*="st-key-reason_"] button[data-selected="false"] p::before {{ opacity: .35; }}

/* タブ: 選んだタブは太い墨の下線 */
.stTabs [role="tablist"] {{ gap: 1.6rem; border-bottom: 1px solid var(--line); }}
.stTabs [role="tab"] {{ padding: .55rem .1rem; }}
.stTabs [role="tab"] p {{ font-size: 1rem; font-weight: 700; color: var(--muted); }}
.stTabs [role="tab"][aria-selected="true"] p {{ color: var(--ink); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--ink); height: 3px; }}

/* 普通のボタン */
.stButton button, .stDownloadButton button {{ font-weight: 700; border-radius: 6px; }}
.stButton button[kind="primary"] {{ background: var(--loss); border-color: var(--loss); }}

.chips {{ display: flex; flex-wrap: wrap; gap: .35rem; margin: .5rem 0 0 !important; padding: 0 !important; list-style: none; }}
.chips li {{ margin: 0 !important; }}
.chips li {{ border: 1px solid var(--line); border-radius: 4px; padding: .1rem .55rem; font-size: .85rem; background: #fff; }}
.chips li span {{ color: var(--muted); margin-right: .4em; }}

.hero h1 {{ font-size: 2.1rem; font-weight: 700; line-height: 1.3; margin: .2rem 0 .35rem; padding: 0; color: var(--ink); }}
/* タイトルの下の細かい設定（trading.store.naming.run_details）。左に売買のルール、右に資金とデータ */
.specs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 0 2.4rem;
         margin: .7rem 0 .6rem; padding: 0; }}
.specs dl {{ margin: 0; padding: 0; border-top: 1px solid var(--line); }}
.specs div {{ display: grid; grid-template-columns: 6.5rem 1fr; gap: 1rem; padding: .38rem 0; border-bottom: 1px solid var(--line); }}
.specs dt {{ color: var(--muted); font-size: .85rem; margin: 0; padding-top: .05rem; }}
.specs dd {{ margin: 0; font-size: .92rem; color: var(--ink); }}
.hero .meta {{ color: var(--muted); font-size: .9rem; margin: 0; }}
.hero .meta b {{ color: var(--ink); font-weight: 700; margin-right: 1.4em; }}
.hero .note {{ margin: .5rem 0 0; font-size: .92rem; color: var(--ink); }}

.score {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); border: 1px solid var(--line);
         border-radius: 8px; overflow: hidden; background: var(--soft); margin: .4rem 0 .6rem; }}
.score section {{ padding: 1.1rem 1.4rem 1.2rem; border-top: 3px solid #b8c0cc; }}
.score section + section {{ border-left: 1px solid var(--line); }}
.score section.oos {{ border-top-color: var(--oos); background: #fff; }}
.score h3 {{ font-size: 1rem; font-weight: 700; margin: 0; padding: 0; color: var(--ink); }}
.score section.oos h3 {{ color: var(--oos); }}
.score section.pt {{ border-top-color: var(--pt); background: #fff; }}
.score section.pt h3 {{ color: var(--pt); }}
.score .span {{ font-size: .8rem; color: var(--muted); margin: .15rem 0 0; }}
.score .pf {{ display: flex; align-items: baseline; gap: .5rem; margin: .7rem 0 .3rem; }}
.score .pf strong {{ font-size: 2.9rem; font-weight: 700; line-height: 1; }}
.score .pf span {{ font-size: .85rem; color: var(--muted); }}
.gauge {{ position: relative; height: 8px; background: #e3e7ec; border-radius: 4px; margin: .4rem 0 1.3rem; }}
.gauge i {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; }}
.gauge b {{ position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; background: var(--ink); }}
.gauge em {{ position: absolute; left: 50%; top: 12px; transform: translateX(-50%); font-size: .7rem;
            font-style: normal; color: var(--muted); }}
.score .judge {{ font-size: .92rem; margin: 0 0 .8rem; }}
.score dl {{ display: grid; grid-template-columns: auto 1fr; gap: .3rem 1rem; margin: 0; padding: 0; font-size: .92rem; }}
.score dt {{ color: var(--muted); margin: 0; padding: 0; }}
.score dd {{ margin: 0; text-align: right; font-weight: 700; }}

.tally {{ display: flex; flex-wrap: wrap; gap: .2rem 1.6rem; font-size: .92rem; color: var(--muted); margin: .2rem 0 .5rem; }}
.tally b {{ color: var(--ink); font-size: 1.05rem; margin-left: .35em; }}

.slip {{ border: 1px solid var(--line); border-left: 5px solid var(--muted); border-radius: 6px;
        padding: .9rem 1.2rem; display: grid; grid-template-columns: 1fr auto; gap: .6rem 1.5rem; align-items: center; }}
.slip.win {{ border-left-color: var(--win); }} .slip.loss {{ border-left-color: var(--loss); }}
.slip .code {{ font-size: 1.7rem; font-weight: 700; line-height: 1.1; }}
.slip .when {{ display: block; color: var(--muted); font-size: .9rem; margin-top: .2rem; }}
.slip .reason {{ display: inline-block; font-size: .8rem; border: 1px solid var(--line); border-radius: 999px;
                padding: .05rem .6rem; margin-left: .6rem; vertical-align: .15rem; }}
.slip .legs {{ display: flex; gap: 1.8rem; margin-top: .45rem; font-size: .95rem; }}
.slip .legs span {{ color: var(--muted); margin-right: .5em; }}
.slip .pnl {{ text-align: right; }}
.slip .pnl strong {{ font-size: 1.9rem; font-weight: 700; display: block; line-height: 1.1; white-space: nowrap; }}
.slip .pnl small {{ color: var(--muted); font-size: .85rem; }}
@media (max-width: 640px) {{ .slip {{ grid-template-columns: 1fr; }} .slip .pnl {{ text-align: left; }} }}
</style>
"""


def strategy_label(name: str) -> str:
    return STRATEGY_NAMES.get(name, name)


def run_title(row) -> str:
    """実行の短いタイトル（trading.store.naming.run_name）。名前のない実行は戦略名で代用する。"""
    return row["name"] or strategy_label(row["strategy"])


def run_specs(row) -> dict[str, str]:
    """タイトルの下に並べる細かい設定（項目 → 値）。古い実行にはない。"""
    d = row.get("details") or {}
    return json.loads(d) if isinstance(d, str) else d


def specs_text(row, sep: str = "　／　") -> str:
    return sep.join(f"{k}: {v}" for k, v in run_specs(row).items()) or params_text(row["params"])


def params_text(params: str | dict) -> str:
    p = json.loads(params) if isinstance(params, str) else params
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "なし"


def num(v: float, fmt: str) -> str:
    """書式付きの数値。負号は読みやすいマイナス記号にする。"""
    return format(v, fmt).replace("-", "−")


def tone(v: float) -> str:
    return "win" if v > 0 else "loss" if v < 0 else "flat"


def judge(pf: float) -> tuple[str, str]:
    """プロフィットファクターを、損失 1 円あたりの利益で言い換える。"""
    if pd.isna(pf):
        return "負けた取引がありません。", "win"
    each = f"損失 1 円につき利益 {pf:.2f} 円。"
    if pf >= 1.3:
        return "利益が出ています。" + each, "win"
    if pf >= 1.0:
        return "わずかにプラスです。" + each, "win"
    return "負け越しています。" + each, "loss"


def paper_note(paper: dict | None) -> str:
    """ペーパートレード（PT）の始め方（実行の設定の paper。scripts/paper_trade.py が書く）。"""
    if not paper:
        return PERIOD_TITLES["PT"][1]
    start, live = (pd.Timestamp(paper[k]).strftime("%Y/%m/%d") for k in ("start", "live_from"))
    text = f"{paper['capital'] / 1e4:,.0f} 万円で始め、実際の注文は出さずに記録している成績。"
    return text + (f"{live} までは手元の日足で後から計算し、その後は毎日の大引けの後に記録" if live > start else "毎日の大引けの後に記録")


def scorecard(results: pd.DataFrame, equity: pd.DataFrame, paper: dict | None = None) -> str:
    """期間ごとの成績を横に並べた表（HTML）。主役はプロフィットファクターとそのゲージ。"""
    order = [p for p in charts.PERIODS if p in results.index]
    cols = []
    for period in order:
        r = results.loc[period]
        title, sub = PERIOD_TITLES.get(period, (period, ""))
        if period == "PT":
            sub = paper_note(paper)
        days = equity.loc[equity["period"] == period, "date"]
        span = f"{days.min():%Y/%m/%d} 〜 {days.max():%Y/%m/%d}　{sub}" if len(days) else sub
        if not r["num_trades"]:
            body = '<p class="judge flat">取引はありませんでした。</p>'
        else:
            pf = r["profit_factor"]
            text, cls = judge(pf)
            fill = 100 if pd.isna(pf) else min(pf / 2, 1) * 100
            color = charts.WIN if cls == "win" else charts.LOSS
            rows = [("取引数", f"{r['num_trades']:,} 回", ""),
                    *([("資産の増減", num(r["total_return"], "+.1%"), tone(r["total_return"]))] if period == "PT" else []),
                    ("勝率", f"{r['win_rate']:.1%}", ""),
                    ("1 回あたり平均", num(r["avg_ret_bp"], "+.1f") + " bp", tone(r["avg_ret_bp"])),
                    ("損益合計", num(r["total_pnl"], "+,.0f") + " 円", tone(r["total_pnl"])),
                    ("最大ドローダウン", num(r["max_drawdown"], ".2%"), "")]
            body = (f'<div class="pf"><strong class="{cls}">{"—" if pd.isna(pf) else f"{pf:.2f}"}</strong>'
                    f'<span>プロフィットファクター</span></div>'
                    f'<div class="gauge"><i style="width:{fill:.1f}%;background:{color}"></i><b></b><em>1.0</em></div>'
                    f'<p class="judge {cls}">{text}</p>'
                    + "<dl>" + "".join(f'<dt>{k}</dt><dd class="{c}">{v}</dd>' for k, v, c in rows) + "</dl>")
        cols.append(f'<section class="{period.lower()}"><h3>{title}</h3><p class="span">{span}</p>{body}</section>')
    return f'<div class="score">{"".join(cols)}</div>'


def trade_slip(t: pd.Series, kind: str) -> str:
    """選んだ取引の伝票（HTML）。"""
    reason = charts.REASON_NAMES.get(t["exit_reason"], t["exit_reason"])
    e, x = t["entry_date"], t["exit_date"]
    if kind == "intraday":
        when = f"{e:%Y/%m/%d}（{WEEKDAYS[e.weekday()]}）"
        names = ("空売り", "買い戻し") if t.get("side") == "short" else ("買い", "売り")
        legs = [(names[0], f"{e:%H:%M}", t["entry_price"]), (names[1], f"{x:%H:%M}", t["exit_price"])]
        ret = num(t["ret"] * 1e4, "+.1f") + " bp"
    else:
        when = f"{e:%Y/%m/%d} 〜 {x:%Y/%m/%d}（{(x - e).days} 日）"
        names = ("空売り", "買い戻し") if t.get("side") == "short" else ("買い", "売り")
        legs = [(names[0], f"{e:%m/%d}", t["entry_price"]), (names[1], f"{x:%m/%d}", t["exit_price"])]
        ret = num(t["ret"] * 100, "+.2f") + "%"
    legs_html = "".join(f"<div><span>{k} {at}</span>{p:,.1f}</div>" for k, at, p in legs)
    size = (f'{int(t["shares"]):,} 株' if float(t["shares"]).is_integer()   # 指数の買い増しは金額で建てるので株数は小数
            else f'建てた額 {t["shares"] * t["entry_price"] / 1e4:,.0f} 万円')
    return (f'<div class="slip {tone(t["pnl"])}"><div>'
            f'<span class="code">{html.escape(str(t["code"]))}</span><span class="reason">{reason}</span>'
            f'<span class="when">{when}</span><div class="legs">{legs_html}</div></div>'
            f'<div class="pnl"><strong class="{tone(t["pnl"])}">{num(t["pnl"], "+,.0f")} 円</strong>'
            f'<small>{ret}　{size}</small></div></div>')


def pnl_style(v) -> str:
    if pd.isna(v) or v == 0:
        return ""
    if v > 0:
        return f"color: {charts.WIN}; background-color: rgba(207,58,43,0.07); font-weight: 700"
    return f"color: {charts.LOSS}; background-color: rgba(31,95,174,0.07); font-weight: 700"


@st.cache_data(show_spinner="5 分足を読み込み中...")
def load_intraday_day(code: str, data_dir: str, day: pd.Timestamp) -> pd.DataFrame:
    bars = load_5min(code, data_dir)
    return bars[bars.index.normalize() == day]


@st.cache_data(show_spinner="日足を読み込み中...")
def load_daily_prices(data: dict, code: str) -> pd.DataFrame:
    if code.startswith("^"):  # 日経平均の買い増しの取引は yfinance の指数
        return get_source("yfinance").get_daily(code, "2000-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    if data.get("dir"):  # 多銘柄の実行は手元の日足 CSV を使う
        return load_daily_csv(code, data["dir"])
    return get_source(data["source"]).get_daily(code, data["start"], data["end"])


def chart_bands(run: dict, prices: pd.DataFrame) -> pd.DataFrame | None:
    """取引チャートに重ねるバンド。params から戦略が受け取る引数だけを渡す（資金制約ありの実行は枠・資金なども入っている）。
    併用（legs）は 1 つ目の手法のバンドに、2 つ目以降の手法の下のバンドを lower_2, lower_3 … として足す。"""
    def make(name: str, params: dict) -> pd.DataFrame | None:
        cls = STRATEGIES.get(BAND_STRATEGY.get(name, name))
        if cls is None or not hasattr(cls, "bands"):
            return None
        accepted = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in accepted}).bands(prices)

    params = run["params"]
    if "legs" not in params:
        return make(run["strategy"], params)
    legs = [make("bb_touch_fade", q) for q in params["legs"]]
    if legs[0] is None:
        return None
    return legs[0].assign(**{f"lower_{i}": b["lower"] for i, b in enumerate(legs[1:], 2)})


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


st.html(CSS)

# ---- DB の読み込み（保存先はサイドバーで変えられる） ----
with st.sidebar:
    db_path = st.text_input("DB ファイル", DEFAULT_DB, help="結果を保存した SQLite ファイルの場所")
if not Path(db_path).exists():
    st.info(f"{db_path} がまだありません。`scripts/run_backtest.py` か `scripts/run_intraday.py` を実行すると作られます。")
    st.stop()
store = ResultStore(db_path)
runs = store.runs()
if runs.empty:
    st.info("保存された実行結果がありません。`scripts/run_backtest.py` か `scripts/run_intraday.py` を実行してください。")
    st.stop()
runs["title"] = runs.apply(run_title, axis=1)
runs["label"] = [f"#{i} {t}" for i, t in zip(runs["run_id"], runs["title"])]
heads = runs.drop_duplicates("run_id").set_index("run_id")  # 新しい実行が先頭

# ---- 上部: 画面の切り替え ----
requested = st.query_params.get("run", "")  # URL の ?run=ID で開く実行を指定できる
req_id = int(requested) if requested.isdigit() and int(requested) in heads.index else None
kinds = [k for k in ("intraday", "daily") if (heads["kind"] == k).any()]
start_mode = heads.at[req_id, "kind"] if req_id else heads.iloc[0]["kind"]

topbar = st.container(key="topbar")
top_l, top_c, top_r = topbar.columns([2, 6, 1], vertical_alignment="bottom")
top_l.html('<p class="brand">売買の検証<small>バックテストの結果を確かめる</small></p>')
with top_c.container(key="modebar"):
    mode = st.segmented_control(
        "画面", [*kinds, "compare"], default=start_mode, key="mode", label_visibility="collapsed",
        format_func=lambda m: (f"{MODE_ICONS[m]} {MODES[m]}（{(heads['kind'] == m).sum()}）" if m != "compare"
                               else f"{MODE_ICONS[m]} {MODES[m]}"))
mode = mode or start_mode
with top_r.container(key="glossary").popover("用語", icon=":material/help:", width="stretch"):
    st.markdown(
        "- **プロフィットファクター (PF)**: 勝ちの合計 ÷ 負けの合計。1 を超えれば利益、1.3 以上が一つの目安\n"
        "- **bp**: 0.01%。100 万円の取引で 1bp = 100 円\n"
        "- **ドローダウン**: それまでの最高値からの下落幅\n"
        "- **調整期間（IS）**: パラメータを決めるのに使った期間\n"
        "- **検証期間（OOS）**: 調整に使っていない期間。ここで勝てるかが本番の目安\n"
        "- **決済理由**: シグナル＝戦略の売り条件、目標＝狙った値まで戻った、損切り＝損切り価格に到達、期限＝保有日数の上限、大引け＝その日の終わりに手仕舞い\n"
        "- **色**: 朱＝利益、藍＝損失（日本の証券アプリと同じ）"
    )

# ---- 実行を比べる ----
if mode == "compare":
    st.html('<div class="hero"><h1>実行を比べる</h1><p class="meta">調整期間（IS）で良くても検証期間（OOS）で崩れる場合は、'
            "過去データに合わせすぎている可能性があります。</p></div>")
    runs = runs.sort_values(["run_id", "period"])  # 期間は ALL → IS → OOS の順
    plot(charts.compare_runs(runs))
    table = runs.assign(created_at=runs["created_at"].str.replace("T", " ").str[:16],
                        kind=runs["kind"].map(MODES), strategy=runs["title"],
                        params=runs.apply(specs_text, axis=1), period=runs["period"].map(charts.PERIOD_NAMES))
    st.dataframe(
        table[["run_id", "created_at", "kind", "strategy", "params", "period", "num_trades", "total_pnl", "win_rate",
               "avg_ret_bp", "profit_factor", "max_drawdown", "note"]]
        .style.map(pnl_style, subset=["total_pnl", "avg_ret_bp"])
        .map(lambda v: pnl_style(v - 1) if pd.notna(v) else "", subset=["profit_factor"]),
        hide_index=True, width="stretch",
        column_config={
            "run_id": "ID", "created_at": "実行日時", "kind": "種類", "strategy": "タイトル", "params": "設定",
            "period": "期間", "num_trades": st.column_config.NumberColumn("取引数", format="%d"),
            "total_pnl": st.column_config.NumberColumn("損益(円)", format="%+,.0f"),
            "win_rate": st.column_config.NumberColumn("勝率", format="percent"),
            "avg_ret_bp": st.column_config.NumberColumn("平均(bp)", format="%+.1f"),
            "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("最大DD", format="percent"),
            "note": "メモ",
        },
    )
    st.stop()

# ---- 実行の選択 ----
ids = heads.index[heads["kind"] == mode].tolist()
default_id = req_id if req_id in ids else ids[0]
state_key = f"run_{mode}"
if st.session_state.get(state_key) not in ids:
    st.session_state[state_key] = default_id
run_id = st.session_state[state_key]
if len(ids) > 1:
    oos_pf = runs[runs["period"] == "OOS"].set_index("run_id")["profit_factor"]
    pt = runs[runs["period"] == "PT"].set_index("run_id")

    def pick(i: int) -> None:
        st.session_state[state_key] = i

    def differs(i: int) -> str:
        """同じ戦略のほかの実行と値が違うパラメータ（カードで見分けるため）。"""
        same = [json.loads(p) for j, p in heads.loc[ids, "params"].items() if heads.at[j, "strategy"] == heads.at[i, "strategy"]]
        mine = json.loads(heads.at[i, "params"])
        keys = [k for k, v in mine.items() if any(o.get(k) != v for o in same)]
        keys.sort(key=lambda k: sum(o.get(k) == mine[k] for o in same))  # 珍しい値ほど先に
        return "　".join(f"{k}={mine[k]}" for k in keys[:3]).replace("_", r"\_") or "　"

    with st.container(horizontal=True, key="runcards"):
        for i in ids:
            pf = oos_pf.get(i)
            if pf is None or pd.isna(pf):
                mark, score = "none", "検証期間なし"
            else:
                mark, score = ("win" if pf >= 1 else "loss"), f"検証 PF *{pf:.2f}*"
            if i in pt.index:   # ペーパートレードは資産の増減で示す（取引が少ないうちは PF が決まらない）
                score += f"　PT *{pt.at[i, 'total_return']:+.1%}*"
            when = heads.at[i, "created_at"][5:16].replace("T", " ").replace("-", "/")
            # タイトルを太字に、その下に資金とデータの範囲を 1 行ずつ（全部の設定はマウスを乗せると出る）
            specs = run_specs(heads.loc[i])
            detail = "\n\n".join(v for k, v in specs.items() if k in ("資金", "データ")) or differs(i)
            st.button(f"#{i}　{when}\n\n**{heads.at[i, 'title']}**\n\n{detail}\n\n{score}",
                      key=f"rc_{i}_{mark}{'_sel' if i == run_id else ''}", on_click=pick, args=(i,),
                      help=specs_text(heads.loc[i], "  \n"))
st.query_params["run"] = str(run_id)

run = store.run(run_id)
results = runs[runs["run_id"] == run_id].set_index("period")
trades = store.trades(run_id)
equity = store.equity(run_id)
kind = run["kind"]

# ---- 見出しと成績表 ----
note = f'<p class="note">📝 {html.escape(run["note"])}</p>' if run["note"] else ""
if run_specs(run):   # 細かい設定はタイトルの下に項目ごとに並べる（パラメータそのものは「設定・メモ」のタブ）
    rule = {k: v for k, v in run_specs(run).items() if k.startswith(RULE_SPECS)}
    rest = {k: v for k, v in run_specs(run).items() if k not in rule}
    specs = "".join("<dl>" + "".join(f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>"
                                     for k, v in group.items()) + "</dl>" for group in (rule, rest) if group)
    specs = f'<div class="specs">{specs}</div>'
else:
    specs = "".join(f'<li><span>{html.escape(str(k))}</span>{html.escape(str(v))}</li>' for k, v in run["params"].items())
    specs = f'<ul class="chips">{specs}</ul>'
st.html(f'<div class="hero"><h1>{html.escape(run_title(run))}</h1>{specs}'
        f'<p class="meta">実行 ID <b>#{run_id}</b>実行日時 <b>{run["created_at"][:16].replace("T", " ")}</b></p>'
        f'{note}</div>')
paper = run["config"].get("paper")
st.html(scorecard(results, equity, paper))

periods = [p for p in charts.PERIODS if p in results.index]
period = "すべて"
if len(periods) > 1:
    period = st.segmented_control(
        "表示する期間", ["すべて", *periods], default="すべて", key=f"period_{run_id}",
        format_func=lambda p: PERIOD_TITLES[p][0] if p in PERIOD_TITLES else p) or "すべて"
view = trades if period == "すべて" else trades[trades["period"] == period]

tab_names = [f":material/receipt_long: 取引一覧（{len(view):,} 件）", ":material/show_chart: 損益の推移",
             ":material/apartment: 銘柄別",
             ":material/date_range: " + ("時間帯・曜日" if kind == "intraday" else "曜日"), ":material/tune: 設定・メモ"]
tab_trades, tab_overview, tab_stock, tab_time, tab_config = st.tabs(tab_names)

# ---- 取引一覧: 左に一覧、右に選んだ取引のチャート ----
with tab_trades:
    if view.empty:
        st.info("この期間には取引がありません。")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            f1, f2 = st.columns([3, 2], vertical_alignment="bottom")
            outcome = f1.segmented_control("勝ち負け", ["すべて", "勝ち", "負け"], default="すべて",
                                           format_func=lambda o: {"勝ち": ":material/trending_up: 勝ち", "負け": ":material/trending_down: 負け"}.get(o, o),
                                           key=f"outcome_{run_id}") or "すべて"
            code = f2.text_input("銘柄コード", placeholder="例: 7203", icon=":material/search:", key=f"code_{run_id}").strip().upper()
            all_reasons = sorted(view["exit_reason"].unique())
            reasons = all_reasons
            if len(all_reasons) > 1:
                st.html("<style>" + "".join(
                    f'[class*="st-key-reason_"] button:nth-child({n + 1}) {{ --dot: '
                    f'{charts.REASON_COLORS.get(charts.REASON_NAMES.get(r, r), charts.MUTED)}; }}'
                    for n, r in enumerate(all_reasons)) + "</style>")
                reasons = st.pills("決済理由", all_reasons, selection_mode="multi", default=all_reasons,
                                   key=f"reason_{run_id}", format_func=lambda r: charts.REASON_NAMES.get(r, r))
            t = view[view["exit_reason"].isin(reasons)]
            if outcome != "すべて":
                t = t[t["pnl"] > 0] if outcome == "勝ち" else t[t["pnl"] <= 0]
            if code:
                t = t[t["code"].str.upper().str.contains(code, regex=False)]
            t = t.sort_values("entry_date", ascending=False).reset_index(drop=True)

            total = t["pnl"].sum()
            st.html(f'<div class="tally"><span>表示<b>{len(t):,} 件</b></span>'
                    f'<span>損益<b class="{tone(total)}">{num(total, "+,.0f")} 円</b></span>'
                    f'<span>勝率<b>{(t["pnl"] > 0).mean():.1%}</b></span></div>' if len(t) else
                    '<div class="tally"><span>条件に合う取引がありません。絞り込みを変えてください。</span></div>')

            if kind == "intraday":
                shown = pd.DataFrame({
                    "日付": [f"{d:%m/%d}（{WEEKDAYS[d.weekday()]}）" for d in t["entry_date"]],
                    "銘柄": t["code"], "向き": t["side"].map({"long": "買い", "short": "空売り"}),
                    "建て": t["entry_date"].dt.strftime("%H:%M"),
                    "手仕舞い": t["exit_date"].dt.strftime("%H:%M"), "損益": t["pnl"], "リターン": t["ret"] * 1e4})
                ret_cfg = st.column_config.NumberColumn("bp", format="%+.1f", help="1 回のリターン（0.01% 単位）")
            else:
                shown = pd.DataFrame({
                    "銘柄": t["code"], "向き": t["side"].map({"long": "買い", "short": "空売り"}),
                    "建て": t["entry_date"].dt.strftime("%y/%m/%d"),
                    "手仕舞い": t["exit_date"].dt.strftime("%y/%m/%d"),
                    "日数": (t["exit_date"] - t["entry_date"]).dt.days, "損益": t["pnl"], "リターン": t["ret"] * 100})
                ret_cfg = st.column_config.NumberColumn("%", format="%+.2f", help="1 回のリターン（%）")
            shown["決済"] = t["exit_reason"].map(lambda r: charts.REASON_NAMES.get(r, r))
            if period == "すべて" and len(periods) > 1:
                shown.insert(0, "期間", t["period"].map({"IS": "調整", "OOS": "検証", "PT": "PT"}))
            event = st.dataframe(
                shown.style.map(pnl_style, subset=["損益", "リターン"]),
                hide_index=True, width="stretch", height=560, on_select="rerun", selection_mode="single-row",
                key=f"table_{run_id}_{period}",
                column_config={"損益": st.column_config.NumberColumn("損益(円)", format="%+,.0f"),
                               "リターン": ret_cfg, "日数": st.column_config.NumberColumn(format="%d")})
            st.download_button("表示中の取引を CSV で保存", t.drop(columns=["id", "run_id"]).to_csv(index=False)
                               .encode("utf-8-sig"), file_name=f"run{run_id}_trades.csv", mime="text/csv",
                               icon=":material/download:")

        with right:
            rows = event.selection.rows if len(t) else []
            if not len(t):
                st.caption("一覧で取引を選ぶと、ここにチャートが表示されます。")
            else:
                selected = t.iloc[rows[0]] if rows else t.iloc[0]
                st.html(trade_slip(selected, kind))
                st.caption("一覧の行をクリックすると、その取引に切り替わります。▲ 買い　▽ 売り"
                           + ("　線は VWAP" if kind == "intraday" else ""))
                try:
                    if kind == "intraday":
                        bars = load_intraday_day(selected["code"], run["config"]["data"]["dir"],
                                                 selected["entry_date"].normalize())
                        plot(charts.intraday_trade(bars, session_vwap(bars), selected))
                    else:
                        cfg = run["config"]   # 併用の実行は手法ごとの設定（legs）の中に data がある
                        prices = load_daily_prices(cfg.get("data") or cfg["legs"][0]["data"], selected["code"])
                        sma = tuple(v for k, v in run["params"].items() if k in ("fast", "slow"))
                        plot(charts.daily_trades(prices, trades[trades["code"] == selected["code"]], selected, sma,
                                                 chart_bands(run, prices)))
                except Exception as e:  # noqa: BLE001  元データが見つからなくても画面は表示し続ける
                    st.warning(f"チャート用の価格データを読み込めませんでした: {e}")

with tab_overview:
    if trades.empty:
        st.info("取引がないため、グラフはありません。")
    else:
        # 資金制約ありの実行は params の capital が元手（config の initial_cash は 1 銘柄分のことがある）
        initial_cash = run["params"].get("capital") or run["config"].get("backtest", {}).get("initial_cash", 0)
        plot(charts.overview(charts.pnl_curve(kind, initial_cash, trades, equity), trades, paper and paper["live_from"]))
        if not view.empty:
            c1, c2 = st.columns(2)
            with c1:
                plot(charts.return_histogram(view))
            with c2:
                plot(charts.exit_reasons(view))
            plot(charts.monthly_pnl(view))

with tab_stock:
    if view.empty:
        st.info("この期間には取引がありません。")
    else:
        by_stock = (view.groupby("code")
                    .agg(取引数=("pnl", "size"), 損益=("pnl", "sum"), 勝率=("pnl", lambda x: (x > 0).mean()),
                         平均bp=("ret", lambda x: x.mean() * 1e4))
                    .reset_index().sort_values("損益", ascending=False))
        if len(by_stock) > 1:
            plot(charts.stock_ranking(by_stock, n=min(15, len(by_stock))))
            plot(charts.stock_scatter(by_stock))
        st.dataframe(by_stock.rename(columns={"code": "銘柄"}).style.map(pnl_style, subset=["損益", "平均bp"]),
                     hide_index=True, width="stretch",
                     column_config={"損益": st.column_config.NumberColumn(format="%+,.0f"),
                                    "勝率": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                    "平均bp": st.column_config.NumberColumn(format="%+.1f")})

with tab_time:
    if view.empty:
        st.info("この期間には取引がありません。")
    else:
        if kind == "intraday":
            plot(charts.by_entry_time(view))
        plot(charts.by_weekday(view))

with tab_config:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**パラメータ**")
        st.json(run["params"])
        st.markdown("**タイトル**（この実行で重視したことが分かる短い名前）")
        name = st.text_input("タイトル", run_title(run), label_visibility="collapsed")
        st.markdown("**メモ**")
        note = st.text_area("メモ", run["note"], label_visibility="collapsed")
        if st.button("タイトルとメモを保存", icon=":material/save:"):
            store.update_name(run_id, name)
            store.update_note(run_id, note)
            st.toast("タイトルとメモを保存しました")
            st.rerun()
    with c2:
        grid = run["config"].pop("grid_results", None)
        universe = run["config"].pop("universe_codes", None)
        if grid:
            st.markdown("**調整期間でのパラメータ探索**（損益合計が最大のものを採用）")
            g = pd.DataFrame([{**x["params"], **{k: v for k, v in x.items() if k != "params"}} for x in grid])
            st.dataframe(g.style.map(pnl_style, subset=["total_pnl", "avg_ret_bp"]), hide_index=True, width="stretch",
                         column_config={"total_pnl": st.column_config.NumberColumn("損益(円)", format="%+,.0f"),
                                        "avg_ret_bp": st.column_config.NumberColumn("平均(bp)", format="%+.1f"),
                                        "num_trades": st.column_config.NumberColumn("取引数", format="%d")})
        if universe:
            with st.expander(f"対象銘柄（{len(universe)} 銘柄）"):
                st.write(", ".join(universe))
        with st.expander("実行時の設定"):
            st.code(json.dumps(run["config"], ensure_ascii=False, indent=2), language="json")
    with st.expander("この実行を削除", icon=":material/delete:"):
        if st.checkbox("削除すると元に戻せないことを確認しました") and st.button("削除する", type="primary"):
            store.delete_run(run_id)
            st.query_params.clear()
            st.rerun()
