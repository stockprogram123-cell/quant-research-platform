"""確認画面に出す実行の名前（短いタイトル）と細かい設定の一覧を、パラメータから決まった形で作る。

タイトル: 「手法：重視したこと」。例: −2σ押し目買い：寄りの約定を予想順に建てる
設定の一覧: {項目: 値}。例: {"手法": "買い −2σ割れ（寄り＋場中）", "絞り込み": "押し目（…）", "手仕舞い": …, "資金": …, "データ": …}
重視したこと（ctx["focus"]）はその実行で確かめたいことで、保存するスクリプトが渡す。渡さなければ運用の仕方から 1 つ選ぶ。
ctx にはほかに params に入っていない情報を渡す（universe: データの範囲、regime_mode: 5 分足の局面の決め方）。
"""

IDLE = {"n225": "日経平均", "market": "市場平均"}
PULLBACK = ("long_min_high250", "long_max_ret5", "long_min_sma200")


def _pct(x: float) -> str:
    return f"{x * 100:.4g}%"


# ---- 日足の ±kσ 逆張り（bb_touch_fade） ----
def bb_signal(p: dict) -> str:
    k, sides = p.get("k", 3.0), p.get("sides", "both")
    legs = []
    if sides in ("long", "both"):
        legs.append(f"買い −{k:g}σ割れ")
    if sides in ("short", "both"):
        legs.append(f"空売り +{k:g}σ超え")
    return "・".join(legs) + ("（寄りだけ）" if p.get("gap_only") else "（寄り＋場中）")


def bb_filters(p: dict) -> list[str]:
    out = []
    if p.get("long_min_p_down") is not None:
        out.append("下げ局面" if p["long_min_p_down"] == 0.5 else f"下げ確率>{p['long_min_p_down']:g}")
    if p.get("short_max_p_down") is not None:
        out.append("上げ局面" if p["short_max_p_down"] == 0.5 else f"下げ確率≦{p['short_max_p_down']:g}")
    r2 = lambda x: f"{x * 100:.2g}%"  # noqa: E731  有効数字 2 桁
    rule = [f"1年高値から{r2(p['long_min_high250'])}以内" if p.get("long_min_high250") is not None else "",
            f"5日で{r2(p['long_max_ret5'])}以下" if p.get("long_max_ret5") is not None else "",
            f"200日線{r2(p['long_min_sma200'])}以上" if p.get("long_min_sma200") is not None else ""]
    if p.get("long_cooldown"):
        rule.append(f"前の割れから{p['long_cooldown']}日空ける")
    if any(rule):
        out.append("押し目（" + "・".join(r for r in rule if r) + "）")
    return out


def bb_exit(p: dict) -> str:
    ek, stop, hold = p.get("exit_k"), p.get("stop_pct"), p.get("max_hold")
    sign = {"long": "−", "short": "+"}.get(p.get("sides", "both"), "±")
    target = "利確なし" if ek is None else "利確25日線" if ek == 0 else f"利確{sign}{ek:g}σ"
    days = {0: "当日大引け", 1: "翌日まで"}.get(hold, f"最長{hold}日")
    return f"{target}・損切{_pct(stop)}・{days}" if stop is not None and stop < 0.9 else f"{target}・{days}"


def bb_core(p: dict) -> str:
    """タイトル用の短い手法名（−3σ割れ / −2σ押し目 / +3σ超え）。"""
    k, sides = p.get("k", 3.0), p.get("sides", "both")
    pullback = any(p.get(c) is not None for c in PULLBACK)
    return {"long": f"−{k:g}σ" + ("押し目" if pullback else "割れ"), "short": f"+{k:g}σ超え"}.get(sides, f"±{k:g}σ")


def bb_method(p: dict) -> str:
    sides = p.get("sides", "both")
    regime = ("下げ局面の" if sides == "long" and p.get("long_min_p_down") is not None else
              "上げ局面の" if sides == "short" and p.get("short_max_p_down") is not None else "")
    verb = {"long": "買い", "short": "の空売り"}.get(sides, "逆張り")
    return regime + bb_core(p) + verb + ("（寄りだけ）" if p.get("gap_only") else "")


# ---- 資金の使い方（simulate_portfolio / simulate_combo / study_ml_rank） ----
def operation(p: dict) -> dict[str, str]:
    if "slots" not in p:
        return {"資金": "銘柄ごとに同じ資金で検証して合算"}
    out = {"資金": f"{p.get('capital', 10_000_000) / 1e4:,.0f}万円を{p['slots']}枠に等分"}
    if p.get("max_per_day") or p.get("day_fraction"):
        out["1日の上限"] = (f"{p['max_per_day']}銘柄まで" if p.get("max_per_day") else
                         f"その日の初めの空き枠の{p['day_fraction']:.0%}まで")
    if p.get("wait_days"):
        out["見送る日"] = f"下げ局面の{p['wait_days']}日目まで"
    if p.get("rank_by"):
        out["注文"] = f"{p['rank_by']}の上位、空き枠×{p.get('orders_per_slot'):g}銘柄にだけ出す"
    if p.get("priority"):
        out["建てる順"] = p["priority"]
    if p.get("idle_in"):
        cost = f"（乗り換えは片道{_pct(p['switch_cost'])}）" if p.get("switch_cost") else ""
        out["待機中の資金"] = IDLE[p["idle_in"]] + cost
    return out


def operation_focus(p: dict) -> str:
    """重視したことが渡されなかったときに、運用の仕方からタイトルに出す 1 つを選ぶ。"""
    if p.get("rank_by"):
        return f"注文を{p['rank_by']}の上位に絞る"
    if p.get("max_per_day"):
        return f"1日{p['max_per_day']}銘柄まで"
    if p.get("day_fraction"):
        return f"1日は空き枠の{p['day_fraction']:.0%}まで"
    if p.get("wait_days"):
        n = p["wait_days"]
        return f"{p['slots']}枠・局面の" + ("初日は見送る" if n == 1 else f"{n}日目まで見送る")
    if "slots" in p:
        return f"{p['slots']}枠" + (f"・待機中は{IDLE[p['idle_in']]}" if p.get("idle_in") else "で運用")
    return "1銘柄ずつ検証"


# ---- 5 分足の寄りの窓（opening_gap） ----
def _gap_side(p: dict) -> str:
    return "空売り" if p.get("side", "short") == "short" else "買い"


def gap_regime(p: dict, ctx: dict) -> str:
    if not p.get("regime"):
        return ""
    how = {"hindsight": "後から見た判定・参考", "confirmed": "前日までで確定"}.get(ctx.get("regime_mode", ""), "")
    return {"down": "下げ相場の日", "up": "上げ相場の日"}.get(p["regime"], p["regime"]) + (f"（{how}）" if how else "")


def gap_details(p: dict, ctx: dict) -> dict[str, str]:
    side = _gap_side(p)
    out = {"手法": f"{side} 寄りの窓{'下げ' if side == '空売り' else '上げ'}{_pct(p.get('min_gap', 0))}以上",
           "局面": gap_regime(p, ctx) or "絞らない"}
    if p.get("stop_pct"):
        out["手仕舞い"] = f"損切{_pct(p['stop_pct'])}・{p.get('exit_time', '大引け')}手仕舞い"
    return {**out, "資金": "銘柄ごとに同じ資金で検証して合算"}


# ---- 日経平均の買い増し（index_overlay。scripts/study_index_crash.py） ----
def index_method(p: dict) -> str:
    sig = p.get("signal", "A")
    if sig == "C":
        return f"{p.get('min_stocks')}銘柄以上が−3σ割れの日に日経平均を買い増し"
    return f"日経平均の{'終値' if sig == 'B' else ''}−{p.get('k', 3.0):g}σ割れで買い増し"


def index_details(p: dict) -> dict[str, str]:
    sig, k, hold = p.get("signal", "A"), p.get("k", 3.0), p.get("max_hold")
    how = {"A": f"日経平均（25日線）の−{k:g}σに場中で触れたら買い増す（寄りで下なら寄り）",
           "B": f"日経平均の終値が−{k:g}σを割ったら翌日の寄りで買い増す",
           "C": f"−3σに触れた貸借銘柄（売買代金1億円以上）が{p.get('min_stocks')}銘柄以上の日の"
                f"{'翌日の寄り' if p.get('entry') == 'next_open' else '大引け'}で買い増す"}[sig]
    out = {"手法": how, "絞り込み": "下げ局面" if p.get("long_min_p_down") is not None else "",
           "手仕舞い": bb_exit(p) if sig == "A" else f"終値が−1σに戻った日の大引け・最長{hold}日",
           "資金": f"{p.get('capital', 10_000_000) / 1e4:,.0f}万円をいつも日経平均で持ち、合図の間は{1 + p.get('extra', 1.0):g}倍"}
    out["費用"] = f"買い増し分は信用の金利 年{p.get('rate', 0) * 100:.3g}%・売り買い片道{_pct(p.get('switch_cost', 0))}"
    return out


# ---- 大引けで判断して買う -kσ 割れ（bb_close。scripts/study_bb2_close.py） ----
def close_method(p: dict) -> str:
    return f"終値−{p.get('k', 2.0):g}σ割れを{p.get('entry', '引け')}で買い" + ("（押し目）" if p.get("rules") else "")


def open_method(p: dict) -> str:
    return f"寄りで−{p.get('k', 2.0):g}σ割れを買い" + ("（押し目）" if p.get("rules") else "")


def close_details(p: dict, strategy: str = "bb_close") -> dict[str, str]:
    first = "（最初の割れだけ）" if p.get("first_break") else ""
    k = p.get("k", 2.0)
    how = (f"寄りで−{k:g}σの値段より下で始まった日{first}に寄りで買う（8:59 の気配で判断）" if strategy == "bb_open" else
           f"終値が−{k:g}σ以下で引けた日{first}に{p.get('entry', '引け')}の成行で買う")
    out = {"手法": how, "絞り込み": p.get("rules_text", ""), "手仕舞い": p.get("exit", "")}
    out.update(operation({k: v for k, v in p.items() if k != "priority"}))
    if p.get("pick"):
        out["建てる順"] = p["pick"]
    return out


# ---- 少額の口座で 2 つの手法を一緒に運用（small_account。scripts/simulate_small_account.py） ----
def account_details(p: dict) -> dict[str, str]:
    close_p, index_p = p["parts"]
    c, i = close_details(close_p), index_details(index_p)
    w = p.get("index_weight", 1.0)
    if w < 1:
        money = (f"{p['capital'] / 1e4:,.0f}万円のうち個別株に入っていない資金の{w:.0%}を日経平均、残りを現金で持ち、"
                 f"個別株は{p['slots']}枠に等分して乗り換える（{p.get('lot', '100株')}単位）。"
                 f"手法2の合図の間は資産の{index_p.get('extra', 1.0) * w:.0%}分の日経平均を、まず現金で買い増す")
    else:
        money = (f"{p['capital'] / 1e4:,.0f}万円をいつも日経平均で持ち、個別株は{p['slots']}枠に等分して乗り換える"
                 f"（{p.get('lot', '100株')}単位）。手法2の合図の間は日経平均を{1 + index_p.get('extra', 1.0):g}倍")
    return {"手法1": c["手法"], "絞り込み1": c["絞り込み"], "手仕舞い1": c["手仕舞い"], "建てる順": c.get("建てる順", ""),
            "手法2": i["手法"], "絞り込み2": i["絞り込み"], "手仕舞い2": i["手仕舞い"], "資金": money, "費用": i["費用"]}


# ---- まとめ ----
def method_name(strategy: str, params: dict) -> str:
    """タイトルの前半（何をする手法か）。"""
    if "legs" in params:   # 併用: legs は各手法のパラメータ
        return "＋".join(bb_core(q) for q in params["legs"]) + "の併用"
    if strategy == "bb_touch_fade":
        return bb_method(params)
    if strategy == "opening_gap":
        side = _gap_side(params)
        return f"寄りの窓{'下げ' if side == '空売り' else '上げ'}を{side}"
    if strategy == "index_overlay":
        return index_method(params)
    if strategy == "bb_close":
        return close_method(params)
    if strategy == "bb_open":
        return open_method(params)
    if strategy == "small_account":
        return f"−{params.get('k', 2.0):g}σ割れの翌朝買い＋日経平均の買い増し"
    return strategy


def run_name(kind: str, strategy: str, params: dict, ctx: dict | None = None) -> str:
    """短いタイトル「手法：重視したこと」。"""
    ctx = ctx or {}
    focus = ctx.get("focus")
    if not focus:
        if strategy == "opening_gap":
            focus = (gap_regime(params, ctx).replace("の日", "の日だけ").replace("・参考", "")
                     if params.get("regime") else "局面で絞らない")
        elif strategy == "index_overlay":
            focus = ("下げ局面だけ" if params.get("long_min_p_down") is not None else "局面で絞らない") + \
                f"・最長{params.get('max_hold')}日"
        elif strategy == "small_account":
            index_p = params["parts"][1]
            focus = f"{params['capital'] / 1e4:,.0f}万円・{params['slots']}枠" + \
                (f"・日経平均は{params['index_weight']:.0%}" if params.get("index_weight", 1.0) < 1 else "") + \
                (f"・買い増しは日経平均−{index_p['k']:g}σ" if index_p.get("signal") == "B" else "")
        elif strategy in ("bb_close", "bb_open") and params.get("pick") and params["pick"] != "乱数の順":
            focus = f"{params['pick'].split('（')[0]}に{params.get('slots')}枠"
        else:
            focus = operation_focus(params)
    return f"{method_name(strategy, params)}：{focus}"


def run_details(kind: str, strategy: str, params: dict, ctx: dict | None = None) -> dict[str, str]:
    """タイトルの下に並べる細かい設定（項目 → 値）。"""
    ctx, p = ctx or {}, params
    if "legs" in p:
        out = {}
        for i, q in enumerate(p["legs"], 1):
            out[f"手法{i}"] = bb_signal(q)
            if bb_filters(q):
                out[f"絞り込み{i}"] = "・".join(bb_filters(q))
        out["手仕舞い"] = " / ".join(dict.fromkeys(bb_exit(q) for q in p["legs"]))   # 同じ手仕舞いは 1 回だけ
        out.update(operation(p))
    elif strategy == "bb_touch_fade":
        out = {"手法": bb_signal(p), "絞り込み": "・".join(bb_filters(p)), "手仕舞い": bb_exit(p), **operation(p)}
    elif strategy == "opening_gap":
        out = gap_details(p, ctx)
    elif strategy == "index_overlay":
        out = index_details(p)
    elif strategy in ("bb_close", "bb_open"):
        out = close_details(p, strategy)
    elif strategy == "small_account":
        out = account_details(p)
    else:
        out = {"手法": strategy, "パラメータ": ", ".join(f"{k}={v}" for k, v in p.items() if v is not None)}
    if ctx.get("universe"):
        out["データ"] = ctx["universe"]
    return {k: v for k, v in out.items() if v}


def describe_run(kind: str, strategy: str, params: dict, ctx: dict | None = None) -> dict:
    """ResultStore.save_run にそのまま渡せる名前と設定の一覧（save_run(..., **describe_run(...))）。"""
    return {"name": run_name(kind, strategy, params, ctx), "details": run_details(kind, strategy, params, ctx)}


def universe_label(cfg: dict, n_stocks: int) -> str:
    """データの範囲（貸借銘柄に絞ったか・銘柄数）。"""
    return f"{'貸借' if cfg.get('universe', {}).get('margin_only') else ''}{n_stocks:,}銘柄"
