import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_DB = "data/trading.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- daily | intraday
    strategy    TEXT NOT NULL,
    name        TEXT NOT NULL DEFAULT '',  -- 画面に出す短いタイトル（trading.store.naming.run_name）
    details     TEXT NOT NULL DEFAULT '{}',  -- JSON。タイトルの下に並べる細かい設定（項目 → 値、run_details）
    params      TEXT NOT NULL,          -- JSON
    config      TEXT NOT NULL,          -- JSON（実行時の設定ファイルの中身）
    note        TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS results (
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    period        TEXT NOT NULL,        -- ALL | IS | OOS | PT（ペーパートレード）
    num_trades    INTEGER,
    total_pnl     REAL,
    win_rate      REAL,
    avg_ret_bp    REAL,
    profit_factor REAL,
    total_return  REAL,
    sharpe        REAL,
    max_drawdown  REAL,
    exit_reasons  TEXT,                 -- JSON
    PRIMARY KEY (run_id, period)
);
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    period      TEXT NOT NULL,
    code        TEXT NOT NULL,
    entry_date  TEXT NOT NULL,
    exit_date   TEXT NOT NULL,
    shares      INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price  REAL NOT NULL,
    pnl         REAL NOT NULL,          -- 手数料込み（円）
    ret         REAL NOT NULL,          -- 手数料込みリターン
    exit_reason TEXT NOT NULL DEFAULT '',
    side        TEXT NOT NULL DEFAULT 'long'  -- long（買い） | short（空売り）
);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id, period);
CREATE TABLE IF NOT EXISTS equity (
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    period  TEXT NOT NULL,
    date    TEXT NOT NULL,
    equity  REAL NOT NULL,
    PRIMARY KEY (run_id, period, date)
);
"""

RESULT_COLUMNS = ["num_trades", "total_pnl", "win_rate", "avg_ret_bp", "profit_factor",
                  "total_return", "sharpe", "max_drawdown"]
TRADE_COLUMNS = ["code", "entry_date", "exit_date", "shares", "entry_price", "exit_price",
                 "pnl", "ret", "exit_reason", "side"]


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def _clean(value):
    """SQLite に入れられる値にする（NaN → NULL、numpy の数値 → Python の数値）。"""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class ResultStore:
    """バックテストの実行結果（要約・売買履歴・資産推移）を SQLite に保存する。"""

    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # 空売りに対応する前に作った DB には side 列がない
            if "side" not in {r[1] for r in conn.execute("PRAGMA table_info(trades)")}:
                conn.execute("ALTER TABLE trades ADD COLUMN side TEXT NOT NULL DEFAULT 'long'")
            if "name" not in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}:
                conn.execute("ALTER TABLE runs ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            if "details" not in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}:
                conn.execute("ALTER TABLE runs ADD COLUMN details TEXT NOT NULL DEFAULT '{}'")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_run(self, kind: str, strategy: str, params: dict, config: dict,
                 periods: dict[str, tuple[dict, pd.DataFrame, pd.Series]], note: str = "", name: str = "",
                 details: dict[str, str] | None = None) -> int:
        """1 回の実行を保存して run_id を返す。name は画面に出す短いタイトル、details はその下に並べる設定（項目 → 値）。
        どちらも trading.store.naming.describe_run で作れる（save_run(..., **describe_run(...))）。

        periods は {期間名: (summarize の結果, round_trip_table の取引一覧, 日次資産)}。
        """
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (created_at, kind, strategy, name, details, params, config, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), kind, strategy, name, _json(details or {}),
                 _json(params), _json(config), note),
            )
            run_id = cur.lastrowid
            self._insert_periods(conn, run_id, periods)
        return run_id

    def replace_periods(self, run_id: int, periods: dict[str, tuple[dict, pd.DataFrame, pd.Series]]) -> None:
        """run_id の periods にある期間の要約・取引・資産を入れ替える（ペーパートレードで毎日記録を更新するため）。"""
        with self._connect() as conn:
            for table in ("results", "trades", "equity"):
                conn.executemany(f"DELETE FROM {table} WHERE run_id = ? AND period = ?", [(run_id, p) for p in periods])
            self._insert_periods(conn, run_id, periods)

    @staticmethod
    def _insert_periods(conn: sqlite3.Connection, run_id: int,
                        periods: dict[str, tuple[dict, pd.DataFrame, pd.Series]]) -> None:
        for period, (summary, trades, equity) in periods.items():
            conn.execute(
                f"INSERT INTO results (run_id, period, {', '.join(RESULT_COLUMNS)}, exit_reasons) "
                f"VALUES (?, ?, {', '.join('?' * len(RESULT_COLUMNS))}, ?)",
                (run_id, period, *[_clean(summary.get(c)) for c in RESULT_COLUMNS],
                 _json(summary.get("exit_reasons", {}))),
            )
            if not trades.empty:
                rows = trades[TRADE_COLUMNS].copy()
                for col in ("entry_date", "exit_date"):
                    rows[col] = pd.to_datetime(rows[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
                conn.executemany(
                    f"INSERT INTO trades (run_id, period, {', '.join(TRADE_COLUMNS)}) "
                    f"VALUES (?, ?, {', '.join('?' * len(TRADE_COLUMNS))})",
                    [(run_id, period, *map(_clean, r)) for r in rows.itertuples(index=False)],
                )
            conn.executemany(
                "INSERT INTO equity (run_id, period, date, equity) VALUES (?, ?, ?, ?)",
                [(run_id, period, pd.Timestamp(d).strftime("%Y-%m-%d"), float(v))
                 for d, v in equity.items()],
            )

    def runs(self) -> pd.DataFrame:
        """実行一覧（期間ごとの要約付き、新しい順）。"""
        with self._connect() as conn:
            return pd.read_sql(
                "SELECT r.id AS run_id, r.created_at, r.kind, r.strategy, r.name, r.details, r.params, r.note, res.* "
                "FROM runs r LEFT JOIN results res ON res.run_id = r.id "
                "ORDER BY r.id DESC, res.period",
                conn,
            ).loc[:, lambda df: ~df.columns.duplicated()]

    def run(self, run_id: int) -> dict:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run_id={run_id} は存在しません")
        return {**dict(row), "details": json.loads(row["details"]), "params": json.loads(row["params"]),
                "config": json.loads(row["config"])}

    def trades(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM trades WHERE run_id = ? ORDER BY entry_date, code",
                               conn, params=(run_id,), parse_dates=["entry_date", "exit_date"])

    def equity(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT period, date, equity FROM equity WHERE run_id = ? ORDER BY date",
                               conn, params=(run_id,), parse_dates=["date"])

    def update_note(self, run_id: int, note: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE runs SET note = ? WHERE id = ?", (note, run_id))

    def update_config(self, run_id: int, config: dict) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE runs SET config = ? WHERE id = ?", (_json(config), run_id))

    def update_name(self, run_id: int, name: str, details: dict[str, str] | None = None) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE runs SET name = ? WHERE id = ?", (name, run_id))
            if details is not None:
                conn.execute("UPDATE runs SET details = ? WHERE id = ?", (_json(details), run_id))

    def delete_run(self, run_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
