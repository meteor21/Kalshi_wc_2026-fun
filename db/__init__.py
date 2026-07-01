"""SQLite helpers. Single writer assumption."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "state.db"
SCHEMA = Path(__file__).parent / "schema.sql"


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init():
    with conn() as c:
        c.executescript(SCHEMA.read_text())


def log_prediction(c, model, ticker, side, prob, price, edge_val):
    cur = c.execute(
        "INSERT INTO predictions(model,ticker,side,model_prob,market_price,edge) VALUES(?,?,?,?,?,?)",
        (model, ticker, side, prob, price, edge_val))
    return cur.lastrowid


def log_order(c, pred_id, ticker, side, action, count, price_cents, kalshi_id=None, is_parlay=0):
    cur = c.execute(
        "INSERT INTO orders(prediction_id,kalshi_order_id,ticker,side,action,count,price_cents,is_parlay) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (pred_id, kalshi_id, ticker, side, action, count, price_cents, is_parlay))
    return cur.lastrowid


def snapshot_price(c, ticker, yes_bid, yes_ask, volume):
    c.execute("INSERT INTO price_log(ticker,yes_bid,yes_ask,volume) VALUES(?,?,?,?)",
              (ticker, yes_bid, yes_ask, volume))


def log_trigger_event(c, fixture_id, trigger_type, state_json, quotes_json=None):
    cur = c.execute(
        "INSERT INTO trigger_events(fixture_id,trigger_type,state_json,quotes_json) VALUES(?,?,?,?)",
        (fixture_id, trigger_type, state_json, quotes_json))
    return cur.lastrowid


def log_settlement(c, ticker, result, pnl):
    c.execute("INSERT INTO settlements(ticker,result,pnl) VALUES(?,?,?)", (ticker, result, pnl))


def record_equity(c, balance, open_exposure):
    c.execute("INSERT OR REPLACE INTO equity(ts,balance,open_exposure) VALUES(datetime('now'),?,?)",
              (balance, open_exposure))


def daily_realized_pnl(c):
    """Sum of today's settled P&L (UTC). Drives the daily-loss halt in run_cycle."""
    row = c.execute(
        "SELECT COALESCE(SUM(pnl),0.0) FROM settlements WHERE date(ts)=date('now')").fetchone()
    return float(row[0])
