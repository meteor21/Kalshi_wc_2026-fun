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
