-- kalshi-quant operational state
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  model TEXT NOT NULL,            -- soccer_wc | tennis_prematch
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  model_prob REAL NOT NULL,
  market_price REAL NOT NULL,     -- price seen at prediction time
  edge REAL NOT NULL,
  acted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  prediction_id INTEGER REFERENCES predictions(id),
  kalshi_order_id TEXT,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  action TEXT NOT NULL,
  count INTEGER NOT NULL,
  price_cents INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'submitted',  -- submitted|filled|partial|canceled|rejected
  is_parlay INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  order_id INTEGER REFERENCES orders(id),
  count INTEGER NOT NULL,
  price_cents INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settlements (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  ticker TEXT NOT NULL,
  result TEXT NOT NULL,           -- yes|no
  pnl REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS equity (
  ts TEXT PRIMARY KEY DEFAULT (datetime('now')),
  balance REAL NOT NULL,
  open_exposure REAL NOT NULL
);

-- odds snapshots: every price we ever saw (calibration gold)
CREATE TABLE IF NOT EXISTS price_log (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  ticker TEXT NOT NULL,
  yes_bid INTEGER, yes_ask INTEGER, volume INTEGER
);
CREATE INDEX IF NOT EXISTS idx_price_ticker ON price_log(ticker, ts);
