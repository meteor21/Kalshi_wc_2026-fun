"""Kalshi REST client. RSA-PSS auth, demo by default."""
import base64, os, time
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

URLS = {
    "demo": "https://demo-api.kalshi.co",
    "prod": "https://api.elections.kalshi.com",
}
PREFIX = "/trade-api/v2"


def _load_dotenv(path=".env"):
    """Minimal .env loader: KEY=VALUE lines, no quotes/expansion. os.environ wins."""
    try:
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


_load_dotenv()


class KalshiClient:
    def __init__(self, env=None, key_id=None, key_path=None):
        self.env = env or os.environ.get("KALSHI_ENV", "demo")
        if self.env not in URLS:
            raise ValueError(f"env must be demo|prod, got {self.env}")
        self.base = URLS[self.env]
        self.key_id = key_id or os.environ["KALSHI_KEY_ID"]
        path = key_path or os.environ["KALSHI_PRIVATE_KEY_PATH"]
        with open(path, "rb") as f:
            self.key = serialization.load_pem_private_key(f.read(), password=None)
        self.s = requests.Session()

    def _headers(self, method, path):
        ts = str(int(time.time() * 1000))
        msg = (ts + method + path).encode()  # path incl. /trade-api/v2, no query string
        sig = self.key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def _req(self, method, path, params=None, json=None):
        full = PREFIX + path
        r = self.s.request(method, self.base + full, params=params, json=json,
                           headers=self._headers(method, full), timeout=15)
        r.raise_for_status()
        return r.json()

    # --- market data ---
    def markets(self, **params):
        params.setdefault("status", "open")
        return self._req("GET", "/markets", params=params)

    def orderbook(self, ticker, depth=10):
        return self._req("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def events(self, **params):
        return self._req("GET", "/events", params=params)

    # --- portfolio (auth required) ---
    def balance(self):
        return self._req("GET", "/portfolio/balance")

    def positions(self, **params):
        return self._req("GET", "/portfolio/positions", params=params)

    def orders(self, **params):
        return self._req("GET", "/portfolio/orders", params=params)

    def create_order(self, ticker, side, action, count, price_cents, type_="limit", client_order_id=None):
        """side: yes|no. action: buy|sell. price_cents: 1-99 limit price."""
        body = {
            "ticker": ticker, "side": side, "action": action, "type": type_,
            "count": count, "client_order_id": client_order_id or f"kq-{int(time.time()*1000)}",
        }
        body["yes_price" if side == "yes" else "no_price"] = price_cents
        return self._req("POST", "/portfolio/orders", json=body)

    def cancel_order(self, order_id):
        return self._req("DELETE", f"/portfolio/orders/{order_id}")
