"""Auth + balance check. First thing to run after setting .env."""
from execution.kalshi_client import KalshiClient


def main():
    c = KalshiClient()
    print(f"env={c.env} base={c.base}")
    bal = c.balance()
    print(f"balance: {bal}")
    m = c.markets(limit=3)
    for mk in m.get("markets", []):
        print(f"  {mk['ticker']}: {mk.get('title','')[:60]}")


if __name__ == "__main__":
    main()
