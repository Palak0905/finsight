import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker: str, filing_date: str) -> dict:
    """
    Get stock price change after a filing date.
    Returns price on filing day, +1 day, +3 days, +7 days.
    """
    try:
        date = datetime.strptime(filing_date, "%Y-%m-%d")
        start = date - timedelta(days=3)
        end = date + timedelta(days=12)

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            return {"error": "No stock data available"}

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        prices = hist["Close"]

        # Get nearest trading day prices
        def nearest_price(target_date):
            target = pd.Timestamp(target_date)
            idx = prices.index.searchsorted(target)
            if idx >= len(prices):
                idx = len(prices) - 1
            return round(float(prices.iloc[idx]), 2)

        base_price = nearest_price(date)
        price_1d = nearest_price(date + timedelta(days=1))
        price_3d = nearest_price(date + timedelta(days=3))
        price_7d = nearest_price(date + timedelta(days=7))

        def pct_change(new, base):
            if base == 0:
                return 0
            return round(((new - base) / base) * 100, 2)

        return {
            "base_price": base_price,
            "price_1d": price_1d,
            "price_3d": price_3d,
            "price_7d": price_7d,
            "change_1d_pct": pct_change(price_1d, base_price),
            "change_3d_pct": pct_change(price_3d, base_price),
            "change_7d_pct": pct_change(price_7d, base_price),
        }
    except Exception as e:
        return {"error": str(e)}

def correlate_signal(sentiment_score: float, stock_data: dict) -> dict:
    """
    Simple correlation: does positive sentiment → positive price movement?
    """
    if "error" in stock_data:
        return {"verdict": "No stock data", "aligned": None}

    price_move_1d = stock_data.get("change_1d_pct", 0)
    price_move_7d = stock_data.get("change_7d_pct", 0)

    # Signal alignment check
    sentiment_positive = sentiment_score > 0.05
    sentiment_negative = sentiment_score < -0.05
    price_up_1d = price_move_1d > 0
    price_up_7d = price_move_7d > 0

    aligned_1d = (sentiment_positive and price_up_1d) or (sentiment_negative and not price_up_1d)
    aligned_7d = (sentiment_positive and price_up_7d) or (sentiment_negative and not price_up_7d)

    if aligned_1d and aligned_7d:
        verdict = "Strong alignment"
    elif aligned_1d or aligned_7d:
        verdict = "Partial alignment"
    else:
        verdict = "No alignment"

    return {
        "verdict": verdict,
        "aligned_1d": aligned_1d,
        "aligned_7d": aligned_7d,
        "sentiment_direction": "positive" if sentiment_positive else ("negative" if sentiment_negative else "neutral"),
        "price_direction_1d": "up" if price_up_1d else "down",
        "price_direction_7d": "up" if price_up_7d else "down"
    }
