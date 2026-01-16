"""Financial data tools using yfinance."""

from typing import Annotated

import yfinance as yf
from langchain_core.tools import tool


@tool(
    "get_financial_product_data",
    description="Get price data and fundamentals for stocks, stocks, ETFs, crypto, and other financial products. Use this to lookup data for any asset class supported by Yahoo Finance.",
)
def get_financial_product_data(
    symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL', 'VOO', 'BTC-USD', 'GC=F')"],
    period: Annotated[str, "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"] = "1mo",
) -> dict:
    """Get price data and fundamentals for stocks, ETFs, crypto, and other financial products.

    Use this tool to lookup data for any asset class supported by Yahoo Finance, including:
    - Stocks (e.g., AAPL)
    - ETFs (e.g., VOO, QQ)
    - Cryptocurrencies (e.g., BTC-USD)
    - Futures/Commodities (e.g., GC=F for Gold)

    Returns prices, fundamentals (P/E, yield), and ETF-specific data (expense ratio, category).
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period=period)

        # Calculate history summary
        history_summary = {}
        if len(history) > 0:
            history_summary = {
                "period": period,
                "start_price": round(float(history["Close"].iloc[0]), 2),
                "end_price": round(float(history["Close"].iloc[-1]), 2),
                "high": round(float(history["High"].max()), 2),
                "low": round(float(history["Low"].min()), 2),
                "percent_change": round(
                    (
                        (history["Close"].iloc[-1] - history["Close"].iloc[0])
                        / history["Close"].iloc[0]
                    )
                    * 100,
                    2,
                ),
            }

        data = {
            "symbol": symbol.upper(),
            "name": info.get("longName", symbol),
            "type": info.get("quoteType", "Unknown"),
            "current_price": info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("navPrice"),
            "previous_close": info.get("previousClose"),
            "market_cap": info.get("marketCap") or info.get("totalAssets"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "history_summary": history_summary,
        }

        # Add specific fields based on asset type
        quote_type = info.get("quoteType", "").upper()

        if "ETF" in quote_type or "MUTUALFUND" in quote_type:
            data.update(
                {
                    "category": info.get("category"),
                    "expense_ratio": info.get("netExpenseRatio")
                    or info.get("annualReportExpenseRatio"),
                    "family": info.get("fundFamily"),
                    "yield": info.get("yield"),
                    "ytd_return": info.get("ytdReturn"),
                    "beta_3y": info.get("beta3Year"),
                }
            )

        # Add fundamental data for equities if available
        # (Some of these might also apply to ETFs, but are traditional stock metrics)
        data.update(
            {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
        )

        # Clean up None values to keep response clean
        return {k: v for k, v in data.items() if v is not None}

    except Exception as e:
        return {"error": str(e), "symbol": symbol}


@tool
def get_market_overview(
    indices: Annotated[list[str] | None, "List of index symbols to check"] = None,
) -> dict:
    """Get overview of major market indices and their current performance.

    Use this to understand overall market conditions before making recommendations.
    Default indices: S&P 500 (^GSPC), Dow Jones (^DJI), NASDAQ (^IXIC), VIX (^VIX).
    """
    if indices is None:
        indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]

    results = {}
    for symbol in indices:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            results[symbol] = {
                "name": info.get("shortName", symbol),
                "price": info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "change_percent": info.get("regularMarketChangePercent"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}

    return {
        "indices": results,
        "timestamp": "real-time",
        "note": "Market data may be delayed 15-20 minutes",
    }


@tool
def assess_portfolio_risk(
    holdings: Annotated[
        list[dict], "List of holdings with 'symbol' and 'weight' keys (weight as percentage 0-100)"
    ],
) -> dict:
    """Assess the risk profile of a portfolio based on its holdings.

    Each holding should have:
    - symbol: Stock/ETF ticker
    - weight: Percentage of portfolio (0-100)

    Returns risk metrics, sector breakdown, and diversification analysis.
    """
    if not holdings:
        return {"error": "No holdings provided"}

    total_weight = sum(h.get("weight", 0) for h in holdings)
    if abs(total_weight - 100) > 5:
        return {
            "warning": f"Portfolio weights sum to {total_weight}%, not 100%",
            "suggestion": "Please adjust weights to sum to approximately 100%",
        }

    # Collect data for analysis
    volatilities = []
    sector_breakdown = {}
    asset_details = []

    for holding in holdings:
        symbol = holding.get("symbol", "").upper()
        weight = holding.get("weight", 0)

        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="1y")
            info = ticker.info

            detail = {
                "symbol": symbol,
                "weight": weight,
                "name": info.get("longName", symbol),
            }

            if len(history) > 20:
                # Calculate annualized volatility
                returns = history["Close"].pct_change().dropna()
                vol = returns.std() * (252**0.5) * 100  # Annualized, as percentage
                volatilities.append({"symbol": symbol, "weight": weight, "volatility": vol})
                detail["volatility"] = round(vol, 2)

            # Sector tracking
            sector = info.get("sector", "Unknown")
            if (sector == "Unknown" or sector is None) and "ETF" in info.get("quoteType", ""):
                sector = info.get("category", "ETF/Fund")
            sector_breakdown[sector] = sector_breakdown.get(sector, 0) + weight
            detail["sector"] = sector

            asset_details.append(detail)

        except Exception as e:
            asset_details.append({"symbol": symbol, "weight": weight, "error": str(e)})

    # Calculate weighted volatility
    weighted_vol = (
        sum(v["volatility"] * v["weight"] / 100 for v in volatilities) if volatilities else None
    )

    # Determine risk level
    if weighted_vol:
        if weighted_vol < 12:
            risk_level = "Low"
        elif weighted_vol < 18:
            risk_level = "Moderate"
        elif weighted_vol < 25:
            risk_level = "Moderately High"
        else:
            risk_level = "High"
    else:
        risk_level = "Unable to assess"

    return {
        "portfolio_volatility": round(weighted_vol, 2) if weighted_vol else None,
        "risk_level": risk_level,
        "sector_breakdown": {k: round(v, 1) for k, v in sector_breakdown.items()},
        "diversification_score": len(sector_breakdown),
        "holdings_analysis": asset_details,
        "recommendations": _get_risk_recommendations(sector_breakdown, weighted_vol),
    }


def _get_risk_recommendations(sectors: dict, volatility: float | None) -> list[str]:
    """Generate risk management recommendations based on portfolio analysis."""
    recommendations = []

    # Check diversification
    if len(sectors) < 3:
        recommendations.append(
            "Low diversification: Consider adding assets from different sectors to reduce risk"
        )
    elif len(sectors) >= 5:
        recommendations.append("Good sector diversification across the portfolio")

    # Check concentration
    for sector, weight in sectors.items():
        if weight > 40:
            recommendations.append(
                f"High concentration in {sector} ({weight:.1f}%): Consider reducing to below 30%"
            )
        elif weight > 30:
            recommendations.append(
                f"Moderate concentration in {sector} ({weight:.1f}%): Monitor this allocation"
            )

    # Volatility-based recommendations
    if volatility:
        if volatility > 25:
            recommendations.append(
                "High portfolio volatility: Consider adding bonds or low-volatility dividend stocks"
            )
        elif volatility > 20:
            recommendations.append(
                "Moderately high volatility: May be suitable for long time horizons only"
            )
        elif volatility < 10:
            recommendations.append(
                "Low volatility portfolio: Good for capital preservation, may underperform in bull markets"
            )

    if not recommendations:
        recommendations.append("Portfolio appears well-balanced for moderate risk tolerance")

    return recommendations
