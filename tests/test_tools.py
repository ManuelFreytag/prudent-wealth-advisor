"""Tests for financial tools."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.prudent_wealth.tools.calculators import calculate_compound_growth
from src.prudent_wealth.tools.yfinance_tools import (
    assess_portfolio_risk,
    get_market_overview,
    get_stock_data,
)


class TestCompoundGrowthCalculator:
    """Tests for the compound growth calculator tool."""

    def test_basic_compound_growth(self):
        """Test basic compound growth without contributions."""
        result = calculate_compound_growth.invoke(
            {"principal": 10000, "annual_rate": 0.07, "years": 10, "monthly_contribution": 0}
        )

        assert result["initial_investment"] == 10000
        assert result["years"] == 10
        assert result["annual_rate_percent"] == 7.0
        # $10,000 at 7% for 10 years should be roughly $19,672
        assert 19000 < result["future_value"] < 20500
        assert result["total_contributed"] == 10000
        assert result["total_growth"] > 0

    def test_compound_growth_with_contributions(self):
        """Test compound growth with monthly contributions."""
        result = calculate_compound_growth.invoke(
            {"principal": 10000, "annual_rate": 0.07, "years": 10, "monthly_contribution": 500}
        )

        # With $500/month contributions, should be much higher
        assert result["future_value"] > 80000
        assert result["total_contributed"] == 10000 + (500 * 12 * 10)
        assert result["monthly_contribution"] == 500

    def test_year_by_year_breakdown(self):
        """Test that year-by-year breakdown is generated correctly."""
        result = calculate_compound_growth.invoke(
            {"principal": 10000, "annual_rate": 0.07, "years": 5, "monthly_contribution": 0}
        )

        assert "year_by_year" in result
        assert len(result["year_by_year"]) == 5

        # Each year should show growth
        prev_balance = 10000
        for entry in result["year_by_year"]:
            assert entry["balance"] > prev_balance
            prev_balance = entry["balance"]

    def test_zero_principal(self):
        """Test with zero initial investment but monthly contributions."""
        result = calculate_compound_growth.invoke(
            {"principal": 0, "annual_rate": 0.07, "years": 10, "monthly_contribution": 500}
        )

        assert result["initial_investment"] == 0
        assert result["future_value"] > 0
        assert result["total_contributed"] == 500 * 12 * 10

    def test_high_return_rate(self):
        """Test with higher return rate."""
        result = calculate_compound_growth.invoke(
            {"principal": 10000, "annual_rate": 0.12, "years": 10, "monthly_contribution": 0}
        )

        # 12% returns should result in higher growth
        assert result["future_value"] > 30000


class TestUserProfile:
    """Tests for UserProfile model."""

    def test_profile_completeness(self):
        """Test profile completeness checking."""
        from src.prudent_wealth.models import UserProfile

        # Empty profile should be incomplete
        profile = UserProfile()
        assert not profile.is_complete()

        # Partial profile should be incomplete
        profile = UserProfile(age=30)
        assert not profile.is_complete()

        # Complete profile
        profile = UserProfile(age=30, risk_tolerance="moderate", time_horizon_years=20)
        assert profile.is_complete()


class TestGetStockData:
    """Tests for the get_stock_data tool."""

    @patch("src.prudent_wealth.tools.yfinance_tools.yf.Ticker")
    def test_get_stock_data_success(self, mock_ticker_class):
        """Test successful stock data retrieval."""
        # Create mock ticker instance
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Mock info data
        mock_ticker.info = {
            "longName": "Apple Inc.",
            "currentPrice": 175.50,
            "previousClose": 174.00,
            "marketCap": 2800000000000,
            "trailingPE": 28.5,
            "forwardPE": 26.0,
            "dividendYield": 0.005,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 124.17,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }

        # Mock history data
        mock_history = pd.DataFrame(
            {
                "Close": [170.0, 172.0, 174.0, 175.0, 175.5],
                "High": [171.0, 173.0, 175.0, 176.0, 176.5],
                "Low": [169.0, 171.0, 173.0, 174.0, 175.0],
            }
        )
        mock_ticker.history.return_value = mock_history

        result = get_stock_data.invoke({"symbol": "AAPL", "period": "1mo"})

        assert result["symbol"] == "AAPL"
        assert result["name"] == "Apple Inc."
        assert result["current_price"] == 175.50
        assert result["sector"] == "Technology"
        assert "history_summary" in result
        assert result["history_summary"]["start_price"] == 170.0
        assert result["history_summary"]["end_price"] == 175.5

    @patch("src.prudent_wealth.tools.yfinance_tools.yf.Ticker")
    def test_get_stock_data_error(self, mock_ticker_class):
        """Test error handling when stock data retrieval fails."""
        mock_ticker_class.side_effect = Exception("API Error")

        result = get_stock_data.invoke({"symbol": "INVALID"})

        assert "error" in result
        assert result["symbol"] == "INVALID"


class TestGetMarketOverview:
    """Tests for the get_market_overview tool."""

    @patch("src.prudent_wealth.tools.yfinance_tools.yf.Ticker")
    def test_get_market_overview_default_indices(self, mock_ticker_class):
        """Test market overview with default indices."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {
            "shortName": "S&P 500",
            "regularMarketPrice": 4500.0,
            "regularMarketChange": 25.5,
            "regularMarketChangePercent": 0.57,
            "dayHigh": 4520.0,
            "dayLow": 4480.0,
        }

        result = get_market_overview.invoke({})

        assert "indices" in result
        # Should have checked default indices
        assert mock_ticker_class.call_count == 4  # ^GSPC, ^DJI, ^IXIC, ^VIX

    @patch("src.prudent_wealth.tools.yfinance_tools.yf.Ticker")
    def test_get_market_overview_custom_indices(self, mock_ticker_class):
        """Test market overview with custom indices."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {
            "shortName": "Test Index",
            "regularMarketPrice": 1000.0,
            "regularMarketChange": 10.0,
            "regularMarketChangePercent": 1.0,
            "dayHigh": 1010.0,
            "dayLow": 990.0,
        }

        result = get_market_overview.invoke({"indices": ["^GSPC", "^DJI"]})

        assert "indices" in result
        assert mock_ticker_class.call_count == 2

    @patch("src.prudent_wealth.tools.yfinance_tools.yf.Ticker")
    def test_get_market_overview_partial_failure(self, mock_ticker_class):
        """Test market overview handles partial failures gracefully."""

        def side_effect(symbol):
            if symbol == "^GSPC":
                mock = MagicMock()
                mock.info = {"shortName": "S&P 500", "regularMarketPrice": 4500.0}
                return mock
            else:
                raise Exception("API Error")

        mock_ticker_class.side_effect = side_effect

        result = get_market_overview.invoke({"indices": ["^GSPC", "^INVALID"]})

        assert "indices" in result
        assert "^GSPC" in result["indices"]
        assert "^INVALID" in result["indices"]
        assert "error" in result["indices"]["^INVALID"]


class TestAssessPortfolioRisk:
    """Tests for the assess_portfolio_risk tool."""

    @patch("src.prudent_wealth.tools.yfinance_tools.yf.Ticker")
    def test_assess_portfolio_risk_success(self, mock_ticker_class):
        """Test successful portfolio risk assessment."""

        def create_mock_ticker(symbol):
            mock = MagicMock()
            mock.info = {
                "longName": f"{symbol} Corp",
                "sector": "Technology" if symbol == "AAPL" else "Healthcare",
                "quoteType": "EQUITY",
            }
            # Create realistic price history for volatility calculation
            import numpy as np

            np.random.seed(42)
            prices = 100 * (1 + np.random.randn(252).cumsum() * 0.01)
            mock.history.return_value = pd.DataFrame(
                {
                    "Close": prices,
                    "High": prices * 1.01,
                    "Low": prices * 0.99,
                }
            )
            return mock

        mock_ticker_class.side_effect = create_mock_ticker

        result = assess_portfolio_risk.invoke(
            {"holdings": [{"symbol": "AAPL", "weight": 60}, {"symbol": "JNJ", "weight": 40}]}
        )

        assert "portfolio_volatility" in result
        assert "risk_level" in result
        assert "sector_breakdown" in result
        assert "holdings_analysis" in result
        assert "recommendations" in result

    def test_assess_portfolio_risk_no_holdings(self):
        """Test error when no holdings provided."""
        result = assess_portfolio_risk.invoke({"holdings": []})

        assert "error" in result

    def test_assess_portfolio_risk_invalid_weights(self):
        """Test warning when weights don't sum to 100%."""
        result = assess_portfolio_risk.invoke(
            {"holdings": [{"symbol": "AAPL", "weight": 30}]}  # Only 30%, not 100%
        )

        assert "warning" in result

    @patch("src.prudent_wealth.tools.yfinance_tools.yf.Ticker")
    def test_assess_portfolio_risk_etf_detection(self, mock_ticker_class):
        """Test that ETFs are properly categorized."""
        mock = MagicMock()
        mock.info = {
            "longName": "Vanguard S&P 500 ETF",
            "sector": "Unknown",
            "quoteType": "ETF",
        }
        mock.history.return_value = pd.DataFrame(
            {"Close": [100, 101, 102], "High": [101, 102, 103], "Low": [99, 100, 101]}
        )
        mock_ticker_class.return_value = mock

        result = assess_portfolio_risk.invoke({"holdings": [{"symbol": "VOO", "weight": 100}]})

        assert "ETF/Fund" in result.get("sector_breakdown", {})
