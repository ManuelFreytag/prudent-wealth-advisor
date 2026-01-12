"""Financial calculator tools."""

from typing import Annotated

from langchain_core.tools import tool


@tool
def calculate_compound_growth(
    principal: Annotated[float, "Initial investment amount in dollars"],
    annual_rate: Annotated[float, "Expected annual return rate as decimal (e.g., 0.07 for 7%)"],
    years: Annotated[int, "Number of years to grow"],
    monthly_contribution: Annotated[float, "Monthly contribution amount in dollars"] = 0,
) -> dict:
    """Calculate compound growth of an investment over time.

    Use this tool to show users how their money can grow with compound interest.
    Demonstrates the power of regular contributions and long-term investing.

    Returns projections including year-by-year breakdown.
    """
    # Future value of initial principal
    fv_principal = principal * ((1 + annual_rate) ** years)

    # Future value of monthly contributions (annuity formula)
    if monthly_contribution > 0:
        monthly_rate = annual_rate / 12
        n_months = years * 12
        fv_contributions = monthly_contribution * (
            ((1 + monthly_rate) ** n_months - 1) / monthly_rate
        )
    else:
        fv_contributions = 0

    total_future_value = fv_principal + fv_contributions
    total_contributed = principal + (monthly_contribution * 12 * years)
    total_growth = total_future_value - total_contributed

    return {
        "initial_investment": principal,
        "monthly_contribution": monthly_contribution,
        "annual_rate_percent": round(annual_rate * 100, 2),
        "years": years,
        "total_contributed": round(total_contributed, 2),
        "future_value": round(total_future_value, 2),
        "total_growth": round(total_growth, 2),
        "growth_percentage": round((total_growth / total_contributed) * 100, 2)
        if total_contributed > 0
        else 0,
        "year_by_year": _generate_yearly_breakdown(principal, annual_rate, years, monthly_contribution),
    }


def _generate_yearly_breakdown(
    principal: float, rate: float, years: int, monthly_contrib: float
) -> list[dict]:
    """Generate year-by-year growth breakdown."""
    breakdown = []
    balance = principal
    monthly_rate = rate / 12

    # Cap at 10 years for brevity in output
    for year in range(1, min(years + 1, 11)):
        for _ in range(12):
            balance = balance * (1 + monthly_rate) + monthly_contrib

        breakdown.append({"year": year, "balance": round(balance, 2)})

    return breakdown
