"""Tests for the agent graph and nodes."""

import pytest

from src.prudent_wealth.agent.nodes import extract_profile_updates
from src.prudent_wealth.agent.prompts import format_profile_summary
from src.prudent_wealth.models import UserProfile


class TestProfileExtraction:
    """Tests for profile extraction from messages."""

    def test_extract_age(self):
        """Test extracting age from user message."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="I'm 35 years old")]
        updates = extract_profile_updates(messages)
        assert updates.get("age") == 35

    def test_extract_age_alternate_format(self):
        """Test extracting age with alternate phrasing."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="I am 42")]
        updates = extract_profile_updates(messages)
        assert updates.get("age") == 42

    def test_extract_risk_tolerance_conservative(self):
        """Test extracting conservative risk tolerance."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="I prefer low risk investments")]
        updates = extract_profile_updates(messages)
        assert updates.get("risk_tolerance") == "conservative"

    def test_extract_risk_tolerance_aggressive(self):
        """Test extracting aggressive risk tolerance."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="I have aggressive risk tolerance")]
        updates = extract_profile_updates(messages)
        assert updates.get("risk_tolerance") == "aggressive"

    def test_extract_time_horizon(self):
        """Test extracting investment time horizon."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="I want to retire in 25 years")]
        updates = extract_profile_updates(messages)
        assert updates.get("time_horizon_years") == 25

    def test_extract_multiple_fields(self):
        """Test extracting multiple profile fields from conversation."""
        from langchain_core.messages import HumanMessage

        messages = [
            HumanMessage(content="I'm 30 years old"),
            HumanMessage(content="I have moderate risk tolerance"),
            HumanMessage(content="Planning to invest for 20 years"),
        ]
        updates = extract_profile_updates(messages)
        assert updates.get("age") == 30
        assert updates.get("risk_tolerance") == "moderate"
        assert updates.get("time_horizon_years") == 20


class TestProfileSummary:
    """Tests for profile summary formatting."""

    def test_empty_profile_summary(self):
        """Test summary for empty profile."""
        profile = UserProfile()
        summary = format_profile_summary(profile)
        assert "No profile information" in summary

    def test_partial_profile_summary(self):
        """Test summary for partial profile."""
        profile = UserProfile(age=30, risk_tolerance="moderate")
        summary = format_profile_summary(profile)
        assert "Age: 30" in summary
        assert "moderate" in summary
        assert "Missing" in summary
        assert "time horizon" in summary.lower()

    def test_complete_profile_summary(self):
        """Test summary for complete profile."""
        profile = UserProfile(
            age=35,
            risk_tolerance="conservative",
            time_horizon_years=15,
            financial_goals=["retirement", "house down payment"],
        )
        summary = format_profile_summary(profile)
        assert "Age: 35" in summary
        assert "conservative" in summary
        assert "15 years" in summary
        assert "retirement" in summary
        assert "Missing" not in summary
