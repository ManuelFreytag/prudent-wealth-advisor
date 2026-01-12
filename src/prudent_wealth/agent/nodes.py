"""Node functions for the LangGraph agent."""

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..models import AgentState, UserProfile
from .prompts import SYSTEM_PROMPT, format_profile_summary


def check_profile_node(state: AgentState) -> dict:
    """Check if user profile is complete and extract any new profile info from messages."""
    profile = state.get("user_profile") or UserProfile()

    # Try to extract profile info from recent messages
    updates = extract_profile_updates(state.get("messages", []))
    if updates:
        profile = UserProfile(**{**profile.model_dump(), **updates})

    return {
        "profile_complete": profile.is_complete(),
        "user_profile": profile,
    }


def create_agent_node(llm):
    """Create the main agent reasoning node with the given LLM."""

    def agent_node(state: AgentState) -> dict:
        """Main agent reasoning node."""
        profile = state.get("user_profile") or UserProfile()
        messages = state["messages"]

        # Prepare system message with profile context
        system_content = SYSTEM_PROMPT.format(profile_summary=format_profile_summary(profile))

        # Build message list for LLM
        llm_messages = [SystemMessage(content=system_content)] + list(messages)

        # Invoke LLM
        response = llm.invoke(llm_messages)

        return {"messages": [response]}

    return agent_node


def extract_profile_updates(messages: list[BaseMessage]) -> dict:
    """Extract profile information from conversation history."""
    updates = {}

    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue

        content = msg.content.lower()

        # Extract age
        age_patterns = [
            r"i(?:'m| am) (\d{2,3})(?: years old)?",
            r"(\d{2,3}) years old",
            r"age[:\s]+(\d{2,3})",
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, content)
            if age_match:
                age = int(age_match.group(1))
                if 18 <= age <= 120:  # Sanity check
                    updates["age"] = age
                break

        # Extract risk tolerance
        if "conservative" in content and "risk" in content:
            updates["risk_tolerance"] = "conservative"
        elif "aggressive" in content and "risk" in content:
            updates["risk_tolerance"] = "aggressive"
        elif "moderate" in content and "risk" in content:
            updates["risk_tolerance"] = "moderate"
        # Also check for direct statements
        elif "i prefer conservative" in content or "low risk" in content:
            updates["risk_tolerance"] = "conservative"
        elif "i prefer aggressive" in content or "high risk" in content:
            updates["risk_tolerance"] = "aggressive"

        # Extract time horizon
        horizon_patterns = [
            r"(\d+)\s*years?\s*(?:time\s*)?horizon",
            r"horizon\s*(?:of\s*)?(\d+)\s*years?",
            r"invest(?:ing)?\s*for\s*(\d+)\s*years?",
            r"retire\s*in\s*(\d+)\s*years?",
        ]
        for pattern in horizon_patterns:
            horizon_match = re.search(pattern, content)
            if horizon_match:
                updates["time_horizon_years"] = int(horizon_match.group(1))
                break

    return updates
