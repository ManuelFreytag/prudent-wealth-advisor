"""Core data models for the financial advisor agent."""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class UserProfile(BaseModel):
    """Persistent user financial profile collected through conversation."""

    age: int | None = Field(default=None, description="User's current age")
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = Field(
        default=None, description="User's risk tolerance level"
    )
    time_horizon_years: int | None = Field(
        default=None, description="Investment time horizon in years"
    )
    financial_goals: list[str] = Field(default_factory=list, description="User's financial goals")

    def is_complete(self) -> bool:
        """Check if profile has enough information for personalized advice."""
        return (
            self.age is not None
            and self.risk_tolerance is not None
            and self.time_horizon_years is not None
        )


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    destination: Literal["small_talk", "main_agent"] = Field(
        ...,
        description="Given a user input choose to route it to 'small_talk' "
        "or a 'main_agent' capability.",
    )


class AgentState(TypedDict):
    """LangGraph state for the financial advisor agent."""

    intent: Literal["small_talk", "main_agent"]
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: UserProfile
    profile_complete: bool
