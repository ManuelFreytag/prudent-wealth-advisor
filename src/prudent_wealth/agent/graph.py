"""LangGraph agent definition for the financial advisor."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ..config import settings
from ..models import AgentState, UserProfile
from ..tools import (
    assess_portfolio_risk,
    calculate_compound_growth,
    get_market_overview,
    get_financial_product_data,
    web_search,
)
from .nodes import check_profile_node, create_agent_node, create_router_node, create_smalltalk_node


def create_agent_graph(checkpointer=None, temperature: float | None = None):
    """Create the financial advisor agent graph.

    Args:
        checkpointer: Optional checkpointer for state persistence.
                     Defaults to MemorySaver for development.

    Returns:
        Compiled LangGraph agent.
    """
    # Define available tools
    tools = [
        get_financial_product_data,
        get_market_overview,
        assess_portfolio_risk,
        calculate_compound_growth,
        web_search,
    ]

    # Simple LLM for basic questions
    simple_llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", google_api_key=settings.google_api_key
    )

    # Initialize the LLM with tools
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        include_thoughts=True,
    ).bind_tools(tools)

    # Create tool node
    tool_node = ToolNode(tools)

    # Create router node
    router_node = create_router_node(simple_llm)

    smalltalk_node = create_smalltalk_node(simple_llm)

    # Create agent node with the configured LLM
    agent_node = create_agent_node(llm)

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("smalltalk", smalltalk_node)
    workflow.add_node("check_profile", check_profile_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Define edges
    workflow.add_edge("smalltalk", END)
    workflow.add_edge("check_profile", "agent")

    # Conditional routing from agent: financial advice or finish
    def should_advice(state: AgentState) -> str:
        """Determine if the agent should provide financial advice or finish."""
        intent = state["intent"]
        if not intent:
            return "end"

        if intent == "small_talk":
            return "smalltalk"
        if intent == "main_agent":
            return "agent"
        return "end"

    workflow.add_conditional_edges(
        "router",
        should_advice,
        {"smalltalk": "smalltalk", "agent": "agent", "end": END},
    )

    # Conditional routing from agent: use tools or finish
    def should_use_tools(state: AgentState) -> str:
        """Determine if the agent should use tools or finish."""
        messages = state["messages"]
        if not messages:
            return "end"

        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    workflow.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "end": END},
    )

    # Tools loop back to agent for further reasoning
    workflow.add_edge("tools", "agent")

    # Use provided checkpointer (required - handled by api/database.py)
    return workflow.compile(checkpointer=checkpointer)


def get_initial_state() -> AgentState:
    """Get initial state for a new conversation."""
    return {
        "intent": "small_talk",
        "messages": [],
        "user_profile": UserProfile(),
        "profile_complete": False,
    }
