# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   /v1/chat/completions               │   │
│  │                   (OpenAI-compatible)                │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐   │
│  │                  LangGraph Agent                     │   │
│  │  ┌────────────┐    ┌────────────┐    ┌──────────┐   │   │
│  │  │   Check    │───▶│   ReAct    │───▶│  Format  │   │   │
│  │  │  Profile   │    │    Loop    │    │ Response │   │   │
│  │  └────────────┘    └─────┬──────┘    └──────────┘   │   │
│  │                          │                           │   │
│  │         ┌────────────────┼────────────────┐          │   │
│  │         ▼                ▼                ▼          │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │  yfinance  │  │  Google    │  │ Calculator │     │   │
│  │  │   Tool     │  │  Search    │  │   Tools    │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘     │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
│  ┌──────────────────────────▼──────────────────────────┐   │
│  │              PostgreSQL Checkpointer                 │   │
│  │         (Conversation State + User Profiles)         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| LLM | Gemini 3.0 Pro | via `langchain-google-genai` |
| Web Search | Gemini Google Search grounding | Native Gemini capability |
| Financial Data | yfinance | Wrapped as LangGraph tool |
| Orchestration | LangGraph | ReAct pattern with checkpointing |
| Persistence | PostgreSQL | `langgraph-checkpoint-postgres` |
| API | FastAPI | SSE streaming, OpenAI schema |
| Config | pydantic-settings | Type-safe env handling |

## Agent Workflow

### ReAct Loop Pattern

The agent follows a ReAct (Reason + Act) pattern:

1. **Profile Check**: If user profile is incomplete, agent asks clarifying questions before providing financial advice
2. **Reasoning**: Agent thinks through the query (exposed in `<think>` tags for transparency)
3. **Tool Use**: Calls yfinance for market data or Google Search for current information
4. **Response**: Delivers educational, conservative-biased advice

### Graph Nodes

| Node | Purpose |
|------|---------|
| `check_profile` | Verify user profile completeness (age, risk tolerance, time horizon, goals) |
| `react_loop` | Core reasoning and tool-calling loop |
| `format_response` | Apply educational tone, add explanations |

## State Schema

```python
class UserProfile(BaseModel):
    """Persistent user financial profile."""
    age: int | None = None
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = None
    time_horizon_years: int | None = None
    financial_goals: list[str] = []

class AgentState(TypedDict):
    """LangGraph state for the financial advisor agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: UserProfile
    profile_complete: bool
```

## Tools

| Tool | Purpose | Implementation |
|------|---------|----------------|
| `get_stock_data` | Price, fundamentals, history | yfinance wrapper |
| `get_market_overview` | Index performance, sector trends | yfinance batch query |
| `google_search` | Current events, news, grounding | Gemini native search |
| `calculate_compound_growth` | Investment projections | Pure Python |
| `assess_portfolio_risk` | Risk metrics calculation | yfinance + statistics |

## Persistence

### PostgreSQL Checkpointer

- **Conversation State**: Full message history per thread_id
- **User Profiles**: Stored as part of graph state, persisted automatically
- **Thread Management**: Each user session gets a unique thread_id

### Database Schema (managed by LangGraph)

LangGraph's PostgreSQL checkpointer handles schema automatically. User profiles are serialized within the checkpoint state.

## API Design

### Endpoint

```
POST /v1/chat/completions
```

### Request Format (OpenAI-compatible)

```json
{
  "model": "prudent-wealth-steward",
  "messages": [
    {"role": "user", "content": "Should I invest in index funds?"}
  ],
  "stream": true,
  "user": "user_123"
}
```

### Response Stream Format

Server-Sent Events with thinking exposed:

```
data: {"choices": [{"delta": {"content": "<think>User is asking about index funds..."}}]}
data: {"choices": [{"delta": {"content": "</think>"}}]}
data: {"choices": [{"delta": {"content": "Index funds are an excellent choice..."}}]}
data: [DONE]
```

## Dependencies

```toml
[project]
dependencies = [
    "langchain-core>=0.3",
    "langchain-google-genai>=2.0",
    "langgraph>=0.2",
    "langgraph-checkpoint-postgres>=2.0",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic-settings>=2.0",
    "yfinance>=0.2",
    "httpx>=0.27",
    "asyncpg>=0.30",
]
```

## Deployment

### Container Structure

```dockerfile
FROM python:3.14-slim
# Install uv, copy project, run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API authentication |
| `DATABASE_URL` | PostgreSQL connection string |
| `API_TOKEN` | Token for API authentication |
