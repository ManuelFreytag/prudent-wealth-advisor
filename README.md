# Prudent Wealth Steward (Financial Advisor Agent)

Prudent Wealth Steward is an autonomous financial planning agent designed to maximize long-term financial health while minimizing risk. It leverages LangGraph for orchestration, Gemini 3.0 Pro for reasoning, and FastAPI to provide an OpenAI-compatible interface.

The agent follows a conservative, educational approach, prioritizing capital preservation and diversification while explaining the "why" behind its recommendations.

---

## Features

- **Holistic Financial Analysis**: Considers user profile (age, risk tolerance, goals) before advising.
- **ReAct Agent Pattern**: Sophisticated reasoning-action loop for complex financial queries.
- **Integrated Tools**:
    - `yfinance`: Real-time market data, fundamentals, and portfolio risk assessment.
    - `Google Search`: Native Gemini grounding for up-to-date financial news and trends.
    - `Calculators`: Custom tools for compound growth and risk metrics.
- **State Persistence**: Full conversation history and user profile persistence using PostgreSQL (fallback to in-memory for dev).
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints, supporting both streaming and non-streaming.
- **Transparent Reasoning**: Reasoning steps are provided in the `reasoning_content` field (OpenAI-compatible) of the streaming response.
- **Secure**: Built-in Basic Token Authentication.

---

## Tech Stack

- **Language**: Python 3.14
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **LLM**: Gemini 3.0 Pro (via `langchain-google-genai`)
- **API**: FastAPI
- **Database**: PostgreSQL (via `langgraph-checkpoint-postgres`)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)

---

## Installation & Setup

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) installed.
- A Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### 2. Environment Configuration
Create a `.env` file from the example:
```bash
cp .env.example .env
```
Fill in your keys:
```env
GOOGLE_API_KEY=your-api-key
API_TOKEN=your-unique-app-token
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres # Optional for local dev
```

### 3. Install Dependencies
```bash
uv sync --all-extras
```

---

## Running the Application

### Locally
```bash
uv run main.py
```
The API will be available at `http://localhost:8000`.

### Using Docker
```bash
docker-compose up --build
```
This spawns both the FastAPI server and a PostgreSQL instance.

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Chat Completion (Streaming)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Should I invest in index funds?"}],
    "stream": true,
    "chat_id": "unique-session-id"
  }'
```

---

## OpenWebUI Integration

The API is generally OpenAI compatible. However, to get the best experience (including thread persistence and streaming stability), we recommend using the following Pipe and Filter.

### Streaming Proxy Pipe
Use this to wrap the agent in a "Manifold" pipe for OpenWebUI.

```python
"""
title: Prudent Wealth Steward Pipe
author: Manuel Freytag
version: 0.1.0
"""

from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator, Callable, Dict, Any
import requests
import json
import os

class Pipe:
    class Valves(BaseModel):
        # Configuration settings exposed in the OpenWebUI interface
        TARGET_API_URL: str = Field(
            default="http://host.docker.internal:8000/v1/chat/completions",
            description=" The full endpoint URL to send the request to."
        )
        API_KEY: str = Field(
            default="",
            description="Bearer token or API Key for the external endpoint."
        )
        MODEL_ID: str = Field(
            default="prudent-wealth-steward",
            description="The model ID to request from the external API."
        )
        EMIT_THINK_TAGS: bool = Field(
            default=True,
            description="If True, wraps reasoning_content in <think> tags for UI rendering."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.type = "manifold" # Indicates this adds a model to the selector

    def pipes(self) -> list[dict[str, str]]:
        # This registers the model in OpenWebUI's model list
        return [{"id": self.valves.MODEL_ID, "name": f"Proxy: {self.valves.MODEL_ID}"}]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        """
        The main handler function.
        """
        
        # 1. Prepare Headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.API_KEY}"
        }

        # 2. Prepare Payload
        # We replace the model in the body with the one defined in Valves
        payload = {**body, "model": self.valves.MODEL_ID, "stream": True}

        # 3. Request logic within a try/except block for safety
        try:
            response = requests.post(
                self.valves.TARGET_API_URL,
                headers=headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()

            # 4. Return the generator that processes the stream
            return self.stream_generator(response)

        except Exception as e:
            return f"Error: {e}"

    def stream_generator(self, response) -> Generator[str, None, None]:
        """
        Parses the SSE stream and handles reasoning_content vs content.
        """
        reasoning_started = False
        reasoning_ended = False

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                
                # Handle SSE format (usually starts with "data: ")
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:] # Strip "data: "

                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                            
                        delta = choices[0].get("delta", {})
                        
                        # --- Handle Reasoning Content ---
                        reasoning_chunk = delta.get("reasoning_content", "")
                        if reasoning_chunk:
                            # If this is the first piece of reasoning and we haven't started yet
                            if self.valves.EMIT_THINK_TAGS and not reasoning_started:
                                yield "<think>\n"
                                reasoning_started = True
                            
                            yield reasoning_chunk

                        # --- Handle Standard Content ---
                        content_chunk = delta.get("content", "")
                        if content_chunk:
                            # If we were reasoning but now have content, close the tag
                            if self.valves.EMIT_THINK_TAGS and reasoning_started and not reasoning_ended:
                                yield "\n</think>\n"
                                reasoning_ended = True
                                
                            yield content_chunk

                    except json.JSONDecodeError:
                        # Skip unparseable lines (keep-alives, errors)
                        continue
        
        # Cleanup: Ensure <think> is closed if the stream ended while reasoning
        if self.valves.EMIT_THINK_TAGS and reasoning_started and not reasoning_ended:
             yield "\n</think>\n"
```

### Metadata Chat ID Mapper (Filter)
OpenWebUI doesn't natively pass `chat_id` in a way this agent expects for persistence. Use this filter to map the UI `chat_id` to the request metadata.

```python
"""
title: Chat ID Mapper
author: openwebui-fan
version: 1.0
"""
from pydantic import BaseModel, Field
from typing import Optional

class Filter:
    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            chat_id = metadata.get("chat_id")
            if chat_id:
                body["chat_id"] = chat_id
        return body
```

---

## Architecture
For advanced details on how the ReAct loop, profile checking, and persistence layers work, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Development
- **Tests**: `uv run pytest tests/`
- **Linting**: `uv run black .` & `uv run isort .`
