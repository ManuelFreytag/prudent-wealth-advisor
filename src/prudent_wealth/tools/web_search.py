"""Web search tool using Gemini's native Google Search grounding."""

from typing import Annotated

from google.genai.types import GoogleSearch, Tool
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings


@tool
def web_search(
    query: Annotated[str, "Search query for finding current information on the web"],
) -> str:
    """Search the web for current information using Google Search.

    Use this tool when you need:
    - Recent news about markets, companies, or economic events
    - Current information that may have changed since your training data
    - Facts that need to be verified with up-to-date sources
    - Information about recent financial regulations or policy changes

    Returns a summary of relevant search results.
    """
    try:
        # Create a separate LLM instance with Google Search grounding
        search_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

        # Invoke with Google Search grounding enabled using google-genai types
        google_search_tool = Tool(google_search=GoogleSearch())
        response = search_llm.invoke(
            f"Search for and summarize the most relevant and recent information about: {query}",
            tools=[google_search_tool],
        )

        return response.content

    except Exception as e:
        return f"Search failed: {str(e)}. Please try rephrasing your query or proceed without this information."
