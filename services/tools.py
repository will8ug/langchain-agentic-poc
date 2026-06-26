from datetime import date

from langchain.tools import tool
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch


@tool
def get_current_date() -> str:
    """Get the current date. Returns today's date as a string in ISO format (YYYY-MM-DD)."""
    return date.today().isoformat()


def web_search_tool(max_results: int = 5) -> BaseTool:
    return TavilySearch(max_results=max_results)

