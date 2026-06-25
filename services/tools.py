from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch


def web_search_tool(max_results: int = 5) -> BaseTool:
    return TavilySearch(max_results=max_results)

