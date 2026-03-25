import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search(query):
    """
    Search tool for the ReAct Agent
    Input: query (str)
    Output: summarized search results with sources (str)
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Error: Missing TAVILY_API_KEY in environment."

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=False,
        )

        results = response.get("results", [])
        if not results:
            return "No results found."

        lines = []
        for idx, item in enumerate(results[:3], start=1):
            title = (item.get("title") or "Untitled").strip()
            content = (item.get("content") or "No snippet available.").replace("\n", " ").strip()
            snippet = content[:320]
            lines.append(f"[{idx}] {title}: {snippet}")

        return "\n\n".join(lines)

    except Exception as e:
        return f"Error: {str(e)}"