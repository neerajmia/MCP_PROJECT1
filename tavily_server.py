from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
import json



load_dotenv()

# Load Groq API key
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# ✅ Initialize the MCP server with a descriptive name
mcp = FastMCP("TavilyToolServer")

# ✅ Create a TavilySearch instance with max_results=2
#tavily_tool_instance = TavilySearch(max_results=2)

# ✅ Define an MCP tool function
@mcp.tool()


async def tavily_query_tool(user_query: str) -> str:
    """
    Fetch web results using TavilySearch for the given user query.

    Args:
        user_query (str): The user's search query.

    Returns:
        str: The search result or an error message.
    """
    try:
        tavily_tool_instance = TavilySearch(max_results=2)
        # ✅ Await the Tavily tool's async invocation
        result =  tavily_tool_instance.invoke(user_query)
        return json.dumps(result)
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Resource: Greeting
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}! How can I assist you with latest web-search today ??"

if __name__ == "__main__":
    mcp.run()