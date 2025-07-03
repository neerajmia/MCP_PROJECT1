import os
import json
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_community.llms import Ollama
from langchain import LLMChain, PromptTemplate

from langchain_groq import ChatGroq



# ✅ Load environment variables (Optional, if you want to use .env for future keys)
load_dotenv()

# Load Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROG_API_KEY1")

# Initialize LLM
llm = ChatGroq(model="llama3-70b-8192")

# ✅ Design prompt to simulate Co‑Scientist behavior
prompt = PromptTemplate(
    input_variables=["research_goal"],
    template=(
        "You are an AI Co‑Scientist helping researchers consolidate medical research.\n"
        "Given the following research goal:\n\n"
        "{research_goal}\n\n"
        "Please return:\n"
        "1. A concise literature review summary.\n"
        "2. Three novel research hypotheses.\n"
        "3. A proposed experimental study or next research steps.\n"
        "Format your output in clear bullet points or numbered lists."
    ),
)

# ✅ Chain LLM with prompt
co_scientist_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# ✅ Initialize MCP server
mcp = FastMCP("LocalCoScientistToolServer")

@mcp.tool()
async def co_scientist_tool(research_goal: str) -> str:
    """
    Local AI Co-Scientist using Ollama: Returns literature summary, hypotheses, and study suggestions.
    """
    if not research_goal.strip():
        return "❌ Error: Please provide a valid research goal."
    try:
        result = co_scientist_chain.run(research_goal=research_goal)
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return f"Hello, {name}! Welcome to the Local AI CoScientist Tool powered by Ollama."

if __name__ == "__main__":
    mcp.run()