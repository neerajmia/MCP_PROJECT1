
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.tools import tool
from mcp.server.fastmcp import FastMCP


from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()

# ✅ Connect to MySQL
db = SQLDatabase.from_uri("mysql+mysqlconnector://root:Mother04%40@localhost/arcadia")

# Load Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROG_API_KEY1")

# Initialize LLM
llm = ChatGroq(model="llama3-70b-8192")

# ✅ Create SQL Chain
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# ✅ Define SQL Query Tool

## toolname
mcp = FastMCP("QueryManager")

@mcp.tool()
async def mysql_query_tool(user_question: str) -> str:
    """Agentic SQL Tool: Converts user question into SQL and returns results."""
    if not user_question.strip():
        return "Error: Please provide a valid question."
    try:
        result = db_chain.run(user_question)
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"



# Resource: Greeting
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}! How can I assist you with internal database today??"

if __name__ == "__main__":
    mcp.run()