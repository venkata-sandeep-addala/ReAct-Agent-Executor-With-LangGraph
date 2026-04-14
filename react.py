from dotenv import load_dotenv

from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq


load_dotenv()

@tool
def triple(temperature: float) -> float:
    """Returns the triple of the given temperature.

    Args:
        temperature (float): The temperature to be tripled.

    Returns:
        float: The triple of the given temperature.
    """
    return float(temperature) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)


