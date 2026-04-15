from dotenv import load_dotenv

from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage


from nodes import react_agent_reasoning_node, tool_node

load_dotenv()

LAST = -1
AGENT_REASON = "agent_reasoning"
ACT = "act"

def should_continue(state: MessagesState):
    
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, react_agent_reasoning_node)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {ACT: ACT, END: END})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")


def main():
    response = app.invoke({"messages": [HumanMessage(content="What is the temperature in Hyderabad? List it and Triple it.")]})
    print(response['messages'][LAST].content)


if __name__ == "__main__":
    main()
