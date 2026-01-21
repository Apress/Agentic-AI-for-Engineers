import os
import uuid
import streamlit as st
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="Safe Travel Agent", page_icon="✈️")

# ==============================================================================
# 1. SETUP GRAPH (Cached Resource)
# ==============================================================================

# We use @st.cache_resource so this only runs ONCE.
# The 'memory' object will now persist across button clicks.
@st.cache_resource
def setup_graph():
    print("🔄 Initializing Graph and Memory...")
    
    # --- Tools ---
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # --- State ---
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        is_travel_related: bool

    # --- Nodes ---
    def guardrail_node(state: AgentState):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        last_message = state["messages"][-1]
        
        prompt = f"""
        Analyze query: "{last_message.content}"
        Is this related to travel, geography, or culture? 
        Reply YES or NO.
        """
        decision = llm.invoke(prompt).content.strip().upper()
        
        if "YES" in decision:
            return {"is_travel_related": True}
        else:
            return {
                "is_travel_related": False,
                "messages": [AIMessage(content="🚫 I am a Travel Agent only.")]
            }

    def agent_node(state: AgentState):
        llm = ChatOpenAI(model="gpt-4o-mini")
        llm_with_tools = llm.bind_tools(tools)
        sys_msg = SystemMessage(content="You are a helpful travel assistant.")
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    # --- Edges ---
    def check_guardrail(state: AgentState) -> Literal["agent", "__end__"]:
        return "agent" if state["is_travel_related"] else "__end__"

    builder = StateGraph(AgentState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges("guardrail", check_guardrail)
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    # --- Memory ---
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

# Load the graph (This uses the cached version)
graph = setup_graph()

# ==============================================================================
# 2. STREAMLIT UI
# ==============================================================================

st.title("✈️ AI Travel Agent")

# Session State for Thread ID (Keeps track of WHO you are)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Session State for Chat History (Visual only)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle Input
if user_input := st.chat_input("Where do you want to go?"):
    
    # 1. Show User Input
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 2. Run Graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        final_text = ""
        
        # Stream events
        # Note: We stream the graph, which reads/writes to the CACHED memory
        for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config):
            
            # Catch Refusal
            if "guardrail" in event:
                if not event["guardrail"]["is_travel_related"]:
                    final_text = event["guardrail"]["messages"][-1].content
            
            # Catch Agent Response
            if "agent" in event:
                msg = event["agent"]["messages"][-1]
                if not msg.tool_calls:
                    final_text = msg.content
                    
        message_placeholder.markdown(final_text)
        st.session_state.messages.append(AIMessage(content=final_text))