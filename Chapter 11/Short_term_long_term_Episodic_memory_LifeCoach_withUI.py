import os
import shutil
from typing import TypedDict, Annotated, List
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document

# ✅ FIXED: Use standard pydantic instead of langchain_core wrapper
from pydantic import BaseModel, Field 

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ==============================================================================
# 1. MEMORY STORAGE SETUP
# ==============================================================================

# --- A. Episodic Memory (Vector DB) ---
# We use a local vector store to save "Events"
embedding_function = OpenAIEmbeddings()
PERSIST_DIR = "./chroma_memory_db"

# Cleanup previous run for fresh start (Optional)
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

vector_store = Chroma(
    collection_name="episodic_memory",
    embedding_function=embedding_function,
    persist_directory=PERSIST_DIR
)

# --- B. Long-Term Memory (Structured Profile) ---
# In production, this would be a User Table in Postgres
user_profile_db = {
    "user_123": {
        "name": "Unknown",
        "core_goals": [],
        "hobbies": []
    }
}

# ==============================================================================
# 2. STATE & SCHEMA DEFINITIONS
# ==============================================================================

class UserProfileUpdate(BaseModel):
    """Schema for extracting LTM updates from conversation"""
    name: str = Field(description="The user's name if mentioned")
    new_goals: List[str] = Field(description="Any new goals the user mentioned")
    new_hobbies: List[str] = Field(description="Any new hobbies mentioned")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # <--- STM (Short Term)
    user_id: str
    episodic_context: str  # <--- Relevant past events fetched from Vector DB
    user_profile: str      # <--- Structured facts fetched from JSON DB

# ==============================================================================
# 3. GRAPH NODES (The Brain)
# ==============================================================================

# --- NODE 1: RECALL (Fetch LTM & Episodic) ---
def recall_memory_node(state: AgentState):
    user_id = state["user_id"]
    latest_query = state["messages"][-1].content
    
    # 1. Fetch LTM (Profile)
    profile = user_profile_db.get(user_id, {})
    profile_str = f"Name: {profile.get('name')}, Goals: {profile.get('core_goals')}"
    
    # 2. Fetch Episodic (Vector Search)
    # Search for similar past conversations
    docs = vector_store.similarity_search(latest_query, k=2)
    episodic_str = "\n".join([d.page_content for d in docs]) if docs else "No relevant past memories."
    
    print(f"🧠 RECALLING:\n  - Profile: {profile_str}\n  - Past Events: {episodic_str[:50]}...")
    
    return {
        "user_profile": profile_str,
        "episodic_context": episodic_str
    }

# --- NODE 2: GENERATE (The Response) ---
def generate_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Inject ALL memories into the system prompt
    system_prompt = (
        f"You are a Life Coach. \n"
        f"--- LONG TERM INFO ---\n{state['user_profile']}\n"
        f"--- PAST RELEVANT MEMORIES ---\n{state['episodic_context']}\n"
        f"--------------------------\n"
        f"Use this context to give personalized advice."
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

# --- NODE 3: CONSOLIDATE (Save New Memories) ---
def save_memory_node(state: AgentState):
    user_id = state["user_id"]
    last_user_msg = state["messages"][-2].content # The user's input
    last_ai_msg = state["messages"][-1].content   # The AI's response
    
    # 1. Update Episodic Memory (Vector Store)
    # We save the interaction so we can find it later
    memory_text = f"User said: {last_user_msg} | AI advised: {last_ai_msg}"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    vector_store.add_documents([
        Document(page_content=memory_text, metadata={"user_id": user_id, "timestamp": current_time})
    ])
    
    # 2. Update Long Term Memory (Profile Extraction)
    # We ask a "quiet" LLM instance to extract facts
    extractor = ChatOpenAI(model="gpt-4o-mini").with_structured_output(UserProfileUpdate)
    
    extraction_prompt = f"Analyze this conversation and update the user profile: {last_user_msg}"
    update = extractor.invoke(extraction_prompt)
    
    # Update the "Database"
    current_profile = user_profile_db[user_id]
    if update.name != "Unknown":
        current_profile["name"] = update.name
    if update.new_goals:
        current_profile["core_goals"].extend(update.new_goals)
    
    print(f"💾 SAVED: Added to vector DB & Updated Profile: {current_profile}")
    return {} # No state update needed, just side effects

# ==============================================================================
# 4. BUILDING THE GRAPH
# ==============================================================================

builder = StateGraph(AgentState)

builder.add_node("recall", recall_memory_node)
builder.add_node("generate", generate_node)
builder.add_node("save", save_memory_node)

# Flow: Start -> Recall -> Generate -> Save -> End
builder.add_edge(START, "recall")
builder.add_edge("recall", "generate")
builder.add_edge("generate", "save")
builder.add_edge("save", END)

memory = MemorySaver() # This is the STM checkpointer
graph = builder.compile(checkpointer=memory)

# ==============================================================================
# 5. SIMULATION
# ==============================================================================

def chat(message, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n💬 User: {message}")
    
    # We initialize the state with user_id for DB lookups
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "user_id": "user_123" 
    }
    
    # Run the graph
    events = graph.invoke(initial_state, config)
    print(f"🤖 Coach: {events['messages'][-1].content}")

if __name__ == "__main__":
    t_id = "session_1"
    
    # Turn 1: Introduction (Sets LTM)
    chat("Hi, my name is Alex. I want to train for a marathon.", t_id)
    
    # Turn 2: Specific Event (Sets Episodic)
    chat("I went for a run today but my knee hurt really bad.", t_id)
    
    print("\n--- 🕒 TIME PASSES (New Session) ---\n")
    t_id_2 = "session_2" # New Thread ID = STM is wiped!
    
    # Turn 3: Retrieval (Tests LTM + Episodic)
    # Even though STM is gone, it should know my name (LTM) and my knee issue (Episodic)
    chat("I'm thinking of going for a sprint today. What do you think?", t_id_2)