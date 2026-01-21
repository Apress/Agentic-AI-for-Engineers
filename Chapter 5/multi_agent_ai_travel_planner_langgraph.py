from textwrap import dedent
import streamlit as st
import re
import os
from datetime import datetime, timedelta
from icalendar import Calendar, Event
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
load_dotenv()
# -------------------------------
# ⚙️  LLM Configuration (CrewAI)
# -------------------------------
# Using a small local Llama model for fast inference.
# Replace with larger models for higher-quality outputs.
llm = LLM(
    model="ollama/llama3.2:1b",
    temperature=0.3,   # keeps creativity low and structure high
    max_tokens=2048
)

# -------------------------------
# 📅  ICS GENERATOR
# -------------------------------
def generate_ics_content(plan_text: str, start_date: datetime=None) -> bytes:
    """
    Convert a day-by-day itinerary (plain text) into an .ics calendar file.

    Expected input format:
        "Day 1: ...\n Day 2: ...\n ..."

    If days are not detected, creates a single summary event.
    """
    cal = Calendar()
    cal.add('prodid', '-//AI Travel Planner//github.com//')
    cal.add('version', '2.0')

    if start_date is None:
        start_date = datetime.today()

    # Regex to extract "Day X: <content>"
    day_pattern = re.compile(
        r'Day (\d+)[:\s]+(.*?)(?=Day \d+|$)',
        re.DOTALL
    )

    days = day_pattern.findall(plan_text)

    # --- Case 1: No "Day X" structure found ---
    if not days:
        event = Event()
        event.add('summary', 'Travel Itinerary')
        event.add('description', plan_text)
        event.add('dtstart', start_date.date())
        event.add('dtend', start_date.date())
        event.add('dtstamp', datetime.now())
        cal.add_component(event)

    # --- Case 2: Build one calendar event per day ---
    else:
        for day_num, day_content in days:
            current_date = start_date + timedelta(days=int(day_num) - 1)
            event = Event()
            event.add('summary', f"Day {day_num} Itinerary")
            event.add('description', day_content.strip())
            event.add('dtstart', current_date.date())
            event.add('dtend', current_date.date())
            event.add('dtstamp', datetime.now())
            cal.add_component(event)

    return cal.to_ical()

# -------------------------------
# 🌐 STREAMLIT UI
# -------------------------------
st.title("AI Travel Planner")
st.caption("Plan your next adventure with AI Travel Planner using CrewAI agents")

# Session state for saving outputs
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None

serp_api_key = os.getenv("SERPAPI_KEY")

# -------------------------------
# 🔍 DEFINE SEARCH TOOL
# -------------------------------
from crewai.tools import tool

@tool("Performs a Google search and returns raw results.")
def search_google(query: str) -> str:
    """
    Wrapper around SerpAPI.
    Returns a JSON-like dictionary as string.
    """
    from serpapi import GoogleSearch
    params = {"q": query, "api_key": serp_api_key}
    results = GoogleSearch(params).get_dict()
    return str(results)  # ensure downstream agents cannot break formatting


# -----------------------------------------------------------
# 🤖 Researcher Agent — AIR-TIGHT PROMPT DESIGN
# -----------------------------------------------------------
researcher = Agent(
    name="Researcher",
    role="Travel Research Specialist",
    goal=dedent("""\
        Identify 10 factual, relevant, and verifiable travel attractions,
        accommodations, and activity options for the user's destination.
        Never make up hotels, landmarks, or events. Only extract information 
        that clearly exists in the search results.
    """),
    backstory=dedent("""\
        You are a meticulous travel analyst with 20+ years of global experience.
        Your specialty is extracting factual, high-confidence insights without
        hallucination. You always cite only what the search tool returns.
    """),
    tools=[search_google],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# -----------------------------------------------------------
# 🗺️ Planner Agent — AIR-TIGHT PROMPT DESIGN
# -----------------------------------------------------------
planner = Agent(
    name="Planner",
    role="Travel Itinerary Designer",
    goal=dedent("""\
        Transform research results into a structured, daily itinerary.
        
        STRICT RULES:
        - Format each day as: "Day X: <activities>"
        - Do NOT invent hotels, restaurants, or attractions.
        - Only use items explicitly found in the research results.
        - Keep activities feasible (no unrealistic travel distances).
        - Avoid time-specific claims unless confirmed (e.g., opening hours).
        - Produce only plain text. No markdown, emojis, or narrative fluff.
    """),
    backstory=dedent("""\
        A world-renowned travel itinerary architect known for efficient,
        logically structured plans that maximize user enjoyment while
        avoiding fatigue or travel overload.
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# -------------------------------
# 🧑‍💻 USER INPUTS
# -------------------------------
destination = st.text_input("Where do you want to go?")
num_days = st.number_input(
    "How many days?",
    min_value=1,
    max_value=30,
    value=7
)
special_preferences = st.text_area(
    "Any special preferences or constraints?",
    placeholder="Kid-friendly, vegetarian food, art museums, avoid crowds, etc."
)

col1, col2 = st.columns(2)

# -------------------------------
# ▶️ GENERATE ITINERARY BUTTON
# -------------------------------
with col1:
    if st.button("Generate Itinerary"):
        with st.spinner("Planning your trip..."):

            # ------------------------------------------
            # Task 1 — Research Task
            # ------------------------------------------
            research_task = Task(
                description=dedent(f"""\
                    You MUST perform a factual travel research operation 
                    for the destination: {destination}.

                    REQUIREMENTS:
                    - Search specifically for "{destination} top attractions", 
                      hotels, must-see activities, cultural highlights.  
                    - Incorporate user constraints: {special_preferences}
                    - Return EXACTLY 10 items.
                    - Each item must be factual and appear in search results.
                    - No invented names.

                    FORMAT:
                    Return a numbered list of 10 researched items.
                """),
                expected_output="A numbered list of 10 factual travel items.",
                agent=researcher
            )

            # ------------------------------------------
            # Task 2 — Planning Task
            # ------------------------------------------
            plan_task = Task(
                description=dedent(f"""\
                    Build a {num_days}-day itinerary for: {destination}.

                    INPUT:
                    Use ONLY the factual items from the research task:
                    {research_task}

                    STRICT RULES:
                    - Format exactly as:
                      Day 1: ...
                      Day 2: ...
                      ...
                    - No markdown, no emojis.
                    - Do NOT invent attractions, hotels, or events.
                    - Respect user constraints: {special_preferences}
                    - Keep each day's plan realistic and geographically feasible.
                    - Produce plain text only.
                """),
                expected_output="A plain-text, day-by-day itinerary formatted as 'Day X: ...'",
                agent=planner,
                context=[research_task]
            )

            # ------------------------------------------
            # CREW EXECUTION
            # ------------------------------------------
            crew = Crew(
                agents=[researcher, planner],
                tasks=[research_task, plan_task],
                process=Process.sequential
            )
            crew.kickoff()

            # ------------------------------------------
            # Extract plan output
            # ------------------------------------------
            if plan_task.output and hasattr(plan_task.output, 'raw'):
                st.session_state.itinerary = plan_task.output.raw
            else:
                st.error("Failed to generate itinerary.")
                st.session_state.itinerary = None

            # Render itinerary text
            if st.session_state.itinerary:
                st.text(st.session_state.itinerary)

# -------------------------------
# 📥 DOWNLOAD .ICS BUTTON
# -------------------------------
with col2:
    if st.session_state.itinerary:
        ics_content = generate_ics_content(st.session_state.itinerary)
        st.download_button(
            label="Download as Calendar (.ics)",
            data=ics_content,
            file_name="travel_itinerary.ics",
            mime="text/calendar"
        )
