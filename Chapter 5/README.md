
# Chapter 5: Architectural Design Patterns for Agentic Systems

This repository contains the code examples and exercises for **Chapter 5**, focusing on the architectural *scaffolding* that turns Large Language Models (LLMs) into reliable, autonomous agents.

---

## Overview

In this chapter, we explore the shift from **LLM-as-a-tool** to **System-as-an-Agent**.

Rather than treating an LLM as a single callable component, we design **agentic systems** with structure, roles, and control flow. The code in this repository demonstrates a range of architectural topologies—from simple single-agent loops to complex, hierarchical multi-agent systems.

The examples utilize **LangChain**, **LangGraph**, and **CrewAI** to express these patterns in practice.

---

## 📂 Code Repository Map

The notebooks are organized by the **architectural pattern** they demonstrate.

---

### 1. The Single-Agent Loop (Minimalist Autonomy)
*Ref: Captain Byte Example*

The foundational pattern where an agent **perceives → reasons → acts**.  
We explore both stateless versions (fast, predictable) and stateful versions (memory-augmented).

| File Name | Description | Framework |
| :--- | :--- | :--- |
| **`Captain_Byte_Stateless_agent_using_Crewai.ipynb`** | A basic stateless agent that responds instantly with no context retention. | CrewAI |
| **`Caption_Byte_with_Memory_Using_Crewai.ipynb`** | The same agent augmented with Short-Term (ST) and Long-Term (LT) memory to enable continuity. | CrewAI |
| **`Single_agent_Technical_Writer_Crewai.ipynb`** | A functional single-agent workflow applied to technical writing. | CrewAI |

---

### 2. Tool-Augmented Reasoning (ReAct & Tool Use)
*Ref: “Reasoning + External Action”*

These notebooks demonstrate how to connect LLMs to the outside world (APIs, search tools) using the **ReAct** pattern:

**Reason → Act → Observe**

| File Name | Description | Framework |
| :--- | :--- | :--- |
| **`Tool_use_langgraph.ipynb`** | Basic implementation of an agent defining and invoking external tools. | LangGraph |
| **`React_framework_using_States_Langgraph.ipynb`** | ReAct pattern where reasoning traces are interleaved with tool execution. | LangGraph |
| **`Tool_use_and_tool_evaluation_llm_as_judge_using_langgraph.ipynb`** | Advanced pattern where an “LLM Judge” evaluates tool outputs before returning results. | LangGraph |

---

### 3. The Planner–Executor–Reflector (PER) Pattern
*Ref: “Think ahead, then execute”*

This architecture separates work into **three explicit roles**:
1. Planner – decomposes the task  
2. Executor – performs the steps  
3. Reflector / Validator – verifies quality  

| File Name | Description | Framework |
| :--- | :--- | :--- |
| **`Planner_Executor_Validator_using_Langgraph.ipynb`** | Breaks a complex query (e.g., market research) into a plan, executes steps, and validates results. | LangGraph |

---

### 4. Multi-Agent Topologies (Hierarchical & Sequential)
*Ref: Wiring Multiple Agents Together*

These examples show how to orchestrate **multiple agents** working together, either in a sequence or under a supervising agent.

| File Name | Description | Framework |
| :--- | :--- | :--- |
| **`Hierarchical_Content_Creator_using_Crewai.ipynb`** | A **hierarchical** (Manager–Worker) topology where an Editor agent delegates to Fetcher, Analyzer, and Compiler agents. | CrewAI |
| **`Iteration_Writer_Critique_Repeat_Loop_using_Crewai.ipynb`** | A feedback-driven loop where a Critic agent iteratively improves a Writer agent’s output. | CrewAI |
| **`Multi_agent_ai_travel_planner_langgraph.ipynb`** | A collaborative multi-agent travel planner demonstrating state passing between specialized agents. | LangGraph |

---

## Key Concepts Covered

- **Stateless vs Stateful Agents**  
  When to simply react vs when to maintain memory.

- **ReAct Loop**  
  The standard cycle of *Thought → Action → Observation → Result*.

- **Planner–Executor–Reflector (PER)**  
  Reducing fragility through explicit planning and validation.

- **Agent Topologies**
  - *Sequential:* A → B → C pipeline  
  - *Hierarchical:* Supervisor delegates to workers  
  - *Hybrid:* Combinations used in enterprise workflows  

---

## Prerequisites

To run these notebooks, you will need:

1. Python 3.10+
2. API keys for OpenAI (or your preferred LLM provider)
3. Required libraries:
   ```bash
   pip install crewai langchain langgraph
