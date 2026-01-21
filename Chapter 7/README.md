# Chapter 7: Tools & Frameworks for Building Agents

This repository contains the code examples and exercises for **Chapter 7**, focusing on the "toolchain" required to build agentic AI. 

## Overview

[cite_start]In this chapter, we move beyond simple prompts to full system orchestration[cite: 486]. [cite_start]We explore the ecosystem of frameworks that provide the "scaffolding" for agents to think, plan, and coordinate[cite: 486].

The examples cover the four major orchestration patterns discussed in the book:
1.  [cite_start]**LangChain** (General purpose / Swiss-Army Knife) [cite: 810]
2.  [cite_start]**LangGraph** (State-based / Graph control flow) [cite: 911]
3.  [cite_start]**CrewAI** (Role-based / Teamwork) [cite: 1110]
4.  [cite_start]**AutoGen** (Conversational / Feedback loops) [cite: 1154]

## 📂 Code Repository Map

The notebooks are categorized by the framework and architectural pattern they demonstrate.

### 1. CrewAI: Role-Based Orchestration
*Ref: "Think in roles, tasks, and hand-offs"*

[cite_start]These examples demonstrate how to assemble a "crew" of specialized agents that collaborate sequentially or hierarchically[cite: 494].

| File Name | Description | Key Concept |
| :--- | :--- | :--- |
| **`Retail_Chatbot_agent_using_Crewai.ipynb`** | A retail support pipeline where agents specialize in different stages of resolution. | [cite_start]**Role-Based Pipelines**<br>Matches the "Retail support pipeline" discussed in the chapter[cite: 1119]. |
| **`Tools_LinkedIn_Tweet_Crewai.ipynb`** | Demonstrates creating custom tools for agents to interact with social media platforms. | [cite_start]**The Tool Layer**<br>Implementing the "Function first, wrappers second" approach[cite: 742]. |
| **`example_linkedin.txt`** / **`example_threads.txt`** | Sample data files used as inputs for the tool demonstration notebook. | **Data Source** |

### 2. AutoGen: Conversational Multi-Agent Systems
*Ref: "Think in conversations, roles, and feedback loops"*

[cite_start]This framework models workflows as a "group chat" where agents (including critics and user proxies) collaborate to solve tasks[cite: 1157].

| File Name | Description | Key Concept |
| :--- | :--- | :--- |
| **`Autogen_Reflection_and_Blogpost_Writing.ipynb`** | A writing workflow where a "Critic" agent reviews drafts and provides feedback to a "Writer" agent. | [cite_start]**Reflective Feedback**<br>Leverages AutoGen's strength in "Reviewer and critic loops"[cite: 1168]. |

### 3. LangGraph: State-Based Control Flow
*Ref: "Think in states and edges"*

[cite_start]These examples use a graph architecture to manage complex workflows with branching, loops, and persistent state[cite: 911].

| File Name | Description | Key Concept |
| :--- | :--- | :--- |
| **`Writer_critique_reflector_using_Langgraph.ipynb`** | A system implementing a cycle of writing and self-correction using explicit state management. | [cite_start]**The Reflector Pattern**<br>Demonstrates the "ReAct" or "Planner-Executor-Reflector" loops within a graph[cite: 915]. |

### 4. LangChain: The Swiss-Army Knife
*Ref: "Chains, Agents, and Tools"*

[cite_start]Foundational examples using the broad ecosystem of LangChain tools and chains[cite: 810].

| File Name | Description | Key Concept |
| :--- | :--- | :--- |
| **`Chatbot_Using_Tool_Using_Langchain_agent.ipynb`** | A basic agent implementation using the ReAct framework to call external tools. | [cite_start]**ReAct Framework**<br>The standard "Reason → Act → Observe" loop[cite: 826]. |

---

## Key Framework Comparisons

As detailed in the chapter, use this guide to understand why specific frameworks were chosen for these examples:

* [cite_start]**LangChain:** Best for general tool orchestration and rapid prototyping[cite: 1194].
* [cite_start]**LangGraph:** Best for complex branching, retries, and production-grade state management[cite: 1196].
* [cite_start]**CrewAI:** Best for intuitive, role-based teams and natural hand-offs[cite: 1197].
* [cite_start]**AutoGen:** Best for conversational workflows and code-generation tasks with feedback loops[cite: 1198].

## Prerequisites

To run these notebooks, you will need:
1.  Python 3.10+
2.  API Keys for OpenAI (or your preferred LLM provider)
3.  Installation of the specific frameworks:
    ```bash
    pip install crewai autogen pyautogen langchain langgraph
    ```

If you reference the architectural definitions in this code, please cite:
> *Agentic AI for Engineers*, Chapter 7: Tools & Frameworks for building Agents (2025).
