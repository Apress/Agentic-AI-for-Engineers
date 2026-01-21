# Chapter 12: Collaborative Agents (Multi-Agent Systems & Human-AI Teaming)

This repository contains the code examples and exercises for **Chapter 12**, focusing on how **multiple agents—and humans—work together** to solve complex, real-world problems.

If single-agent systems are about individual capability, this chapter is about **teamwork**:
> **Specialization + coordination + oversight = scalable intelligence**

The notebooks in this chapter show how agentic systems evolve from solo reasoning loops into **collaborative systems** with clear roles, coordination strategies, and human supervision.

---

## Overview

Most meaningful problems are not solved by a single expert acting alone—and neither are they solved by a single agent.

In this chapter, we explore:
- **Multi-agent collaboration patterns**
- **Sequential vs parallel execution**
- **Role specialization and orchestration**
- **Human-in-the-loop and human-on-the-loop designs**

The examples demonstrate how different **organizational topologies**—borrowed from human teams—can be implemented using modern agent frameworks.

The notebooks primarily use **LangGraph** and **CrewAI** to express these collaborative structures.

---

## 📂 Code Repository Map

The notebooks are organized by **collaboration pattern**.

---

### 1. Parallel Agent Execution
*Ref: “Multiple specialists working side-by-side”*

These examples show how multiple agents can operate **concurrently**, each tackling a different aspect of a task while sharing context through a common state.

| File Name | Description | Framework |
| :--- | :--- | :--- |
| **`Parallel_Agents_using_Langgraph.ipynb`** | Demonstrates parallel task execution with shared state, enabling faster throughput and independent reasoning paths. | LangGraph |

**Why it matters:**  
Parallelism improves speed and scale, but requires careful coordination to avoid conflicts and inconsistent state.

---

### 2. Sequential & Role-Based Collaboration
*Ref: “Assembly-line intelligence”*

These notebooks implement **sequential multi-agent workflows**, where each agent has a well-defined role and hands off work to the next.

| File Name | Description | Framework |
| :--- | :--- | :--- |
| **`Sequential_Architecture_using_CrewAI.ipynb`** | A role-based pipeline where agents execute tasks in sequence (e.g., research → content creation → quality review). | CrewAI |

**Why it matters:**  
Sequential systems are easier to reason about, debug, and govern—making them ideal for high-stakes or regulated workflows.

---

### 3. Multi_Agent_Architecture_Patterns
*Ref: “Assembly-line intelligence”*

These notebooks implement **different multi-agent workflows**

| File Name | Description | Framework |
| :--- | :--- | :--- |
| **`Multi_Agent_Architecture_Patterns.ipynb`** | Walk through of 8 different multi-agent architectures. | LangChain |

**Why it matters:**  
A clear comparison of different architectures with solid examples.

---
 
## How This Chapter Fits in the Book

Chapter 12 builds directly on:
- Agent architectures (Chapter 5)
- Feedback loops and learning (Chapter 11)

It sets the stage for:
- Evaluation, observability, and deployment (Chapter 13)
- Full human-AI ecosystems in production

This chapter marks the shift from **“building agents”** to **“building teams of agents.”**

---

## Prerequisites

To run these notebooks, you will need:
1. Python 3.10+
2. API keys for your preferred LLM provider
3. Required libraries installed, for example:
   ```bash
   pip install crewai langgraph