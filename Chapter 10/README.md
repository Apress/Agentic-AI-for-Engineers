# Chapter 10: Build Your First AI Agent — Hands-on Coding

This repository contains the complete, step-by-step code examples for **Chapter 10** of *Agentic AI for Engineers*.  
The focus of this chapter is **learning by building**—constructing a real, end-to-end agentic system that evolves from a single agent into a production-grade, supervised multi-agent pipeline.

---

## Overview

Agentic AI is not learned by theory alone. It is learned by **writing code, running it, watching it fail, fixing it, and running it again**.

This chapter serves as the *“hello world”* of agentic engineering—**not trivial**, but **complete**.  
You will build a working **Research Assistant Agent** that:

- Searches the web for academic content
- Synthesizes and summarizes findings
- Critiques its own output
- Iterates until quality thresholds are met
- Applies judgment and confidence scoring
- Incorporates human approval
- Enforces guardrails before publication

The architecture mirrors how **real production agents mature**, layer by layer.

---

## 📂 Code Repository Map

The notebooks are designed to be run **sequentially**.  
Each step introduces a new capability without breaking the previous one.

| Step | File Name | Description | Key Concept |
| :--- | :--- | :--- | :--- |
| **Step 1** | `Step1_Single Agent Research Fetcher.ipynb` | A single autonomous agent that searches and retrieves relevant research papers. | **Single-Agent Autonomy** |
| **Step 2** | `Step2_Multi Agent Researcher.ipynb` | Splits responsibilities across Researcher, Summarizer, and Critic agents. | **Separation of Concerns** |
| **Step 3** | `Step3_Multi_Agent_with_Critic.ipynb` | Adds structured critique and scoring of outputs. | **Self-Evaluation** |
| **Step 4** | `Step4_Multi_Agent_with_Judge.ipynb` | Introduces a Judge LLM to assign confidence scores to final outputs. | **Judgment & Confidence Scoring** |
| **Step 5** | `Step5_Multi_Agent_with_HITL.ipynb` | Adds Human-in-the-Loop approval for high-stakes publishing. | **Human Oversight** |
| **Step 6** | `Step6_Multiagent_with_loop-guardrail-human-feedback-looping.ipynb` | Full production pipeline with retries, guardrails, and branching logic. | **Production-Grade Agentic Workflow** |

---

## Architectural Progression

This chapter deliberately mirrors how real-world agentic systems evolve:

### 1. Single-Agent Reasoning
*Ref: “Start small”*  
A single agent with a clear role, goal, and tools.  
The objective is correctness and predictability—not sophistication.

---

### 2. Multi-Agent Specialization
*Ref: “Divide cognitive labor”*  
Separate agents for:
- Research
- Summarization
- Critique

Each agent becomes easier to prompt, debug, and improve.

---

### 3. Iterative Refinement Loops
*Ref: “Quality emerges through iteration”*  
Agents revise outputs until explicit quality thresholds are met.  
This mirrors how humans write, review, and improve drafts.

---

### 4. Judge LLM
*Ref: “Editor-in-Chief”*  
A Judge LLM evaluates the **final output**, assigns a confidence score, and determines readiness for publication.

This is distinct from critique:
- **Critic:** Improves drafts
- **Judge:** Decides if the work is acceptable

---

### 5. Human-in-the-Loop (HITL)
*Ref: “Autonomy with accountability”*  
Before irreversible actions (e.g., publishing), a human reviewer:
- Approves
- Rejects
- Requests revision

Human oversight is a **design feature**, not a failure mode.

---

### 6. Guardrails & Validation
*Ref: “Fail fast, fail safely”*  
Guardrails enforce:
- Output format validation
- Length constraints
- Schema compliance
- Quality expectations

They prevent low-quality or unsafe outputs from propagating downstream.

---

## Key Concepts Covered

- **Agent Roles & Backstories:** How personality shapes behavior
- **Task Design:** Why explicit expected outputs matter
- **Sequential vs Iterative Crews**
- **Critic vs Judge LLMs**
- **Retry Logic & Thresholds**
- **Human Approval Gates**
- **Guardrails as Governance Mechanisms**
- **Cost-Aware Agent Design**

---

## Prerequisites

To run the notebooks, you will need:

1. **Python 3.10+**
2. **LLM API Key** (OpenAI or compatible provider)
3. **Search API Key** (Serper recommended)
4. Required packages:
   ```bash
   pip install crewai crewai-tools python-dotenv
