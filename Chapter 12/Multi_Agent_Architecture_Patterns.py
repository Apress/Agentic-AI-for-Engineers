# Multi_Agent_Architecture_Patterns.py
"""
Chapter 12: Multi-Agent Collaboration Architectures

Complete code examples demonstrating all major multi-agent patterns:
1. Sequential Pipeline
2. Parallel Fan-Out / Fan-In
3. Hierarchical (Manager-Worker)
4. Debate / Adversarial
5. Voting / Consensus
6. Blackboard (Shared Memory)
7. Market-Based (Auction)
8. Supervisor with Routing

Each architecture includes:
- Clear explanation of when to use it
- Complete working code
- Example output
- Pros and cons

Usage:
    python Multi_Agent_Architecture_Patterns.py --arch sequential
    python Multi_Agent_Architecture_Patterns.py --arch all

┌────────────────────────────────────────────────────────────────────┐
│                          BLACKBOARD                                │
│                                                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  problem:   │ │ Researcher: │ │  Analyst:   │ │  Critic:    │   │
│  │ "AI agents  │ │ "Adoption   │ │ "Key trend: │ │ "Missing:   │   │
│  │ in customer │ │  rate is    │ │  a hybrid   | │  regulatory │   │
│  │  service"   │ │  34%..."    │ │  models..." │ │  concerns"  │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│                                                                    │
│  ┌─────────────┐ ┌─────────────┐                                   │
│  │ Synthesizer:│ │   status:   │                                   │
│  │ "Recommend- │ │  complete   │                                   │
│  │  ations..." │ │             │                                   │
│  └─────────────┘ └─────────────┘                                   │
└───────▲─────────────▲─────────────▲─────────────▲──────────────────┘
        │             │             │             │
   ┌────┴────-┐   ┌───┴────┐   ┌────┴────┐   ┌────┴────-┐
   │Researcher│   │ Analyst│   │ Critic  │   │Synthesic │
   │  read/   │   │  read/ │   │  read/  │   │  read/   │
   │  write   │   │  write │   │  write  │   │  write   │
   └──────────┘   └────────┘   └─────────┘   └─────────-┘
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             │
                      ┌──────▼──────┐
                      │ Controller  │
                      │ "Problem    │
                      │  solved?"   │
                      └─────────────┘
"""

import os
import json
import time
import random
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


# ==============================================================================
# SHARED INFRASTRUCTURE
# ==============================================================================

class AgentRole(Enum):
    """Common agent roles across architectures."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic"
    EDITOR = "editor"
    FACT_CHECKER = "fact_checker"
    MANAGER = "manager"
    SPECIALIST = "specialist"


@dataclass
class AgentMessage:
    """A message passed between agents."""
    sender: str
    recipient: str
    content: str
    message_type: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from an agent's work."""
    agent_name: str
    output: str
    success: bool
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Base class for all agents in our architectures."""
    
    def __init__(
        self, 
        name: str, 
        role: str, 
        system_prompt: str,
        model: str = "gpt-4o-mini"
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.message_history: List[AgentMessage] = []
    
    def process(self, input_text: str, context: Dict = None) -> AgentResult:
        """Process input and return result."""
        start_time = time.time()
        
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_prompt(input_text, context))
        ]
        
        try:
            response = self.llm.invoke(messages)
            output = response.content
            success = True
        except Exception as e:
            output = f"Error: {str(e)}"
            success = False
        
        execution_time = (time.time() - start_time) * 1000
        
        return AgentResult(
            agent_name=self.name,
            output=output,
            success=success,
            execution_time_ms=execution_time,
            metadata={"role": self.role, "context": context}
        )
    
    def _build_prompt(self, input_text: str, context: Dict = None) -> str:
        """Build the prompt with optional context."""
        prompt = input_text
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            prompt = f"Context:\n{context_str}\n\nTask:\n{input_text}"
        return prompt
    
    def __repr__(self):
        return f"Agent({self.name}, role={self.role})"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_result(result: AgentResult, verbose: bool = True):
    """Print an agent result."""
    status = "✅" if result.success else "❌"
    print(f"{status} [{result.agent_name}] ({result.execution_time_ms:.0f}ms)")
    if verbose:
        # Truncate long outputs
        output = result.output[:500] + "..." if len(result.output) > 500 else result.output
        print(f"   {output}\n")


# ==============================================================================
# ARCHITECTURE 1: SEQUENTIAL PIPELINE
# ==============================================================================
"""
SEQUENTIAL PIPELINE

    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Agent 1 │ ──► │ Agent 2 │ ──► │ Agent 3 │ ──► │ Agent 4 │
    └─────────┘     └─────────┘     └─────────┘     └─────────┘
    
When to use:
- Tasks have clear, ordered stages
- Each stage depends on the previous output
- Processing is naturally linear (research → write → edit → publish)

Pros:
- Simple to understand and debug
- Clear data flow
- Easy to add/remove stages

Cons:
- No parallelism (slow for independent tasks)
- Single point of failure at each stage
- Later agents wait for earlier ones
"""

class SequentialPipeline:
    """
    Sequential pipeline where each agent processes the output of the previous one.
    
    Example: Research → Draft → Edit → Fact-Check → Final
    """
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.results: List[AgentResult] = []
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the pipeline."""
        self.agents.append(agent)
        return self  # Allow chaining
    
    def run(self, initial_input: str) -> Dict[str, Any]:
        """Run the sequential pipeline."""
        print_section("SEQUENTIAL PIPELINE")
        print(f"📋 Pipeline: {' → '.join(a.name for a in self.agents)}")
        print(f"📝 Input: {initial_input[:100]}...\n")
        
        self.results = []
        current_input = initial_input
        
        for i, agent in enumerate(self.agents, 1):
            print(f"Stage {i}/{len(self.agents)}: {agent.name}")
            
            # Pass previous output as context
            context = {}
            if self.results:
                context["previous_output"] = self.results[-1].output
            
            result = agent.process(current_input, context)
            self.results.append(result)
            print_result(result)
            
            if not result.success:
                print(f"❌ Pipeline failed at {agent.name}")
                break
            
            # Next agent receives this output
            current_input = result.output
        
        return {
            "final_output": self.results[-1].output if self.results else "",
            "all_results": self.results,
            "success": all(r.success for r in self.results)
        }


def demo_sequential_pipeline():
    """Demonstrate the sequential pipeline architecture."""
    
    # Create specialized agents for each stage
    researcher = BaseAgent(
        name="Researcher",
        role="researcher",
        system_prompt="""You are a research specialist. Given a topic, provide 
        3-5 key facts and findings. Be concise and factual. Format as bullet points."""
    )
    
    writer = BaseAgent(
        name="Writer",
        role="writer",
        system_prompt="""You are a content writer. Take the research provided and 
        write a short, engaging paragraph (3-4 sentences) summarizing the key points. 
        Make it accessible to a general audience."""
    )
    
    editor = BaseAgent(
        name="Editor",
        role="editor",
        system_prompt="""You are an editor. Review the text for clarity, grammar, 
        and flow. Make improvements and return the polished version. 
        Keep the same length but improve quality."""
    )
    
    fact_checker = BaseAgent(
        name="FactChecker",
        role="fact_checker",
        system_prompt="""You are a fact checker. Review the content for accuracy.
        If everything looks accurate, return the content unchanged with a note 
        "✓ Verified". If you find issues, note them and suggest corrections."""
    )
    
    # Build and run pipeline
    pipeline = SequentialPipeline()
    pipeline.add_agent(researcher)\
            .add_agent(writer)\
            .add_agent(editor)\
            .add_agent(fact_checker)
    
    result = pipeline.run("The impact of artificial intelligence on healthcare")
    
    print_section("FINAL OUTPUT")
    print(result["final_output"])
    
    return result


# ==============================================================================
# ARCHITECTURE 2: PARALLEL FAN-OUT / FAN-IN
# ==============================================================================
"""
PARALLEL FAN-OUT / FAN-IN

                    ┌─────────┐
                ┌──►│ Agent A │──┐
                │   └─────────┘  │
    ┌───────┐   │   ┌─────────┐  │   ┌───────────┐
    │ Input │───┼──►│ Agent B │──┼──►│ Aggregator│
    └───────┘   │   └─────────┘  │   └───────────┘
                │   ┌─────────┐  │
                └──►│ Agent C │──┘
                    └─────────┘
    
When to use:
- Multiple independent analyses needed
- Tasks can run in parallel
- Need diverse perspectives on same input

Pros:
- Fast (parallel execution)
- Multiple perspectives
- Fault tolerant (one failure doesn't stop others)

Cons:
- Need good aggregation strategy
- Resource intensive (multiple LLM calls at once)
- Results may conflict
"""

class ParallelFanOutFanIn:
    """
    Parallel execution with aggregation.
    
    Multiple agents process the same input independently,
    then an aggregator combines their outputs.
    """
    
    def __init__(self, aggregator: BaseAgent = None, max_workers: int = 5):
        self.parallel_agents: List[BaseAgent] = []
        self.aggregator = aggregator
        self.max_workers = max_workers
    
    def add_parallel_agent(self, agent: BaseAgent):
        """Add an agent to run in parallel."""
        self.parallel_agents.append(agent)
        return self
    
    def set_aggregator(self, agent: BaseAgent):
        """Set the aggregator agent."""
        self.aggregator = agent
        return self
    
    def run(self, input_text: str) -> Dict[str, Any]:
        """Run parallel agents and aggregate results."""
        print_section("PARALLEL FAN-OUT / FAN-IN")
        print(f"📋 Parallel Agents: {[a.name for a in self.parallel_agents]}")
        print(f"📋 Aggregator: {self.aggregator.name if self.aggregator else 'None'}")
        print(f"📝 Input: {input_text[:100]}...\n")
        
        # Fan-out: Run all agents in parallel
        print("⚡ FAN-OUT: Running agents in parallel...")
        parallel_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(agent.process, input_text): agent 
                for agent in self.parallel_agents
            }
            
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    parallel_results.append(result)
                    print_result(result, verbose=False)
                except Exception as e:
                    print(f"❌ {agent.name} failed: {e}")
        
        # Fan-in: Aggregate results
        print("\n🔄 FAN-IN: Aggregating results...")
        
        if self.aggregator and parallel_results:
            # Prepare aggregation input
            aggregation_input = "Synthesize these different perspectives:\n\n"
            for i, result in enumerate(parallel_results, 1):
                aggregation_input += f"--- Perspective {i} ({result.agent_name}) ---\n"
                aggregation_input += f"{result.output}\n\n"
            
            final_result = self.aggregator.process(aggregation_input)
            print_result(final_result)
            
            return {
                "final_output": final_result.output,
                "parallel_results": parallel_results,
                "aggregated_result": final_result,
                "success": final_result.success
            }
        
        return {
            "final_output": "\n\n".join(r.output for r in parallel_results),
            "parallel_results": parallel_results,
            "success": all(r.success for r in parallel_results)
        }


def demo_parallel_fan_out():
    """Demonstrate parallel fan-out/fan-in architecture."""
    
    # Create agents with different perspectives
    optimist = BaseAgent(
        name="Optimist",
        role="analyst",
        system_prompt="""You analyze topics from an optimistic perspective. 
        Focus on opportunities, benefits, and positive potential.
        Be specific but brief (2-3 sentences)."""
    )
    
    pessimist = BaseAgent(
        name="Pessimist", 
        role="analyst",
        system_prompt="""You analyze topics from a cautious/pessimistic perspective.
        Focus on risks, challenges, and potential downsides.
        Be specific but brief (2-3 sentences)."""
    )
    
    pragmatist = BaseAgent(
        name="Pragmatist",
        role="analyst", 
        system_prompt="""You analyze topics from a practical perspective.
        Focus on realistic implementation and trade-offs.
        Be specific but brief (2-3 sentences)."""
    )
    
    historian = BaseAgent(
        name="Historian",
        role="analyst",
        system_prompt="""You analyze topics from a historical perspective.
        What similar situations happened before? What can we learn?
        Be specific but brief (2-3 sentences)."""
    )
    
    aggregator = BaseAgent(
        name="Synthesizer",
        role="aggregator",
        system_prompt="""You synthesize multiple perspectives into a balanced analysis.
        Acknowledge different viewpoints and provide a nuanced conclusion.
        Create a cohesive summary (1 paragraph)."""
    )
    
    # Build and run
    parallel_system = ParallelFanOutFanIn()
    parallel_system.add_parallel_agent(optimist)\
                   .add_parallel_agent(pessimist)\
                   .add_parallel_agent(pragmatist)\
                   .add_parallel_agent(historian)\
                   .set_aggregator(aggregator)
    
    result = parallel_system.run("The rise of autonomous AI agents in business")
    
    print_section("FINAL SYNTHESIZED OUTPUT")
    print(result["final_output"])
    
    return result


# ==============================================================================
# ARCHITECTURE 3: HIERARCHICAL (MANAGER-WORKER)
# ==============================================================================
"""
HIERARCHICAL (MANAGER-WORKER)

                    ┌─────────┐
                    │ Manager │
                    └────┬────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │Worker A │   │Worker B │   │Worker C │
      └─────────┘   └─────────┘   └─────────┘
    
When to use:
- Complex tasks that need decomposition
- Different specialists for different subtasks
- Need coordination and quality control

Pros:
- Good for complex, multi-part tasks
- Manager can adapt strategy based on results
- Clear accountability

Cons:
- Manager is a bottleneck
- More complex to implement
- Manager quality affects everything
"""

class TaskAssignment(BaseModel):
    """Schema for manager's task assignment."""
    subtasks: List[Dict[str, str]] = Field(
        description="List of subtasks with 'worker' and 'task' fields"
    )
    reasoning: str = Field(description="Why tasks were assigned this way")


class HierarchicalSystem:
    """
    Manager-worker hierarchical system.
    
    A manager decomposes tasks and assigns them to specialist workers,
    then synthesizes their outputs.
    """
    
    def __init__(self):
        self.manager: BaseAgent = None
        self.workers: Dict[str, BaseAgent] = {}
    
    def set_manager(self, agent: BaseAgent):
        """Set the manager agent."""
        self.manager = agent
        return self
    
    def add_worker(self, name: str, agent: BaseAgent):
        """Add a worker agent."""
        self.workers[name] = agent
        return self
    
    def run(self, task: str) -> Dict[str, Any]:
        """Run the hierarchical system."""
        print_section("HIERARCHICAL (MANAGER-WORKER)")
        print(f"👔 Manager: {self.manager.name}")
        print(f"👷 Workers: {list(self.workers.keys())}")
        print(f"📝 Task: {task[:100]}...\n")
        
        # Step 1: Manager decomposes the task
        print("📋 Step 1: Manager decomposing task...")
        
        decomposition_prompt = f"""Decompose this task into subtasks for your team.

Available workers and their specialties:
{self._describe_workers()}

Task: {task}

Assign each subtask to the most appropriate worker.
Return a JSON object with:
- "subtasks": list of {{"worker": "worker_name", "task": "specific task description"}}
- "reasoning": why you assigned tasks this way

Only use workers from the available list."""
        
        manager_result = self.manager.process(decomposition_prompt)
        print_result(manager_result, verbose=False)
        
        # Parse task assignments
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', manager_result.output, re.DOTALL)
            if json_match:
                assignments = json.loads(json_match.group())
            else:
                # Fallback: assign to all workers
                assignments = {
                    "subtasks": [{"worker": w, "task": task} for w in self.workers.keys()],
                    "reasoning": "Default assignment to all workers"
                }
        except:
            assignments = {
                "subtasks": [{"worker": w, "task": task} for w in self.workers.keys()],
                "reasoning": "Fallback assignment"
            }
        
        print(f"   Assignments: {len(assignments['subtasks'])} subtasks")
        
        # Step 2: Workers execute their tasks
        print("\n👷 Step 2: Workers executing tasks...")
        worker_results = {}
        
        for subtask in assignments.get("subtasks", []):
            worker_name = subtask.get("worker", "")
            task_desc = subtask.get("task", "")
            
            if worker_name in self.workers:
                print(f"   → {worker_name}: {task_desc[:50]}...")
                result = self.workers[worker_name].process(task_desc)
                worker_results[worker_name] = result
                print_result(result, verbose=False)
            else:
                print(f"   ⚠️ Unknown worker: {worker_name}")
        
        # Step 3: Manager synthesizes results
        print("\n📊 Step 3: Manager synthesizing results...")
        
        synthesis_prompt = f"""Synthesize these worker outputs into a final response.

Original task: {task}

Worker outputs:
"""
        for worker_name, result in worker_results.items():
            synthesis_prompt += f"\n--- {worker_name} ---\n{result.output}\n"
        
        synthesis_prompt += "\nCreate a cohesive final response that integrates all contributions."
        
        final_result = self.manager.process(synthesis_prompt)
        print_result(final_result)
        
        return {
            "final_output": final_result.output,
            "task_assignments": assignments,
            "worker_results": worker_results,
            "success": final_result.success
        }
    
    def _describe_workers(self) -> str:
        """Generate description of available workers."""
        descriptions = []
        for name, agent in self.workers.items():
            descriptions.append(f"- {name}: {agent.role}")
        return "\n".join(descriptions)


def demo_hierarchical():
    """Demonstrate hierarchical manager-worker architecture."""
    
    # Create manager
    manager = BaseAgent(
        name="ProjectManager",
        role="manager",
        system_prompt="""You are a project manager who decomposes complex tasks 
        and coordinates specialists. You assign work based on each worker's 
        expertise and synthesize their outputs into cohesive deliverables."""
    )
    
    # Create specialist workers
    researcher = BaseAgent(
        name="researcher",
        role="Research Specialist",
        system_prompt="""You are a research specialist. Gather facts, data, and 
        background information on the assigned topic. Be thorough but concise.
        Provide 3-5 key findings."""
    )
    
    analyst = BaseAgent(
        name="analyst",
        role="Data Analyst",
        system_prompt="""You are a data analyst. Analyze trends, patterns, and 
        implications. Provide quantitative insights where possible.
        Focus on actionable analysis."""
    )
    
    writer = BaseAgent(
        name="writer",
        role="Content Writer",
        system_prompt="""You are a content writer. Create clear, engaging content
        based on the assigned topic. Focus on readability and impact."""
    )
    
    critic = BaseAgent(
        name="critic",
        role="Critical Reviewer",
        system_prompt="""You are a critical reviewer. Identify weaknesses, gaps,
        and areas for improvement. Be constructive but thorough."""
    )
    
    # Build system
    system = HierarchicalSystem()
    system.set_manager(manager)\
          .add_worker("researcher", researcher)\
          .add_worker("analyst", analyst)\
          .add_worker("writer", writer)\
          .add_worker("critic", critic)
    
    result = system.run(
        "Create a brief analysis of how generative AI is transforming software development"
    )
    
    print_section("FINAL OUTPUT")
    print(result["final_output"])
    
    return result


# ==============================================================================
# ARCHITECTURE 4: DEBATE / ADVERSARIAL
# ==============================================================================
"""
DEBATE / ADVERSARIAL

    ┌─────────┐         ┌─────────┐
    │ Agent A │◄───────►│ Agent B │
    │  (Pro)  │ Debate  │ (Con)   │
    └─────────┘         └─────────┘
          │                 │
          └────────┬────────┘
                   ▼
             ┌───────────┐
             │   Judge   │
             └───────────┘
    
When to use:
- Need to explore multiple sides of an issue
- Want to stress-test ideas
- Decision-making with trade-offs

Pros:
- Surfaces hidden assumptions
- Produces more balanced output
- Good for complex decisions

Cons:
- Can be slow (multiple rounds)
- May not converge
- Debate quality depends on agent capabilities
"""

class DebateSystem:
    """
    Adversarial debate system.
    
    Two agents debate a topic, with a judge making the final decision.
    """
    
    def __init__(self, max_rounds: int = 3):
        self.pro_agent: BaseAgent = None
        self.con_agent: BaseAgent = None
        self.judge: BaseAgent = None
        self.max_rounds = max_rounds
        self.debate_history: List[Dict] = []
    
    def set_pro_agent(self, agent: BaseAgent):
        self.pro_agent = agent
        return self
    
    def set_con_agent(self, agent: BaseAgent):
        self.con_agent = agent
        return self
    
    def set_judge(self, agent: BaseAgent):
        self.judge = agent
        return self
    
    def run(self, topic: str) -> Dict[str, Any]:
        """Run the debate."""
        print_section("DEBATE / ADVERSARIAL")
        print(f"🎤 Pro: {self.pro_agent.name}")
        print(f"🎤 Con: {self.con_agent.name}")
        print(f"⚖️ Judge: {self.judge.name}")
        print(f"📝 Topic: {topic}")
        print(f"🔄 Max Rounds: {self.max_rounds}\n")
        
        self.debate_history = []
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n{'─'*50}")
            print(f"ROUND {round_num}")
            print(f"{'─'*50}")
            
            # Pro argument
            pro_context = self._build_context("pro", topic)
            pro_result = self.pro_agent.process(
                f"Present your argument FOR: {topic}",
                context=pro_context
            )
            self.debate_history.append({
                "round": round_num,
                "side": "pro",
                "argument": pro_result.output
            })
            print(f"\n✅ PRO ({self.pro_agent.name}):")
            print(f"   {pro_result.output[:300]}...")
            
            # Con argument
            con_context = self._build_context("con", topic)
            con_result = self.con_agent.process(
                f"Present your argument AGAINST: {topic}",
                context=con_context
            )
            self.debate_history.append({
                "round": round_num,
                "side": "con", 
                "argument": con_result.output
            })
            print(f"\n❌ CON ({self.con_agent.name}):")
            print(f"   {con_result.output[:300]}...")
        
        # Judge makes final decision
        print(f"\n{'─'*50}")
        print("JUDGMENT")
        print(f"{'─'*50}")
        
        judgment_prompt = self._build_judgment_prompt(topic)
        judgment_result = self.judge.process(judgment_prompt)
        
        print(f"\n⚖️ JUDGE ({self.judge.name}):")
        print(f"   {judgment_result.output}")
        
        return {
            "final_output": judgment_result.output,
            "debate_history": self.debate_history,
            "topic": topic,
            "success": judgment_result.success
        }
    
    def _build_context(self, side: str, topic: str) -> Dict:
        """Build context from debate history for an agent."""
        context = {"topic": topic}
        
        # Add opponent's previous arguments
        opponent_side = "con" if side == "pro" else "pro"
        opponent_args = [
            h["argument"] for h in self.debate_history 
            if h["side"] == opponent_side
        ]
        
        if opponent_args:
            context["opponent_previous_arguments"] = "\n\n".join(opponent_args)
            context["instruction"] = "Respond to and counter the opponent's arguments while strengthening your position."
        
        return context
    
    def _build_judgment_prompt(self, topic: str) -> str:
        """Build the prompt for the judge."""
        prompt = f"""You are judging a debate on: {topic}

Here is the full debate:

"""
        for entry in self.debate_history:
            side = "PRO" if entry["side"] == "pro" else "CON"
            prompt += f"\n[Round {entry['round']} - {side}]\n{entry['argument']}\n"
        
        prompt += """

Based on the arguments presented, provide:
1. A summary of the strongest points from each side
2. Your assessment of which side made the more compelling case
3. A balanced conclusion that acknowledges the nuances

Be fair and analytical in your judgment."""
        
        return prompt


def demo_debate():
    """Demonstrate debate/adversarial architecture."""
    
    pro_agent = BaseAgent(
        name="TechAdvocate",
        role="pro_debater",
        system_prompt="""You argue IN FAVOR of technology and progress.
        Make compelling arguments with evidence and logic.
        Anticipate and counter opposing arguments.
        Be persuasive but fair. Keep arguments to 2-3 key points."""
    )
    
    con_agent = BaseAgent(
        name="CautiousSkeptic",
        role="con_debater",
        system_prompt="""You argue AGAINST or express caution about technology claims.
        Highlight risks, unintended consequences, and overlooked factors.
        Counter the opponent's arguments with evidence.
        Be persuasive but fair. Keep arguments to 2-3 key points."""
    )
    
    judge = BaseAgent(
        name="NeutralJudge",
        role="judge",
        system_prompt="""You are a neutral judge evaluating debates.
        Assess arguments based on logic, evidence, and persuasiveness.
        Acknowledge valid points from both sides.
        Provide a balanced, well-reasoned verdict."""
    )
    
    debate = DebateSystem(max_rounds=2)
    debate.set_pro_agent(pro_agent)\
          .set_con_agent(con_agent)\
          .set_judge(judge)
    
    result = debate.run(
        "AI agents should be given more autonomy in making business decisions"
    )
    
    return result


# ==============================================================================
# ARCHITECTURE 5: VOTING / CONSENSUS
# ==============================================================================
"""
VOTING / CONSENSUS

    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Agent 1 │   │ Agent 2 │   │ Agent 3 │
    └────┬────┘   └────┬────┘   └────┬────┘
         │             │             │
         ▼             ▼             ▼
        Vote         Vote          Vote
         │             │             │
         └─────────────┼─────────────┘
                       ▼
                  ┌─────────┐
                  │ Tally & │
                  │ Decide  │
                  └─────────┘
    
When to use:
- Need high reliability/confidence
- Reducing individual agent errors
- Democratic decision making

Pros:
- More reliable than single agent
- Catches individual errors
- Builds confidence in results

Cons:
- Expensive (multiple LLM calls)
- Slow (wait for all votes)
- May still fail if agents have correlated errors
"""

class VoteResponse(BaseModel):
    """Schema for agent voting response."""
    decision: str = Field(description="The decision or answer")
    confidence: float = Field(description="Confidence 0-1")
    reasoning: str = Field(description="Brief reasoning")


class VotingSystem:
    """
    Voting/consensus system.
    
    Multiple agents vote on a decision, and the system aggregates votes.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.voters: List[BaseAgent] = []
        self.threshold = threshold  # For majority
    
    def add_voter(self, agent: BaseAgent):
        """Add a voting agent."""
        self.voters.append(agent)
        return self
    
    def run(self, question: str, options: List[str] = None) -> Dict[str, Any]:
        """Run the voting process."""
        print_section("VOTING / CONSENSUS")
        print(f"🗳️ Voters: {[v.name for v in self.voters]}")
        print(f"❓ Question: {question}")
        if options:
            print(f"📋 Options: {options}")
        print()
        
        votes = []
        
        # Collect votes
        print("Collecting votes...")
        for voter in self.voters:
            vote_prompt = f"""Question: {question}

{"Options: " + ", ".join(options) if options else "Provide your answer."}

Respond with:
1. Your decision/answer
2. Your confidence (0-1)
3. Brief reasoning (1-2 sentences)"""
            
            result = voter.process(vote_prompt)
            
            # Parse vote (simplified - in production use structured output)
            vote = {
                "voter": voter.name,
                "raw_response": result.output,
                "success": result.success
            }
            votes.append(vote)
            
            print(f"   🗳️ {voter.name}: {result.output[:100]}...")
        
        # Tally votes
        print("\n📊 Tallying votes...")
        tally = self._tally_votes(votes, options)
        
        # Determine winner
        winner = max(tally.items(), key=lambda x: x[1]["count"]) if tally else ("No consensus", {"count": 0})
        
        print(f"\n📊 Results:")
        for option, data in tally.items():
            bar = "█" * int(data["percentage"] * 20)
            print(f"   {option}: {bar} {data['count']}/{len(votes)} ({data['percentage']:.0%})")
        
        consensus_reached = winner[1]["count"] / len(votes) >= self.threshold
        
        print(f"\n{'✅' if consensus_reached else '⚠️'} Winner: {winner[0]} ({winner[1]['count']}/{len(votes)} votes)")
        
        return {
            "final_output": winner[0],
            "votes": votes,
            "tally": tally,
            "consensus_reached": consensus_reached,
            "success": consensus_reached
        }
    
    def _tally_votes(self, votes: List[Dict], options: List[str] = None) -> Dict:
        """Tally the votes."""
        tally = {}
        
        for vote in votes:
            response = vote["raw_response"].lower()
            
            if options:
                # Match against known options
                for option in options:
                    if option.lower() in response:
                        if option not in tally:
                            tally[option] = {"count": 0, "voters": []}
                        tally[option]["count"] += 1
                        tally[option]["voters"].append(vote["voter"])
                        break
            else:
                # Extract first significant word/phrase as the vote
                # This is simplified - production would use structured output
                first_line = response.split("\n")[0][:50]
                if first_line not in tally:
                    tally[first_line] = {"count": 0, "voters": []}
                tally[first_line]["count"] += 1
                tally[first_line]["voters"].append(vote["voter"])
        
        # Calculate percentages
        total = len(votes)
        for option in tally:
            tally[option]["percentage"] = tally[option]["count"] / total if total > 0 else 0
        
        return tally


def demo_voting():
    """Demonstrate voting/consensus architecture."""
    
    # Create diverse voters
    voter1 = BaseAgent(
        name="Analyst1",
        role="voter",
        system_prompt="""You are a careful analyst who votes based on data and logic.
        Consider all options thoroughly before deciding."""
    )
    
    voter2 = BaseAgent(
        name="Analyst2",
        role="voter",
        system_prompt="""You are an experienced analyst who votes based on practical considerations.
        Focus on what's most likely to succeed in practice."""
    )
    
    voter3 = BaseAgent(
        name="Analyst3",
        role="voter",
        system_prompt="""You are a risk-aware analyst who considers potential downsides.
        Vote for options with the best risk/reward balance."""
    )
    
    voter4 = BaseAgent(
        name="Analyst4",
        role="voter",
        system_prompt="""You are an innovative analyst who values new approaches.
        Consider both traditional and novel solutions."""
    )
    
    voter5 = BaseAgent(
        name="Analyst5",
        role="voter",
        system_prompt="""You are a pragmatic analyst focused on implementation.
        Vote for options that are most feasible to execute."""
    )
    
    # Build system
    voting = VotingSystem(threshold=0.5)
    voting.add_voter(voter1)\
          .add_voter(voter2)\
          .add_voter(voter3)\
          .add_voter(voter4)\
          .add_voter(voter5)
    
    result = voting.run(
        question="What's the best approach for a company starting with AI agents?",
        options=[
            "Start with a single simple agent and iterate",
            "Build a comprehensive multi-agent system from the start",
            "Use an off-the-shelf agent platform",
            "Hire consultants to build custom solution"
        ]
    )
    
    return result


# ==============================================================================
# ARCHITECTURE 6: BLACKBOARD (SHARED MEMORY)
# ==============================================================================
"""
BLACKBOARD (SHARED MEMORY)

    ┌─────────────────────────────────────┐
    │           BLACKBOARD                │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
    │  │Data1│ │Data2│ │Data3│ │Data4│   │
    │  └─────┘ └─────┘ └─────┘ └─────┘   │
    └────▲─────────▲─────────▲─────────▲─┘
         │         │         │         │
    ┌────┴───┐ ┌───┴────┐ ┌──┴─────┐ ┌─┴──────┐
    │Agent A │ │Agent B │ │Agent C │ │Agent D │
    └────────┘ └────────┘ └────────┘ └────────┘
    
When to use:
- Complex problems requiring shared state
- Agents need to build on each other's work
- Incremental refinement of a solution

Pros:
- Flexible collaboration
- Agents can work asynchronously
- Supports incremental progress

Cons:
- Need careful concurrency management
- Can become chaotic without structure
- Harder to trace causality
"""

class Blackboard:
    """Shared memory space for agents."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.history: List[Dict] = []
    
    def write(self, key: str, value: Any, author: str):
        """Write data to the blackboard."""
        self.data[key] = value
        self.history.append({
            "action": "write",
            "key": key,
            "author": author,
            "timestamp": datetime.now().isoformat()
        })
    
    def read(self, key: str = None) -> Any:
        """Read from the blackboard."""
        if key:
            return self.data.get(key)
        return self.data.copy()
    
    def get_summary(self) -> str:
        """Get a text summary of blackboard contents."""
        if not self.data:
            return "Blackboard is empty."
        
        summary = "Current Blackboard State:\n"
        for key, value in self.data.items():
            value_str = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            summary += f"\n[{key}]\n{value_str}\n"
        return summary


class BlackboardSystem:
    """
    Blackboard architecture with shared memory.
    
    Agents read from and write to a shared blackboard,
    building on each other's contributions.
    """
    
    def __init__(self, max_iterations: int = 5):
        self.blackboard = Blackboard()
        self.agents: List[BaseAgent] = []
        self.controller: BaseAgent = None
        self.max_iterations = max_iterations
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent that can read/write to blackboard."""
        self.agents.append(agent)
        return self
    
    def set_controller(self, agent: BaseAgent):
        """Set the controller that decides when to stop."""
        self.controller = agent
        return self
    
    def run(self, initial_problem: str) -> Dict[str, Any]:
        """Run the blackboard system."""
        print_section("BLACKBOARD (SHARED MEMORY)")
        print(f"👥 Agents: {[a.name for a in self.agents]}")
        print(f"🎮 Controller: {self.controller.name if self.controller else 'Round-robin'}")
        print(f"📝 Problem: {initial_problem[:100]}...")
        print(f"🔄 Max Iterations: {self.max_iterations}\n")
        
        # Initialize blackboard
        self.blackboard.write("problem", initial_problem, "system")
        self.blackboard.write("status", "in_progress", "system")
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'─'*50}")
            print(f"ITERATION {iteration}")
            print(f"{'─'*50}")
            
            # Each agent gets a turn to contribute
            for agent in self.agents:
                # Agent reads blackboard and decides contribution
                blackboard_state = self.blackboard.get_summary()
                
                prompt = f"""You are collaborating on a problem via a shared blackboard.

{blackboard_state}

Your role: {agent.role}

Review the blackboard and make a contribution:
- Add new insights, analysis, or solutions
- Build on what others have contributed
- Identify gaps or issues in existing content

Provide your contribution in a clear, focused format.
If you have nothing new to add, say "PASS"."""
                
                result = agent.process(prompt)
                
                if "PASS" not in result.output.upper():
                    # Write contribution to blackboard
                    key = f"{agent.name}_contribution_{iteration}"
                    self.blackboard.write(key, result.output, agent.name)
                    print(f"✍️ {agent.name} contributed: {result.output[:100]}...")
                else:
                    print(f"⏭️ {agent.name} passed")
            
            # Controller checks if we should stop
            if self.controller:
                check_prompt = f"""Review the blackboard state and decide if the problem is solved.

{self.blackboard.get_summary()}

Respond with either:
- "CONTINUE" if more work is needed
- "COMPLETE" if the problem is adequately solved

Then briefly explain why."""
                
                control_result = self.controller.process(check_prompt)
                
                if "COMPLETE" in control_result.output.upper():
                    print(f"\n🎮 Controller: Problem solved!")
                    self.blackboard.write("status", "complete", "controller")
                    break
                else:
                    print(f"\n🎮 Controller: Continuing...")
        
        # Generate final summary
        print(f"\n{'─'*50}")
        print("FINAL BLACKBOARD STATE")
        print(f"{'─'*50}")
        print(self.blackboard.get_summary())
        
        return {
            "final_output": self.blackboard.get_summary(),
            "blackboard_data": self.blackboard.data,
            "history": self.blackboard.history,
            "iterations": iteration,
            "success": self.blackboard.read("status") == "complete"
        }


def demo_blackboard():
    """Demonstrate blackboard architecture."""
    
    # Create specialist agents
    researcher = BaseAgent(
        name="Researcher",
        role="Finds facts and data",
        system_prompt="""You research and provide factual information.
        Add relevant facts, statistics, or background to the blackboard.
        Build on existing research, don't repeat it."""
    )
    
    analyst = BaseAgent(
        name="Analyst",
        role="Analyzes implications",
        system_prompt="""You analyze information and draw insights.
        Look at facts on the blackboard and explain what they mean.
        Identify patterns, trends, and implications."""
    )
    
    critic = BaseAgent(
        name="Critic",
        role="Identifies problems",
        system_prompt="""You critically evaluate contributions.
        Find gaps, inconsistencies, or weak arguments on the blackboard.
        Suggest what's missing or needs improvement."""
    )
    
    synthesizer = BaseAgent(
        name="Synthesizer",
        role="Combines insights",
        system_prompt="""You synthesize multiple contributions.
        Combine insights from the blackboard into coherent conclusions.
        Create actionable recommendations."""
    )
    
    controller = BaseAgent(
        name="Controller",
        role="Decides completion",
        system_prompt="""You evaluate if a problem has been adequately solved.
        Look for: sufficient research, good analysis, addressed criticisms, clear conclusions.
        Be reasonably demanding but not perfectionist."""
    )
    
    # Build system
    system = BlackboardSystem(max_iterations=3)
    system.add_agent(researcher)\
          .add_agent(analyst)\
          .add_agent(critic)\
          .add_agent(synthesizer)\
          .set_controller(controller)
    
    result = system.run(
        "What are the key considerations for deploying AI agents in customer service?"
    )
    
    return result


# ==============================================================================
# ARCHITECTURE 7: MARKET-BASED (AUCTION)
# ==============================================================================
"""
MARKET-BASED (AUCTION)

    ┌────────────────────────────────────────┐
    │            AUCTIONEER                  │
    │         "Who can do this task?"        │
    └────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Agent A │  │ Agent B │  │ Agent C │
    │ Bid: 90 │  │ Bid: 75 │  │ Bid: 85 │
    └─────────┘  └─────────┘  └─────────┘
         │
         └──► Winner executes task
    
When to use:
- Agents have different capabilities/costs
- Want optimal task assignment
- Dynamic resource allocation

Pros:
- Efficient resource allocation
- Self-organizing
- Handles varying agent capabilities

Cons:
- Complex bidding logic
- May not optimize global objectives
- Gaming/manipulation possible
"""

class Bid(BaseModel):
    """Schema for agent bid."""
    confidence: float = Field(description="Confidence in completing task well (0-1)")
    estimated_quality: float = Field(description="Expected output quality (0-1)")
    reasoning: str = Field(description="Why this agent is suited for the task")


class MarketBasedSystem:
    """
    Market-based/auction system.
    
    Tasks are auctioned to agents who bid based on their
    confidence and capabilities.
    """
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.task_history: List[Dict] = []
    
    def add_agent(self, agent: BaseAgent, specialty: str = "general"):
        """Add an agent with optional specialty."""
        agent.specialty = specialty
        self.agents.append(agent)
        return self
    
    def run(self, task: str) -> Dict[str, Any]:
        """Run the auction for a task."""
        print_section("MARKET-BASED (AUCTION)")
        print(f"👥 Agents: {[(a.name, a.specialty) for a in self.agents]}")
        print(f"📝 Task: {task[:100]}...\n")
        
        # Step 1: Collect bids
        print("📢 AUCTION: Collecting bids...")
        bids = []
        
        for agent in self.agents:
            bid_prompt = f"""You are bidding on a task in an auction.

Task: {task}

Your specialty: {agent.specialty}

Evaluate whether you're suited for this task and submit a bid:
1. Confidence (0-1): How confident are you that you can do this well?
2. Estimated Quality (0-1): What quality output can you produce?
3. Reasoning: Why are you suited (or not) for this task?

Be honest - bidding high on unsuitable tasks wastes resources.
Respond in format:
Confidence: X.X
Quality: X.X
Reasoning: ..."""
            
            result = agent.process(bid_prompt)
            
            # Parse bid (simplified)
            try:
                lines = result.output.lower().split("\n")
                confidence = 0.5
                quality = 0.5
                
                for line in lines:
                    if "confidence" in line:
                        nums = [float(s) for s in line.split() if s.replace(".", "").isdigit()]
                        if nums:
                            confidence = min(1.0, max(0.0, nums[0]))
                    elif "quality" in line:
                        nums = [float(s) for s in line.split() if s.replace(".", "").isdigit()]
                        if nums:
                            quality = min(1.0, max(0.0, nums[0]))
                
                bid = {
                    "agent": agent,
                    "confidence": confidence,
                    "quality": quality,
                    "score": (confidence + quality) / 2,
                    "raw_response": result.output
                }
            except:
                bid = {
                    "agent": agent,
                    "confidence": 0.5,
                    "quality": 0.5,
                    "score": 0.5,
                    "raw_response": result.output
                }
            
            bids.append(bid)
            print(f"   🎯 {agent.name}: confidence={bid['confidence']:.2f}, quality={bid['quality']:.2f}, score={bid['score']:.2f}")
        
        # Step 2: Select winner
        print("\n🏆 WINNER SELECTION...")
        winner_bid = max(bids, key=lambda b: b["score"])
        winner = winner_bid["agent"]
        
        print(f"   Winner: {winner.name} (score: {winner_bid['score']:.2f})")
        
        # Step 3: Winner executes task
        print(f"\n⚡ EXECUTION: {winner.name} working on task...")
        
        execution_result = winner.process(task)
        print_result(execution_result)
        
        # Record
        self.task_history.append({
            "task": task,
            "bids": bids,
            "winner": winner.name,
            "result": execution_result
        })
        
        return {
            "final_output": execution_result.output,
            "winner": winner.name,
            "winning_bid": winner_bid,
            "all_bids": bids,
            "success": execution_result.success
        }


def demo_market_based():
    """Demonstrate market-based/auction architecture."""
    
    # Create agents with different specialties
    code_agent = BaseAgent(
        name="CodeExpert",
        role="programmer",
        system_prompt="""You are a programming expert. You excel at code-related tasks,
        algorithms, and technical implementation. Bid high on coding tasks."""
    )
    code_agent.specialty = "programming"
    
    writing_agent = BaseAgent(
        name="WriteExpert",
        role="writer",
        system_prompt="""You are a writing expert. You excel at content creation,
        documentation, and communication. Bid high on writing tasks."""
    )
    writing_agent.specialty = "writing"
    
    analysis_agent = BaseAgent(
        name="AnalysisExpert",
        role="analyst",
        system_prompt="""You are an analysis expert. You excel at data analysis,
        research synthesis, and strategic thinking. Bid high on analysis tasks."""
    )
    analysis_agent.specialty = "analysis"
    
    generalist = BaseAgent(
        name="Generalist",
        role="generalist",
        system_prompt="""You are a generalist who can handle various tasks adequately.
        Bid moderately on most tasks - you're reliable but not specialized."""
    )
    generalist.specialty = "general"
    
    # Build system
    market = MarketBasedSystem()
    market.add_agent(code_agent)\
          .add_agent(writing_agent)\
          .add_agent(analysis_agent)\
          .add_agent(generalist)
    
    # Run auctions for different tasks
    print("\n" + "="*70)
    print("AUCTION 1: Technical Task")
    print("="*70)
    result1 = market.run("Write a Python function to implement binary search with error handling")
    
    print("\n" + "="*70)
    print("AUCTION 2: Writing Task")
    print("="*70)
    result2 = market.run("Write a compelling product description for an AI assistant tool")
    
    print("\n" + "="*70)
    print("AUCTION 3: Analysis Task")
    print("="*70)
    result3 = market.run("Analyze the pros and cons of microservices vs monolithic architecture")
    
    return {"auction1": result1, "auction2": result2, "auction3": result3}


# ==============================================================================
# ARCHITECTURE 8: SUPERVISOR WITH DYNAMIC ROUTING
# ==============================================================================
"""
SUPERVISOR WITH DYNAMIC ROUTING

                    ┌────────────┐
                    │ Supervisor │
                    └──────┬─────┘
                           │
              ┌────────────┼────────────┐
              │ Route based on task type │
              ▼            ▼            ▼
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │ Agent A │  │ Agent B │  │ Agent C │
         │ (Code)  │  │ (Write) │  │(Research)│
         └─────────┘  └─────────┘  └─────────┘
    
When to use:
- Different task types need different specialists
- Routing decisions are complex
- Need intelligent orchestration

Pros:
- Intelligent task routing
- Flexible and adaptive
- Supervisor learns patterns

Cons:
- Supervisor is a bottleneck
- Routing errors cascade
- More complex than simple patterns
"""

class RoutingDecision(BaseModel):
    """Schema for supervisor routing decision."""
    selected_agent: str = Field(description="Name of agent to route to")
    reasoning: str = Field(description="Why this agent was selected")
    subtask: str = Field(description="Refined task for the selected agent")


class SupervisorRouter:
    """
    Supervisor that routes tasks to appropriate specialist agents.
    
    The supervisor analyzes incoming tasks and routes them to
    the most suitable agent, potentially breaking complex tasks
    into multiple routed subtasks.
    """
    
    def __init__(self):
        self.supervisor: BaseAgent = None
        self.specialists: Dict[str, BaseAgent] = {}
        self.routing_history: List[Dict] = []
    
    def set_supervisor(self, agent: BaseAgent):
        """Set the supervisor agent."""
        self.supervisor = agent
        return self
    
    def add_specialist(self, name: str, agent: BaseAgent, description: str):
        """Add a specialist agent."""
        agent.description = description
        self.specialists[name] = agent
        return self
    
    def run(self, task: str) -> Dict[str, Any]:
        """Run with supervisor routing."""
        print_section("SUPERVISOR WITH DYNAMIC ROUTING")
        print(f"👔 Supervisor: {self.supervisor.name}")
        print(f"👷 Specialists: {list(self.specialists.keys())}")
        print(f"📝 Task: {task[:100]}...\n")
        
        # Step 1: Supervisor analyzes and routes
        print("🔀 Supervisor analyzing task...")
        
        routing_prompt = f"""Analyze this task and decide which specialist should handle it.

Available specialists:
{self._describe_specialists()}

Task: {task}

Decide:
1. Which specialist is best suited? (must be one of: {list(self.specialists.keys())})
2. Why is this specialist best?
3. What specific subtask should they focus on?

Respond in format:
Selected: [agent name]
Reasoning: [why]
Subtask: [refined task for the agent]"""
        
        routing_result = self.supervisor.process(routing_prompt)
        
        # Parse routing decision
        selected_agent = None
        subtask = task
        
        lines = routing_result.output.split("\n")
        for line in lines:
            line_lower = line.lower()
            if "selected:" in line_lower:
                for name in self.specialists.keys():
                    if name.lower() in line_lower:
                        selected_agent = name
                        break
            elif "subtask:" in line_lower:
                subtask = line.split(":", 1)[-1].strip()
        
        # Fallback to first specialist if parsing fails
        if not selected_agent:
            selected_agent = list(self.specialists.keys())[0]
        
        print(f"   → Routed to: {selected_agent}")
        print(f"   → Subtask: {subtask[:100]}...")
        
        # Step 2: Specialist executes
        print(f"\n⚡ {selected_agent} executing task...")
        
        specialist = self.specialists[selected_agent]
        execution_result = specialist.process(subtask)
        print_result(execution_result)
        
        # Step 3: Supervisor reviews (optional quality check)
        print("📋 Supervisor reviewing result...")
        
        review_prompt = f"""Review this specialist's output.

Original task: {task}
Specialist: {selected_agent}
Their output: {execution_result.output}

Provide:
1. Quality assessment (Good/Needs Improvement/Poor)
2. Any additional notes or refinements

Keep review brief."""
        
        review_result = self.supervisor.process(review_prompt)
        print(f"   {review_result.output[:200]}...")
        
        # Record routing
        self.routing_history.append({
            "task": task,
            "routed_to": selected_agent,
            "subtask": subtask,
            "result": execution_result.output
        })
        
        return {
            "final_output": execution_result.output,
            "routed_to": selected_agent,
            "supervisor_review": review_result.output,
            "routing_history": self.routing_history,
            "success": execution_result.success
        }
    
    def _describe_specialists(self) -> str:
        """Generate description of specialists."""
        descriptions = []
        for name, agent in self.specialists.items():
            desc = getattr(agent, 'description', agent.role)
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)


def demo_supervisor_router():
    """Demonstrate supervisor with dynamic routing."""
    
    # Create supervisor
    supervisor = BaseAgent(
        name="Supervisor",
        role="supervisor",
        system_prompt="""You are a task supervisor who routes work to specialists.
        Analyze incoming tasks and decide which specialist is best suited.
        Consider each specialist's strengths when routing.
        Review outputs for quality."""
    )
    
    # Create specialists
    coder = BaseAgent(
        name="coder",
        role="programmer",
        system_prompt="""You are a programming specialist. Write clean, efficient code.
        Include comments and handle edge cases. Focus only on coding tasks."""
    )
    
    writer = BaseAgent(
        name="writer",
        role="content_creator",
        system_prompt="""You are a content writing specialist. Create engaging, clear content.
        Focus on readability and impact. Handle all writing and documentation tasks."""
    )
    
    researcher = BaseAgent(
        name="researcher",
        role="researcher",
        system_prompt="""You are a research specialist. Find information, analyze data,
        and provide well-sourced insights. Handle research and analysis tasks."""
    )
    
    planner = BaseAgent(
        name="planner",
        role="strategic_planner",
        system_prompt="""You are a planning specialist. Create strategies, roadmaps,
        and action plans. Handle planning and organizational tasks."""
    )
    
    # Build system
    router = SupervisorRouter()
    router.set_supervisor(supervisor)\
          .add_specialist("coder", coder, "Programming, code writing, debugging")\
          .add_specialist("writer", writer, "Content creation, documentation, copywriting")\
          .add_specialist("researcher", researcher, "Research, analysis, fact-finding")\
          .add_specialist("planner", planner, "Strategy, planning, roadmaps")
    
    # Test with different task types
    print("\n" + "="*70)
    print("TASK 1: Coding Task")
    print("="*70)
    result1 = router.run("Create a function that validates email addresses using regex")
    
    print("\n" + "="*70)
    print("TASK 2: Writing Task")
    print("="*70)
    result2 = router.run("Write a welcome email for new users of our AI platform")
    
    print("\n" + "="*70)
    print("TASK 3: Research Task")
    print("="*70)
    result3 = router.run("What are the main differences between GPT-4 and Claude 3?")
    
    print("\n" + "="*70)
    print("TASK 4: Planning Task")
    print("="*70)
    result4 = router.run("Create a 3-month roadmap for launching an AI chatbot product")
    
    return {"task1": result1, "task2": result2, "task3": result3, "task4": result4}


# ==============================================================================
# COMPARISON & SUMMARY
# ==============================================================================

def print_architecture_comparison():
    """Print a comparison table of all architectures."""
    
    print_section("ARCHITECTURE COMPARISON")
    
    comparison = """
┌──────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│ Architecture     │ Best For            │ Pros                │ Cons                │
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Sequential       │ Linear workflows    │ Simple, debuggable  │ Slow, no parallel   │
│ Pipeline         │ Clear stages        │ Easy to modify      │ Single point failure│
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Parallel         │ Independent tasks   │ Fast, fault tolerant│ Resource intensive  │
│ Fan-Out/Fan-In   │ Multiple perspectives│ Diverse outputs    │ Need good aggregator│
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Hierarchical     │ Complex tasks       │ Good decomposition  │ Manager bottleneck  │
│ Manager-Worker   │ Need coordination   │ Clear accountability│ Depends on manager  │
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Debate           │ Decision making     │ Surfaces assumptions│ Slow, may not       │
│ Adversarial      │ Stress-testing ideas│ Balanced output     │ converge            │
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Voting           │ High-stakes decisions│ Reliable, catches  │ Expensive, slow     │
│ Consensus        │ Error reduction     │ individual errors   │ Correlated errors   │
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Blackboard       │ Complex problems    │ Flexible, async     │ Can be chaotic      │
│ Shared Memory    │ Incremental work    │ Builds on others    │ Hard to trace       │
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Market-Based     │ Resource allocation │ Self-organizing     │ Gaming possible     │
│ Auction          │ Variable capabilities│ Efficient allocation│ Complex bidding    │
├──────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ Supervisor       │ Mixed task types    │ Intelligent routing │ Supervisor overhead │
│ Router           │ Need specialization │ Adaptive            │ Routing errors      │
└──────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘

SELECTION GUIDE:

- Simple, ordered tasks → Sequential Pipeline
- Need multiple perspectives → Parallel Fan-Out
- Complex decomposition → Hierarchical
- Important decisions → Debate or Voting
- Evolving solutions → Blackboard
- Variable agent costs → Market-Based
- Mixed task types → Supervisor Router
"""
    print(comparison)


# ==============================================================================
# MAIN
# ==============================================================================

def run_all_demos():
    """Run all architecture demonstrations."""
    
    print("\n" + "="*70)
    print("   MULTI-AGENT ARCHITECTURE DEMONSTRATIONS")
    print("   Chapter 12: Multi-Agent Collaboration")
    print("="*70)
    
    demos = [
        ("Sequential Pipeline", demo_sequential_pipeline),
        ("Parallel Fan-Out/Fan-In", demo_parallel_fan_out),
        ("Hierarchical (Manager-Worker)", demo_hierarchical),
        ("Debate/Adversarial", demo_debate),
        ("Voting/Consensus", demo_voting),
        ("Blackboard (Shared Memory)", demo_blackboard),
        ("Market-Based (Auction)", demo_market_based),
        ("Supervisor with Routing", demo_supervisor_router),
    ]
    
    results = {}
    
    for name, demo_fn in demos:
        print(f"\n\n{'#'*70}")
        print(f"# DEMO: {name}")
        print(f"{'#'*70}")
        
        try:
            result = demo_fn()
            results[name] = {"success": True, "result": result}
            print(f"\n✅ {name} demo completed successfully")
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            print(f"\n❌ {name} demo failed: {e}")
        
        input("\nPress Enter to continue to next demo...")
    
    # Print comparison
    print_architecture_comparison()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Architecture Demos")
    parser.add_argument(
        "--arch", 
        choices=[
            "sequential", "parallel", "hierarchical", "debate",
            "voting", "blackboard", "market", "supervisor", "all", "compare"
        ],
        default="compare",
        help="Which architecture to demonstrate"
    )
    
    args = parser.parse_args()
    
    if args.arch == "all":
        run_all_demos()
    elif args.arch == "compare":
        print_architecture_comparison()
    elif args.arch == "sequential":
        demo_sequential_pipeline()
    elif args.arch == "parallel":
        demo_parallel_fan_out()
    elif args.arch == "hierarchical":
        demo_hierarchical()
    elif args.arch == "debate":
        demo_debate()
    elif args.arch == "voting":
        demo_voting()
    elif args.arch == "blackboard":
        demo_blackboard()
    elif args.arch == "market":
        demo_market_based()
    elif args.arch == "supervisor":
        demo_supervisor_router()