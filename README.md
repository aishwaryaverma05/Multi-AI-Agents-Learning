# Multi-AI-Agents-Learning 
A hands-on learning repository exploring **Multi-Agent AI Systems** using Python, LangChain, LangGraph, and Groq LLMs.

This repository contains two implementations:
1. Simple Multi-Agent System
2. Supervisor-Based Multi-Agent System

The goal is to understand how multiple AI agents can collaborate, divide responsibilities, and complete a task through a structured workflow.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Technologies Used](#-technologies-used)
- [What Is a Multi-Agent System?](#-what-is-a-multi-agent-system)
- [1. Simple Multi-Agent System](#1-simple-multi-agent-system)
- [2. Supervisor-Based Multi-Agent System](#2-supervisor-based-multi-agent-system)
- [Important Concepts](#-important-concepts)
- [Project Workflow](#-project-workflow)
- [Setup and Execution](#-setup-and-execution)
- [Key Learnings](#-key-learnings)
- [Limitations and Improvements](#-limitations-and-improvements)
- [Interview Questions](#-important-interview-questions)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

The project explores how an AI application can use multiple specialized agents instead of asking a single LLM to perform every task.

Two architectures are implemented in the notebook:

| Architecture | Agents / Components | Main Idea |
|---|---|---|
| Simple Multi-Agent | Researcher and Writer | A fixed sequence of tasks |
| Supervisor-Based Multi-Agent | Supervisor, Researcher, Analyst, Writer | A supervisor chooses which agent works next |

### Example use case

**Task:** What are the benefits and risks of AI in healthcare?

The supervisor-based system is designed to:

1. Research the topic.
2. Analyze the research findings.
3. Prepare a structured report.
4. Finish when the report is complete.

This is a learning project for understanding agent collaboration and workflow orchestration.

---

## 🛠️ Technologies Used

- **Python:** Core programming language.
- **LangChain:** Connects LLMs with prompts, messages, and tools.
- **LangGraph:** Builds stateful workflows using graph nodes and edges.
- **Groq:** Provides access to an LLM through its API.
- **Tavily Search:** Intended web-search tool in the simple-agent implementation.
- **python-dotenv:** Loads environment variables from a `.env` file.
- **Jupyter Notebook:** Used for experimentation and learning.

---

## 🧠 What Is a Multi-Agent System?

A Multi-Agent AI System consists of multiple AI agents that collaborate to accomplish a task.

Each agent can have a specific responsibility, instructions, and access to particular tools.

For example:

- Researcher → gathers information.
- Analyst → interprets information and identifies insights.
- Writer → prepares the final report.
- Supervisor → coordinates the workflow and decides the next step.

### Why use multiple agents?

A complex task can be divided into smaller, manageable tasks. Specialized prompts help each agent focus on its responsibility.

However, multiple agents are not always necessary. For a simple task, a single LLM call may be faster, cheaper, and easier to maintain.

---

## 1. Simple Multi-Agent System

### Architecture

```text
          User Task
              |
              v
      Researcher Agent
              |
              v
        Writer Agent
              |
              v
         Final Output
```

### Agent 1: Researcher

The Researcher Agent is responsible for gathering information related to the user's task.

In the notebook, it:

1. Receives the task through the message state.
2. Uses a system message describing its research role.
3. Calls the LLM with the `search_web` tool bound to it.
4. Returns the response and sets `next_agent` to `"writer"`.

The notebook defines a Tavily-based search tool intended to retrieve relevant web results.

### Agent 2: Writer

The Writer Agent converts the available conversation and research information into a readable summary.

Its instructions specify that it should:

- Focus on the research findings.
- Produce clean, structured text.
- Avoid calling tools.
- Return the final response without JSON or tool-call schemas.

It sets `next_agent` to `"end"`.

### Building the graph

LangGraph is used to define the workflow:

```python
workflow = StateGraph(MessagesState)

workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

final_work = workflow.compile()
```

**Explanation:**

- `StateGraph` defines the workflow.
- `add_node()` registers an agent as a graph node.
- `set_entry_point()` defines the starting node.
- `add_edge()` connects nodes in a fixed sequence.
- `END` marks the end of the workflow.
- `compile()` builds the executable graph.

### Key learning

This implementation demonstrates a fixed workflow: the Researcher is followed by the Writer.

The graph does not currently use `next_agent` to make routing decisions. Also, the tool-execution node is defined separately but is not connected to this graph, so the complete tool-calling loop needs to be implemented and tested.

---

## 2. Supervisor-Based Multi-Agent System

This implementation introduces a Supervisor Agent that coordinates a team of specialized agents.

### Architecture

```text
              User Task
                  |
                  v
             Supervisor
                  |
          Select Next Agent
                  |
          +-------+-------+
          |       |       |
          v       v       v
      Researcher Analyst Writer
          |       |       |
          +-------+-------+
                  |
                  v
             Supervisor
                  |
          Continue or End
                  |
                  v
             Final Report
```

The Supervisor decides which agent should work next. After an agent completes its task, control returns to the Supervisor.

### Agent 1: Supervisor

The Supervisor is responsible for orchestration, not for performing all the research or writing itself.

It checks the workflow state:

```python
has_research = bool(state.get("research_data", ""))
has_analysis = bool(state.get("analysis", ""))
has_report = bool(state.get("final_report", ""))
```

These variables indicate whether each stage has produced data.

The Supervisor uses a prompt chain to ask the LLM to select one of the following:

- `researcher`
- `analyst`
- `writer`
- `DONE`

The returned decision is parsed to determine the next agent.

The Supervisor returns:

- A status message.
- `next_agent`, identifying the next node.
- `current_task`, storing the task being processed.

**Important:** The Supervisor uses an LLM to propose the next step, while the Python routing function enforces the actual graph transition.

### Agent 2: Researcher

The Researcher gathers information about the assigned topic using an LLM prompt.

The prompt requests:

1. Key facts and background.
2. Current trends or developments.
3. Important statistics or data points.
4. Relevant examples or case studies.

The generated information is stored in:

```python
research_data
```

The agent returns control to the Supervisor.

**Important distinction:** In this supervisor-based implementation, the Researcher generates research content using the LLM. It does not use the Tavily search tool defined in the simple-agent section.

### Agent 3: Analyst

The Analyst receives the research data and interprets it.

Its prompt requests:

1. Key insights and patterns.
2. Strategic implications.
3. Risks and opportunities.
4. Actionable recommendations.

The analysis is stored in:

```python
analysis
```

The Analyst then returns control to the Supervisor.

### Agent 4: Writer

The Writer uses the research findings and analysis to create a structured executive report.

The report contains five sections:

1. Executive Summary
2. Key Findings
3. Analysis & Insights
4. Recommendations
5. Conclusion

The final report is stored in:

```python
final_report
```

The notebook also adds a timestamp, task title, and a message identifying the report as an output of the multi-agent system.

### How does the Supervisor know what to do next?

The workflow uses a custom state object:

```python
class SupervisorState(MessagesState):
    next_agent: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""
```

The state preserves the information required across agent transitions.

The Supervisor checks which outputs exist and requests a decision from the LLM. The router then uses `next_agent` and `task_complete` to determine the next graph node.

---

## 🔄 Project Workflow

The supervisor-based workflow is designed to follow these steps:

**Step 1 — User Input**

The user submits a task, such as:

"What are the benefits and risks of AI in healthcare?"

**Step 2 — Supervisor**

The Supervisor checks the current state and decides which agent should act next.

**Step 3 — Research**

The Researcher generates background information and relevant findings. The output is saved in `research_data`.

**Step 4 — Analysis**

The Analyst examines the research and generates insights, risks, opportunities, and recommendations. The output is saved in `analysis`.

**Step 5 — Report Generation**

The Writer combines the research and analysis into a structured report. The output is saved in `final_report`.

**Step 6 — Routing and Completion**

After each agent finishes, the router returns control to the Supervisor. When the workflow is complete, execution should terminate.

---

## 📚 Important Concepts

### 1. LLM

A Large Language Model generates text and other outputs based on input messages and learned patterns.

In this project, the LLM performs research-style generation, analysis, report writing, and supervisor decision-making.

### 2. Agent

An agent is an LLM-driven component configured for a particular responsibility.

An agent may use tools, follow instructions, and participate in a larger workflow.

### 3. State

State is the shared data passed between graph nodes.

In this project, it contains the messages, research data, analysis, final report, and routing information.

### 4. Node

A node represents a step in the workflow. Here, the Supervisor, Researcher, Analyst, and Writer are separate nodes.

### 5. Edge

An edge connects one node to another.

A normal edge represents a fixed transition. A conditional edge chooses a transition based on the current state or routing function.

### 6. Conditional Routing

Conditional routing allows the workflow to choose different paths at runtime.

In this project, the `router()` function returns the next agent based on `next_agent` and `task_complete`.

### 7. Tool Calling

Tool calling allows an LLM to request an external function, such as a web-search tool.

The simple-agent implementation defines a Tavily search tool and binds it to the Researcher LLM. A working tool execution loop is needed to execute tool calls and return their results to the LLM.

### 8. Supervisor Pattern

The Supervisor Pattern uses a coordinating agent to assign work to specialized agents and manage workflow progression.

It is useful when tasks need multiple stages or when the next step depends on previous results.

---

## ⚙️ Setup and Execution

### Prerequisites

- Python installed.
- A Groq API key.
- A Tavily API key if using the web-search tool.
- Jupyter Notebook or JupyterLab.

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Multi-AI-Agents-Learning
```

Replace the placeholder with the actual repository URL.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

Create a `requirements.txt` file with the libraries used by the notebook. For example:

```text
langchain
langchain-core
langchain-community
langchain-groq
langgraph
python-dotenv
jupyter
tavily-python
```

Install them:

```bash
pip install -r requirements.txt
```

Package versions may need to be pinned after the notebook has been tested successfully.

### Step 4: Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Never upload the real `.env` file or expose API keys in notebook cells or GitHub commits.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
.ipynb_checkpoints/
```

### Step 5: Open the notebook

```bash
jupyter notebook multiagent.ipynb
```

Run the cells in order, beginning with the imports and environment setup.

**Execution status:** The notebook documents both architectures, but the end-to-end workflows require debugging and validation before they can be considered fully runnable. See the limitations below.

---

## 💡 Key Learnings

Through this project, I explored:

- Designing multiple specialized AI agents.
- Using LangChain to interact with LLMs.
- Building workflows with LangGraph's `StateGraph`.
- Passing shared state between nodes.
- Connecting nodes using normal and conditional edges.
- Implementing a Supervisor-Based orchestration pattern.
- Separating research, analysis, and report generation.
- Understanding the difference between tool calling and direct LLM generation.
- Managing API keys through environment variables.

---

## ⚠️ Limitations and Improvements

The notebook is a learning implementation, not a production-ready agent system.

Before treating it as a fully working project, the following issues should be addressed:

1. **Web-search function:** The `search_web()` function accepts `Query`, but calls `search.invoke(query)`. The variable names do not match and need to be corrected.

2. **Tool execution:** The simple workflow defines a tool-execution function, but does not add it as a node or route tool calls through it. The Researcher-to-Writer transition is currently fixed.

3. **Supervisor input:** The example invokes the graph with a `HumanMessage` directly. The graph should receive an input matching its declared state, such as a `messages` list.

4. **Task preservation:** The supervisor reads the latest message as the task, but later messages may be status messages. The original task should be preserved explicitly in `current_task`.

5. **Routing and termination:** The supervisor's decision parsing and the router's completion logic should be tested together to ensure that every stage runs in the intended order and the graph terminates correctly.

6. **API configuration:** Verify that the environment variables are present and that the selected Groq model is available to the account.

7. **Error handling:** Add handling for API failures, empty outputs, and unexpected supervisor decisions.

8. **Output validation:** Confirm that the final report exists and includes the required sections before marking the task complete.

---

## 🎤 Important Interview Questions

### Q1. What is a Multi-Agent AI System?

A system in which multiple specialized agents collaborate to complete a task. Each agent has a specific responsibility, and a workflow coordinates their execution.

### Q2. Why use multiple agents instead of one LLM?

Multiple agents can separate complex work into specialized stages, making responsibilities easier to organize. However, multi-agent systems also introduce extra latency, cost, and coordination complexity.

### Q3. What is LangGraph?

LangGraph is a framework for building stateful, graph-based workflows. It supports nodes, edges, conditional routing, and shared state.

### Q4. What is the difference between LangChain and LangGraph?

LangChain provides components for working with LLMs, prompts, and tools. LangGraph helps orchestrate multi-step workflows, including cycles and conditional transitions.

### Q5. What is the role of the Supervisor Agent?

The Supervisor coordinates the other agents, examines the workflow state, and decides which agent should work next.

### Q6. What is the difference between a fixed workflow and supervisor-based orchestration?

A fixed workflow follows predefined transitions. A supervisor-based workflow uses the current state and a decision-making step to select the next agent.

### Q7. What is state in LangGraph?

State is the shared information passed between graph nodes. In this project, it includes messages, research findings, analysis, the final report, and routing fields.

### Q8. What are nodes and edges?

Nodes represent processing steps, such as agents. Edges define how execution moves between those steps.

### Q9. What is conditional routing?

Conditional routing chooses the next node based on a function's output or the current state.

### Q10. How does the Researcher Agent differ from the Analyst Agent?

The Researcher gathers or generates relevant information. The Analyst interprets that information and identifies insights, risks, opportunities, and recommendations.

### Q11. What is tool calling?

Tool calling lets an LLM request the execution of a function or external capability. The application must execute the tool and return its result to the model.

### Q12. Is this project production-ready?

No. It is a learning project. The graph input, tool execution, task preservation, routing, error handling, and end-to-end execution need validation.

---

## 🚀 Future Improvements

- Fix and test the simple-agent tool-calling loop.
- Validate the complete supervisor-based workflow.
- Add structured outputs for agent responses.
- Add retries and error handling.
- Add graph visualization.
- Add persistent checkpoints where appropriate.
- Compare a single-agent solution with a multi-agent solution.
- Add more real-world tasks and test cases.

---

## 📂 Repository Structure

```text
Multi-AI-Agents-Learning/
│
├── README.md
├── multiagent.ipynb
├── requirements.txt
└── .gitignore
```

---

## 📝 Final Note

This repository is part of my ongoing learning journey in Generative AI and agentic AI.

The main objective is to understand how specialized agents, shared state, tools, and graph-based orchestration can work together to solve multi-step tasks.

**Learning focus:** Multi-Agent Systems | LangChain | LangGraph | Groq LLMs | Supervisor Pattern
