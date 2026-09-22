# Multi-Agent Customer Support

A Weekly Buildathon project for the **Gen AI Architect Program** demonstrating a sequential multi-agent customer support workflow built with **CrewAI + Streamlit**.

## 1. Project Overview

The application accepts two inputs from the user:

1. A customer-support knowledge-base file (`.txt`)
2. A customer question/query

Three CrewAI agents then execute **sequentially**:

```text
Customer Query + Support Source File
                |
                v
     +-----------------------+
     | 1. Assistant Agent    |
     | Internal knowledge    |
     | from uploaded file    |
     +-----------+-----------+
                 |
                 | Answer 1
                 v
     +-----------------------+
     | 2. Web Search         |
     | Assistant             |
     | Searches the web      |
     +-----------+-----------+
                 |
                 | Answer 2
                 v
     +-----------------------+
     | 3. Entry Agent        |
     | Saves query + both    |
     | answers to TXT file   |
     +-----------+-----------+
                 |
                 v
             Streamlit UI
```

The main learning objective is to demonstrate how a **sequential CrewAI workflow passes outputs from one task into later tasks**.

---

## 2. Buildathon Requirements Covered

| Requirement | Implementation |
|---|---|
| CrewAI framework | Yes |
| Exactly 3 agents | Yes |
| Sequential processing | `Process.sequential` |
| Agent 1: Assistant | Answers only from uploaded support source |
| Agent 2: Web Search Assistant | Uses `SerperDevTool` |
| Agent 3: Entry Agent | Saves query and both answers |
| Streamlit UI | Yes |
| Inbound customer-support file | `.txt` upload using Streamlit |
| API keys from environment | `.env` |
| Single application file | `app.py` |
| Persistent interaction log | `customer_support_entries.txt` |

---

## 3. Agent Responsibilities

### Agent 1 — Assistant

**Role:** Internal customer-support assistant.

Agent 1 receives the uploaded support knowledge base directly inside its task.

Rules:

- Use only the uploaded source file.
- Do not browse the web.
- Do not use general LLM knowledge.
- Do not invent or assume missing information.
- If the source does not contain the answer, state that the information is not available in the customer-support knowledge base.

This makes Agent 1 a **source-grounded support agent**.

### Agent 2 — Web Search Assistant

**Role:** External/web research assistant.

Agent 2:

- Receives Agent 1's task output as context.
- Uses `SerperDevTool`.
- Searches the web for the customer's query.
- Produces a separate web-grounded answer.
- Can cite or mention useful web sources.

This agent is deliberately separate from Agent 1 so the application can demonstrate the difference between:

```text
Internal company knowledge
vs.
External web information
```

### Agent 3 — Entry Agent

**Role:** Final record/persistence agent.

Agent 3 receives:

- Original customer query
- Agent 1 output
- Agent 2 output

It then uses the custom `Save Customer Support Entry` tool to append the complete interaction to:

```text
customer_support_entries.txt
```

The Entry Agent also returns the previous answers to the UI.

---

## 4. Sequential Task Handoff

The application uses explicit CrewAI task context.

### Task 1

```python
assistant_task
```

Produces:

```text
Answer 1
```

### Task 2

```python
web_search_task
context=[assistant_task]
```

This allows Agent 2 to receive Agent 1's output.

### Task 3

```python
entry_task
context=[assistant_task, web_search_task]
```

This allows Agent 3 to receive both earlier outputs.

The crew is executed as:

```python
crew = Crew(
    agents=[assistant, web_search_assistant, entry_agent],
    tasks=[assistant_task, web_search_task, entry_task],
    process=Process.sequential,
)
```

Then:

```python
result = crew.kickoff(inputs={"query": query})
```

---

## 5. Project Structure

```text
CrewAI/
│
├── app.py
├── customer_support_source.txt
├── requirements.txt
├── .gitignore
├── .env                    # local secrets; NOT committed
├── .venv/                  # local Python environment; NOT committed
│
└── customer_support_entries.txt
                             # generated at runtime; NOT committed
```

### File descriptions

| File | Purpose |
|---|---|
| `app.py` | Complete Streamlit + CrewAI application |
| `customer_support_source.txt` | Sample customer-support knowledge base |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Prevents secrets, virtual environments and generated files from Git |
| `.env` | API credentials and model configuration |
| `customer_support_entries.txt` | Runtime interaction log |

---

## 6. Prerequisites

Recommended Python version:

```text
Python 3.12.x
```

Tested project environment:

```text
Python 3.12.10
CrewAI 1.15.22
CrewAI Tools 1.15.22
```

Python 3.14 should not be used for this environment because the CrewAI version used by this project requires Python below 3.14.

---

## 7. Create the Virtual Environment

From the project directory:

```cmd
py -3.12 -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

Verify:

```cmd
python --version
```

Expected:

```text
Python 3.12.10
```

---

## 8. Install Dependencies

Run:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify CrewAI:

```cmd
pip show crewai
```

Verify CrewAI tools:

```cmd
pip show crewai-tools
```

Expected versions:

```text
crewai       1.15.22
crewai-tools 1.15.22
```

---

## 9. Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
OPENAI_MODEL=openai/gpt-4o-mini
```

### API key usage

```text
OPENAI_API_KEY
    |
    +--> CrewAI LLM

SERPER_API_KEY
    |
    +--> SerperDevTool
```

Never hard-code API keys in `app.py`.

Never commit `.env` to GitHub.

---

## 10. Run the Application

Start Streamlit:

```cmd
streamlit run app.py
```

Open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

---

## 11. Using the Application

### Step 1 — Upload the support file

Upload:

```text
customer_support_source.txt
```

The Streamlit application reads the file directly as an inbound upload.

The source file does **not** have to remain beside `app.py` at runtime.

### Step 2 — Enter a customer query

Example:

```text
Can I cancel my order after it has been shipped?
```

### Step 3 — Run the crew

Click:

```text
Run 3-Agent Support
```

You should see progress:

```text
➡️ Running Assistant...
✅ Assistant completed

➡️ Running Web Search Assistant...
✅ Web Search Assistant completed

➡️ Running Entry Agent...
✅ Entry Agent completed
```

### Step 4 — Review the results

The UI displays:

```text
Answer 1 — Assistant
```

followed by the source-grounded response.

Then:

```text
Answer 2 — Web Search Assistant
```

followed by the web-grounded response.

The Entry Agent output confirms that the interaction was saved.

---

## 12. Recommended Validation Tests

### Test A — Information clearly present in source

Query:

```text
Can I cancel my order after it has been shipped?
```

Agent 1 should answer from the uploaded source.

The sample source says that an order can be cancelled only if it has not been shipped.

### Test B — Information not present in source

Query:

```text
Do you accept PayPal?
```

The sample knowledge base does not contain PayPal information.

Agent 1 should respond that the requested information is not available in the customer-support knowledge base.

Agent 2 can independently search the web.

This is an important test of Agent 1's grounding rule.

---

## 13. Runtime Output File

After a successful run, the application creates:

```text
customer_support_entries.txt
```

A typical record contains:

```text
================================================================================
Timestamp: 2026-09-23T01:00:00

Customer Query:
Can I cancel my order after it has been shipped?

Answer 1 — Assistant:
...

Answer 2 — Web Search Assistant:
...
```

New interactions are appended to the same file.

This file is excluded from Git through `.gitignore`.

---

## 14. Security / Git Rules

The following should NOT be committed:

```text
.env
.venv/
__pycache__/
customer_support_entries.txt
```

The sample source file may be committed for the buildathon:

```text
customer_support_source.txt
```

For a production customer-support application, review source files and generated logs before publishing them because they may contain confidential information.

---

## 15. Git Setup

Initialize Git once:

```cmd
git init
```

Add the GitHub remote:

```cmd
git remote add origin https://github.com/mohanaraajanai/multi-agent-customer-support.git
```

Verify:

```cmd
git remote -v
```

---

## 16. Check Before Committing

Run:

```cmd
git status
```

The intended files are:

```text
.gitignore
app.py
customer_support_source.txt
requirements.txt
README.md
```

The following should NOT appear as files to commit:

```text
.env
.venv/
customer_support_entries.txt
```

---

## 17. Commit and Push

Stage:

```cmd
git add app.py customer_support_source.txt requirements.txt README.md .gitignore
```

Commit:

```cmd
git commit -m "Update multi-agent customer support workflow"
```

Use the `main` branch:

```cmd
git branch -M main
```

Push:

```cmd
git push -u origin main
```

Repository:

```text
https://github.com/mohanaraajanai/multi-agent-customer-support
```

---

## 18. Troubleshooting

### `ImportError: cannot import name 'tool' from 'crewai'`

Use:

```python
from crewai.tools import tool
```

not:

```python
from crewai import tool
```

### `'tuple' object has no attribute 'kickoff'`

`build_crew()` must return:

```python
return crew
```

not:

```python
return crew, assistant_task, web_search_task, entry_task
```

The caller must use:

```python
crew = build_crew(...)
result = crew.kickoff(...)
```

### Missing API key

Check `.env`:

```env
OPENAI_API_KEY=...
SERPER_API_KEY=...
```

Then restart Streamlit.

### Agent 1 uses information outside the source

Check that:

- Agent 1 has no web-search tool.
- The complete uploaded source is included in `assistant_task`.
- The task explicitly states that only the supplied knowledge base may be used.

---

## 19. Learning Outcome

This project demonstrates the following GenAI architecture concepts:

- Multi-agent systems
- Agent roles and specialization
- Sequential CrewAI workflows
- Task-to-task context passing
- Tool assignment to individual agents
- Grounded LLM responses
- Web search augmentation
- Agent-based persistence
- Streamlit user interfaces
- Environment-variable-based secret management
- Git/GitHub project organization

The core design principle is:

```text
Specialized Agent 1
       +
Specialized Agent 2
       +
Persistence Agent
       =
Sequential Multi-Agent Workflow
```

---

## 20. Future Extensions

Possible extensions after the buildathon:

- Support PDF knowledge-base uploads
- Chunk large support documents
- Add semantic/vector retrieval for Agent 1
- Add source citations in the UI
- Add conversation history
- Add human approval before final response
- Add evaluation datasets
- Add LangSmith tracing/evaluation
- Containerize with Docker
- Deploy Streamlit + backend on a VPS
