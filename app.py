import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool

# =========================================================
# Configuration
# =========================================================
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "customer_support_entries.txt"
MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
MAX_SOURCE_CHARS = 120_000


# =========================================================
# Environment helpers
# =========================================================
def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Add it to your .env file and restart Streamlit."
        )
    return value


def read_uploaded_support_file(uploaded_file) -> str:
    """Read the uploaded UTF-8 text support knowledge base."""
    if uploaded_file is None:
        raise ValueError("Please upload a customer support source file.")

    try:
        content = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "The customer support source file must be UTF-8 encoded text."
        ) from exc

    content = content.strip()

    if not content:
        raise ValueError("The uploaded customer support source file is empty.")

    if len(content) > MAX_SOURCE_CHARS:
        raise ValueError(
            f"The uploaded source file is too large. Maximum supported size is "
            f"{MAX_SOURCE_CHARS:,} characters."
        )

    return content


# =========================================================
# Tool used only by Agent 3
# =========================================================
@tool("Save Customer Support Entry")
def save_customer_support_entry(entry_text: str) -> str:
    """Append the complete customer-support interaction to a text file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        file.write(entry_text.strip() + "\n")

    return f"Entry saved successfully to {LOG_FILE.name}."


# =========================================================
# Build the exactly-three-agent sequential crew
# =========================================================
def build_crew(user_query: str, support_source: str, status_box):
    # Validate both keys before creating the crew.
    require_env("OPENAI_API_KEY")
    require_env("SERPER_API_KEY")

    # CrewAI reads the OpenAI key from the environment.
    llm = LLM(
        model=MODEL_NAME,
        temperature=0.2,
    )

    # Agent 2 is the only agent that receives the web-search tool.
    search_tool = SerperDevTool()

    # -----------------------------------------------------
    # Agent 1 - Assistant
    # -----------------------------------------------------
    assistant = Agent(
        role="Assistant",
        goal=(
            "Answer the customer's query using ONLY the uploaded customer support "
            "knowledge base. Never use web search, general model knowledge, or "
            "unstated assumptions."
        ),
        backstory=(
            "You are the first-line customer support assistant. The uploaded "
            "customer support source is your only source of truth. If the requested "
            "information is not present, explicitly say it is not available in the "
            "customer support knowledge base."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # -----------------------------------------------------
    # Agent 2 - Web Search Assistant
    # -----------------------------------------------------
    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Search the web for the customer's query and produce a second answer "
            "grounded in relevant web results."
        ),
        backstory=(
            "You are the second-line web research assistant. Use the web-search "
            "tool to find relevant and current information. You may use Agent 1's "
            "answer as context, but independently research the query."
        ),
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # -----------------------------------------------------
    # Agent 3 - Entry Agent
    # -----------------------------------------------------
    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Save the customer query and both previous answers into a text file, "
            "then return both answers clearly."
        ),
        backstory=(
            "You are the final support-record agent. You receive Agent 1 and Agent 2 "
            "outputs through task context. You must use the save tool once to persist "
            "the complete interaction, then report the two answers."
        ),
        tools=[save_customer_support_entry],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # -----------------------------------------------------
    # Task 1 - Agent 1: source-grounded direct answer
    # -----------------------------------------------------
    assistant_task = Task(
        description=(
            "Answer the customer's query using ONLY the customer support knowledge "
            "base provided below.\n\n"
            "STRICT RULES:\n"
            "1. Do not browse the web.\n"
            "2. Do not use general model knowledge.\n"
            "3. Do not invent, infer, or assume facts not stated in the source.\n"
            "4. If the answer is not explicitly supported by the source, respond "
            "with exactly: \"This information is not available in the customer "
            "support knowledge base.\"\n"
            "5. Keep the answer concise and customer-friendly.\n\n"
            "CUSTOMER SUPPORT KNOWLEDGE BASE\n"
            "--------------------------------\n"
            f"{support_source}\n"
            "--------------------------------\n\n"
            "CUSTOMER QUERY\n"
            "--------------\n"
            f"{user_query}"
        ),
        expected_output=(
            "One concise answer grounded only in the uploaded customer support "
            "knowledge base, or the required unavailable-information message."
        ),
        agent=assistant,
    )

    # -----------------------------------------------------
    # Task 2 - Agent 2: web-grounded answer
    # -----------------------------------------------------
    web_search_task = Task(
        description=(
            "Research the customer's query using the web-search tool. Produce a "
            "second answer grounded in the web results. Use Agent 1's output as "
            "context, but independently research the query. Clearly distinguish "
            "web findings from the internal support answer. Where useful, include "
            "source names or URLs.\n\n"
            "CUSTOMER QUERY\n"
            "--------------\n"
            "{query}"
        ),
        expected_output=(
            "A concise web-grounded answer supported by relevant search results, "
            "including source names or links when useful."
        ),
        agent=web_search_assistant,
        context=[assistant_task],
    )

    # -----------------------------------------------------
    # Task 3 - Agent 3: save + return both answers
    # -----------------------------------------------------
    entry_task = Task(
        description=(
            "Using the outputs of both previous tasks, create one complete customer "
            "support record. You MUST call the 'Save Customer Support Entry' tool "
            "exactly once. The text passed to the tool must contain:\n\n"
            "1. Original customer query\n"
            "2. Answer 1 - Assistant answer\n"
            "3. Answer 2 - Web Search Assistant answer\n\n"
            "After the save succeeds, return both answers clearly to the user and "
            "briefly confirm that the interaction was saved. Do not invent a third "
            "answer.\n\n"
            "CUSTOMER QUERY\n"
            "--------------\n"
            "{query}"
        ),
        expected_output=(
            "A final response containing Answer 1, Answer 2, and a short save "
            "confirmation."
        ),
        agent=entry_agent,
        context=[assistant_task, web_search_task],
    )

    # -----------------------------------------------------
    # Progress callback
    # -----------------------------------------------------
    task_names = ["Assistant", "Web Search Assistant", "Entry Agent"]
    state = {"completed": 0}

    def on_task_complete(task_output):
        index = state["completed"]

        if index < len(task_names):
            status_box.write(f"✅ {task_names[index]} completed")

        state["completed"] += 1

        if state["completed"] < len(task_names):
            status_box.write(f"➡️ Running {task_names[state['completed']]}...")
        else:
            status_box.update(
                label="3-agent support crew completed",
                state="complete",
                expanded=False,
            )

    # -----------------------------------------------------
    # IMPORTANT: return ONLY the Crew object.
    # -----------------------------------------------------
    crew = Crew(
        agents=[assistant, web_search_assistant, entry_agent],
        tasks=[assistant_task, web_search_task, entry_task],
        process=Process.sequential,
        task_callback=on_task_complete,
        verbose=False,
    )

    return crew


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(
    page_title="Multi-Agent Customer Support",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Multi-Agent Customer Support")
st.caption("CrewAI · Sequential Process · 3 Agents · Streamlit")

with st.expander("How this buildathon demo works", expanded=False):
    st.markdown(
        """
        **Agent 1 — Assistant**  
        Answers ONLY from the uploaded customer-support source file.

        **Agent 2 — Web Search Assistant**  
        Receives Agent 1's task output and searches the web.

        **Agent 3 — Entry Agent**  
        Receives both earlier task outputs, saves the interaction to
        `customer_support_entries.txt`, and returns both answers.
        """
    )

# ---------------------------------------------------------
# 1. Inbound support source file
# ---------------------------------------------------------
st.subheader("1. Customer Support Source File")

uploaded_file = st.file_uploader(
    "Upload the customer-support knowledge base",
    type=["txt"],
    accept_multiple_files=False,
    max_upload_size=5,
    help="Agent 1 will use ONLY the contents of this uploaded text file.",
)

support_source = None

if uploaded_file is not None:
    try:
        support_source = read_uploaded_support_file(uploaded_file)
        st.success(
            f"Source loaded: `{uploaded_file.name}` "
            f"({len(support_source):,} characters)"
        )

        with st.expander("Preview source file", expanded=False):
            st.text(support_source)

    except ValueError as exc:
        st.error(str(exc))

# ---------------------------------------------------------
# 2. Customer query
# ---------------------------------------------------------
st.subheader("2. Customer Query")

query = st.text_area(
    "Enter the customer's question",
    placeholder="Example: Can I cancel my order after it has been shipped?",
    height=120,
)

run_button = st.button(
    "Run 3-Agent Support",
    type="primary",
    use_container_width=True,
)

if run_button:
    if uploaded_file is None or support_source is None:
        st.warning("Please upload a valid customer support source file first.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a customer query.")
        st.stop()

    status_box = st.status(
        "Starting 3-agent support crew...",
        expanded=True,
    )
    status_box.write("➡️ Running Assistant...")

    try:
        crew = build_crew(
            user_query=query.strip(),
            support_source=support_source,
            status_box=status_box,
        )

        # build_crew returns a Crew object, so kickoff is called directly on it.
        result = crew.kickoff(inputs={"query": query.strip()})

        # CrewAI exposes each completed task through result.tasks_output.
        task_outputs = getattr(result, "tasks_output", None)

        if not task_outputs or len(task_outputs) < 3:
            raise RuntimeError(
                "Crew completed, but the expected three task outputs were not returned."
            )

        answer_1 = task_outputs[0].raw
        answer_2 = task_outputs[1].raw
        final_output = task_outputs[2].raw

        st.subheader("Answer 1 — Assistant")
        st.write(answer_1)

        st.subheader("Answer 2 — Web Search Assistant")
        st.write(answer_2)

        st.subheader("Entry Agent")
        st.success("The Entry Agent completed the final step and saved the interaction.")
        st.write(final_output)

        if LOG_FILE.exists():
            st.info(f"Saved support log: `{LOG_FILE.name}`")

    except Exception as exc:
        status_box.update(
            label="Crew execution failed",
            state="error",
            expanded=True,
        )
        st.error(str(exc))
        st.caption(
            "Check OPENAI_API_KEY, SERPER_API_KEY, package versions, "
            "and network/API access."
        )
