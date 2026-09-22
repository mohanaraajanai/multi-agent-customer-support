import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool

# ---------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "customer_support_entries.txt"
MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
MAX_SOURCE_CHARS = 120_000


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Add it to your environment or .env file."
        )
    return value


def read_uploaded_support_file(uploaded_file) -> str:
    """Read the inbound customer-support knowledge file as UTF-8 text."""
    if uploaded_file is None:
        raise ValueError("Please upload a customer support source file.")

    try:
        content = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "The support source file must be a UTF-8 encoded text file."
        ) from exc

    content = content.strip()

    if not content:
        raise ValueError("The uploaded support source file is empty.")

    if len(content) > MAX_SOURCE_CHARS:
        raise ValueError(
            f"The uploaded support source file is too large. "
            f"Maximum supported text length is {MAX_SOURCE_CHARS:,} characters."
        )

    return content


# ---------------------------------------------------------
# Custom tool used ONLY by the Entry Agent
# ---------------------------------------------------------
@tool("Save Customer Support Entry")
def save_customer_support_entry(entry_text: str) -> str:
    """
    Append one complete customer-support interaction to the text log.
    The entry must contain the customer query, Answer 1 from the Assistant,
    and Answer 2 from the Web Search Assistant.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write(entry_text.strip() + "\n")

    return f"Entry saved successfully to {LOG_FILE.name}."


# ---------------------------------------------------------
# Crew construction
# ---------------------------------------------------------
def build_crew(user_query: str, support_source: str, status_box):
    openai_key = require_env("OPENAI_API_KEY")
    serper_key = require_env("SERPER_API_KEY")

    llm = LLM(
        model=MODEL_NAME,
        api_key=openai_key,
        temperature=0.2,
    )

    # The web-search capability belongs only to Agent 2.
    search_tool = SerperDevTool(api_key=serper_key)

    # -----------------------------
    # Agent 1 - Assistant
    # -----------------------------
    assistant = Agent(
        role="Assistant",
        goal=(
            "Answer the customer's query using ONLY the supplied customer support "
            "knowledge base. Do not use general model knowledge, web search, or "
            "unstated assumptions."
        ),
        backstory=(
            "You are the first-line customer support assistant. The uploaded "
            "customer support source file is your only authority. When the source "
            "does not contain the requested information, explicitly say that the "
            "information is not available in the customer support knowledge base."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # -----------------------------
    # Agent 2 - Web Search Assistant
    # -----------------------------
    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Search the web for the customer's query, verify current information, "
            "and provide a web-grounded customer-support answer."
        ),
        backstory=(
            "You are the research-oriented second-line support assistant. "
            "You use the available web search tool, prefer reliable sources, "
            "and clearly distinguish current web findings from general knowledge."
        ),
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # -----------------------------
    # Agent 3 - Entry Agent
    # -----------------------------
    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Create a durable text record containing the customer query, both "
            "previous answers, and return both answers to the user."
        ),
        backstory=(
            "You are the final support-record agent. You receive the outputs of "
            "the first two agents, save the complete interaction using the file "
            "writing tool, and then return both answers clearly."
        ),
        tools=[save_customer_support_entry],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # -----------------------------
    # Task 1 - Direct answer
    # -----------------------------
    assistant_task = Task(
        description=(
            "Answer the following customer query using ONLY the customer support "
            "knowledge base provided below.\n\n"
            "STRICT RULES:\n"
            "- Do not browse the web.\n"
            "- Do not use general model knowledge.\n"
            "- Do not invent, infer, or assume facts not stated in the source.\n"
            "- If the answer is not explicitly supported by the source, say: "
            '"This information is not available in the customer support knowledge base."\n'
            "- Keep the answer concise and customer-friendly.\n\n"
            "CUSTOMER SUPPORT KNOWLEDGE BASE:\n"
            "------------------------------\n"
            f"{support_source}\n"
            "------------------------------\n\n"
            "CUSTOMER QUERY:\n"
            f"{user_query}"
        ),
        expected_output=(
            "A concise customer-support answer grounded strictly in the uploaded "
            "customer support knowledge base, or the exact unavailable-information "
            "message when the source does not contain the answer."
        ),
        agent=assistant,
    )

    # -----------------------------
    # Task 2 - Web-grounded answer
    # -----------------------------
    web_search_task = Task(
        description=(
            "Research the same customer query using your web search tool. "
            "Produce a second answer grounded in current web results. "
            "Use the previous Assistant answer as context, but independently "
            "verify important claims rather than blindly copying it. "
            "Where useful, include source names and URLs.\n\n"
            "Customer query:\n{query}"
        ),
        expected_output=(
            "A web-grounded answer to the query, supported by relevant web results "
            "and a concise list of useful source links or source names."
        ),
        agent=web_search_assistant,
        context=[assistant_task],
    )

    # -----------------------------
    # Task 3 - Save + return both
    # -----------------------------
    entry_task = Task(
        description=(
            "Create the final customer-support record using the outputs of BOTH "
            "previous tasks. You must call the 'Save Customer Support Entry' tool "
            "exactly once. Pass it one complete text entry containing:\n"
            "1) the original customer query\n"
            "2) Answer 1 - the Assistant's answer\n"
            "3) Answer 2 - the Web Search Assistant's answer\n\n"
            "After the save succeeds, return BOTH answers to the user. Preserve the "
            "meaning and wording of both previous answers; do not invent a third answer.\n\n"
            "Customer query:\n{query}"
        ),
        expected_output=(
            "A final response containing Answer 1 and Answer 2, plus a brief confirmation "
            "that the complete interaction was saved to the text file."
        ),
        agent=entry_agent,
        context=[assistant_task, web_search_task],
    )

    task_names = ["Assistant", "Web Search Assistant", "Entry Agent"]
    state = {"completed": 0}

    def on_task_complete(task_output):
        completed = state["completed"]
        finished_name = task_names[completed]
        state["completed"] += 1

        status_box.write(f"✅ {finished_name} completed")

        if state["completed"] < len(task_names):
            next_name = task_names[state["completed"]]
            status_box.write(f"➡️ Running {next_name}...")
        else:
            status_box.update(
                label="3-agent support crew completed",
                state="complete",
                expanded=False,
            )

    crew = Crew(
        agents=[assistant, web_search_assistant, entry_agent],
        tasks=[assistant_task, web_search_task, entry_task],
        process=Process.sequential,
        task_callback=on_task_complete,
        verbose=False,
    )

    return crew, assistant_task, web_search_task, entry_task


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
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
        **Agent 1 — Assistant:** answers ONLY from the uploaded customer-support source file.\n\n
        **Agent 2 — Web Search Assistant:** receives Agent 1's output and searches the web.\n\n
        **Agent 3 — Entry Agent:** receives both prior outputs, saves the full interaction to
        `customer_support_entries.txt`, and returns both answers.
        """
    )

# ---------------------------------------------------------
# Inbound customer-support source file
# ---------------------------------------------------------
st.subheader("1. Customer Support Source File")

uploaded_file = st.file_uploader(
    "Upload the customer-support knowledge base",
    type=["txt"],
    accept_multiple_files=False,
    max_upload_size=5,
    help="Agent 1 will use ONLY the text in this uploaded file.",
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
# Customer query
# ---------------------------------------------------------
st.subheader("2. Customer Query")

query = st.text_area(
    "Enter the customer's question",
    placeholder="Example: Can I cancel my order after it has been shipped?",
    height=120,
)

run_button = st.button("Run 3-Agent Support", type="primary", use_container_width=True)

if run_button:
    if uploaded_file is None or support_source is None:
        st.warning("Please upload a valid customer support source file first.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a customer query.")
        st.stop()

    status_box = st.status("Starting 3-agent support crew...", expanded=True)
    status_box.write("➡️ Running Assistant...")

    try:
        crew = build_crew(
            user_query=query.strip(),
            support_source=support_source,
            status_box=status_box,
        )

        result = crew.kickoff(inputs={"query": query.strip()})

        # Individual task outputs are used so the UI shows the two requested answers
        # exactly as produced by Agents 1 and 2.
        task_outputs = getattr(result, "tasks_output", [])

        if len(task_outputs) < 3:
            raise RuntimeError("Crew completed, but the expected three task outputs were not returned.")

        answer_1 = task_outputs[0].raw
        answer_2 = task_outputs[1].raw
        final_output = task_outputs[2].raw

        st.subheader("Answer 1 — Assistant")
        st.write(answer_1)

        st.subheader("Answer 2 — Web Search Assistant")
        st.write(answer_2)

        st.subheader("Entry Agent")
        st.success("The Entry Agent completed the final step.")
        st.write(final_output)

        if LOG_FILE.exists():
            st.info(f"Saved support log: `{LOG_FILE}`")

    except Exception as exc:
        status_box.update(
            label="Crew execution failed",
            state="error",
            expanded=True,
        )
        st.error(str(exc))
        st.caption(
            "Check OPENAI_API_KEY, SERPER_API_KEY, package installation, and network/API access."
        )
