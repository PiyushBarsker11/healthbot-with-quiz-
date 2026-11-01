"""
Every step of the HealthBot flow is its own node, each with a single
responsibility, exactly per the rubric.

Nodes that need a human response use LangGraph's `interrupt()`. Calling
interrupt(payload) pauses graph execution and returns `payload` to the
caller (Streamlit) as part of an `__interrupt__` entry in the result.
Execution resumes exactly where it paused when the caller invokes the
graph again with `Command(resume=<value>)` - the interrupt() call then
returns <value> instead of pausing again.
"""

from langgraph.types import interrupt
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage

from state import HealthBotState

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
tavily = TavilySearch(max_results=5, topic="general")


# ---------------------------------------------------------------------
# 1. Ask the patient what topic they want to learn about
# ---------------------------------------------------------------------
def get_topic_node(state: HealthBotState) -> dict:
    topic = interrupt({"prompt": "What health topic or condition would you like to learn about?"})
    return {"topic": topic}


# ---------------------------------------------------------------------
# 2. Search Tavily for reputable medical info
# ---------------------------------------------------------------------
def search_node(state: HealthBotState) -> dict:
    query = f"{state['topic']} patient education reputable medical source"
    results = tavily.invoke({"query": query})
    # results["results"] is a list of {title, url, content, ...}
    formatted = "\n\n".join(
        f"Source: {r['title']} ({r['url']})\n{r['content']}"
        for r in results.get("results", [])
    )
    return {"search_results": formatted}


# ---------------------------------------------------------------------
# 3. Summarize into patient-friendly language
# ---------------------------------------------------------------------
SUMMARY_PROMPT = """You are a patient education assistant. Using ONLY the \
information below (no outside knowledge), write a clear, patient-friendly \
3-4 paragraph explanation of "{topic}". Avoid jargon; explain any medical \
term you must use. Do not add facts that aren't in the source material.

SOURCE MATERIAL:
{search_results}
"""


def summarize_node(state: HealthBotState) -> dict:
    prompt = SUMMARY_PROMPT.format(topic=state["topic"], search_results=state["search_results"])
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"summary": response.content}


# ---------------------------------------------------------------------
# 4. Let the patient read the summary, wait until they're ready
# ---------------------------------------------------------------------
def comprehension_ready_node(state: HealthBotState) -> dict:
    interrupt({"prompt": "Take your time reading the summary above. Click 'Ready' when you'd like a comprehension check.", "summary": state["summary"]})
    return {}


# ---------------------------------------------------------------------
# 5. Generate one quiz question from the summary ONLY
# ---------------------------------------------------------------------
QUIZ_PROMPT = """Based ONLY on the following patient-friendly summary, write \
ONE quiz question that tests the patient's understanding. The question must \
be answerable using only the summary. Return only the question text, no \
preamble, no answer.

SUMMARY:
{summary}
"""


def quiz_node(state: HealthBotState) -> dict:
    response = llm.invoke([SystemMessage(content=QUIZ_PROMPT.format(summary=state["summary"]))])
    return {"quiz_question": response.content.strip()}


# ---------------------------------------------------------------------
# 6. Present the question, wait for the patient's answer
# ---------------------------------------------------------------------
def get_answer_node(state: HealthBotState) -> dict:
    answer = interrupt({"prompt": "Answer the question:", "question": state["quiz_question"]})
    return {"quiz_answer": answer}


# ---------------------------------------------------------------------
# 7. Grade the answer using ONLY the summary, with citations
# ---------------------------------------------------------------------
GRADE_PROMPT = """You are grading a patient's answer to a comprehension question. \
Use ONLY the summary below as your source of truth - do not use outside \
medical knowledge.

SUMMARY:
{summary}

QUESTION:
{question}

PATIENT'S ANSWER:
{answer}

Respond in this exact format:
Grade: <A/B/C/D/F>
Explanation: <1-2 sentences on why, quoting or closely referencing the \
relevant part of the summary that supports or corrects the answer>
"""


def grade_node(state: HealthBotState) -> dict:
    prompt = GRADE_PROMPT.format(
        summary=state["summary"],
        question=state["quiz_question"],
        answer=state["quiz_answer"],
    )
    response = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    grade, explanation = "N/A", response
    if "Grade:" in response and "Explanation:" in response:
        grade_part, explanation_part = response.split("Explanation:", 1)
        grade = grade_part.replace("Grade:", "").strip()
        explanation = explanation_part.strip()

    return {"grade": grade, "explanation": explanation}


# ---------------------------------------------------------------------
# 8. Present the grade, ask: new topic or exit?
# ---------------------------------------------------------------------
def continue_node(state: HealthBotState) -> dict:
    choice = interrupt({
        "prompt": "Would you like to learn about another topic, or exit?",
        "grade": state["grade"],
        "explanation": state["explanation"],
    })
    return {"want_new_topic": choice == "new"}


# ---------------------------------------------------------------------
# 9. Reset state before looping back, for privacy + accuracy
# ---------------------------------------------------------------------
def reset_node(state: HealthBotState) -> dict:
    return {
        "topic": "",
        "search_results": "",
        "summary": "",
        "quiz_question": "",
        "quiz_answer": "",
        "grade": "",
        "explanation": "",
        "want_new_topic": None,
    }
