"""
Streamlit front-end. Each rerun either:
  (a) starts the graph for the first time, or
  (b) resumes it from its last interrupt() with Command(resume=...)

The graph's own state (topic, summary, etc.) lives in LangGraph's
checkpointer, keyed by thread_id - NOT in st.session_state. Streamlit
only needs to remember: the thread_id, and the latest interrupt payload
so it knows what to render.
"""

import uuid
import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from graph import build_graph

load_dotenv("config.env")

st.set_page_config(page_title="HealthBot", page_icon="🩺")
st.title("🩺 HealthBot — Patient Education Assistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.graph = build_graph()
    st.session_state.interrupt = None
    st.session_state.done = False

config = {"configurable": {"thread_id": st.session_state.thread_id}}
graph = st.session_state.graph


def advance(resume_value=None):
    """Run the graph until it either finishes or hits the next interrupt()."""
    if resume_value is None:
        result = graph.invoke({}, config)
    else:
        result = graph.invoke(Command(resume=resume_value), config)

    if "__interrupt__" in result:
        st.session_state.interrupt = result["__interrupt__"][0].value
    else:
        st.session_state.interrupt = None
        st.session_state.done = True


# Kick off the graph on first load
if st.session_state.interrupt is None and not st.session_state.done:
    advance()

payload = st.session_state.interrupt

# ---------------------------------------------------------------------
# Render whichever step the graph is currently paused at
# ---------------------------------------------------------------------
if st.session_state.done:
    st.success("Session ended. Refresh the page to start over.")

elif payload and "question" not in payload and "grade" not in payload and "summary" not in payload:
    # get_topic step
    st.write(payload["prompt"])
    topic = st.text_input("Topic")
    if st.button("Submit", disabled=not topic):
        advance(topic)
        st.rerun()

elif payload and "summary" in payload:
    # comprehension_ready step
    st.subheader(f"About: {payload.get('summary', '')[:0]}")  # no-op placeholder for clarity
    st.markdown(payload["summary"])
    st.info(payload["prompt"])
    if st.button("I'm ready for the quiz"):
        advance("ready")
        st.rerun()

elif payload and "question" in payload:
    # get_answer step
    st.markdown(f"**Quiz question:** {payload['question']}")
    answer = st.text_area("Your answer")
    if st.button("Submit answer", disabled=not answer):
        advance(answer)
        st.rerun()

elif payload and "grade" in payload:
    # continue_check step
    st.markdown(f"**Grade:** {payload['grade']}")
    st.markdown(f"**Explanation:** {payload['explanation']}")
    col1, col2 = st.columns(2)
    if col1.button("Learn about another topic"):
        advance("new")
        st.rerun()
    if col2.button("Exit"):
        advance("exit")
        st.rerun()
