
import streamlit as st
from dotenv import load_dotenv

from db_save import save_conversation, save_feedback
from src.rag_helper_project_rewrite import ask, ragger

load_dotenv()

st.set_page_config(page_title="Earnings Call RAG", page_icon="📈", layout="wide")
st.title("📈 Earnings Call RAG Assistant")
st.caption("Ask questions about earnings call transcripts, guidance, risks, and forward-looking statements.")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top-K chunks", min_value=1, max_value=10, value=5, step=1)
    show_prompt = st.checkbox("Show prompt", value=False)
    st.markdown("---")
    st.write("**Example queries**")
    st.code(
        "What did Apple say about margins?\nWhat risks did management mention?\nHow was guidance for next quarter?",
        language="text",
    )

query = st.text_input("Enter your question", placeholder="e.g. What did Apple say about margins?")

ask_col, clear_col = st.columns([1, 1])
ask_clicked = ask_col.button("Ask", type="primary")
clear_clicked = clear_col.button("Clear")

if clear_clicked:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_conversation_id", None)
    st.rerun()

if ask_clicked and query.strip():
    with st.spinner("Searching transcripts and generating an answer..."):
        result = ask(query.strip(), limit=int(top_k))
        st.session_state["last_result"] = result
        st.session_state["last_query"] = query.strip()

        conversation_id = save_conversation(ragger.last_call, query.strip(), source="earnings_rag")
        st.session_state["last_conversation_id"] = conversation_id

result = st.session_state.get("last_result")

if result:
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    sources = result.get("sources", [])
    if sources:
        rows = []
        for i, src in enumerate(sources, start=1):
            rows.append(
                {
                    "rank": i,
                    "symbol": src.get("symbol"),
                    "year": src.get("year"),
                    "quarter": src.get("quarter"),
                    "text": src.get("text", "")[:250],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No sources returned.")

    if show_prompt:
        st.subheader("Prompt")
        st.code(result.get("prompt", ""), language="text")

    st.subheader("Feedback")
    feedback_col1, feedback_col2 = st.columns(2)

    if feedback_col1.button("👍 Helpful"):
        cid = st.session_state.get("last_conversation_id")
        if cid:
            save_feedback(cid, "user", score=1)
            st.success("Thanks for the feedback!")
        else:
            st.warning("No conversation id found.")

    if feedback_col2.button("👎 Not helpful"):
        cid = st.session_state.get("last_conversation_id")
        if cid:
            save_feedback(cid, "user", score=-1)
            st.warning("Thanks — this helps improve retrieval and prompting.")
        else:
            st.warning("No conversation id found.")
else:
    st.info("Ask a question to search the earnings transcripts.")
