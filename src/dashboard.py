import pandas as pd
import streamlit as st

from db_query import get_conversations, get_stats

st.set_page_config(page_title="Earnings RAG Dashboard", page_icon="📊", layout="wide")
st.title("📊 Earnings RAG Dashboard")
st.caption("Monitoring for transcript questions, latency, cost, and feedback.")

stats = get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", stats.total)
col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
col3.metric("Total cost", f"${stats.total_cost:.4f}")
col4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")

col5, col6, col7 = st.columns(3)
col5.metric("User feedback", stats.total_feedback)
col6.metric("Thumbs up", stats.user_feedback_up)
col7.metric("Thumbs down", stats.user_feedback_down)

records = get_conversations(limit=200)
df = pd.DataFrame([r.__dict__ for r in records])

if not df.empty:
    df["created_at"] = pd.to_datetime(df["created_at"])

    st.subheader("Response time over time")
    st.line_chart(df, x="created_at", y="response_time")

    st.subheader("Cost over time")
    st.line_chart(df, x="created_at", y="cost")

    st.subheader("Token usage over time")
    token_df = df[["created_at", "prompt_tokens", "completion_tokens", "total_tokens"]].copy()
    token_df = token_df.fillna(0)
    st.line_chart(token_df, x="created_at", y=["prompt_tokens", "completion_tokens", "total_tokens"])

    st.subheader("Recent conversations")
    display_df = df[["created_at", "query", "answer", "response_time", "cost", "model"]].copy()
    display_df["query"] = display_df["query"].astype(str).str.slice(0, 100)
    display_df["answer"] = display_df["answer"].astype(str).str.slice(0, 160)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No conversations found yet. Ask a question in the app first.")
