"""Admissions chatbot — Phase 4. RAG over ASU corpus via Groq + ChromaDB."""
import time

import streamlit as st

st.set_page_config(
    page_title="Admissions Chatbot",
    page_icon=":speech_balloon:",
    layout="wide",
)
st.title("Admissions Chatbot")
st.caption("RAG · all-MiniLM-L6-v2 embeddings · Groq llama-3.1-8b-instant · ChromaDB")

SAMPLE_QUESTIONS = [
    "What are the application deadlines?",
    "What GPA do I need to get admitted?",
    "How do I apply for financial aid?",
    "What majors are available?",
    "Is there on-campus housing for freshmen?",
    "How do I contact the admissions office?",
]

# ── Load retriever (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def _load_retriever():
    from src.rag.retrieval import retrieve
    try:
        retrieve("admissions", k=1)
        return retrieve, None
    except Exception as e:
        return None, str(e)


retriever_fn, load_error = _load_retriever()

if load_error:
    st.error(
        f"Knowledge base unavailable: {load_error}\n\n"
        "Run `python src/rag/ingest.py` to build the vector index first."
    )
    st.stop()

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latencies" not in st.session_state:
    st.session_state.latencies = []

# ── Sample question chips ────────────────────────────────────────────────────
st.markdown("**Try a question:**")
cols = st.columns(len(SAMPLE_QUESTIONS))
prompt_from_chip = None
for col, q in zip(cols, SAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        prompt_from_chip = q

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src['source']}` (score: {src['score']:.3f})")

# ── Input ────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about admissions, deadlines, programs...") or prompt_from_chip

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()

        with st.spinner("Searching knowledge base..."):
            chunks = retriever_fn(user_input, k=5)

        if not chunks:
            answer = (
                "I couldn't find relevant information. "
                "Please contact the admissions office directly."
            )
            st.markdown(answer)
            sources = []
        else:
            from src.rag.generation import generate

            placeholder = st.empty()
            full_response = ""
            try:
                for token in generate(user_input, chunks, stream=True):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                answer = full_response
            except Exception as e:
                answer = f"[Generation error: {e}]"
                placeholder.markdown(answer)

            sources = chunks

        elapsed = time.perf_counter() - t0
        st.session_state.latencies.append(elapsed)

        if sources:
            with st.expander("Sources used", expanded=False):
                for src in sources:
                    st.markdown(f"- `{src['source']}` (score: {src['score']:.3f})")

        st.caption(f"Response time: {elapsed:.2f}s · Groq llama-3.1-8b-instant")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.latencies = []
        st.rerun()
    if st.session_state.latencies:
        avg_lat = sum(st.session_state.latencies) / len(st.session_state.latencies)
        st.metric("Avg response time", f"{avg_lat:.2f}s")
        st.metric("Messages", len(st.session_state.messages) // 2)
    st.markdown("---")
    st.caption(
        "Powered by [Groq](https://groq.com) + all-MiniLM-L6-v2 embeddings · "
        "Knowledge base: public university admissions corpus (redacted for deployment)"
    )
    st.caption(
        "Answers from fixed corpus only. For authoritative information, "
        "contact the admissions office."
    )
