"""RAG generation — Phase 4.

Groq llama-3.1-8b-instant, streaming. System prompt enforces:
- Answer only from provided context
- Cite source chunks
- Defer to human advisors for complex cases
Fallback: local Ollama.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_TOKENS = 512
MAX_CONTEXT_CHARS = 6_000

SYSTEM_PROMPT = """\
You are an admissions assistant for a public research university.

Rules you must follow without exception:
1. Answer ONLY using the context sections provided. If the answer is not present, respond: "I don't have that information — please contact the admissions office directly."
2. Cite the source for every factual claim: e.g. "According to [source]: ..."
3. For questions about individual application status, financial aid decisions, or deadline exceptions, always say: "For personalized guidance, please speak with an admissions advisor."
4. Be concise. Use bullet points for multi-part answers.\
"""


def _get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GROQ_API_KEY not found in environment or Streamlit secrets."
        )


def _build_user_message(question: str, chunks: list[dict]) -> str:
    parts = []
    budget = MAX_CONTEXT_CHARS
    for chunk in chunks:
        snippet = chunk["text"][: budget // max(len(chunks), 1)]
        parts.append(f"[Source: {chunk['source']}]\n{snippet}")
        budget -= len(snippet)
        if budget <= 0:
            break
    context = "\n\n---\n\n".join(parts)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(
    question: str,
    context_chunks: list[dict],
    stream: bool = True,
) -> Iterator[str] | str:
    """Generate answer grounded in context_chunks. Streams by default."""
    message = _build_user_message(question, context_chunks)
    try:
        return _groq_generate(message, stream=stream)
    except Exception as groq_err:
        try:
            return _ollama_generate(message, stream=stream)
        except Exception:
            err_text = f"[Generation unavailable: {groq_err}]"
            if stream:
                def _err() -> Iterator[str]:
                    yield err_text
                return _err()
            return err_text


def _groq_generate(message: str, stream: bool) -> Iterator[str] | str:
    from groq import Groq

    client = Groq(api_key=_get_api_key())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    if stream:
        def _stream() -> Iterator[str]:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                stream=True,
                max_tokens=MAX_TOKENS,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _stream()
    else:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            stream=False,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content


def _ollama_generate(message: str, stream: bool) -> Iterator[str] | str:
    import json

    import requests

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "stream": stream,
    }
    if stream:
        def _stream() -> Iterator[str]:
            r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
        return _stream()
    else:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return r.json()["message"]["content"]
