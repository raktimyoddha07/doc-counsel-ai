import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory


def make_langchain_session_id(*, user_id: int, document_id: int) -> str:
    # Session-scoping mirrors the UI: chat memory is per (user, PDF session).
    return f"auditlens:user:{user_id}:doc:{document_id}"


def citations_from_answer_text(answer_text: str) -> List[int]:
    pages: List[int] = []
    for m in re.finditer(r"\[Page\s+(\d+)\]", answer_text or ""):
        try:
            pages.append(int(m.group(1)))
        except Exception:
            pass
    return sorted(set(pages))


@asynccontextmanager
async def _langchain_history(
    *,
    connection_string: str,
    table_name: str,
    session_id: str,
):
    # `langchain_postgres` expects psycopg connection objects, not raw connection strings.
    import psycopg

    async_conn = await psycopg.AsyncConnection.connect(connection_string)
    try:
        history = PostgresChatMessageHistory(
            table_name=table_name,
            session_id=session_id,
            async_connection=async_conn,
        )
        yield history
    finally:
        try:
            await async_conn.close()
        except Exception:
            pass


async def ensure_langchain_message_history_table(*, connection_string: str, table_name: str) -> None:
    import psycopg

    async_conn = await psycopg.AsyncConnection.connect(connection_string)
    try:
        await PostgresChatMessageHistory.acreate_tables(async_conn, table_name)
    finally:
        try:
            await async_conn.close()
        except Exception:
            pass


async def get_recent_chat_context(
    *,
    connection_string: str,
    table_name: str,
    session_id: str,
    limit: int = 3,
) -> str:
    """
    Returns a short, passive chat history snippet to help the model interpret follow-up questions
    within the same PDF session.
    """
    async with _langchain_history(
        connection_string=connection_string,
        table_name=table_name,
        session_id=session_id,
    ) as history:
        messages = await history.aget_messages()

    # Pair up (Human -> AI).
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(messages) - 1):
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i + 1], AIMessage):
            q = messages[i].content or ""
            a = messages[i + 1].content or ""
            if q.strip() and a.strip():
                pairs.append((q, a))

    if not pairs:
        return ""

    # Keep last `limit` pairs, but render in chronological order.
    pairs = pairs[-limit:]
    parts = [f"Q: {q}\nA: {a}" for (q, a) in pairs]
    return "\n\n".join(parts).strip()


async def add_chat_messages(
    *,
    connection_string: str,
    table_name: str,
    session_id: str,
    question: str,
    assistant_answer: str,
) -> None:
    async with _langchain_history(
        connection_string=connection_string,
        table_name=table_name,
        session_id=session_id,
    ) as history:
        await history.aadd_messages(
            [HumanMessage(content=question), AIMessage(content=assistant_answer)]
        )


async def list_document_chats_from_history(
    *,
    connection_string: str,
    table_name: str,
    session_id: str,
) -> List[Dict[str, Any]]:
    async with _langchain_history(
        connection_string=connection_string,
        table_name=table_name,
        session_id=session_id,
    ) as history:
        messages = await history.aget_messages()

    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(messages) - 1):
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i + 1], AIMessage):
            q = messages[i].content or ""
            a = messages[i + 1].content or ""
            if q.strip() and a.strip():
                pairs.append((q, a))

    out: List[Dict[str, Any]] = []
    # Preserve chronological order.
    for idx, (q, a) in enumerate(pairs, start=1):
        out.append(
            {
                "id": idx,
                "created_at": "",  # not available via PostgresChatMessageHistory API
                "question": q,
                "assistant_answer": a,
                "citation_pages": citations_from_answer_text(a),
            }
        )
    return out

