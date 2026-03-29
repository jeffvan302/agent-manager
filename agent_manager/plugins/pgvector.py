"""Optional adapter for pgvector (PostgreSQL vector) retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_manager.memory.retrieval import BaseRetriever, RetrievalResult
from agent_manager.plugins.base import Plugin
from agent_manager.tools.builtins.retrieval import RetrieveDocumentsTool


class PgVectorRetrieverAdapter(BaseRetriever):
    """Wrap a PostgreSQL connection with pgvector as an agent_manager retriever.

    The adapter expects:
    - A DB-API 2.0 compatible connection (e.g. ``psycopg2`` or ``psycopg``).
    - A table with columns: ``id``, ``content``, ``embedding``, ``metadata`` (JSONB).
    - An embedding function ``embed_fn(str) -> list[float]``.

    Usage::

        import psycopg2
        conn = psycopg2.connect(dsn)
        retriever = PgVectorRetrieverAdapter(
            connection=conn,
            table_name="documents",
            embed_fn=my_embed_fn,
        )
    """

    def __init__(
        self,
        *,
        connection: Any,
        table_name: str = "documents",
        embed_fn: Any,
        id_column: str = "id",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
    ) -> None:
        self._conn = connection
        self._table = table_name
        self._embed_fn = embed_fn
        self._id_col = id_column
        self._content_col = content_column
        self._embedding_col = embedding_column
        self._metadata_col = metadata_column

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self._embed_fn(query)
        embedding_literal = self._format_vector(query_embedding)

        where_clauses: list[str] = []
        params: list[Any] = []

        if metadata_filter:
            for key, value in metadata_filter.items():
                where_clauses.append(f"{self._metadata_col}->>%s = %s")
                params.extend([str(key), str(value)])

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = (
            f"SELECT {self._id_col}, {self._content_col}, {self._metadata_col}, "
            f"{self._embedding_col} <-> '{embedding_literal}'::vector AS distance "
            f"FROM {self._table} "
            f"{where_sql} "
            f"ORDER BY distance ASC "
            f"LIMIT %s"
        )
        params.append(max(top_k, 1))

        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()

        results: list[RetrievalResult] = []
        for row in rows:
            doc_id, content, metadata_raw, distance = row
            metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
            score = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievalResult(
                    id=str(doc_id),
                    content=str(content),
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    @staticmethod
    def _format_vector(vector: Any) -> str:
        """Format a vector as a pgvector literal ``[0.1,0.2,...]``."""
        values = [str(float(v)) for v in vector]
        return "[" + ",".join(values) + "]"


class PgVectorRetrievalPlugin(Plugin):
    """Register a pgvector-backed retriever and retrieval tool on a session."""

    name = "pgvector-retrieval"
    description = "Expose a pgvector PostgreSQL table through the unified retrieval tool."

    def __init__(
        self,
        *,
        connection: Any,
        table_name: str = "documents",
        embed_fn: Any,
        retriever_name: str = "pgvector",
        set_default: bool = True,
        id_column: str = "id",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
    ) -> None:
        self._adapter = PgVectorRetrieverAdapter(
            connection=connection,
            table_name=table_name,
            embed_fn=embed_fn,
            id_column=id_column,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_column=metadata_column,
        )
        self._retriever_name = retriever_name
        self._set_default = set_default

    def register(self, target: Any) -> None:
        target.register_retriever(
            self._retriever_name,
            self._adapter,
            make_default=self._set_default,
        )
        target.register_tool(RetrieveDocumentsTool(self._adapter), replace=True)
