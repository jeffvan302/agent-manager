"""Optional adapters for TinyDB-backed tools."""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from agent_manager.plugins.base import Plugin
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec

READ_OPERATIONS = frozenset({"all", "count", "get", "search"})
WRITE_OPERATIONS = frozenset({"insert", "remove", "update", "upsert"})
DEFAULT_OPERATIONS = tuple(sorted(READ_OPERATIONS | WRITE_OPERATIONS))


class TinyDBToolAdapter(BaseTool):
    """Expose a TinyDB database or table as a configurable runtime tool."""

    def __init__(
        self,
        *,
        database: Any | None = None,
        path: str | Path | None = None,
        table_name: str | None = None,
        tool_name: str = "tinydb",
        description: str | None = None,
        allowed_operations: Iterable[str] | None = None,
    ) -> None:
        if database is None and path is None:
            raise ValueError("TinyDBToolAdapter requires a database object or a path.")
        self._database = database
        self._path = Path(path).resolve(strict=False) if path is not None else None
        self._table_name = table_name
        self._client: Any | None = None
        self._allowed_operations = self._normalize_operations(allowed_operations)
        permissions = ["database:query"]
        if self._allowed_operations & WRITE_OPERATIONS:
            permissions.append("database:write")
        self.spec = ToolSpec(
            name=tool_name,
            description=description
            or self._default_description(tool_name=tool_name, table_name=table_name),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(sorted(self._allowed_operations)),
                    },
                    "document": {"type": "object"},
                    "fields": {"type": "object"},
                    "query": {"type": "object"},
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["action"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "table": {"type": "string"},
                    "count": {"type": "integer"},
                    "documents": {"type": "array"},
                    "document": {"type": "object"},
                    "doc_ids": {"type": "array"},
                },
            },
            tags=["database", "tinydb"],
            permissions=permissions,
        )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        action = str(arguments.get("action", "")).strip().lower()
        if action not in self._allowed_operations:
            return ToolResult(
                tool_name=self.spec.name,
                ok=False,
                output={},
                error=(
                    f"Unsupported TinyDB action '{action}'. "
                    f"Allowed actions: {', '.join(sorted(self._allowed_operations))}."
                ),
            )

        try:
            payload = await asyncio.to_thread(self._execute_action, action, arguments)
        except Exception as exc:
            return ToolResult(
                tool_name=self.spec.name,
                ok=False,
                output={},
                error=str(exc),
            )

        return ToolResult(
            tool_name=self.spec.name,
            ok=True,
            output=payload,
            metadata={"action": action, "table": payload.get("table")},
        )

    def _execute_action(self, action: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        table = self._resolve_table()
        table_name = self._resolved_table_name()
        filters = self._coerce_query(arguments.get("query"))
        doc_ids = self._coerce_doc_ids(arguments.get("doc_ids"))
        predicate = self._build_predicate(filters)

        if action == "insert":
            document = self._coerce_mapping(arguments.get("document"), field_name="document")
            inserted_id = table.insert(document)
            return {
                "action": action,
                "table": table_name,
                "count": 1,
                "doc_ids": [inserted_id],
            }

        if action == "upsert":
            document = self._coerce_mapping(arguments.get("document"), field_name="document")
            if doc_ids:
                updated_ids = table.update(document, doc_ids=doc_ids)
                if updated_ids:
                    return {
                        "action": action,
                        "table": table_name,
                        "count": len(updated_ids),
                        "doc_ids": list(updated_ids),
                    }
                inserted_id = table.insert(document)
                return {
                    "action": action,
                    "table": table_name,
                    "count": 1,
                    "doc_ids": [inserted_id],
                }
            if predicate is None:
                inserted_id = table.insert(document)
                return {
                    "action": action,
                    "table": table_name,
                    "count": 1,
                    "doc_ids": [inserted_id],
                }
            upserted_ids = table.upsert(document, predicate if predicate is not None else None)
            return {
                "action": action,
                "table": table_name,
                "count": len(upserted_ids),
                "doc_ids": list(upserted_ids),
            }

        if action == "all":
            documents = [self._serialize_document(doc) for doc in table.all()]
            return {
                "action": action,
                "table": table_name,
                "count": len(documents),
                "documents": documents,
            }

        if action == "count":
            documents = self._collect_matching_documents(table, predicate, doc_ids)
            return {
                "action": action,
                "table": table_name,
                "count": len(documents),
            }

        if action == "search":
            documents = self._collect_matching_documents(table, predicate, doc_ids)
            limit = max(int(arguments.get("limit", 20)), 1)
            serialized = [self._serialize_document(doc) for doc in documents[:limit]]
            return {
                "action": action,
                "table": table_name,
                "count": len(serialized),
                "documents": serialized,
            }

        if action == "get":
            document = self._get_single_document(table, predicate, doc_ids)
            return {
                "action": action,
                "table": table_name,
                "count": 0 if document is None else 1,
                "document": self._serialize_document(document) if document is not None else None,
            }

        if action == "update":
            fields = self._coerce_mapping(arguments.get("fields"), field_name="fields")
            self._require_filter(action, filters, doc_ids)
            updated_ids = table.update(
                fields,
                cond=predicate if doc_ids is None else None,
                doc_ids=doc_ids,
            )
            return {
                "action": action,
                "table": table_name,
                "count": len(updated_ids),
                "doc_ids": list(updated_ids),
            }

        if action == "remove":
            self._require_filter(action, filters, doc_ids)
            removed_ids = table.remove(
                cond=predicate if doc_ids is None else None,
                doc_ids=doc_ids,
            )
            return {
                "action": action,
                "table": table_name,
                "count": len(removed_ids),
                "doc_ids": list(removed_ids),
            }

        raise ValueError(f"Unhandled TinyDB action '{action}'.")

    def _resolve_table(self) -> Any:
        database = self._resolve_database()
        if self._table_name and hasattr(database, "table"):
            return database.table(self._table_name)
        return database

    def _resolve_database(self) -> Any:
        if self._database is not None:
            return self._database
        if self._client is None:
            from tinydb import TinyDB  # type: ignore[import-not-found]

            self._client = TinyDB(self._path)
        return self._client

    def _resolved_table_name(self) -> str:
        return self._table_name or "_default"

    def _collect_matching_documents(
        self,
        table: Any,
        predicate: Any,
        doc_ids: list[int] | None,
    ) -> list[Any]:
        if doc_ids:
            documents = [
                table.get(doc_id=doc_id)
                for doc_id in doc_ids
            ]
            return [doc for doc in documents if doc is not None]
        if predicate is not None:
            return list(table.search(predicate))
        return list(table.all())

    def _get_single_document(
        self,
        table: Any,
        predicate: Any,
        doc_ids: list[int] | None,
    ) -> Any | None:
        if doc_ids:
            return table.get(doc_id=doc_ids[0])
        if predicate is not None:
            return table.get(predicate)
        documents = table.all()
        return documents[0] if documents else None

    def _build_predicate(self, filters: dict[str, Any]) -> Any | None:
        if not filters:
            return None

        def _predicate(document: Mapping[str, Any]) -> bool:
            return all(self._lookup_field(document, key) == expected for key, expected in filters.items())

        return _predicate

    def _lookup_field(self, document: Mapping[str, Any], key: str) -> Any:
        current: Any = document
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    def _serialize_document(self, document: Any) -> dict[str, Any]:
        if document is None:
            return {}
        payload = dict(document) if isinstance(document, Mapping) else {"value": document}
        doc_id = getattr(document, "doc_id", None)
        if doc_id is not None:
            payload["doc_id"] = doc_id
        return payload

    def _coerce_query(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("TinyDB query must be an object mapping field names to values.")
        return {str(key): item for key, item in value.items()}

    def _coerce_mapping(self, value: Any, *, field_name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"TinyDB field '{field_name}' must be an object.")
        return {str(key): item for key, item in value.items()}

    def _coerce_doc_ids(self, value: Any) -> list[int] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("TinyDB 'doc_ids' must be an array of integers.")
        return [int(item) for item in value]

    def _require_filter(
        self,
        action: str,
        filters: Mapping[str, Any],
        doc_ids: list[int] | None,
    ) -> None:
        if filters or doc_ids:
            return
        raise ValueError(
            f"TinyDB action '{action}' requires either 'query' or 'doc_ids' to avoid bulk writes."
        )

    def _normalize_operations(self, value: Iterable[str] | None) -> set[str]:
        raw = value or DEFAULT_OPERATIONS
        operations = {str(item).strip().lower() for item in raw if str(item).strip()}
        if not operations:
            raise ValueError("TinyDB allowed operations must not be empty.")
        unsupported = operations - (READ_OPERATIONS | WRITE_OPERATIONS)
        if unsupported:
            raise ValueError(
                f"Unsupported TinyDB operations: {', '.join(sorted(unsupported))}."
            )
        return operations

    def _default_description(self, *, tool_name: str, table_name: str | None) -> str:
        table_label = table_name or "the default TinyDB table"
        return (
            f"Query and mutate {table_label} through the '{tool_name}' TinyDB wrapper."
        )


class TinyDBToolsPlugin(Plugin):
    """Register a TinyDB-backed tool on an AgentSession."""

    name = "tinydb-tools"
    description = "Expose a TinyDB database or table through the unified tool registry."

    def __init__(
        self,
        *,
        database: Any | None = None,
        path: str | Path | None = None,
        table_name: str | None = None,
        tool_name: str = "tinydb",
        description: str | None = None,
        allowed_operations: Iterable[str] | None = None,
        replace: bool = True,
    ) -> None:
        self._database = database
        self._path = path
        self._table_name = table_name
        self._tool_name = tool_name
        self._description = description
        self._allowed_operations = tuple(allowed_operations or DEFAULT_OPERATIONS)
        self._replace = replace

    def is_available(self) -> bool:
        if self._database is not None:
            return True
        return importlib.util.find_spec("tinydb") is not None

    def register(self, target: Any) -> None:
        target.register_tool(
            TinyDBToolAdapter(
                database=self._database,
                path=self._path,
                table_name=self._table_name,
                tool_name=self._tool_name,
                description=self._description,
                allowed_operations=self._allowed_operations,
            ),
            replace=self._replace,
        )
