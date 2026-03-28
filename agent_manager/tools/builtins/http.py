"""Built-in HTTP request tool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec


class HttpRequestTool(BaseTool):
    spec = ToolSpec(
        name="http_request",
        description="Send an HTTP request and return the response payload.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "default": "GET"},
                "headers": {"type": "object"},
                "body": {"type": "string"},
                "json": {"type": "object"},
                "timeout_seconds": {"type": "number", "default": 30},
                "max_chars": {"type": "integer", "default": 20000},
            },
            "required": ["url"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "status": {"type": "integer"},
                "headers": {"type": "object"},
                "body": {"type": "string"},
                "truncated": {"type": "boolean"},
            },
            "required": ["url", "status", "headers", "body", "truncated"],
        },
        tags=["network", "http"],
        permissions=["network:request"],
        timeout_seconds=30.0,
    )

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        url = str(arguments["url"])
        method = str(arguments.get("method", "GET")).upper()
        headers = self._coerce_headers(arguments.get("headers"))
        timeout = float(arguments.get("timeout_seconds", self.spec.timeout_seconds))
        max_chars = max(int(arguments.get("max_chars", 20000)), 1)

        raw_json = arguments.get("json")
        raw_body = arguments.get("body")
        data: bytes | None = None
        if isinstance(raw_json, Mapping):
            data = json.dumps(dict(raw_json)).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        elif raw_body is not None:
            data = str(raw_body).encode("utf-8")

        try:
            response = await asyncio.to_thread(
                self._request,
                method,
                url,
                headers,
                data,
                timeout,
                max_chars,
            )
        except TimeoutError as exc:
            raise exc

        ok = 200 <= response["status"] < 400
        return ToolResult(
            tool_name=self.spec.name,
            ok=ok,
            output=response,
            error=None if ok else f"HTTP request failed with status {response['status']}.",
        )

    def _coerce_headers(self, value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            return {}
        return {str(key): str(item) for key, item in value.items()}

    def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None,
        timeout: float,
        max_chars: int,
    ) -> dict[str, Any]:
        request = Request(url=url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                response_body = response.read().decode("utf-8", errors="replace")
                response_headers = dict(response.headers.items())
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            status = exc.code
            response_headers = dict(exc.headers.items()) if exc.headers else {}
        except URLError as exc:
            return {
                "url": url,
                "status": 0,
                "headers": {},
                "body": "",
                "truncated": False,
                "reason": str(exc.reason),
            }
        except OSError as exc:
            raise TimeoutError(f"HTTP request failed: {exc}") from exc

        truncated = len(response_body) > max_chars
        if truncated:
            response_body = response_body[:max_chars]
        return {
            "url": url,
            "status": status,
            "headers": response_headers,
            "body": response_body,
            "truncated": truncated,
        }
