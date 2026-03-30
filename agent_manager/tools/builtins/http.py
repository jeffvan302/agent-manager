"""Built-in HTTP request tool."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_manager.errors import ConfigurationError
from agent_manager.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec


class HttpRequestTool(BaseTool):
    spec = ToolSpec(
        name="http_request",
        description="Send an HTTP request or browser-backed page fetch and return the response payload.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "default": "GET"},
                "engine": {"type": "string", "default": "browser"},
                "response_format": {"type": "string", "default": "text"},
                "headers": {"type": "object"},
                "body": {"type": "string"},
                "json": {"type": "object"},
                "timeout_seconds": {"type": "number", "default": 30},
                "max_chars": {"type": "integer", "default": 20000},
                "headless": {"type": "boolean", "default": True},
                "cookie_file": {"type": "string"},
                "output_path": {"type": "string"},
                "page_format": {"type": "string", "default": "A4"},
                "print_background": {"type": "boolean", "default": True},
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
                "engine": {"type": "string"},
                "content_format": {"type": "string"},
                "path": {"type": ["string", "null"]},
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
        engine = str(arguments.get("engine", "browser")).strip().lower() or "browser"
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

        if self._should_use_browser_fetch(
            url=url,
            method=method,
            engine=engine,
            raw_json=raw_json,
            raw_body=raw_body,
        ):
            response = await asyncio.to_thread(
                self._browser_fetch_request,
                url,
                str(arguments.get("response_format", "text")).strip().lower() or "text",
                self._coerce_bool(arguments.get("headless", True)),
                arguments.get("cookie_file"),
                arguments.get("output_path"),
                str(arguments.get("page_format", "A4")),
                self._coerce_bool(arguments.get("print_background", True)),
                max_chars,
            )
        else:
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

    def _should_use_browser_fetch(
        self,
        *,
        url: str,
        method: str,
        engine: str,
        raw_json: Any,
        raw_body: Any,
    ) -> bool:
        if engine == "raw":
            return False
        if method != "GET":
            return False
        if raw_json is not None or raw_body is not None:
            return False
        return url.startswith(("http://", "https://"))

    def _coerce_headers(self, value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            return {}
        return {str(key): str(item) for key, item in value.items()}

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _browser_fetch_request(
        self,
        url: str,
        response_format: str,
        headless: bool,
        cookie_file: Any,
        output_path: Any,
        page_format: str,
        print_background: bool,
        max_chars: int,
    ) -> dict[str, Any]:
        try:
            from google_search_tool import get_page, get_pdf
        except ImportError as exc:  # pragma: no cover - depends on local package
            raise ConfigurationError(
                "Browser-backed http_request requires the installed 'google_search_tool' package. "
                "See google_search_tool.md for setup."
            ) from exc

        normalized_format = response_format if response_format in {"text", "html", "pdf"} else "text"
        if normalized_format == "pdf":
            result = get_pdf(
                url,
                output_path=output_path,
                headless=headless,
                cookie_file=cookie_file,
                page_format=page_format,
                print_background=print_background,
            )
            pdf_bytes = bytes(result.get("pdf_bytes", b""))
            body = base64.b64encode(pdf_bytes).decode("ascii")
            truncated = len(body) > max_chars
            if truncated:
                body = body[:max_chars]
            return {
                "url": str(result.get("url", url)),
                "status": 200,
                "headers": {
                    "Content-Type": "application/pdf",
                    "Content-Transfer-Encoding": "base64",
                },
                "body": body,
                "truncated": truncated,
                "engine": "browser",
                "content_format": "pdf",
                "path": result.get("path"),
            }

        result = get_page(
            url,
            headless=headless,
            cookie_file=cookie_file,
        )
        body = str(result.get("html" if normalized_format == "html" else "text", ""))
        truncated = len(body) > max_chars
        if truncated:
            body = body[:max_chars]
        return {
            "url": str(result.get("url", url)),
            "status": 200,
            "headers": {
                "Content-Type": (
                    "text/html; charset=utf-8"
                    if normalized_format == "html"
                    else "text/plain; charset=utf-8"
                )
            },
            "body": body,
            "truncated": truncated,
            "engine": "browser",
            "content_format": normalized_format,
            "path": None,
        }

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
                "engine": "raw",
                "content_format": "text",
                "path": None,
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
            "engine": "raw",
            "content_format": "text",
            "path": None,
        }
