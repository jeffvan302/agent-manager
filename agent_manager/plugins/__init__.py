"""Plugin exports."""

from agent_manager.plugins.base import Plugin
from agent_manager.plugins.chroma import ChromaRetrieverAdapter, ChromaRetrievalPlugin
from agent_manager.plugins.embeddings import (
    BaseEmbeddingProvider,
    CallableEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from agent_manager.plugins.export import (
    export_all,
    to_langchain_tool_definition,
    to_mcp_tool_definition,
    to_openai_function,
    to_openapi_schema,
)
from agent_manager.plugins.faiss_store import FAISSDocument, FAISSRetrieverAdapter, FAISSRetrievalPlugin
from agent_manager.plugins.langchain import LangChainToolAdapter, LangChainToolsPlugin
from agent_manager.plugins.llamaindex import (
    LlamaIndexRetrieverAdapter,
    LlamaIndexRetrievalPlugin,
)
from agent_manager.plugins.mcp import MCPToolAdapter, MCPToolsPlugin
from agent_manager.plugins.openapi import OpenAPIOperation, OpenAPIToolAdapter, OpenAPIToolsPlugin
from agent_manager.plugins.pgvector import PgVectorRetrieverAdapter, PgVectorRetrievalPlugin
from agent_manager.plugins.registry import PluginRegistry
from agent_manager.plugins.tinydb import TinyDBToolAdapter, TinyDBToolsPlugin

__all__ = [
    "BaseEmbeddingProvider",
    "CallableEmbeddingProvider",
    "ChromaRetrieverAdapter",
    "ChromaRetrievalPlugin",
    "FAISSDocument",
    "FAISSRetrieverAdapter",
    "FAISSRetrievalPlugin",
    "LangChainToolAdapter",
    "LangChainToolsPlugin",
    "LlamaIndexRetrieverAdapter",
    "LlamaIndexRetrievalPlugin",
    "MCPToolAdapter",
    "MCPToolsPlugin",
    "OpenAIEmbeddingProvider",
    "OpenAPIOperation",
    "OpenAPIToolAdapter",
    "OpenAPIToolsPlugin",
    "PgVectorRetrieverAdapter",
    "PgVectorRetrievalPlugin",
    "Plugin",
    "PluginRegistry",
    "SentenceTransformerEmbeddingProvider",
    "TinyDBToolAdapter",
    "TinyDBToolsPlugin",
    "export_all",
    "to_langchain_tool_definition",
    "to_mcp_tool_definition",
    "to_openai_function",
    "to_openapi_schema",
]
