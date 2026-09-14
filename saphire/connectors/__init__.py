"""Connectors turn a company's existing systems into Saphire tools and environments.

  sql.SQLConnector        – tables in any SQLAlchemy-supported database -> lookup/search/list tools
  openapi.OpenAPIConnector – REST endpoints described by an OpenAPI 3 spec -> tools (httpx)
  files.FileDataConnector – CSV/JSON exports -> in-memory tables -> tools
  mcp (saphire.sdk.tools.registry_from_mcp) – Model Context Protocol servers
  DataEnvironment          – wrap any registry + task list into an Environment with verifiers
"""
from .data_env import DataEnvironment, TaskVerifier
from .files import FileDataConnector
from .openapi import OpenAPIConnector
from .sql import SQLConnector

__all__ = ["DataEnvironment", "TaskVerifier", "FileDataConnector", "OpenAPIConnector", "SQLConnector"]
