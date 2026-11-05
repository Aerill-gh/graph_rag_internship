"""Local ingestion-only MCP endpoint; does not register chatbot tools."""

import argparse
from contextvars import ContextVar
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from neo4j import GraphDatabase

from .config import PROJECT_ROOT, get_rest_api_port
from .database_mcp.contracts import (
    ChunkReadResponse,
    ChunkWrite,
    ChunkWriteResponse,
    CorpusSummaryResponse,
    ExploreEntityRequest,
    ExploreEntityResponse,
    ExtractionWrite,
    ExtractionWriteResponse,
    FindEntitiesRequest,
    FindEntitiesResponse,
    HealthResponse,
    QueryGraphRequest,
    QueryGraphResponse,
    SchemaResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
)
from .database_mcp.config import (
    DatabaseSettings,
    Neo4jDesktopSettings,
    database_mcp_url,
    load_database_settings,
    load_neo4j_desktop_settings,
)


logger = logging.getLogger(__name__)


class DatabaseBackend(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def health_check(self) -> HealthResponse: ...
    def ensure_schema(self) -> SchemaResponse: ...
    def upsert_chunk(self, request: ChunkWrite) -> ChunkWriteResponse: ...
    def upsert_extraction(self, request: ExtractionWrite) -> ExtractionWriteResponse: ...
    def get_chunk(self, chunk_id: str) -> ChunkReadResponse: ...
    def search_documents(
        self, request: SearchDocumentsRequest,
    ) -> SearchDocumentsResponse: ...
    def find_entities(self, request: FindEntitiesRequest) -> FindEntitiesResponse: ...
    def explore_entity(self, request: ExploreEntityRequest) -> ExploreEntityResponse: ...
    def query_graph(self, request: QueryGraphRequest) -> QueryGraphResponse: ...
    def corpus_summary(self) -> CorpusSummaryResponse: ...


def create_mcp_server(backend: DatabaseBackend | None = None) -> FastMCP:
    backend_context: ContextVar[DatabaseBackend | None] = ContextVar(
        "database_mcp_backend", default=None,
    )

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[None]:
        session_backend = backend
        if session_backend is None:
            from graphrag.database_mcp.config import load_database_settings
            from graphrag.database_mcp.neo4j_backend import Neo4jBackend

            session_backend = Neo4jBackend(load_database_settings())
        token = backend_context.set(session_backend)
        try:
            await to_thread.run_sync(session_backend.open)
            yield None
        finally:
            backend_context.reset(token)
            await to_thread.run_sync(session_backend.close)

    server = FastMCP(
        "Zava database ingestion",
        host="127.0.0.1",
        port=8002,
        streamable_http_path="/mcp",
        stateless_http=False,
        json_response=True,
        lifespan=lifespan,
    )

    def database() -> DatabaseBackend:
        active_backend = backend_context.get()
        if active_backend is None:
            raise RuntimeError("Database MCP backend is unavailable.")
        return active_backend

    read_annotations = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False,
        idempotentHint=True, openWorldHint=False,
    )
    write_annotations = ToolAnnotations(
        readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    )

    @server.tool(annotations=read_annotations)
    async def health_check() -> HealthResponse:
        """Verify the MCP server can connect to its database."""
        return await to_thread.run_sync(database().health_check)

    @server.tool(annotations=write_annotations)
    async def ensure_schema() -> SchemaResponse:
        """Ensure source constraints and the existing 1536-dimensional vector index."""
        return await to_thread.run_sync(database().ensure_schema)

    @server.tool(annotations=write_annotations)
    async def upsert_chunk(request: ChunkWrite) -> ChunkWriteResponse:
        """Store a validated source chunk; preserve its vector if no embedding is supplied."""
        validated = ChunkWrite.model_validate(request.model_dump())
        return await to_thread.run_sync(database().upsert_chunk, validated)

    @server.tool(annotations=write_annotations)
    async def upsert_extraction(request: ExtractionWrite) -> ExtractionWriteResponse:
        """Atomically store allowed graph facts linked to an existing source chunk."""
        validated = ExtractionWrite.model_validate(request.model_dump())
        return await to_thread.run_sync(database().upsert_extraction, validated)

    @server.tool(annotations=read_annotations)
    async def get_chunk(chunk_id: str) -> ChunkReadResponse:
        """Read one source chunk, its vector, and up to 50 source-backed entities and facts."""
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Source chunk ID cannot be blank.")
        return await to_thread.run_sync(database().get_chunk, chunk_id)

    @server.tool(annotations=read_annotations)
    async def search_documents(
        embedding: list[float], top_k: int = 5,
    ) -> SearchDocumentsResponse:
        """Search indexed document chunks using a 1536-dimensional query embedding."""
        request = SearchDocumentsRequest(embedding=embedding, top_k=top_k)
        return await to_thread.run_sync(database().search_documents, request)

    @server.tool(annotations=read_annotations)
    async def find_entities(name: str, limit: int = 10) -> FindEntitiesResponse:
        """Find allowed ontology entities by a case-insensitive name substring."""
        request = FindEntitiesRequest(name=name, limit=limit)
        return await to_thread.run_sync(database().find_entities, request)

    @server.tool(annotations=read_annotations)
    async def explore_entity(
        name: str, label: str, limit: int = 20,
    ) -> ExploreEntityResponse:
        """Read bounded, source-backed facts attached to one named graph entity."""
        request = ExploreEntityRequest(name=name, label=label, limit=limit)
        return await to_thread.run_sync(database().explore_entity, request)

    @server.tool(annotations=read_annotations)
    async def query_graph(
        pattern: str,
        source_names: list[str],
        target_name: str = "",
        match_mode: Literal["any", "all", "difference"] = "any",
        limit: int = 20,
    ) -> QueryGraphResponse:
        """Query bounded graph facts using an allowlisted ontology pattern, never raw Cypher."""
        request = QueryGraphRequest(
            pattern=pattern, source_names=source_names, target_name=target_name,
            match_mode=match_mode, limit=limit,
        )
        return await to_thread.run_sync(database().query_graph, request)

    @server.tool(annotations=read_annotations)
    async def corpus_summary() -> CorpusSummaryResponse:
        """Return bounded corpus counts and a digest without exposing full graph contents."""
        return await to_thread.run_sync(database().corpus_summary)

    return server


@dataclass(frozen=True)
class LocalService:
    name: str
    port: int
    probe: str | None = None


def _resolve_dbms_home(
    database: str, desktop: Neo4jDesktopSettings,
) -> Path:
    executable = "neo4j.bat" if os.name == "nt" else "neo4j"

    def supports_database(path: Path) -> bool:
        return (
            (path / "bin" / executable).is_file()
            and (path / "data" / "databases" / database).is_dir()
        )

    if desktop.dbms_path is not None:
        path = desktop.dbms_path.resolve()
        if not supports_database(path):
            raise RuntimeError(
                f"Configured Neo4j DBMS path does not contain database '{database}'."
            )
        return path

    dbms_root = desktop.home / "Data" / "dbmss"
    matches = [path for path in dbms_root.glob("dbms-*") if supports_database(path)]
    if not matches:
        raise RuntimeError(
            f"Could not find a Neo4j Desktop DBMS containing database '{database}' "
            f"under {dbms_root}. Set NEO4J_DESKTOP_DBMS_PATH in .env.database-mcp."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple Neo4j Desktop DBMS instances contain '{database}'. "
            "Set NEO4J_DESKTOP_DBMS_PATH in .env.database-mcp to select one."
        )
    return matches[0].resolve()


def _resolve_java_home(desktop: Neo4jDesktopSettings) -> Path:
    java_name = "java.exe" if os.name == "nt" else "java"

    def is_java_home(path: Path) -> bool:
        return (path / "bin" / java_name).is_file()

    configured = desktop.java_home or (Path(os.environ["JAVA_HOME"]) if os.getenv("JAVA_HOME") else None)
    if configured is not None:
        path = configured.resolve()
        if not is_java_home(path):
            raise RuntimeError("NEO4J_JAVA_HOME must point to a Java runtime directory.")
        return path

    runtime_root = desktop.home / "Cache" / "runtime"
    runtimes = [path for path in runtime_root.iterdir() if path.is_dir()] if runtime_root.is_dir() else []

    def runtime_rank(path: Path) -> tuple[int, str]:
        match = re.search(r"(?:zulu|jdk|jre|java)[-_]?(\d+)", path.name, re.IGNORECASE)
        major = int(match.group(1)) if match else 0
        return (0 if major == 21 else 1 if major == 17 else 2, path.name)

    for path in sorted(runtimes, key=runtime_rank):
        if is_java_home(path):
            return path.resolve()
    raise RuntimeError(
        f"No Java runtime was found under {runtime_root}. Set NEO4J_JAVA_HOME "
        "in .env.database-mcp."
    )


def _database_endpoint(uri: str) -> tuple[str, int]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "The GraphRAG launcher requires a local Neo4j URI without embedded credentials."
        )
    return parsed.hostname, parsed.port or 7687


def _local_port(url: str, path: str) -> int:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path.rstrip("/") != path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("GraphRAG service URLs must point to the matching local HTTP endpoint.")
    port = parsed.port or 80
    if not 1 <= port <= 65535:
        raise ValueError("GraphRAG service URLs must contain a valid port.")
    return port


def _services() -> list[LocalService]:
    result = [
        LocalService("mcp", _local_port(database_mcp_url(), "/mcp")),
        LocalService("ingestion_api", get_rest_api_port(), "/openapi.json"),
    ]
    if len({service.port for service in result}) != len(result):
        raise ValueError("The MCP and GraphRAG API services must use different ports.")
    return result


def _neo4j_environment(home: Path, java_home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["NEO4J_HOME"] = str(home)
    environment["NEO4J_CONF"] = str(home / "conf")
    environment["JAVA_HOME"] = str(java_home)
    return environment


def _start_dbms(
    home: Path, desktop_home: Path, environment: dict[str, str],
) -> subprocess.Popen:
    logger.info("Starting Neo4j Desktop DBMS.")
    if os.name == "nt":
        wrapper = desktop_home / "Cache" / "neo4j-start.ps1"
        if not wrapper.is_file():
            raise RuntimeError(f"Neo4j Desktop start wrapper was not found: {wrapper}")
        command = [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-NoLogo",
            "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass", "-File",
            str(wrapper), "-binPath", str(home / "bin" / "neo4j.ps1"),
            "-logsPath", str(home / "logs" / "neo4j-relate.log"),
        ]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        command = [str(home / "bin" / "neo4j"), "console"]
        creationflags = 0
    return subprocess.Popen(
        command, cwd=home, env=environment,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL if os.name == "nt" else None,
        stderr=subprocess.DEVNULL if os.name == "nt" else None,
    )


def _database_is_ready(settings: DatabaseSettings) -> bool:
    try:
        with GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
            connection_timeout=1,
        ) as driver:
            driver.verify_connectivity()
            with driver.session(database=settings.database) as session:
                session.run("RETURN 1").consume()
        return True
    except Exception:
        return False


def _wait_for_database(
    settings: DatabaseSettings, process: subprocess.Popen | None, timeout: float = 60,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _database_is_ready(settings):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError("Neo4j Desktop DBMS exited before the database became ready.")
        time.sleep(0.5)
    raise TimeoutError(f"Neo4j database '{settings.database}' did not become ready in {timeout} seconds.")


def _assert_port_available(port: int) -> None:
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} is already in use.") from None


def _start_service(service: LocalService, environment: dict[str, str]) -> subprocess.Popen:
    child_environment = environment.copy()
    source_path = str(PROJECT_ROOT / "src")
    child_environment["PYTHONPATH"] = source_path + os.pathsep + child_environment.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-m", "graphrag.main", "--service", service.name, "--port", str(service.port)],
        cwd=PROJECT_ROOT,
        env=child_environment,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def _wait_for_service(
    service: LocalService, process: subprocess.Popen, timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{service.name} exited during startup; check its output above.")
        try:
            if service.probe is None:
                with socket.create_connection(("127.0.0.1", service.port), timeout=0.5):
                    return
            else:
                with opener.open(
                    f"http://127.0.0.1:{service.port}{service.probe}", timeout=0.5,
                ) as response:
                    if response.status == 200:
                        return
        except (OSError, URLError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"{service.name} did not become ready within {timeout} seconds.")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _monitor_stack(
    processes: list[subprocess.Popen], database: DatabaseSettings,
) -> None:
    while True:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A GraphRAG service stopped unexpectedly.")
        if not _database_is_ready(database):
            raise RuntimeError("The Neo4j database stopped unexpectedly.")
        time.sleep(1)


def _run_service(name: str, port: int) -> None:
    import uvicorn

    if name == "mcp":
        app = create_mcp_server().streamable_http_app()
    else:
        from .ingestion_api import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_config=None)


def run_local_stack() -> int:
    owned_processes: list[subprocess.Popen] = []
    started_dbms = False
    dbms_process: subprocess.Popen | None = None
    try:
        database = load_database_settings()
        desktop = load_neo4j_desktop_settings()
        services = _services()
        _, bolt_port = _database_endpoint(database.uri)
        if bolt_port in {service.port for service in services}:
            raise ValueError("The Neo4j Bolt port must differ from the MCP and GraphRAG API ports.")
        dbms_home = _resolve_dbms_home(database.database, desktop)
        java_home = _resolve_java_home(desktop)
        environment = _neo4j_environment(dbms_home, java_home)

        for service in services:
            _assert_port_available(service.port)

        if _database_is_ready(database):
            logger.info("Using the already-running Neo4j Desktop DBMS.")
        else:
            dbms_process = _start_dbms(dbms_home, desktop.home, environment)
            started_dbms = True
            _wait_for_database(database, dbms_process)

        for service in services:
            logger.info("Starting %s on port %s.", service.name, service.port)
            process = _start_service(service, os.environ.copy())
            owned_processes.append(process)
            _wait_for_service(service, process)

        logger.info(
            "GraphRAG ready. MCP: %s; ingestion API: http://127.0.0.1:%s/docs. "
            "Press Ctrl+C to stop MCP and the API; Neo4j Desktop remains running.",
            database_mcp_url(),
            next(service.port for service in services if service.name == "ingestion_api"),
        )
        _monitor_stack(owned_processes, database)
    except KeyboardInterrupt:
        logger.info("Stopping GraphRAG services.")
        return 0
    except Exception as exc:
        logger.error("GraphRAG startup failed (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        for process in reversed(owned_processes):
            _stop_process(process)
        if started_dbms:
            logger.info("Neo4j Desktop DBMS remains running; stop it from Neo4j Desktop when no longer needed.")
        if owned_processes:
            logger.info("GraphRAG services stopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the local GraphRAG database services.")
    parser.add_argument("--service", choices=("mcp", "ingestion_api"), help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.service:
        if arguments.port is None or not 1 <= arguments.port <= 65535:
            parser.error("--port must be a valid port when starting a service.")
        _run_service(arguments.service, arguments.port)
        return 0
    return run_local_stack()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    raise SystemExit(main())
