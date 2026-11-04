import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatbot.config import get_api_urls, get_rest_api_port as chatbot_port, get_ui_port
from graphrag.config import get_rest_api_port as graphrag_port
from graphrag.database_mcp.config import database_mcp_url


logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


@dataclass(frozen=True)
class Service:
    name: str
    port: int
    probe: str | None = None


def local_port(url: str, path: str) -> int:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path.rstrip("/") != path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Service URLs must point to the matching local HTTP endpoint.")
    port = parsed.port if parsed.port is not None else 80
    if not 1 <= port <= 65535:
        raise ValueError("Service URLs must contain a valid port.")
    return port


def services() -> list[Service]:
    chat_port = chatbot_port()
    chat_url, backend_url = get_api_urls()
    if local_port(chat_url, "/chat") != chat_port or local_port(backend_url, "") != chat_port:
        raise ValueError("Chatbot URLs must match CHATBOT_PORT.")
    result = [
        Service("mcp", local_port(database_mcp_url(), "/mcp")),
        Service("graphrag", graphrag_port(), "/openapi.json"),
        Service("chatbot", chat_port, "/openapi.json"),
        Service("ui", get_ui_port(), "/_stcore/health"),
    ]
    if len({service.port for service in result}) != len(result):
        raise ValueError("Each service must use a different port.")
    return result


def command(service: Service) -> list[str]:
    if service.name == "ui":
        return [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "src" / "ui" / "streamlit_chat.py"),
            "--server.address", "127.0.0.1", "--server.port", str(service.port),
            "--server.headless", "true",
        ]
    return [sys.executable, str(PROJECT_ROOT / "main.py"), "--service", service.name]


def run_service(service: Service) -> None:
    import uvicorn

    if service.name == "mcp":
        from graphrag.database_mcp.server import create_mcp_server

        app = create_mcp_server().streamable_http_app()
    elif service.name == "graphrag":
        from graphrag.ingestion_api import app
    else:
        from chatbot.chat_bot_main import app
    uvicorn.run(app, host="127.0.0.1", port=service.port, log_config=None)


def wait_ready(service: Service, processes: list[subprocess.Popen], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A service exited during startup. Check its output above.")
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


def stop_services(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            except ProcessLookupError:
                pass
    for process in reversed(processes):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the local Zava system.")
    parser.add_argument("--service", choices=("mcp", "graphrag", "chatbot"), help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    processes: list[subprocess.Popen] = []
    try:
        configured_services = services()
        if arguments.service:
            run_service(next(service for service in configured_services if service.name == arguments.service))
            return 0
        for service in configured_services:
            with socket.socket() as listener:
                try:
                    listener.bind(("127.0.0.1", service.port))
                except OSError:
                    raise RuntimeError(f"Port {service.port} for {service.name} is unavailable.") from None
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        environment["CHATBOT_API_URL"], environment["BACKEND_API_URL"] = get_api_urls()
        for service in configured_services:
            logger.info("Starting %s on port %s.", service.name, service.port)
            processes.append(subprocess.Popen(
                command(service), cwd=PROJECT_ROOT, env=environment,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            ))
            wait_ready(service, processes)
        logger.info("System ready. UI: http://127.0.0.1:%s (Ctrl+C to stop).", configured_services[-1].port)
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A service stopped unexpectedly. Stopping the system.")
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Stopping the system.")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Startup or service failure (%s).", type(exc).__name__)
        if isinstance(exc, (RuntimeError, TimeoutError)):
            logger.error("%s", exc)
        return 1
    finally:
        stop_services(processes)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    raise SystemExit(main())