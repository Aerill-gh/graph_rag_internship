import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import main as launcher
from chatbot import config as chat_config
from graphrag import config as graph_config


class EntrypointConfigTests(unittest.TestCase):
    def test_chatbot_loads_project_root_env(self):
        with patch.object(chat_config, "load_dotenv") as load:
            chat_config._load_env()
        load.assert_called_once_with(Path(__file__).resolve().parents[1] / ".env")

    def test_default_ports(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(chat_config, "_load_env"):
            self.assertEqual(chat_config.get_rest_api_port(), 8000)
            self.assertEqual(chat_config.get_ui_port(), 8501)
            self.assertEqual(graph_config.get_rest_api_port(), 8001)

    def test_custom_ports(self):
        environment = {"CHATBOT_PORT": "8100", "GRAPHRAG_PORT": "8101", "UI_PORT": "8600"}
        with patch.dict(os.environ, environment, clear=True), patch.object(chat_config, "_load_env"):
            self.assertEqual(chat_config.get_rest_api_port(), 8100)
            self.assertEqual(chat_config.get_ui_port(), 8600)
            self.assertEqual(graph_config.get_rest_api_port(), 8101)

    def test_invalid_ports(self):
        for value in ("invalid", "0", "65536"):
            for variable, module in (("CHATBOT_PORT", chat_config), ("GRAPHRAG_PORT", graph_config)):
                with self.subTest(variable=variable, value=value):
                    with patch.dict(os.environ, {variable: value}, clear=True):
                        with patch.object(chat_config, "_load_env"), self.assertRaises(ValueError):
                            module.get_rest_api_port()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.services = [launcher.Service("mcp", 8002), launcher.Service("chatbot", 8000, "/openapi.json")]
        for name, value in (("services", self.services), ("get_api_urls", ("http://127.0.0.1:8000/chat", "http://127.0.0.1:8000"))):
            mocked = patch.object(launcher, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        bind_patch = patch.object(launcher.socket, "socket")
        bind_patch.start()
        self.addCleanup(bind_patch.stop)
        self.first = MagicMock()
        self.second = MagicMock()
        self.first.poll.return_value = None
        self.second.poll.return_value = None

    def test_ctrl_c_starts_in_order_and_stops_all_children(self):
        with patch.object(launcher.subprocess, "Popen", side_effect=[self.first, self.second]) as start:
            with patch.object(launcher, "wait_ready") as ready:
                with patch.object(launcher.time, "sleep", side_effect=KeyboardInterrupt):
                    self.assertEqual(launcher.main([]), 0)
        self.assertEqual([call.args[0] for call in ready.call_args_list], self.services)
        self.assertEqual(start.call_args_list[0].args[0][:2], [sys.executable, str(launcher.PROJECT_ROOT / "main.py")])
        self.assertEqual(start.call_args_list[0].kwargs["cwd"], launcher.PROJECT_ROOT)
        self.assertEqual(start.call_args_list[1].kwargs["env"]["CHATBOT_API_URL"], "http://127.0.0.1:8000/chat")
        for process in (self.first, self.second):
            process.send_signal.assert_called_once()
            process.terminate.assert_not_called()
            process.wait.assert_called_once_with(timeout=5)

    def test_partial_startup_failure_cleans_up(self):
        with patch.object(launcher.subprocess, "Popen", side_effect=[self.first, OSError("private")]):
            with patch.object(launcher, "wait_ready"):
                with self.assertLogs(launcher.logger, level="ERROR") as output:
                    self.assertEqual(launcher.main([]), 1)
        self.first.send_signal.assert_called_once()
        self.assertNotIn("private", " ".join(output.output))

    def test_readiness_failure_cleans_up(self):
        with patch.object(launcher.subprocess, "Popen", return_value=self.first):
            with patch.object(launcher, "wait_ready", side_effect=TimeoutError("startup timed out")):
                self.assertEqual(launcher.main([]), 1)
        self.first.send_signal.assert_called_once()

    def test_runtime_child_exit_stops_other_child(self):
        self.first.poll.return_value = 1
        with patch.object(launcher.subprocess, "Popen", side_effect=[self.first, self.second]):
            with patch.object(launcher, "wait_ready"):
                self.assertEqual(launcher.main([]), 1)
        self.second.send_signal.assert_called_once()

    def test_busy_port_starts_no_children(self):
        launcher.socket.socket.return_value.__enter__.return_value.bind.side_effect = OSError()
        with patch.object(launcher.subprocess, "Popen") as start:
            self.assertEqual(launcher.main([]), 1)
        start.assert_not_called()

    def test_shutdown_timeout_kills_child(self):
        self.first.wait.side_effect = [subprocess.TimeoutExpired("child", 5), subprocess.TimeoutExpired("child", 5), 0]
        launcher.stop_services([self.first])
        self.first.terminate.assert_called_once()
        self.first.kill.assert_called_once()

    def test_streamlit_command_uses_current_python_and_absolute_path(self):
        command = launcher.command(launcher.Service("ui", 8501))
        self.assertEqual(command[:3], [sys.executable, "-m", "streamlit"])
        self.assertIn(str(launcher.PROJECT_ROOT / "src" / "ui" / "streamlit_chat.py"), command)


class ReadinessTests(unittest.TestCase):
    def test_early_child_exit_fails_without_network_call(self):
        process = MagicMock()
        process.poll.return_value = 1
        with patch.object(launcher.socket, "create_connection") as connect:
            with self.assertRaises(RuntimeError):
                launcher.wait_ready(launcher.Service("mcp", 8002), [process])
        connect.assert_not_called()

    def test_mcp_checks_transport_only(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(launcher.socket, "create_connection") as connect:
            launcher.wait_ready(launcher.Service("mcp", 8002), [process])
        connect.assert_called_once_with(("127.0.0.1", 8002), timeout=0.5)

    def test_http_probe_uses_non_billable_endpoint(self):
        with patch.object(launcher, "build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.status = 200
            launcher.wait_ready(launcher.Service("chatbot", 8000, "/openapi.json"), [])
        opener.return_value.open.assert_called_once_with("http://127.0.0.1:8000/openapi.json", timeout=0.5)

    def test_timeout_is_bounded(self):
        with patch.object(launcher.time, "monotonic", side_effect=[0, 31]):
            with self.assertRaises(TimeoutError):
                launcher.wait_ready(launcher.Service("mcp", 8002), [])


class ServiceSettingsTests(unittest.TestCase):
    def test_four_services_use_the_existing_configs(self):
        environment = {"CHATBOT_PORT": "8100", "GRAPHRAG_PORT": "8101", "UI_PORT": "8600"}
        with patch.dict(os.environ, environment, clear=True), patch.object(chat_config, "_load_env"):
            with patch.object(launcher, "database_mcp_url", return_value="http://127.0.0.1:8102/mcp"):
                configured = launcher.services()
        self.assertEqual([(service.name, service.port) for service in configured], [
            ("mcp", 8102), ("graphrag", 8101), ("chatbot", 8100), ("ui", 8600),
        ])

    def test_conflicting_ports_are_rejected(self):
        with patch.dict(os.environ, {"GRAPHRAG_PORT": "8000"}, clear=True):
            with patch.object(chat_config, "_load_env"), patch.object(launcher, "database_mcp_url", return_value="http://127.0.0.1:8002/mcp"):
                with self.assertRaises(ValueError):
                    launcher.services()

    def test_mismatched_ui_backend_url_is_not_silently_overridden(self):
        environment = {"CHATBOT_PORT": "8100", "CHATBOT_API_URL": "http://127.0.0.1:8000/chat"}
        with patch.dict(os.environ, environment, clear=True), patch.object(chat_config, "_load_env"):
            with self.assertRaises(ValueError):
                launcher.services()

    def test_local_urls_reject_wrong_endpoint_or_credentials(self):
        for url in ("https://localhost:8000/chat", "http://example.com:8000/chat", "http://user:secret@localhost:8000/chat", "http://localhost:8000/other", "http://localhost:0/chat"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                launcher.local_port(url, "/chat")

    def test_local_url_port(self):
        self.assertEqual(launcher.local_port("http://localhost:8123/mcp", "/mcp"), 8123)

    def test_custom_chatbot_urls_default_to_custom_port(self):
        with patch.dict(os.environ, {"CHATBOT_PORT": "8100"}, clear=True):
            with patch.object(chat_config, "_load_env"):
                self.assertEqual(chat_config.get_api_urls(), ("http://127.0.0.1:8100/chat", "http://127.0.0.1:8100"))


class ServiceBootstrapTests(unittest.TestCase):
    def test_api_services_use_the_existing_apps(self):
        for name, module in (("chatbot", "chatbot.main"), ("graphrag", "graphrag.ingestion_api")):
            fake_module = MagicMock()
            with self.subTest(service=name), patch.dict(sys.modules, {module: fake_module}):
                with patch("uvicorn.run") as run:
                    launcher.run_service(launcher.Service(name, 8100))
                run.assert_called_once_with(fake_module.app, host="127.0.0.1", port=8100, log_config=None)

    def test_mcp_uses_the_existing_factory_and_http_lifespan(self):
        with patch("graphrag.main.create_mcp_server") as create:
            with patch("uvicorn.run") as run:
                launcher.run_service(launcher.Service("mcp", 8102))
        create.assert_called_once_with()
        run.assert_called_once_with(create.return_value.streamable_http_app.return_value, host="127.0.0.1", port=8102, log_config=None)


if __name__ == "__main__":
    unittest.main()