import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from graphrag import main as launcher
from graphrag.database_mcp.config import DatabaseSettings, Neo4jDesktopSettings


class DesktopResolutionTests(unittest.TestCase):
    def test_resolves_dbms_by_logical_database_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            desktop_home = Path(temporary_directory)
            dbms_home = desktop_home / "Data" / "dbmss" / "dbms-target"
            (dbms_home / "bin").mkdir(parents=True)
            (dbms_home / "bin" / "neo4j.bat").touch()
            (dbms_home / "data" / "databases" / "graphragfull").mkdir(parents=True)
            desktop = Neo4jDesktopSettings(home=desktop_home)

            self.assertEqual(launcher._resolve_dbms_home("graphragfull", desktop), dbms_home)

    def test_ambiguous_dbms_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            desktop_home = Path(temporary_directory)
            for name in ("dbms-one", "dbms-two"):
                dbms_home = desktop_home / "Data" / "dbmss" / name
                (dbms_home / "bin").mkdir(parents=True)
                (dbms_home / "bin" / "neo4j.bat").touch()
                (dbms_home / "data" / "databases" / "graphragfull").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "Multiple Neo4j Desktop DBMS"):
                launcher._resolve_dbms_home("graphragfull", Neo4jDesktopSettings(home=desktop_home))


class LocalStackTests(unittest.TestCase):
    def setUp(self):
        self.database = DatabaseSettings(
            uri="bolt://127.0.0.1:7687",
            username="neo4j",
            password="test-password",
            database="graphragfull",
        )
        self.desktop = Neo4jDesktopSettings(home=Path("desktop"))
        self.mcp_process = MagicMock()
        self.api_process = MagicMock()
        self.dbms_process = MagicMock()
        for process in (self.mcp_process, self.api_process, self.dbms_process):
            process.poll.return_value = None

    def test_starts_database_mcp_and_api_in_order_then_cleans_up(self):
        events = []
        services = [
            launcher.LocalService("mcp", 8002),
            launcher.LocalService("ingestion_api", 8001, "/openapi.json"),
        ]
        with (
            patch.object(launcher, "load_database_settings", return_value=self.database),
            patch.object(launcher, "load_neo4j_desktop_settings", return_value=self.desktop),
            patch.object(launcher, "_services", return_value=services),
            patch.object(launcher, "_database_endpoint", return_value=("127.0.0.1", 7687)),
            patch.object(launcher, "_resolve_dbms_home", return_value=Path("dbms")),
            patch.object(launcher, "_resolve_java_home", return_value=Path("java")),
            patch.object(launcher, "_database_is_ready", return_value=False),
            patch.object(launcher, "_start_dbms", side_effect=lambda *args: events.append("neo4j") or self.dbms_process),
            patch.object(launcher, "_wait_for_database", side_effect=lambda *args: events.append("bolt-ready")),
            patch.object(launcher, "_assert_port_available"),
            patch.object(launcher, "_start_service") as start_service,
            patch.object(launcher, "_wait_for_service", side_effect=lambda service, process: events.append(service.name)),
            patch.object(launcher, "_monitor_stack", side_effect=KeyboardInterrupt),
            patch.object(launcher, "_stop_process", side_effect=lambda process: events.append(process)),
            patch.object(launcher, "database_mcp_url", return_value="http://127.0.0.1:8002/mcp"),
        ):
            start_service.side_effect = lambda service, environment: (
                events.append(f"start-{service.name}")
                or (self.mcp_process if service.name == "mcp" else self.api_process)
            )
            self.assertEqual(launcher.run_local_stack(), 0)

        self.assertEqual(events[:6], [
            "neo4j", "bolt-ready", "start-mcp", "mcp", "start-ingestion_api", "ingestion_api",
        ])
        self.assertEqual(events[6:], [self.api_process, self.mcp_process])

    def test_reuses_running_dbms_without_stopping_it(self):
        services = [
            launcher.LocalService("mcp", 8002),
            launcher.LocalService("ingestion_api", 8001, "/openapi.json"),
        ]
        with (
            patch.object(launcher, "load_database_settings", return_value=self.database),
            patch.object(launcher, "load_neo4j_desktop_settings", return_value=self.desktop),
            patch.object(launcher, "_services", return_value=services),
            patch.object(launcher, "_database_endpoint", return_value=("127.0.0.1", 7687)),
            patch.object(launcher, "_resolve_dbms_home", return_value=Path("dbms")),
            patch.object(launcher, "_resolve_java_home", return_value=Path("java")),
            patch.object(launcher, "_database_is_ready", return_value=True),
            patch.object(launcher, "_start_dbms") as start_dbms,
            patch.object(launcher, "_assert_port_available"),
            patch.object(
                launcher, "_start_service",
                side_effect=[self.mcp_process, self.api_process],
            ),
            patch.object(launcher, "_wait_for_service"),
            patch.object(launcher, "_monitor_stack", side_effect=KeyboardInterrupt),
            patch.object(launcher, "_stop_process") as stop_process,
            patch.object(launcher, "database_mcp_url", return_value="http://127.0.0.1:8002/mcp"),
        ):
            self.assertEqual(launcher.run_local_stack(), 0)

        start_dbms.assert_not_called()
        self.assertEqual(stop_process.call_args_list, [
            unittest.mock.call(self.api_process),
            unittest.mock.call(self.mcp_process),
        ])


if __name__ == "__main__":
    unittest.main()