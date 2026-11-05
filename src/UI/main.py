"""Streamlit UI entry point. Starts the chat application with configuration from .env."""
import subprocess
import sys
from pathlib import Path

# Add src/ to path for module discovery
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    from ui.config import get_ui_port
except ModuleNotFoundError:
    from UI.config import get_ui_port


def main() -> None:
    """Start the Streamlit chat application with configured port."""
    ui_port = get_ui_port()
    streamlit_app = Path(__file__).resolve().parent / "streamlit_chat.py"

    # Run streamlit with the configured port
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(streamlit_app),
            "--server.port",
            str(ui_port),
            "--server.address",
            "127.0.0.1",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
