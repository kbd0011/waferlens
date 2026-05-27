"""Hugging Face Spaces entrypoint - reuses the package Streamlit app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from waferlens.ui import app  # noqa: F401,E402
