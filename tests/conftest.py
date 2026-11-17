"""Shared pytest configuration for the D0FUS test suite.

Tests live in this folder as ``test_*.py`` files (see ``[tool.pytest.ini_options]``
in ``pyproject.toml``). Shared fixtures go in this file.
"""
import os

# Headless, quiet runs: no GUI windows from matplotlib, no startup banner.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("D0FUS_NO_BANNER", "1")
