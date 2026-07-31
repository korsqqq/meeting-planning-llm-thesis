# conftest.py
"""Pytest bootstrap: put the repository root on sys.path so `import src...`
resolves regardless of the working directory pytest is launched from."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
