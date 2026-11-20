"""Agar tes bisa mengimpor modul dari root repo dan folder bench/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
