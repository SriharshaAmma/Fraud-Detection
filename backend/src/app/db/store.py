"""Simple file-based logger for predictions (dev)."""

import json
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

OUT_FILE = LOG_PATH / "predictions.log"


def log_prediction(record: Dict[str, Any]):
    with OUT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
