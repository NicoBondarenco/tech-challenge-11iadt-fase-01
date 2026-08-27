
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
THRESHOLD_ARTIFACT = MODELS_DIR / "threshold.json"

FALLBACK_CLASSIFICATION_THRESHOLD = 0.50

COST_FALSE_NEGATIVE = 100
COST_FALSE_POSITIVE = 1

MIN_RECALL_TARGET = 0.95


def load_threshold() -> float:
    if THRESHOLD_ARTIFACT.exists():
        try:
            with THRESHOLD_ARTIFACT.open(encoding="utf-8") as handle:
                return float(json.load(handle)["threshold"])
        except (KeyError, ValueError, OSError):
            pass
    return FALLBACK_CLASSIFICATION_THRESHOLD


def save_threshold(threshold: float, metadata: dict | None = None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": round(float(threshold), 4)}
    if metadata:
        payload.update(metadata)
    with THRESHOLD_ARTIFACT.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return THRESHOLD_ARTIFACT


def clinical_cost(false_negatives: int, false_positives: int) -> int:
    return false_negatives * COST_FALSE_NEGATIVE + false_positives * COST_FALSE_POSITIVE


DEFAULT_CLASSIFICATION_THRESHOLD = load_threshold()
