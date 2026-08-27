
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

try:
    from .config import DEFAULT_CLASSIFICATION_THRESHOLD
    from .preprocessing import DEFAULT_IMAGE_SIZE, make_dataset
except ImportError:
    from config import DEFAULT_CLASSIFICATION_THRESHOLD
    from preprocessing import DEFAULT_IMAGE_SIZE, make_dataset


@dataclass(frozen=True)
class PredictionResult:
    label: str
    malignant_score: float
    threshold: float


def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def label_from_score(score: float, threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD) -> str:
    return "MALIGNO" if score >= threshold else "BENIGNO"


def predict_with_model(
    model: tf.keras.Model,
    image_path: str,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> PredictionResult:
    dataset = make_dataset([image_path], labels=None, batch_size=1, image_size=image_size)
    score = float(model.predict(dataset, verbose=0).reshape(-1)[0])
    return PredictionResult(
        label=label_from_score(score, threshold),
        malignant_score=score,
        threshold=threshold,
    )


def predict_image(
    image_path: str,
    model_path: str,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> PredictionResult:
    model = load_model(model_path)
    return predict_with_model(model, image_path, threshold=threshold, image_size=image_size)
