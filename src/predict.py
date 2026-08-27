
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import DEFAULT_CLASSIFICATION_THRESHOLD
    from .dataset import ROOT
    from .inference import predict_image
    from .preprocessing import DEFAULT_IMAGE_SIZE
except ImportError:
    from config import DEFAULT_CLASSIFICATION_THRESHOLD
    from dataset import ROOT
    from inference import predict_image
    from preprocessing import DEFAULT_IMAGE_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict benign/malignant score for one cropped mammographic finding.")
    parser.add_argument("image_path")
    parser.add_argument("--model", default=str(ROOT / "models" / "mobilenetv2_cbis_ddsm_best.keras"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_CLASSIFICATION_THRESHOLD)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")
    result = predict_image(str(image_path), args.model, args.threshold, tuple(args.image_size))
    print(f"Predicao da alteracao: {result.label}")
    print(f"Score para classe maligna produzido pelo classificador: {result.malignant_score * 100:.1f}%")


if __name__ == "__main__":
    main()
