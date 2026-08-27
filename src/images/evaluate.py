
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, clinical_cost, load_threshold
    from .dataset import DATA_DIR, ROOT, resolve_repo_path
    from .preprocessing import DEFAULT_IMAGE_SIZE, make_dataset
except ImportError:
    from config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, clinical_cost, load_threshold
    from dataset import DATA_DIR, ROOT, resolve_repo_path
    from preprocessing import DEFAULT_IMAGE_SIZE, make_dataset


def metrics_by_abnormality(
    rows: list[dict[str, str]], y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, dict[str, object]]:
    tipos = sorted({(row.get("abnormality_type") or "desconhecido").strip() for row in rows})
    resultado: dict[str, dict[str, object]] = {}

    for tipo in tipos:
        mascara = np.array(
            [(row.get("abnormality_type") or "desconhecido").strip() == tipo for row in rows]
        )
        if mascara.sum() == 0:
            continue
        alvo, pontuacao = y_true[mascara], scores[mascara]
        predito = (pontuacao >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(alvo, predito, labels=[0, 1]).ravel()

        resultado[tipo] = {
            "n": int(mascara.sum()),
            "prevalencia_maligno": float(alvo.mean()),
            "roc_auc": float(roc_auc_score(alvo, pontuacao)) if len(set(alvo.tolist())) == 2 else None,
            "recall_maligno": float(recall_score(alvo, predito, zero_division=0)),
            "precision_maligno": float(precision_score(alvo, predito, zero_division=0)),
            "falsos_negativos": int(fn),
            "falsos_positivos": int(fp),
            "custo_clinico": clinical_cost(int(fn), int(fp)),
        }
    return resultado


def read_master_dataset(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Benigno", "Maligno"])
    ax.set_yticks([0, 1], labels=["Benigno", "Maligno"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mammography classifier.")
    parser.add_argument("--dataset", default=str(DATA_DIR / "cv_master_dataset.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" / "mobilenetv2_cbis_ddsm_best.keras"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Se omitido, usa o threshold ajustado gravado em models/threshold.json.")
    parser.add_argument("--abnormality", default=None,
                        help="Filtra por tipo de lesao (mass / calcification). Omitido = todos.")
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    rows = [row for row in read_master_dataset(Path(args.dataset)) if row["split"] == args.split]
    if args.abnormality:
        alvo = args.abnormality.strip().lower()
        rows = [row for row in rows if (row.get("abnormality_type") or "").strip().lower() == alvo]
    if not rows:
        raise ValueError(f"Nenhuma linha encontrada para split={args.split}.")

    threshold = args.threshold if args.threshold is not None else load_threshold()
    print(f"Threshold em uso: {threshold:.4f}")

    model = tf.keras.models.load_model(args.model)
    dataset = make_dataset(
        [resolve_repo_path(row["image_path"]) for row in rows],
        [int(row["label"]) for row in rows],
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
    )
    scores = model.predict(dataset).reshape(-1)
    y_true = np.array([int(row["label"]) for row in rows])
    y_pred = (scores >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "split": args.split,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity_malignant": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores) if len(set(y_true.tolist())) == 2 else None,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["BENIGN", "MALIGNANT"],
            zero_division=0,
            output_dict=True,
        ),
        "custo_clinico": clinical_cost(int(matrix[1, 0]), int(matrix[0, 1])),
        "custo_premissa": {"falso_negativo": COST_FALSE_NEGATIVE, "falso_positivo": COST_FALSE_POSITIVE},
        "por_tipo_lesao": metrics_by_abnormality(rows, y_true, scores, threshold),
        "false_negatives_note": "Falso negativo = imagem maligna classificada como benigna.",
    }

    prediction_rows: list[dict[str, object]] = []
    for row, score, predicted in zip(rows, scores, y_pred):
        actual = int(row["label"])
        prediction_rows.append(
            {
                "image_path": row["image_path"],
                "patient_id": row["patient_id"],
                "pathology": row["pathology"],
                "actual_label": actual,
                "predicted_label": int(predicted),
                "malignant_score": float(score),
                "correct": actual == int(predicted),
                "error_type": (
                    "false_negative"
                    if actual == 1 and int(predicted) == 0
                    else "false_positive"
                    if actual == 0 and int(predicted) == 1
                    else "correct"
                ),
            }
        )

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    sufixo = f"_{args.abnormality}" if args.abnormality else ""
    metrics_path = reports_dir / f"metrics_{args.split}{sufixo}.json"
    predictions_path = reports_dir / f"predictions_{args.split}{sufixo}.csv"
    matrix_path = reports_dir / f"confusion_matrix_{args.split}{sufixo}.png"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    save_predictions(predictions_path, prediction_rows)
    save_confusion_matrix(matrix_path, matrix)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predicoes salvas em: {predictions_path}")
    print(f"Matriz de confusao salva em: {matrix_path}")


if __name__ == "__main__":
    main()
