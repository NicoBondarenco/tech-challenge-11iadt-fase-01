
from __future__ import annotations

import argparse
import csv
from pathlib import Path


try:
    from .dataset import ROOT
except ImportError:
    from dataset import ROOT


REPORTS_DIR = ROOT / "reports"
DEFAULT_INPUT = REPORTS_DIR / "predictions_validation.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "threshold_analysis_validation.csv"
DEFAULT_PLOT = REPORTS_DIR / "threshold_analysis_validation.png"


def read_predictions(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    predictions: list[dict[str, float | int | str]] = []
    for row in rows:
        predictions.append(
            {
                "image_path": row.get("image_path", ""),
                "actual_label": int(row["actual_label"]),
                "malignant_score": float(row["malignant_score"]),
            }
        )
    return predictions


def metrics_for_threshold(predictions: list[dict[str, float | int | str]], threshold: float) -> dict[str, float | int]:
    y_true = [int(row["actual_label"]) for row in predictions]
    y_pred = [1 if float(row["malignant_score"]) >= threshold else 0 for row in predictions]

    tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 1 and predicted == 1)
    tn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 0 and predicted == 0)
    fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 0 and predicted == 1)
    fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 1 and predicted == 0)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy,
        "precision_malignant": precision,
        "recall_malignant": recall,
        "specificity_benign": specificity,
        "f1_score": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def analyze_thresholds(predictions: list[dict[str, float | int | str]]) -> list[dict[str, float | int]]:
    return [metrics_for_threshold(predictions, threshold / 100) for threshold in range(30, 81)]


def write_results(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_recommended_threshold(
    rows: list[dict[str, float | int]],
    min_recall: float = 0.85,
) -> dict[str, float | int] | None:
    candidates = [row for row in rows if float(row["recall_malignant"]) >= min_recall]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["specificity_benign"]),
            float(row["f1_score"]),
            -int(row["false_positives"]),
            float(row["threshold"]),
        ),
    )


def row_for_threshold(rows: list[dict[str, float | int]], threshold: float) -> dict[str, float | int]:
    rounded = round(threshold, 2)
    for row in rows:
        if float(row["threshold"]) == rounded:
            return row
    raise ValueError(f"Threshold {threshold:.2f} nao encontrado.")


def save_plot(path: Path, rows: list[dict[str, float | int]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nao encontrado; grafico de threshold nao foi gerado.")
        return

    thresholds = [float(row["threshold"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [float(row["recall_malignant"]) for row in rows], label="Recall maligno")
    ax.plot(thresholds, [float(row["precision_malignant"]) for row in rows], label="Precision maligno")
    ax.plot(thresholds, [float(row["specificity_benign"]) for row in rows], label="Specificity benigno")
    ax.plot(thresholds, [float(row["f1_score"]) for row in rows], label="F1-score")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold atual 0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metrica")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_metric(value: float | int) -> str:
    return f"{float(value):.3f}"


def print_comparison(current: dict[str, float | int], recommended: dict[str, float | int] | None) -> None:
    if recommended is None:
        print("Nenhum threshold entre 0.30 e 0.80 manteve recall maligno >= 0.85.")
        recommended = current

    print("\nComparacao de thresholds")
    print(f"{'Metrica':<28} {'Atual 0.50':>14} {'Sugerido ' + format_metric(recommended['threshold']):>18}")
    print("-" * 62)
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision_malignant", "Precision maligno"),
        ("recall_malignant", "Recall maligno"),
        ("specificity_benign", "Specificity benigno"),
        ("f1_score", "F1-score"),
    ]:
        print(f"{label:<28} {format_metric(current[key]):>14} {format_metric(recommended[key]):>18}")

    print("\nMatriz de confusao - atual 0.50")
    print(f"TN={current['true_negatives']}  FP={current['false_positives']}")
    print(f"FN={current['false_negatives']}  TP={current['true_positives']}")

    print(f"\nMatriz de confusao - sugerido {float(recommended['threshold']):.2f}")
    print(f"TN={recommended['true_negatives']}  FP={recommended['false_positives']}")
    print(f"FN={recommended['false_negatives']}  TP={recommended['true_positives']}")

    print("\nErros")
    print(f"Falsos positivos atual/sugerido: {current['false_positives']} / {recommended['false_positives']}")
    print(f"Falsos negativos atual/sugerido: {current['false_negatives']} / {recommended['false_negatives']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation thresholds without re-running the model.")
    parser.add_argument("--predictions", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--plot", default=str(DEFAULT_PLOT))
    parser.add_argument("--min-recall", type=float, default=0.85)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {predictions_path}. Execute primeiro: python -m src.evaluate --split validation"
        )

    predictions = read_predictions(predictions_path)
    rows = analyze_thresholds(predictions)
    write_results(Path(args.output), rows)
    save_plot(Path(args.plot), rows)

    current = row_for_threshold(rows, 0.5)
    recommended = choose_recommended_threshold(rows, min_recall=args.min_recall)
    print_comparison(current, recommended)
    print(f"\nResultados salvos em: {args.output}")
    print(f"Grafico salvo em: {args.plot}")
    print("Observacao: o conjunto de teste nao foi usado para escolher threshold.")


if __name__ == "__main__":
    main()
