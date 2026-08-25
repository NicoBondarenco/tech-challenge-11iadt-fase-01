
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import tensorflow as tf

try:
    from .dataset import DATA_DIR, ROOT, build_master_dataset, resolve_repo_path, summarize, _write_csv
    from .model import build_mobilenetv2_classifier, compile_binary_classifier
    from .preprocessing import DEFAULT_IMAGE_SIZE, make_dataset
except ImportError:
    from dataset import DATA_DIR, ROOT, build_master_dataset, resolve_repo_path, summarize, _write_csv
    from model import build_mobilenetv2_classifier, compile_binary_classifier
    from preprocessing import DEFAULT_IMAGE_SIZE, make_dataset


def read_master_dataset(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_master_dataset(path: Path, image_type: str, validation_size: float, seed: int) -> list[dict[str, str]]:
    if path.exists():
        return read_master_dataset(path)
    rows = build_master_dataset(image_type=image_type, validation_size=validation_size, seed=seed)
    _write_csv(path, rows)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    return read_master_dataset(path)


def split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def class_weight(rows: list[dict[str, str]]) -> dict[int, float]:
    counts = Counter(int(row["label"]) for row in rows)
    total = sum(counts.values())
    return {label: total / (len(counts) * count) for label, count in counts.items()}


def build_callbacks(
    checkpoint_path: Path,
    log_path: Path,
    monitor: str,
    monitor_mode: str,
    patience: int,
    append_log: bool = False,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=monitor_mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_path), append=append_log),
    ]


def find_transfer_base(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(isinstance(child, tf.keras.layers.Conv2D) for child in layer.layers):
            return layer
    raise ValueError("Base convolucional nao encontrada no modelo.")


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def best_metric_value(history: dict[str, list[float]], monitor: str, monitor_mode: str) -> float | None:
    values = history.get(monitor)
    if not values:
        return None
    return max(values) if monitor_mode == "max" else min(values)


def is_better(candidate: float | None, reference: float | None, monitor_mode: str) -> bool:
    if candidate is None:
        return False
    if reference is None:
        return True
    return candidate > reference if monitor_mode == "max" else candidate < reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for mammography classification.")
    parser.add_argument("--dataset", default=str(DATA_DIR / "cv_master_dataset.csv"))
    parser.add_argument("--image-type", choices=["full", "cropped"], default="cropped")
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--monitor", default="val_roc_auc")
    parser.add_argument("--monitor-mode", choices=["min", "max"], default="max")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abnormality", default=None,
                        help="Treina so em 'mass' ou 'calcification'. Omitido = ambos.")
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    dataset_path = Path(args.dataset)
    rows = ensure_master_dataset(dataset_path, args.image_type, args.validation_size, args.seed)
    if args.abnormality:
        alvo = args.abnormality.strip().lower()
        rows = [row for row in rows if (row.get("abnormality_type") or "").strip().lower() == alvo]
        if not rows:
            raise ValueError(f"Nenhuma linha para abnormality={args.abnormality}.")
        print(f"Treinando somente em '{alvo}': {len(rows)} imagens.")

    train_rows = split_rows(rows, "train")
    validation_rows = split_rows(rows, "validation")

    image_size = tuple(args.image_size)
    train_ds = make_dataset(
        [resolve_repo_path(row["image_path"]) for row in train_rows],
        [int(row["label"]) for row in train_rows],
        batch_size=args.batch_size,
        image_size=image_size,
        shuffle=True,
        augment=True,
    )
    validation_ds = make_dataset(
        [resolve_repo_path(row["image_path"]) for row in validation_rows],
        [int(row["label"]) for row in validation_rows],
        batch_size=args.batch_size,
        image_size=image_size,
    )

    model = build_mobilenetv2_classifier(image_size=image_size, learning_rate=args.learning_rate, freeze_base=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    sufixo = f"_{args.abnormality.strip().lower()}" if args.abnormality else ""
    checkpoint_path = model_dir / f"mobilenetv2_cbis_ddsm{sufixo}_best.keras"
    fine_tune_checkpoint_path = model_dir / f"mobilenetv2_cbis_ddsm{sufixo}_fine_tune_best.keras"
    training_log_path = model_dir / f"training_log{sufixo}.csv"

    print("Fase 1: treinando somente o cabecalho de classificacao.")
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=build_callbacks(
            checkpoint_path,
            training_log_path,
            monitor=args.monitor,
            monitor_mode=args.monitor_mode,
            patience=args.patience,
        ),
        class_weight=class_weight(train_rows),
    )
    histories = [history]
    phase1_history = merge_histories(history)
    phase1_best = best_metric_value(phase1_history, args.monitor, args.monitor_mode)

    if args.fine_tune_epochs > 0:
        print("Fase 2: fine-tuning das camadas superiores da base convolucional.")
        base_model = find_transfer_base(model)
        base_model.trainable = True
        for layer in base_model.layers[: args.fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        treinaveis = sum(1 for layer in base_model.layers if layer.trainable)
        print(f"Camadas destravadas na base: {treinaveis} de {len(base_model.layers)}")
        compile_binary_classifier(model, args.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=validation_ds,
            initial_epoch=args.epochs,
            epochs=args.epochs + args.fine_tune_epochs,
            callbacks=build_callbacks(
                fine_tune_checkpoint_path,
                training_log_path,
                monitor=args.monitor,
                monitor_mode=args.monitor_mode,
                patience=args.patience,
                append_log=True,
            ),
            class_weight=class_weight(train_rows),
        )
        histories.append(fine_tune_history)
        fine_tune_best = best_metric_value(fine_tune_history.history, args.monitor, args.monitor_mode)
        if is_better(fine_tune_best, phase1_best, args.monitor_mode):
            shutil.copy2(fine_tune_checkpoint_path, checkpoint_path)
            print(f"Fine-tuning superou a Fase 1 em {args.monitor}; melhor global atualizado.")
        else:
            print(f"Fine-tuning nao superou a Fase 1 em {args.monitor}; melhor global preservado.")

    history_path = model_dir / f"training_history{sufixo}.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(merge_histories(*histories), handle, indent=2)
    model.save(model_dir / f"mobilenetv2_cbis_ddsm{sufixo}_last.keras")
    print(f"Melhor modelo salvo em: {checkpoint_path}")
    print(f"Historico salvo em: {history_path}")


if __name__ == "__main__":
    main()
