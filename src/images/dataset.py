
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "img"

CASE_FILES = {
    "mass_train": ("mass_case_description_train_set.csv", "mass", "train"),
    "mass_test": ("mass_case_description_test_set.csv", "mass", "test"),
    "calc_train": ("calc_case_description_train_set.csv", "calcification", "train"),
    "calc_test": ("calc_case_description_test_set.csv", "calcification", "test"),
}

PATHOLOGY_TO_LABEL = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}

IMAGE_TYPE_TO_CASE_COLUMN = {
    "full": "image file path",
    "cropped": "cropped image file path",
    "roi": "ROI mask file path",
}

IMAGE_TYPE_TO_SERIES = {
    "full": "full mammogram images",
    "cropped": "cropped images",
    "roi": "ROI mask images",
}


@dataclass(frozen=True)
class DicomImage:
    patient_id: str
    series_description: str
    image_path: str
    file_path: str
    local_image_path: str | None
    rows: int | None
    columns: int | None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Nao ha linhas para salvar.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _path_segment(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\\", "/").split("/")[0].strip()


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.replace("\\", "/").split())


def jpeg_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        if handle.read(2) != bytes([0xFF, 0xD8]):
            return None
        while True:
            byte = handle.read(1)
            if not byte:
                return None
            if byte != bytes([0xFF]):
                continue
            while byte == bytes([0xFF]):
                byte = handle.read(1)
            if not byte:
                return None
            marker = byte[0]
            if marker == 0 or marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) < 2:
                return None
            length = struct.unpack(">H", length_bytes)[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                handle.read(1)
                height, width = struct.unpack(">HH", handle.read(4))
                return width, height
            handle.seek(length - 2, os.SEEK_CUR)


def physical_image_index(img_dir: Path = IMG_DIR) -> dict[str, list[tuple[Path, tuple[int, int] | None]]]:
    index: dict[str, list[tuple[Path, tuple[int, int] | None]]] = defaultdict(list)
    for path in sorted(img_dir.glob("*.jpg")):
        original_stem = path.stem.split("_")[0]
        original_basename = f"{original_stem}{path.suffix.lower()}"
        index[original_basename].append((path, jpeg_size(path)))
    return index


def resolve_local_image(
    dicom_image_path: str,
    columns: int | None,
    rows: int | None,
    image_index: dict[str, list[tuple[Path, tuple[int, int] | None]]],
) -> str | None:
    basename = Path(dicom_image_path.replace("\\", "/")).name
    expected_size = (columns, rows) if columns and rows else None
    candidates = image_index.get(basename, [])
    if expected_size:
        matches = [path for path, size in candidates if size == expected_size]
        if len(matches) == 1:
            return str(matches[0])
    if len(candidates) == 1:
        return str(candidates[0][0])
    return None


def to_repo_relative(path: str | Path | None) -> str:
    if path is None:
        return ""

    texto = str(path).replace("\\", "/")

    marcador = f"/{ROOT.name}/"
    if marcador in texto:
        return texto.split(marcador, 1)[1]

    if "/data/" in texto:
        return "data/" + texto.split("/data/", 1)[1]

    try:
        return Path(texto).resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return Path(texto).as_posix()


def resolve_repo_path(path: str | Path) -> str:
    caminho = Path(str(path))
    return str(caminho if caminho.is_absolute() else (ROOT / caminho))


def load_dicom_images(data_dir: Path = DATA_DIR, img_dir: Path = IMG_DIR) -> list[DicomImage]:
    image_index = physical_image_index(img_dir)
    rows = _read_csv(data_dir / "dicom_info.csv")
    images: list[DicomImage] = []
    for row in rows:
        columns = _safe_int(row.get("Columns"))
        image_rows = _safe_int(row.get("Rows"))
        local_path = resolve_local_image(row.get("image_path", ""), columns, image_rows, image_index)
        images.append(
            DicomImage(
                patient_id=row.get("PatientID", ""),
                series_description=row.get("SeriesDescription", "") or "",
                image_path=row.get("image_path", ""),
                file_path=row.get("file_path", ""),
                local_image_path=local_path,
                rows=image_rows,
                columns=columns,
            )
        )
    return images


def load_case_rows(data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for _, (filename, abnormality_group, official_split) in CASE_FILES.items():
        for row in _read_csv(data_dir / filename):
            row["source_csv"] = filename
            row["abnormality_group"] = abnormality_group
            row["official_split"] = official_split
            cases.append(row)
    return cases


def build_master_dataset(
    image_type: str = "cropped",
    data_dir: Path = DATA_DIR,
    img_dir: Path = IMG_DIR,
    validation_size: float = 0.2,
    seed: int = 42,
) -> list[dict[str, object]]:
    if image_type == "roi":
        raise ValueError("ROI mask images nao devem ser usadas como imagens de classificacao.")
    if image_type not in ("full", "cropped"):
        raise ValueError("image_type deve ser 'full' ou 'cropped'.")

    cases = load_case_rows(data_dir)
    dicom_images = load_dicom_images(data_dir, img_dir)
    dicom_by_patient_and_type: dict[tuple[str, str], list[DicomImage]] = defaultdict(list)
    for image in dicom_images:
        dicom_by_patient_and_type[(image.patient_id, image.series_description)].append(image)

    case_column = IMAGE_TYPE_TO_CASE_COLUMN[image_type]
    series_description = IMAGE_TYPE_TO_SERIES[image_type]
    rows: list[dict[str, object]] = []

    for case in cases:
        pathology = case.get("pathology", "")
        if pathology not in PATHOLOGY_TO_LABEL:
            continue
        dicom_patient_id = _path_segment(case.get(case_column))
        matches = dicom_by_patient_and_type.get((dicom_patient_id, series_description), [])
        resolved = [image for image in matches if image.local_image_path]
        if len(resolved) != 1:
            continue
        image = resolved[0]
        rows.append(
            {
                "image_path": to_repo_relative(image.local_image_path),
                "patient_id": case.get("patient_id", ""),
                "pathology": pathology,
                "label": PATHOLOGY_TO_LABEL[pathology],
                "abnormality_type": case.get("abnormality type", case.get("abnormality_group", "")),
                "image_type": image_type,
                "official_split": case.get("official_split", ""),
                "split": case.get("official_split", ""),
                "source_csv": case.get("source_csv", ""),
                "dicom_patient_id": dicom_patient_id,
                "series_description": image.series_description,
                "case_image_path": _clean_text(case.get(case_column)),
                "left_or_right_breast": case.get("left or right breast", ""),
                "image_view": case.get("image view", ""),
                "abnormality_id": case.get("abnormality id", ""),
                "rows": image.rows or "",
                "columns": image.columns or "",
            }
        )

    assign_validation_split(rows, validation_size=validation_size, seed=seed)
    return rows


def assign_validation_split(rows: list[dict[str, object]], validation_size: float = 0.2, seed: int = 42) -> None:
    test_patients = {str(row["patient_id"]) for row in rows if row["official_split"] == "test"}
    train_patients = sorted(
        {
            str(row["patient_id"])
            for row in rows
            if row["official_split"] == "train" and str(row["patient_id"]) not in test_patients
        }
    )
    rng = random.Random(seed)
    rng.shuffle(train_patients)
    validation_count = max(1, round(len(train_patients) * validation_size))
    validation_patients = set(train_patients[:validation_count])
    for row in rows:
        if row["official_split"] == "test":
            row["split"] = "test"
        elif str(row["patient_id"]) in test_patients:
            row["split"] = "excluded_official_overlap"
        elif str(row["patient_id"]) in validation_patients:
            row["split"] = "validation"
        else:
            row["split"] = "train"


def validate_no_patient_leakage(rows: list[dict[str, object]]) -> dict[str, object]:
    patients_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        if split.startswith("excluded"):
            continue
        patients_by_split[split].add(str(row["patient_id"]))
    overlaps: dict[str, list[str]] = {}
    splits = sorted(patients_by_split)
    for idx, left in enumerate(splits):
        for right in splits[idx + 1 :]:
            overlap = sorted(patients_by_split[left] & patients_by_split[right])
            if overlap:
                overlaps[f"{left}_x_{right}"] = overlap
    return {
        "patients_by_split": {split: len(patients) for split, patients in patients_by_split.items()},
        "overlaps": overlaps,
        "has_leakage": bool(overlaps),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "pathology": dict(Counter(str(row["pathology"]) for row in rows)),
        "label": dict(Counter(str(row["label"]) for row in rows)),
        "abnormality_type": dict(Counter(str(row["abnormality_type"]) for row in rows)),
        "official_split": dict(Counter(str(row["official_split"]) for row in rows)),
        "split": dict(Counter(str(row["split"]) for row in rows)),
        "image_type": dict(Counter(str(row["image_type"]) for row in rows)),
        "leakage": validate_no_patient_leakage(rows),
    }


def inventory(data_dir: Path = DATA_DIR, img_dir: Path = IMG_DIR) -> dict[str, object]:
    cases = load_case_rows(data_dir)
    dicom_images = load_dicom_images(data_dir, img_dir)
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    dirs = [path for path in ROOT.rglob("*") if path.is_dir()]
    return {
        "project_files": len(files),
        "project_dirs": len(dirs),
        "csv_files": sorted(str(path.relative_to(ROOT)) for path in data_dir.glob("*.csv")),
        "image_dir": str(img_dir),
        "image_extensions": dict(Counter(path.suffix.lower() for path in img_dir.iterdir() if path.is_file())),
        "dicom_series_descriptions": dict(Counter(image.series_description or "<NA>" for image in dicom_images)),
        "case_rows": len(cases),
        "case_patients": len({row["patient_id"] for row in cases}),
        "case_pathology": dict(Counter(row["pathology"] for row in cases)),
        "case_abnormality_type": dict(Counter(row.get("abnormality type", "") for row in cases)),
        "case_official_split": dict(Counter(row["official_split"] for row in cases)),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the CBIS-DDSM computer vision dataset index.")
    parser.add_argument("--image-type", choices=["full", "cropped"], default="cropped")
    parser.add_argument("--output", default=str(DATA_DIR / "cv_master_dataset.csv"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory", action="store_true", help="Print repository/dataset inventory before building.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.inventory:
        print(json.dumps(inventory(), ensure_ascii=False, indent=2))

    rows = build_master_dataset(args.image_type, validation_size=args.validation_size, seed=args.seed)
    _write_csv(Path(args.output), rows)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    print(f"Dataset mestre salvo em: {args.output}")


if __name__ == "__main__":
    main()
