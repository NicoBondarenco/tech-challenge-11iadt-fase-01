
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import to_repo_relative  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte image_path para caminho relativo.")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "cv_master_dataset.csv"))
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    caminho = Path(args.dataset)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    with caminho.open(newline="", encoding="utf-8") as handle:
        linhas = list(csv.DictReader(handle))
    if not linhas:
        raise ValueError("CSV vazio.")

    if not args.no_backup:
        backup = caminho.with_suffix(".csv.bak")
        shutil.copy2(caminho, backup)
        print(f"Backup: {backup}")

    convertidas = 0
    for linha in linhas:
        original = linha.get("image_path", "")
        novo = to_repo_relative(original)
        if novo != original:
            convertidas += 1
        linha["image_path"] = novo

    with caminho.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.DictWriter(handle, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"{convertidas} de {len(linhas)} linhas convertidas.")
    print(f"Exemplo: {linhas[0]['image_path']}")

    faltando = [
        linha["image_path"] for linha in linhas[:50]
        if not (ROOT / linha["image_path"]).exists()
    ]
    if faltando:
        print(f"\nAviso: {len(faltando)} das 50 primeiras imagens nao foram encontradas em disco.")
        print("Isso e esperado se data/img/ nao estiver presente nesta copia.")


if __name__ == "__main__":
    main()
