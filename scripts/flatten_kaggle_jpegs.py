
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINO_PADRAO = ROOT / "data" / "img"


def achatar(origem: Path, destino: Path, mover: bool = False) -> dict[str, int]:
    destino.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(origem.rglob("*.jpg")) + sorted(origem.rglob("*.jpeg"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum .jpg encontrado em {origem}")

    contador: dict[str, int] = defaultdict(int)
    copiados = 0

    for arquivo in arquivos:
        nome = arquivo.stem
        indice = contador[nome]
        contador[nome] += 1

        final = f"{nome}.jpg" if indice == 0 else f"{nome}_{indice}.jpg"
        alvo = destino / final

        if alvo.exists():
            continue
        if mover:
            shutil.move(str(arquivo), alvo)
        else:
            shutil.copy2(arquivo, alvo)
        copiados += 1

    colisoes = sum(1 for nome, total in contador.items() if total > 1)
    return {
        "encontrados": len(arquivos),
        "gravados": copiados,
        "nomes_unicos": len(contador),
        "nomes_com_colisao": colisoes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Achata jpeg/ do CBIS-DDSM em data/img/.")
    parser.add_argument("--origem", required=True,
                        help="Pasta 'jpeg' extraida do dataset do Kaggle.")
    parser.add_argument("--destino", default=str(DESTINO_PADRAO))
    parser.add_argument("--mover", action="store_true",
                        help="Move em vez de copiar (economiza ~6 GB de disco).")
    args = parser.parse_args()

    origem = Path(args.origem).expanduser().resolve()
    if not origem.is_dir():
        raise NotADirectoryError(f"Origem invalida: {origem}")

    resumo = achatar(origem, Path(args.destino).resolve(), mover=args.mover)

    for chave, valor in resumo.items():
        print(f"  {chave:.<22} {valor}")
    print(f"\nDestino: {args.destino}")
    print("Proximo passo: python -m src.data_prep")


if __name__ == "__main__":
    main()
