
from pathlib import Path

import numpy as np
from PIL import Image


RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "dados" / "phenobench"

NOMES_CLASSES = {
    0: "Solo / fundo",
    1: "Planta cultivada",
    2: "Planta daninha",
}


def analisar_conjunto(conjunto):
    pasta = DATASET / conjunto / "mascaras"
    arquivos = sorted(pasta.glob("*.png"))

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhuma mascara encontrada em: {pasta}"
        )

    contagem_pixels = {}
    imagens_com_daninhas = 0
    imagens_com_cultura = 0

    print(f"\n=== {conjunto.upper()} ===")
    print("Analisando", len(arquivos), "mascaras...")

    for arquivo in arquivos:

        with Image.open(arquivo) as imagem:
            mascara = np.array(imagem)

        if mascara.ndim != 2:
            raise ValueError(
                f"Mascara com formato inesperado: {arquivo}"
            )

        # Contar pixels de cada classe
        contagens = np.bincount(mascara.ravel())

        for classe in np.flatnonzero(contagens):
            classe = int(classe)

            contagem_pixels[classe] = (
                contagem_pixels.get(classe, 0)
                + int(contagens[classe])
            )

        # Verificar se a imagem possui plantas
        if len(contagens) > 1 and contagens[1] > 0:
            imagens_com_cultura += 1

        if len(contagens) > 2 and contagens[2] > 0:
            imagens_com_daninhas += 1

    total_pixels = sum(contagem_pixels.values())

    print("\nDISTRIBUICAO DOS PIXELS:")

    for classe, quantidade in sorted(contagem_pixels.items()):

        nome = NOMES_CLASSES.get(classe, "Classe a verificar")
        percentual = quantidade / total_pixels * 100

        print(
            f"Classe {classe} ({nome}): "
            f"{percentual:.3f}%"
        )

    print("\nPRESENCA DE PLANTAS NAS IMAGENS:")

    print(
        "Imagens com cultura (classe 1):",
        imagens_com_cultura,
        "de",
        len(arquivos)
    )

    print(
        "Imagens com daninhas (classe 2):",
        imagens_com_daninhas,
        "de",
        len(arquivos)
    )

    print(
        "Imagens sem classe 2:",
        len(arquivos) - imagens_com_daninhas
    )


print("AGROSELETIVO - ANALISE DO DATASET")

analisar_conjunto("train")
analisar_conjunto("val")

print("\nAnalise finalizada!")
