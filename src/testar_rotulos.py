
from pathlib import Path

import numpy as np
from PIL import Image


# Localizar o dataset de treinamento
RAIZ = Path(__file__).resolve().parent.parent

PASTA = RAIZ / "dados" / "phenobench" / "train" / "mascaras"


def converter_rotulos(mascara):
    # Garantir que somente as classes esperadas existem
    valores = np.unique(mascara)

    if not np.all(np.isin(valores, [0, 1, 2, 3, 4])):
        raise ValueError(f"Classes desconhecidas: {valores}")

    # Criar uma copia para preservar a mascara original
    convertida = mascara.copy()

    # Agrupar as plantas parcialmente visiveis
    convertida[mascara == 3] = 1
    convertida[mascara == 4] = 2

    return convertida


# Procurar exemplos que possuam as classes especiais
arquivos = sorted(PASTA.glob("*.png"))

encontrados = {}

for arquivo in arquivos:

    with Image.open(arquivo) as imagem:
        mascara = np.array(imagem)

    for classe in [3, 4]:
        if classe not in encontrados and np.any(mascara == classe):
            encontrados[classe] = arquivo

    if len(encontrados) == 2:
        break


print("AGROSELETIVO - TESTE DOS ROTULOS")

for classe in [3, 4]:

    if classe not in encontrados:
        print(f"\nNenhum exemplo da classe {classe} encontrado.")
        continue

    arquivo = encontrados[classe]

    with Image.open(arquivo) as imagem:
        original = np.array(imagem)

    convertida = converter_rotulos(original)

    print(f"\nEXEMPLO COM CLASSE {classe}")
    print("Arquivo:", arquivo.name)
    print("Classes antes:", np.unique(original))
    print("Classes depois:", np.unique(convertida))

    print(
        f"Pixels da classe {classe} original:",
        np.count_nonzero(original == classe)
    )

    print(
        "Conversao realizada corretamente:",
        np.all(np.isin(np.unique(convertida), [0, 1, 2]))
    )

print("\nTeste finalizado!")
