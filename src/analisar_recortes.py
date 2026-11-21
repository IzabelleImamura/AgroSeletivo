
import random

import numpy as np

from dataset_agroseletivo import PhenoBenchDataset


# Tornar o experimento reproduzivel
random.seed(42)

# Carregar o dataset de treinamento
dataset = PhenoBenchDataset("train")

# Quantidade de recortes que vamos analisar
quantidade = 100

# Selecionar 100 fotografias diferentes
indices = random.sample(
    range(len(dataset)),
    quantidade
)

recortes_com_daninhas = 0
recortes_sem_daninhas = 0

percentuais_daninhas = []

print("AGROSELETIVO - ANALISE DOS RECORTES")
print("Quantidade de recortes:", quantidade)

for indice in indices:

    # O dataset seleciona um recorte aleatorio
    imagem, mascara = dataset[indice]

    # Converter a mascara para NumPy
    dados = mascara.numpy()

    # Contar pixels de plantas daninhas
    pixels_daninhas = np.count_nonzero(
        dados == 2
    )

    total_pixels = dados.size

    percentual = (
        pixels_daninhas / total_pixels
    ) * 100

    percentuais_daninhas.append(percentual)

    if pixels_daninhas > 0:
        recortes_com_daninhas += 1
    else:
        recortes_sem_daninhas += 1


print("\nRESULTADOS")

print(
    "Recortes com daninhas:",
    recortes_com_daninhas
)

print(
    "Recortes sem daninhas:",
    recortes_sem_daninhas
)

print(
    "Percentual com daninhas:",
    f"{recortes_com_daninhas / quantidade * 100:.2f}%"
)

print(
    "Media de pixels de daninhas:",
    f"{np.mean(percentuais_daninhas):.4f}%"
)

print(
    "Maior percentual encontrado:",
    f"{np.max(percentuais_daninhas):.4f}%"
)

print(
    "Menor percentual encontrado:",
    f"{np.min(percentuais_daninhas):.4f}%"
)

print("\nAnalise finalizada!")
