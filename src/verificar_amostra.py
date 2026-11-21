
from pathlib import Path

import numpy as np
from PIL import Image


# 1. Localizar a pasta principal do projeto
RAIZ = Path(__file__).resolve().parent.parent

# 2. Nome da fotografia e da mascara
nome_arquivo = "05-15_00028_P0030852.png"

# 3. Definir os caminhos dos arquivos
caminho_imagem = RAIZ / "dados" / "amostras" / "imagens" / nome_arquivo
caminho_mascara = RAIZ / "dados" / "amostras" / "mascaras" / nome_arquivo

# 4. Verificar se os arquivos existem
if not caminho_imagem.exists():
    raise FileNotFoundError(f"Imagem nao encontrada: {caminho_imagem}")

if not caminho_mascara.exists():
    raise FileNotFoundError(f"Mascara nao encontrada: {caminho_mascara}")

# 5. Abrir a fotografia e a mascara
with Image.open(caminho_imagem) as imagem:
    tamanho_imagem = imagem.size
    modo_imagem = imagem.mode

with Image.open(caminho_mascara) as mascara:
    tamanho_mascara = mascara.size
    modo_mascara = mascara.mode
    dados_mascara = np.array(mascara)

# 6. Mostrar informacoes
print("=== AGROSELETIVO - VERIFICACAO DOS DADOS ===")

print("\nFOTOGRAFIA")
print("Dimensoes:", tamanho_imagem)
print("Modo:", modo_imagem)

print("\nMASCARA")
print("Dimensoes:", tamanho_mascara)
print("Modo:", modo_mascara)
print("Formato dos dados:", dados_mascara.shape)

# 7. Conferir se possuem as mesmas dimensoes
if tamanho_imagem == tamanho_mascara:
    print("\nDimensoes compativeis!")
else:
    print("\nATENCAO: dimensoes diferentes!")

# 8. Verificar os valores da mascara
if dados_mascara.ndim == 2:

    valores, quantidades = np.unique(
        dados_mascara, return_counts=True
    )

    nomes_classes = {
        0: "Solo / fundo",
        1: "Planta cultivada",
        2: "Planta daninha",
        3: "Planta cultivada parcial",
        4: "Planta daninha parcial",
    }

    print("\nCLASSES ENCONTRADAS:")

    for valor, quantidade in zip(valores, quantidades):
        nome = nomes_classes.get(int(valor), "Desconhecida")
        print(f"Classe {valor} ({nome}): {quantidade} pixels")

else:
    print("\nATENCAO: a mascara nao possui um unico canal.")
