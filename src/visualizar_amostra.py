
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Localizar os arquivos do projeto
RAIZ = Path(__file__).resolve().parent.parent

nome = "05-15_00028_P0030852.png"

caminho_imagem = RAIZ / "dados" / "amostras" / "imagens" / nome
caminho_mascara = RAIZ / "dados" / "amostras" / "mascaras" / nome

# Abrir a fotografia
with Image.open(caminho_imagem) as arquivo:
    imagem = arquivo.convert("RGB")

# Ler a mascara preservando seus valores numericos
with Image.open(caminho_mascara) as arquivo:
    mascara = np.array(arquivo)

# Cores apenas para visualizacao
cores = np.array([
    [40, 40, 40],     # 0 - Solo: cinza
    [40, 190, 80],    # 1 - Cultura: verde
    [240, 65, 65],    # 2 - Daninha: vermelho
    [140, 220, 120],  # 3 - Cultura parcial
    [255, 170, 50],   # 4 - Daninha parcial
], dtype=np.uint8)

# Verificar os valores antes de aplicar as cores
if mascara.ndim != 2:
    raise ValueError("A mascara deve ter apenas um canal.")

if np.any(mascara > 4) or np.any(mascara < 0):
    raise ValueError("A mascara possui classes desconhecidas.")

# Criar uma imagem colorida a partir da mascara
mascara_colorida = Image.fromarray(
    cores[mascara.astype(int)]
)

# Criar sobreposicao sobre a fotografia
original = np.array(imagem).astype(np.float32)
colorida = np.array(mascara_colorida).astype(np.float32)

mistura = (
    original * 0.55 + colorida * 0.45
).astype(np.uint8)

# Manter o solo com a cor original da fotografia
sobreposicao = np.where(
    (mascara > 0)[..., None],
    mistura,
    original.astype(np.uint8)
)

sobreposicao = Image.fromarray(sobreposicao)

# Redimensionar para facilitar a visualizacao
tamanho = (480, 480)

imagens = [
    imagem.resize(tamanho),
    mascara_colorida.resize(tamanho, Image.Resampling.NEAREST),
    sobreposicao.resize(tamanho),
]

titulos = [
    "Fotografia original",
    "Mascara real",
    "Sobreposicao da mascara real",
]

# Montar as tres imagens lado a lado
painel = Image.new("RGB", (1460, 530), "white")
desenho = ImageDraw.Draw(painel)

for i, (foto, titulo) in enumerate(zip(imagens, titulos)):
    x = i * 490
    painel.paste(foto, (x, 40))
    desenho.text((x + 10, 15), titulo, fill="black")

# Salvar o resultado
saida = RAIZ / "resultados" / "visualizacao_amostra.png"
painel.save(saida)

print("Visualizacao criada com sucesso!")
print("Arquivo salvo em:", saida)
