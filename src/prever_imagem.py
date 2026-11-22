
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp

from PIL import Image


# ==========================================
# 1. CONFIGURACOES E CAMINHOS
# ==========================================

RAIZ = Path(__file__).resolve().parent.parent

NOME_IMAGEM = "05-15_00028_P0030852.png"

CAMINHO_IMAGEM = (
    RAIZ / "dados" / "amostras"
    / "imagens" / NOME_IMAGEM
)

CAMINHO_MODELO = (
    RAIZ / "modelos"
    / "unet_completo_ponderado_melhor.pt"
)

PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

DISPOSITIVO = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("AGROSELETIVO - PRIMEIRA PREVISAO")
print("Dispositivo:", DISPOSITIVO)


# ==========================================
# 2. CARREGAR O MODELO TREINADO
# ==========================================

if not CAMINHO_MODELO.is_file():
    raise FileNotFoundError(
        f"Modelo nao encontrado: {CAMINHO_MODELO}"
    )

modelo = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=3,
    classes=3,
    activation=None
)

checkpoint = torch.load(
    CAMINHO_MODELO,
    map_location="cpu",
    weights_only=True
)

modelo.load_state_dict(checkpoint["modelo"])

modelo = modelo.to(DISPOSITIVO)
modelo.eval()

print("\nModelo carregado com sucesso!")
print("Epoca do treinamento:", checkpoint["epoca"])


# ==========================================
# 3. ABRIR A FOTOGRAFIA
# ==========================================

if not CAMINHO_IMAGEM.is_file():
    raise FileNotFoundError(
        f"Fotografia nao encontrada: {CAMINHO_IMAGEM}"
    )

with Image.open(CAMINHO_IMAGEM) as arquivo:
    fotografia = np.array(
        arquivo.convert("RGB")
    )

altura, largura = fotografia.shape[:2]

print("\nFOTOGRAFIA")
print("Arquivo:", NOME_IMAGEM)
print("Dimensoes:", largura, "x", altura)

if (altura, largura) != (1024, 1024):
    raise ValueError(
        "Este primeiro teste utiliza imagens de 1024x1024."
    )


# ==========================================
# 4. PREPARAR A NORMALIZACAO
# ==========================================

media = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

desvio = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# ==========================================
# 5. REALIZAR A PREVISAO
# ==========================================

mascara_prevista = np.zeros(
    (1024, 1024),
    dtype=np.uint8
)

with torch.inference_mode():

    for y in (0, 512):
        for x in (0, 512):

            # Separar um recorte de 512x512
            recorte = np.ascontiguousarray(
                fotografia[y:y+512, x:x+512]
            )

            # Converter fotografia em tensor
            tensor = torch.from_numpy(recorte)

            tensor = tensor.permute(2, 0, 1).float()
            tensor = tensor / 255.0

            # Aplicar a mesma normalizacao do treino
            tensor = (tensor - media) / desvio

            entrada = tensor.unsqueeze(0).to(
                DISPOSITIVO
            )

            # Executar a U-Net treinada
            saida = modelo(entrada)

            previsao = torch.argmax(
                saida, dim=1
            )[0].cpu().numpy()

            # Reconstruir a mascara completa
            mascara_prevista[
                y:y+512, x:x+512
            ] = previsao


# ==========================================
# 6. TRANSFORMAR AS CLASSES EM CORES
# ==========================================

cores = np.array([
    [64, 64, 64],    # Solo: cinza
    [32, 182, 90],   # Cultura: verde
    [240, 68, 68]    # Daninhas: vermelho
], dtype=np.uint8)

mascara_colorida = cores[mascara_prevista]

# Criar sobreposicao na fotografia original
sobreposicao = fotografia.copy()

regioes_plantas = mascara_prevista > 0

sobreposicao[regioes_plantas] = (
    0.55 * fotografia[regioes_plantas]
    + 0.45 * mascara_colorida[regioes_plantas]
).astype(np.uint8)


# ==========================================
# 7. SALVAR OS RESULTADOS
# ==========================================

caminho_mascara = (
    PASTA_RESULTADOS / "mascara_prevista.png"
)

caminho_sobreposicao = (
    PASTA_RESULTADOS / "sobreposicao_prevista.png"
)

Image.fromarray(mascara_colorida).save(
    caminho_mascara
)

Image.fromarray(sobreposicao).save(
    caminho_sobreposicao
)

print("\nPREVISAO FINALIZADA!")

print(
    "Classes identificadas:",
    np.unique(mascara_prevista).tolist()
)

print("Mascara salva em:", caminho_mascara)
print("Sobreposicao salva em:", caminho_sobreposicao)
