
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PhenoBenchDataset(Dataset):

    def __init__(self, conjunto, tamanho_recorte=512):

        if conjunto not in ("train", "val"):
            raise ValueError("Conjunto deve ser train ou val.")

        self.conjunto = conjunto
        self.tamanho_recorte = tamanho_recorte

        raiz = Path(__file__).resolve().parent.parent

        pasta = raiz / "dados" / "phenobench" / conjunto

        self.pasta_imagens = pasta / "imagens"
        self.pasta_mascaras = pasta / "mascaras"

        self.arquivos = sorted(
            self.pasta_imagens.glob("*.png")
        )

        if not self.arquivos:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada em {self.pasta_imagens}"
            )

        # Normalizacao adequada para encoder
        # pre-treinado em ImageNet
        self.media = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        self.desvio = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

    def __len__(self):

        if self.conjunto == "train":
            return len(self.arquivos)

        # Na validacao, cada imagem gera 4 recortes
        return len(self.arquivos) * 4

    def __getitem__(self, indice):

        if self.conjunto == "train":
            indice_imagem = indice
        else:
            indice_imagem = indice // 4

        caminho_imagem = self.arquivos[indice_imagem]

        caminho_mascara = (
            self.pasta_mascaras / caminho_imagem.name
        )

        # Carregar fotografia RGB
        with Image.open(caminho_imagem) as arquivo:
            imagem = np.array(
                arquivo.convert("RGB")
            )

        # Carregar mascara numerica
        with Image.open(caminho_mascara) as arquivo:
            mascara = np.array(arquivo)

        if imagem.shape[:2] != (1024, 1024):
            raise ValueError(
                f"Dimensoes inesperadas: {caminho_imagem}"
            )

        if mascara.shape != (1024, 1024):
            raise ValueError(
                f"Mascara invalida: {caminho_mascara}"
            )

        # Converter cinco classes em tres
        if not np.all(
            np.isin(np.unique(mascara), [0, 1, 2, 3, 4])
        ):
            raise ValueError(
                f"Classes inesperadas: {caminho_mascara}"
            )

        mascara = mascara.copy()

        mascara[mascara == 3] = 1
        mascara[mascara == 4] = 2

        tamanho = self.tamanho_recorte

        if tamanho != 512:
            raise ValueError(
                "Nesta primeira versao usamos recortes de 512."
            )

        if self.conjunto == "train":

            # Escolher um recorte aleatorio
            x = random.randint(0, 1024 - tamanho)
            y = random.randint(0, 1024 - tamanho)

        else:

            # Quatro recortes fixos na validacao
            quadrante = indice % 4

            x = (quadrante % 2) * tamanho
            y = (quadrante // 2) * tamanho

        imagem = imagem[
            y:y + tamanho,
            x:x + tamanho
        ].copy()

        mascara = mascara[
            y:y + tamanho,
            x:x + tamanho
        ].copy()

        # Converter fotografia para tensor
        imagem = torch.from_numpy(imagem)

        imagem = imagem.permute(2, 0, 1).float()

        imagem = imagem / 255.0

        # Normalizar para o encoder pre-treinado
        imagem = (imagem - self.media) / self.desvio

        # Converter mascara para tensor inteiro
        mascara = torch.from_numpy(
            mascara.astype(np.int64)
        )

        return imagem, mascara


if __name__ == "__main__":

    print("AGROSELETIVO - TESTE DO DATASET")

    treino = PhenoBenchDataset("train")
    validacao = PhenoBenchDataset("val")

    print("\nIMAGENS")
    print("Treinamento:", len(treino.arquivos))
    print("Validacao:", len(validacao.arquivos))

    print("\nAMOSTRAS")
    print("Treinamento:", len(treino))
    print("Validacao:", len(validacao))

    carregador = DataLoader(
        treino,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    imagens, mascaras = next(iter(carregador))

    print("\nTESTE DO PYTORCH")
    print("Formato das imagens:", imagens.shape)
    print("Formato das mascaras:", mascaras.shape)
    print("Tipo das imagens:", imagens.dtype)
    print("Tipo das mascaras:", mascaras.dtype)

    print(
        "Classes presentes no lote:",
        torch.unique(mascaras).tolist()
    )

    print("\nDataset preparado com sucesso!")
