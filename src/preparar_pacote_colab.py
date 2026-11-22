
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import random


# Pasta principal do AgroSeletivo
RAIZ = Path(__file__).resolve().parent.parent

# Dataset original
DATASET = RAIZ / "dados" / "phenobench"

# Salvar o ZIP fora do repositorio Git
SAIDA = RAIZ.parent / "AgroSeletivo_dados_piloto.zip"

# Quantidade de imagens por conjunto
QUANTIDADES = {
    "train": 200,
    "val": 50,
}

# Garantir uma selecao reproduzivel
gerador = random.Random(42)

print("AGROSELETIVO - PREPARACAO DOS DADOS")
print("Criando pacote experimental...\n")

with ZipFile(SAIDA, "w", compression=ZIP_STORED) as pacote:

    for conjunto, quantidade in QUANTIDADES.items():

        pasta = DATASET / conjunto

        pasta_imagens = pasta / "imagens"
        pasta_mascaras = pasta / "mascaras"

        # Encontrar todas as fotografias
        imagens = sorted(pasta_imagens.glob("*.png"))

        # Conferir a correspondencia com as mascaras
        for imagem in imagens:
            mascara = pasta_mascaras / imagem.name

            if not mascara.is_file():
                raise FileNotFoundError(
                    f"Mascara nao encontrada: {mascara}"
                )

        if len(imagens) < quantidade:
            raise ValueError(
                f"Imagens insuficientes em {conjunto}"
            )

        # Selecionar fotografias aleatoriamente
        selecionadas = sorted(
            gerador.sample(imagens, quantidade)
        )

        # Guardar os nomes selecionados para reproducao
        manifesto = "\n".join(
            imagem.name for imagem in selecionadas
        )

        pacote.writestr(
            f"{conjunto}/arquivos_selecionados.txt",
            manifesto
        )

        # Adicionar fotografias e mascaras ao ZIP
        for imagem in selecionadas:

            mascara = pasta_mascaras / imagem.name

            pacote.write(
                imagem,
                arcname=f"{conjunto}/imagens/{imagem.name}"
            )

            pacote.write(
                mascara,
                arcname=f"{conjunto}/mascaras/{mascara.name}"
            )

        print(
            f"{conjunto}: {len(selecionadas)} pares adicionados"
        )

# Conferir o tamanho do arquivo gerado
tamanho_mb = SAIDA.stat().st_size / (1024 ** 2)

print("\nPACOTE CRIADO")
print("Arquivo:", SAIDA)
print(f"Tamanho: {tamanho_mb:.2f} MB")
print("Total: 250 fotografias e 250 mascaras")
print("Semente da selecao: 42")
print("\nPreparacao concluida!")
