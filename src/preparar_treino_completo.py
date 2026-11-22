
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

# ==========================================
# 1. DEFINIR OS CAMINHOS
# ==========================================

RAIZ = Path(__file__).resolve().parent.parent

PASTA_TREINO = RAIZ / "dados" / "phenobench" / "train"

PASTA_IMAGENS = PASTA_TREINO / "imagens"
PASTA_MASCARAS = PASTA_TREINO / "mascaras"

# Salvar fora do repositorio GitHub
SAIDA = RAIZ.parent / "AgroSeletivo_treino_completo.zip"


# ==========================================
# 2. VERIFICAR OS ARQUIVOS
# ==========================================

imagens = sorted(PASTA_IMAGENS.glob("*.png"))
mascaras = sorted(PASTA_MASCARAS.glob("*.png"))

nomes_imagens = {arquivo.name for arquivo in imagens}
nomes_mascaras = {arquivo.name for arquivo in mascaras}

if len(imagens) != 1407:
    raise ValueError("Quantidade inesperada de imagens!")

if nomes_imagens != nomes_mascaras:
    raise ValueError("Imagens e mascaras nao correspondem!")

print("AGROSELETIVO - TREINAMENTO COMPLETO")
print("Fotografias:", len(imagens))
print("Mascaras:", len(mascaras))


# ==========================================
# 3. CRIAR O PACOTE
# ==========================================

with ZipFile(SAIDA, "w", compression=ZIP_STORED) as pacote:

    for indice, imagem in enumerate(imagens, start=1):

        mascara = PASTA_MASCARAS / imagem.name

        pacote.write(
            imagem,
            arcname=f"train/imagens/{imagem.name}"
        )

        pacote.write(
            mascara,
            arcname=f"train/mascaras/{mascara.name}"
        )

        if indice % 200 == 0:
            print(f"Processados: {indice}/1407")


# ==========================================
# 4. CONFERIR O PACOTE
# ==========================================

with ZipFile(SAIDA, "r") as pacote:
    quantidade_arquivos = len(pacote.namelist())

if quantidade_arquivos != 2814:
    raise ValueError("Quantidade incorreta no ZIP!")

tamanho_gb = SAIDA.stat().st_size / (1024 ** 3)

print("\nPACOTE CRIADO COM SUCESSO!")
print("Arquivo:", SAIDA)
print("Total de arquivos:", quantidade_arquivos)
print(f"Tamanho: {tamanho_gb:.2f} GB")
