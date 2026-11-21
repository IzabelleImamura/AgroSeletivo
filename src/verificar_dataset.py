
from pathlib import Path

import numpy as np
from PIL import Image


# Localizar a pasta principal do projeto
RAIZ = Path(__file__).resolve().parent.parent

# Localizar o dataset
DATASET = RAIZ / "dados" / "phenobench"


def verificar_conjunto(nome_conjunto):
    print(f"\n=== CONJUNTO: {nome_conjunto.upper()} ===")

    pasta = DATASET / nome_conjunto

    pasta_imagens = pasta / "imagens"
    pasta_mascaras = pasta / "mascaras"

    # Verificar se as pastas existem
    if not pasta_imagens.is_dir():
        print("ERRO: pasta de imagens nao encontrada!")
        return

    if not pasta_mascaras.is_dir():
        print("ERRO: pasta de mascaras nao encontrada!")
        return

    # Listar os arquivos PNG
    imagens = sorted(pasta_imagens.glob("*.png"))
    mascaras = sorted(pasta_mascaras.glob("*.png"))

    print("Quantidade de imagens:", len(imagens))
    print("Quantidade de mascaras:", len(mascaras))

    # Comparar os nomes dos arquivos
    nomes_imagens = {arquivo.stem for arquivo in imagens}
    nomes_mascaras = {arquivo.stem for arquivo in mascaras}

    sem_mascara = nomes_imagens - nomes_mascaras
    sem_imagem = nomes_mascaras - nomes_imagens

    print("Imagens sem mascara:", len(sem_mascara))
    print("Mascaras sem imagem:", len(sem_imagem))

    if sem_mascara:
        print("Exemplos sem mascara:", sorted(sem_mascara)[:5])

    if sem_imagem:
        print("Exemplos sem imagem:", sorted(sem_imagem)[:5])

    if not sem_mascara and not sem_imagem and imagens:
        print("Correspondencia dos nomes: OK!")

    # Examinar um par de arquivos
    nomes_pares = sorted(nomes_imagens & nomes_mascaras)

    if not nomes_pares:
        print("Nenhum par encontrado.")
        return

    nome = nomes_pares[0]

    caminho_imagem = pasta_imagens / f"{nome}.png"
    caminho_mascara = pasta_mascaras / f"{nome}.png"

    with Image.open(caminho_imagem) as imagem:
        tamanho_imagem = imagem.size

    with Image.open(caminho_mascara) as mascara:
        tamanho_mascara = mascara.size
        modo_mascara = mascara.mode
        dados = np.array(mascara)

    print("\nEXEMPLO ANALISADO")
    print("Arquivo:", nome)
    print("Dimensoes da imagem:", tamanho_imagem)
    print("Dimensoes da mascara:", tamanho_mascara)
    print("Modo da mascara:", modo_mascara)
    print("Valores encontrados:", np.unique(dados))

    if tamanho_imagem == tamanho_mascara:
        print("Dimensoes do exemplo: OK!")
    else:
        print("ATENCAO: dimensoes diferentes!")


print("AGROSELETIVO - VERIFICACAO DO PHENOBENCH")

verificar_conjunto("train")
verificar_conjunto("val")

print("\nVerificacao inicial finalizada!")
