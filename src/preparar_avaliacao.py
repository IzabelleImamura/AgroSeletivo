
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

# ==========================================
# 1. DEFINIR CAMINHOS
# ==========================================

RAIZ = Path(__file__).resolve().parent.parent

PASTA_VAL = RAIZ / "dados" / "phenobench" / "val"

ZIP_PILOTO = RAIZ.parent / "AgroSeletivo_dados_piloto.zip"

SAIDA = RAIZ.parent / "AgroSeletivo_avaliacao_722.zip"


# ==========================================
# 2. IDENTIFICAR AS 50 IMAGENS DO PILOTO
# ==========================================

with ZipFile(ZIP_PILOTO, "r") as pacote:
    conteudo = pacote.read(
        "val/arquivos_selecionados.txt"
    ).decode("utf-8")

nomes_piloto = {
    nome.strip()
    for nome in conteudo.splitlines()
    if nome.strip()
}

if len(nomes_piloto) != 50:
    raise ValueError("Esperadas 50 imagens no piloto.")


# ==========================================
# 3. SEPARAR AS IMAGENS RESTANTES
# ==========================================

pasta_imagens = PASTA_VAL / "imagens"
pasta_mascaras = PASTA_VAL / "mascaras"

nomes_todos = {
    arquivo.name
    for arquivo in pasta_imagens.glob("*.png")
}

if len(nomes_todos) != 772:
    raise ValueError("Validacao original incompleta.")

if not nomes_piloto.issubset(nomes_todos):
    raise ValueError("Imagens do piloto nao encontradas.")

nomes_reservados = sorted(nomes_todos - nomes_piloto)

assert len(nomes_reservados) == 722
assert not (set(nomes_reservados) & nomes_piloto)

print("AGROSELETIVO - AVALIACAO RESERVADA")
print("Validacao original:", len(nomes_todos))
print("Utilizadas no piloto:", len(nomes_piloto))
print("Reservadas para avaliacao:", len(nomes_reservados))


# ==========================================
# 4. CONFERIR OS PARES
# ==========================================

for nome in nomes_reservados:
    if not (pasta_mascaras / nome).is_file():
        raise FileNotFoundError(f"Mascara ausente: {nome}")


# ==========================================
# 5. CRIAR O PACOTE DE AVALIACAO
# ==========================================

with ZipFile(SAIDA, "w", compression=ZIP_STORED) as pacote:

    pacote.writestr(
        "avaliacao/arquivos_reservados.txt",
        "\n".join(nomes_reservados)
    )

    for nome in nomes_reservados:

        pacote.write(
            pasta_imagens / nome,
            arcname=f"avaliacao/imagens/{nome}"
        )

        pacote.write(
            pasta_mascaras / nome,
            arcname=f"avaliacao/mascaras/{nome}"
        )


# ==========================================
# 6. CONFERIR O RESULTADO
# ==========================================

with ZipFile(SAIDA, "r") as pacote:
    quantidade = len(pacote.namelist())

assert quantidade == 1445

print("\nPACOTE CRIADO COM SUCESSO!")
print("Arquivo:", SAIDA)
print("Total de arquivos:", quantidade)
print(
    "Tamanho:",
    f"{SAIDA.stat().st_size / (1024**3):.2f} GB"
)
print("Nenhuma fotografia do piloto foi incluida.")
