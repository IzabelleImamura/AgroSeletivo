from pathlib import Path
import hashlib
import html
import io
import os
import urllib.request

import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError


# ==========================================
# 1. CONFIGURACAO DA APLICACAO
# ==========================================

st.set_page_config(
    page_title="AgroSeletivo",
    layout="wide",
    initial_sidebar_state="collapsed"
)

RAIZ = Path(__file__).resolve().parent

CAMINHO_MODELO = (
    RAIZ / "modelos" /
    "unet_completo_ponderado_melhor.pt"
)

TAMANHO_RECORTE = 512
MAX_PIXELS = 12_000_000


# ==========================================
# 2. IDENTIDADE VISUAL
# ==========================================

st.html("""
<style>

/* ==========================================
   2.1. ESTRUTURA PRINCIPAL
   ========================================== */

[data-testid="stAppViewContainer"] {
    background: #171d1b;
    color: #edf0ee;
}

[data-testid="stHeader"] {
    background: #171d1b;
}

.block-container {
    max-width: 1160px !important;
    padding-top: 5rem !important;
    padding-bottom: 2rem !important;
}


/* ==========================================
   2.2. CABECALHO
   ========================================== */

.agro-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;

    padding-bottom: 22px;
    margin-bottom: 26px;

    border-bottom: 1px solid #303b35;
}

.agro-brand {
    color: #edf0ee;
    font-size: 25px;
    font-weight: 750;
    letter-spacing: -0.9px;
}

.agro-brand span {
    color: #78bf94;
}

.agro-header-description {
    color: #9eada4;
    font-size: 12px;
}


/* ==========================================
   2.3. TITULOS E DESCRICOES
   ========================================== */

.agro-title {
    color: #edf0ee;
    font-size: 23px;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.3;
}

.agro-description {
    color: #9ca9a1;
    font-size: 13px;
    line-height: 1.6;

    margin-top: 6px;
    margin-bottom: 18px;
}


/* ==========================================
   2.4. CAIXA EXTERNA DE ENVIO
   ========================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #202824;
    border: 1px solid #38463d !important;
    border-radius: 8px !important;
}


/* ==========================================
   2.5. UPLOAD INTEGRADO
   ========================================== */

/* Remover a segunda borda do upload */
[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    align-items: center !important;

    gap: 12px !important;
    min-height: 52px !important;
    padding: 2px 0 !important;

    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
}

/* Ocultar as instrucoes padrao */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}

/* Icone de fotografia */
[data-testid="stFileUploaderDropzone"]::before {
    content: "";

    order: 0;
    flex: 0 0 38px;

    width: 38px;
    height: 38px;
    border-radius: 6px;

    background-color: #34463a;
    background-repeat: no-repeat;
    background-position: center;
    background-size: 21px 21px;

    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b7d5bd' stroke-width='1.6'%3E%3Crect x='3' y='3' width='18' height='18' rx='3'/%3E%3Ccircle cx='9' cy='9' r='1.6'/%3E%3Cpath d='m5 18 5-6 3 3 2-2 4 5'/%3E%3C/svg%3E");

    pointer-events: none;
}

/* Texto da caixa */
[data-testid="stFileUploaderDropzone"]::after {
    content: "Selecionar fotografia · PNG ou JPG";

    order: 1;
    flex: 1;

    font-size: 12px;
    font-weight: 500;
    color: #dce7df;

    pointer-events: none;
}

/* Posicionar o seletor nativo a direita */
[data-testid="stFileUploaderDropzone"] > div {
    order: 2;
    margin-left: auto;
}

/* Botao nativo para selecionar arquivo */
[data-testid="stFileUploaderDropzone"] button {
    background: #29382e !important;
    color: #e4ece5 !important;

    border: 1px solid #45574b !important;
    border-radius: 6px !important;

    min-height: 38px !important;
    padding: 7px 14px !important;

    font-size: 12px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #34483a !important;
    border-color: #78bf94 !important;
}


/* Ocultar apenas o botao de adicionar mais arquivos */
[data-testid="stFileUploader"] button[aria-label="Add files"] {
    display: none !important;
}


/* ==========================================
   2.6. INFORMACOES DA FOTOGRAFIA
   ========================================== */

.agro-file-name {
    color: #edf0ee;
    font-size: 13px;
    font-weight: 650;

    overflow-wrap: anywhere;
}

.agro-file-details {
    color: #94a79a;
    font-size: 11px;
    margin-top: 4px;
}


/* ==========================================
   2.7. STATUS DA ANALISE
   ========================================== */

.agro-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;

    color: #a2c5ac;
    font-size: 12px;
    font-weight: 600;

    white-space: nowrap;
}

.agro-status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;

    background: #65bd88;
    border-radius: 50%;
}


/* ==========================================
   2.8. BOTOES PRINCIPAIS
   ========================================== */

div.stButton > button[kind="primary"] {
    background: #7ab992;
    border: 1px solid #7ab992;
    color: #15221a;

    border-radius: 6px;
    font-weight: 650;
    min-height: 42px;
}

div.stButton > button[kind="primary"]:hover {
    background: #95c9a5;
    border-color: #95c9a5;
    color: #15221a;
}

div.stButton > button[kind="secondary"] {
    background: transparent;
    color: #dce7df;

    border: 1px solid #45574b;
    border-radius: 6px;

    min-height: 42px;
}

div.stButton > button[kind="secondary"]:hover {
    background: #29382e;
    color: #ffffff;
    border-color: #76917d;
}


/* ==========================================
   2.9. BOTOES DE DOWNLOAD
   ========================================== */

div.stDownloadButton > button {
    background: transparent;
    color: #dce7df;

    border: 1px solid #45574b;
    border-radius: 6px;

    min-height: 42px;
}

div.stDownloadButton > button:hover {
    background: #29382e;
    color: #ffffff;
    border-color: #76917d;
}


/* ==========================================
   2.10. RESULTADOS
   ========================================== */

.agro-results-heading {
    border-top: 1px solid #303b35;

    padding-top: 18px;
    margin-top: 20px;
    margin-bottom: 12px;
}

.agro-results-title {
    color: #edf0ee;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.3px;
}

.agro-image-title {
    color: #edf0ee;
    font-size: 13px;
    font-weight: 650;
    margin-top: 9px;
}

.agro-image-caption {
    color: #92a49a;
    font-size: 11px;
    margin-top: 4px;
}

[data-testid="stImage"] img {
    border-radius: 6px;
}


/* ==========================================
   2.11. LEGENDA
   ========================================== */

.agro-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;

    margin-top: 20px;
    margin-bottom: 16px;

    color: #bbc7be;
    font-size: 12px;
}

.agro-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 7px;
}

.agro-legend-dot {
    display: inline-block;

    width: 10px;
    height: 10px;

    border-radius: 2px;
}


/* ==========================================
   2.12. RODAPE
   ========================================== */

.agro-footer {
    margin-top: 28px;
    padding-top: 17px;

    border-top: 1px solid #303b35;

    color: #82968a;
    font-size: 11px;
    line-height: 1.7;
}


/* ==========================================
   2.13. RESPONSIVIDADE
   ========================================== */


@media (max-width: 650px) {

    /* 1. Espacamento geral */
    .block-container {
        padding-top: 5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1.5rem !important;
    }

    /* 2. Cabecalho */
    .agro-header {
        padding-bottom: 16px;
        margin-bottom: 18px;
    }

    .agro-brand {
        font-size: 21px;
        line-height: 1.4;
        overflow: visible;
    }

    .agro-header-description {
        display: none;
    }

    /* 3. Titulo da pagina */
    .agro-title {
        font-size: 20px;
        line-height: 1.4;
        overflow: visible;
    }

    .agro-description {
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    /* 4. Upload compacto */
    [data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 10px !important;
        min-height: 48px !important;
        padding: 4px !important;
        border: none !important;
        background: transparent !important;
    }

    /* Esconder elementos nativos desnecessarios */
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    /* Icone personalizado */
    [data-testid="stFileUploaderDropzone"]::before {
        flex: 0 0 36px;
        width: 36px;
        height: 36px;
    }

    /* Texto compacto */
    [data-testid="stFileUploaderDropzone"]::after {
        content: "Selecionar fotografia";
        font-size: 11px;
        min-width: 0;
    }

    /* Botao nativo */
    [data-testid="stFileUploaderDropzone"] button {
        min-height: 36px !important;
        padding: 6px 10px !important;
        font-size: 11px !important;
        white-space: nowrap;
    }

    /* 5. Legenda */
    .agro-legend {
        gap: 12px;
    }
}


</style>
""")


# ==========================================
# 3. GERENCIAMENTO DA APLICACAO
# ==========================================

# Guardar a fotografia e o resultado entre
# as atualizacoes automaticas do Streamlit.

if "foto_atual" not in st.session_state:
    st.session_state["foto_atual"] = None

if "resultado_atual" not in st.session_state:
    st.session_state["resultado_atual"] = None

if "versao_upload" not in st.session_state:
    st.session_state["versao_upload"] = 0


def limpar_analise():

    # Remover a fotografia anterior
    st.session_state["foto_atual"] = None

    # Remover o resultado anterior
    st.session_state["resultado_atual"] = None

    # Reiniciar o componente de upload
    st.session_state["versao_upload"] += 1

    # Atualizar a interface
    st.rerun()


# ==========================================
# 4. CARREGAR A REDE NEURAL
# ==========================================


@st.cache_resource(show_spinner=False)
def carregar_modelo():

    # Baixar o modelo apenas se ele nao existir localmente
    if not CAMINHO_MODELO.is_file():

        url = (
            "https://github.com/IzabelleImamura/AgroSeletivo/"
            "releases/download/v1.0.0/"
            "unet_completo_ponderado_melhor.pt"
        )

        CAMINHO_MODELO.parent.mkdir(
            parents=True, exist_ok=True
        )

        temporario = CAMINHO_MODELO.with_suffix(".download")

        try:
            requisicao = urllib.request.Request(
                url,
                headers={"User-Agent": "AgroSeletivo/1.0"}
            )

            with urllib.request.urlopen(
                requisicao, timeout=120
            ) as resposta, open(temporario, "wb") as arquivo:

                while True:
                    bloco = resposta.read(1024 * 1024)
                    if not bloco:
                        break
                    arquivo.write(bloco)

            os.replace(temporario, CAMINHO_MODELO)

        finally:
            if temporario.exists():
                temporario.unlink()

    # Mesma arquitetura utilizada no treinamento
    modelo = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=3,
        activation=None
    )

    # Carregar os pesos treinados
    checkpoint = torch.load(
        CAMINHO_MODELO,
        map_location="cpu",
        weights_only=True
    )

    modelo.load_state_dict(
        checkpoint["modelo"]
    )

    # Colocar o modelo em modo de avaliacao
    modelo.eval()

    return modelo


# ==========================================
# 5. ABRIR E VALIDAR A FOTOGRAFIA
# ==========================================

def abrir_fotografia(dados):

    with Image.open(io.BytesIO(dados)) as imagem:

        largura, altura = imagem.size

        # Evitar fotografias grandes demais
        if largura * altura > MAX_PIXELS:

            raise ValueError(
                "A fotografia excede o limite "
                "de 12 megapixels."
            )

        # Corrigir orientacao e converter para RGB
        fotografia = ImageOps.exif_transpose(
            imagem
        ).convert("RGB")

    return fotografia


# ==========================================
# 6. EXECUTAR A SEGMENTACAO
# ==========================================

def analisar_fotografia(fotografia, modelo):

    imagem = np.array(
        fotografia.convert("RGB")
    )

    altura, largura = imagem.shape[:2]

    # Mascara inicialmente vazia
    mascara_prevista = np.zeros(
        (altura, largura),
        dtype=np.uint8
    )

    # ======================================
    # 6.1. NORMALIZACAO
    # ======================================

    media = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    desvio = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    # ======================================
    # 6.2. PROCESSAMENTO POR RECORTES
    # ======================================

    with torch.inference_mode():

        for y in range(
            0, altura, TAMANHO_RECORTE
        ):

            for x in range(
                0, largura, TAMANHO_RECORTE
            ):

                # Extrair um recorte de ate 512x512
                recorte = imagem[
                    y:y + TAMANHO_RECORTE,
                    x:x + TAMANHO_RECORTE
                ]

                altura_recorte, largura_recorte = (
                    recorte.shape[:2]
                )

                # Completar bordas quando necessario
                recorte = np.pad(
                    recorte,
                    (
                        (
                            0,
                            TAMANHO_RECORTE - altura_recorte
                        ),
                        (
                            0,
                            TAMANHO_RECORTE - largura_recorte
                        ),
                        (0, 0)
                    ),
                    mode="edge"
                )

                recorte = np.ascontiguousarray(
                    recorte
                )

                # Converter imagem em tensor
                tensor = torch.from_numpy(
                    recorte
                )

                tensor = tensor.permute(
                    2, 0, 1
                ).float() / 255.0

                # Mesma normalizacao utilizada no treino
                tensor = (
                    tensor - media
                ) / desvio

                entrada = tensor.unsqueeze(0)

                # Executar o modelo treinado
                saida = modelo(entrada)

                # Selecionar a classe de cada pixel
                previsao = torch.argmax(
                    saida,
                    dim=1
                )[0].cpu().numpy()

                # Remover bordas artificiais
                previsao = previsao[
                    :altura_recorte,
                    :largura_recorte
                ]

                # Colocar a previsao na mascara final
                mascara_prevista[
                    y:y + altura_recorte,
                    x:x + largura_recorte
                ] = previsao

    # ======================================
    # 6.3. CRIAR A MASCARA COLORIDA
    # ======================================

    cores = np.array([
        [64, 64, 64],     # Solo
        [32, 182, 90],    # Planta cultivada
        [240, 68, 68]     # Planta daninha
    ], dtype=np.uint8)

    mascara_colorida = cores[
        mascara_prevista
    ]

    # ======================================
    # 6.4. CRIAR A SOBREPOSICAO
    # ======================================

    sobreposicao = imagem.copy()

    regioes_plantas = mascara_prevista > 0

    sobreposicao[regioes_plantas] = (
        0.55 * imagem[regioes_plantas]
        + 0.45 * mascara_colorida[regioes_plantas]
    ).astype(np.uint8)

    return (
        Image.fromarray(mascara_colorida),
        Image.fromarray(sobreposicao)
    )


# ==========================================
# 7. PREPARAR IMAGEM PARA DOWNLOAD
# ==========================================

def imagem_para_png(imagem):

    memoria = io.BytesIO()

    imagem.save(
        memoria,
        format="PNG"
    )

    return memoria.getvalue()


# ==========================================
# 8. CABECALHO DO AGROSELETIVO
# ==========================================

st.html("""
<div class="agro-header">

    <div class="agro-brand">
        Agro<span>Seletivo</span>
    </div>

    <div class="agro-header-description">
        Análise de imagens agrícolas
    </div>

</div>
""")


# ==========================================
# 9. AREA DE ENVIO
# ==========================================

st.html("""
<div class="agro-title">
    Analisar fotografia
</div>

<div class="agro-description">
    Selecione uma imagem para identificar áreas
    de cultivo e possíveis plantas daninhas.
</div>
""")


# ==========================================
# 9.1. ESTADO INICIAL: SELECIONAR IMAGEM
# ==========================================

foto_atual = st.session_state["foto_atual"]

# O botao de analise comeca desativado
analisar = False

if foto_atual is None:

    # Upload e botao dentro da mesma caixa
    with st.container(border=True):

        coluna_upload, coluna_botao = st.columns(
            [5.8, 1.6],
            gap="small",
            vertical_alignment="center"
        )

        # ==================================
        # SELECIONAR ARQUIVO
        # ==================================

        with coluna_upload:

            
            arquivo = st.file_uploader(
                "Selecionar fotografia",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key=(
                    f"upload_"
                    f"{st.session_state['versao_upload']}"
                )
            )


        # ==================================
        # BOTAO NA MESMA CAIXA
        # ==================================

        with coluna_botao:

            st.button(
                "Analisar imagem",
                type="primary",
                disabled=True,
                use_container_width=True,
                key="analisar_sem_foto"
            )

    # ======================================
    # VALIDAR A FOTOGRAFIA ENVIADA
    # ======================================


    
    if arquivo is not None:

        try:
            dados = arquivo.getvalue()

            with st.spinner("Preparando fotografia..."):
                fotografia_teste = abrir_fotografia(dados)

                identificador = hashlib.sha256(
                    dados
                ).hexdigest()

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError
        ) as erro:

            st.error(
                f"Não foi possível abrir a imagem: {erro}"
            )

        else:
            st.session_state["foto_atual"] = {
                "nome": arquivo.name,
                "dados": dados,
                "identificador": identificador
            }

            st.rerun()


# ==========================================
# 9.2. ESTADO COM FOTOGRAFIA SELECIONADA
# ==========================================

else:

    resultado = st.session_state["resultado_atual"]

    # Conferir se esta fotografia ja foi analisada
    imagem_analisada = (
        resultado is not None
        and resultado["identificador"]
        == foto_atual["identificador"]
    )

    fotografia = abrir_fotografia(
        foto_atual["dados"]
    )

    # Uma unica caixa externa para tudo
    with st.container(border=True):

        coluna_arquivo, coluna_controles = st.columns(
            [5.5, 3.2],
            gap="small",
            vertical_alignment="center"
        )

        # ==================================
        # 9.2.1. MINIATURA E NOME
        # ==================================

        with coluna_arquivo:

            coluna_previa, coluna_nome = st.columns(
                [0.7, 8],
                gap="small",
                vertical_alignment="center"
            )

            # Miniatura pequena
            with coluna_previa:

                st.image(
                    fotografia,
                    width=44
                )

            # Nome logo ao lado da miniatura
            with coluna_nome:

                nome_seguro = html.escape(
                    foto_atual["nome"]
                )

                st.html(
                    f"""
                    <div class="agro-file-name">
                        {nome_seguro}
                    </div>

                    <div class="agro-file-details">
                        {fotografia.width} ×
                        {fotografia.height} pixels
                    </div>
                    """
                )

        # ==================================
        # 9.2.2. CONTROLES A DIREITA
        # ==================================

        with coluna_controles:

            # ------------------------------
            # ANTES DE ANALISAR
            # ------------------------------

            if not imagem_analisada:

                coluna_trocar, coluna_analisar = (
                    st.columns(
                        [1, 1.25],
                        gap="small",
                        vertical_alignment="center"
                    )
                )

                with coluna_trocar:

                    if st.button(
                        "Trocar imagem",
                        key="trocar_antes",
                        use_container_width=True
                    ):

                        limpar_analise()

                with coluna_analisar:

                    analisar = st.button(
                        "Analisar imagem",
                        type="primary",
                        key="analisar_foto",
                        use_container_width=True
                    )

            # ------------------------------
            # DEPOIS DE ANALISAR
            # ------------------------------

            else:

                coluna_status, coluna_trocar = (
                    st.columns(
                        [1.25, 1.2],
                        gap="small",
                        vertical_alignment="center"
                    )
                )

                # Status proximo ao botao
                with coluna_status:

                    st.html(
                        """
                        <div class="agro-status">
                            <span class="agro-status-dot">
                            </span>
                            Análise concluída
                        </div>
                        """
                    )

                # Botao compacto para nova fotografia
                with coluna_trocar:

                    if st.button(
                        "Trocar imagem",
                        key="trocar_depois",
                        use_container_width=True
                    ):

                        limpar_analise()


    # ======================================
    # 10. EXECUTAR A ANALISE
    # ======================================

    if analisar:

        try:

            with st.spinner(
                "Processando fotografia..."
            ):

                # Carregar a U-Net treinada
                modelo = carregar_modelo()

                # Executar a segmentacao
                mascara, sobreposicao = (
                    analisar_fotografia(
                        fotografia,
                        modelo
                    )
                )

        except Exception as erro:

            st.error(
                "Não foi possível concluir "
                f"a análise: {erro}"
            )

        else:

            # Guardar resultados na sessao
            st.session_state["resultado_atual"] = {
                "identificador": foto_atual["identificador"],
                "nome": foto_atual["nome"],
                "mascara": mascara,
                "sobreposicao": sobreposicao
            }

            # Atualizar a interface
            # para mostrar "Analise concluida"
            st.rerun()


    # ======================================
    # 11. MOSTRAR OS RESULTADOS
    # ======================================

    if imagem_analisada:

        st.html("""
        <div class="agro-results-heading">

            <div class="agro-results-title">
                Resultado da análise
            </div>

            <div class="agro-description"
                 style="margin-bottom:0">

                Visualizações produzidas pelo
                modelo a partir da fotografia.

            </div>

        </div>
        """)

        # ==================================
        # 11.1. TRES VISUALIZACOES
        # ==================================

        coluna1, coluna2, coluna3 = st.columns(
            3,
            gap="medium"
        )

        # ------------------------------
        # FOTOGRAFIA ORIGINAL
        # ------------------------------

        with coluna1:

            st.image(
                fotografia,
                use_container_width=True
            )

            st.html("""
            <div class="agro-image-title">
                Original
            </div>

            <div class="agro-image-caption">
                Fotografia enviada
            </div>
            """)

        # ------------------------------
        # MASCARA PREVISTA
        # ------------------------------

        with coluna2:

            st.image(
                resultado["mascara"],
                use_container_width=True
            )

            st.html("""
            <div class="agro-image-title">
                Máscara prevista
            </div>

            <div class="agro-image-caption">
                Classificação por pixel
            </div>
            """)

        # ------------------------------
        # SOBREPOSICAO
        # ------------------------------

        with coluna3:

            st.image(
                resultado["sobreposicao"],
                use_container_width=True
            )

            st.html("""
            <div class="agro-image-title">
                Sobreposição
            </div>

            <div class="agro-image-caption">
                Regiões identificadas
            </div>
            """)


        # ==================================
        # 11.2. LEGENDA DAS CLASSES
        # ==================================

        st.html("""
        <div class="agro-legend">

            <div class="agro-legend-item">

                <span class="agro-legend-dot"
                      style="background:#36c477">
                </span>

                Cultivo

            </div>

            <div class="agro-legend-item">

                <span class="agro-legend-dot"
                      style="background:#ed6868">
                </span>

                Daninhas

            </div>

            <div class="agro-legend-item">

                <span class="agro-legend-dot"
                      style="background:#535954">
                </span>

                Solo

            </div>

        </div>
        """)


        # ==================================
        # 11.3. DOWNLOAD DOS RESULTADOS
        # ==================================

        nome_base = Path(
            resultado["nome"]
        ).stem

        coluna_espaco, coluna_download1, coluna_download2 = (
            st.columns(
                [3, 1.4, 1.7],
                gap="small"
            )
        )

        # ------------------------------
        # BAIXAR MASCARA
        # ------------------------------

        with coluna_download1:

            st.download_button(
                "Baixar máscara",
                data=imagem_para_png(
                    resultado["mascara"]
                ),
                file_name=(
                    f"{nome_base}_mascara.png"
                ),
                mime="image/png",
                use_container_width=True
            )

        # ------------------------------
        # BAIXAR SOBREPOSICAO
        # ------------------------------

        with coluna_download2:

            st.download_button(
                "Baixar sobreposição",
                data=imagem_para_png(
                    resultado["sobreposicao"]
                ),
                file_name=(
                    f"{nome_base}_sobreposicao.png"
                ),
                mime="image/png",
                use_container_width=True
            )


# ==========================================
# 12. RODAPE
# ==========================================

st.html("""
<div class="agro-footer">

    AgroSeletivo · Modelo de segmentação
    U-Net com MobileNetV2, treinado no PhenoBench.

    <div style="margin-top:4px">

        O desempenho em fotografias de
        outras lavouras ainda não foi validado.

    </div>

</div>
""")
