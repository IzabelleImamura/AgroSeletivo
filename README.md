# AgroSeletivo

**Prova de conceito acadêmica de inteligência artificial para segmentação semântica de lavouras.** O objetivo é classificar cada pixel de uma fotografia RGB em **solo/fundo**, **planta cultivada** ou **planta daninha**, e exibir máscara colorida e sobreposição.

> **Situação (21/11/2026):** scripts de preparação e testes publicados neste repositório; modelo final treinado e aplicativo Streamlit executado localmente. O notebook completo e a versão final do `app.py` **ainda precisam ser enviados do ambiente local**. O checkpoint final permanece fora do GitHub. Portanto, este repositório, isoladamente, ainda não é reprodução completa do sistema final.

## Resultados registrados

| Métrica | 50 imagens de validação de desenvolvimento | 722 imagens reservadas para avaliação |
|---|---:|---:|
| IoU solo/fundo | 99,12% | 99,09% |
| IoU planta cultivada | 92,81% | 93,10% |
| IoU planta daninha | 56,12% | **55,32%** |
| Precisão daninha | 63,84% | 63,09% |
| Recall daninha | 82,28% | 81,78% |

As 722 imagens são disjuntas por nome das 50 de desenvolvimento, mas provêm da mesma base **PhenoBench**. Os resultados **não** demonstram generalização para outros campos, safras ou espécies.

## Dados e método

- Dataset: [PhenoBench](https://www.phenobench.org/dataset.html), 1.407 imagens de treinamento e 772 de validação, imagens 1024×1024.
- Rótulos originais: 0 fundo, 1 cultura, 2 daninha, 3 cultura parcial, 4 daninha parcial. O projeto reagrupa 3→1 e 4→2.
- Recortes usados para o modelo: 512×512 pixels; fotos normalizadas com médias/desvios do ImageNet.
- Arquitetura: **U-Net**, encoder **MobileNetV2**, três classes.
- Aprendizado: modelo iniciado com encoder pré-treinado no ImageNet, depois ajustado aos rótulos agrícolas; uso de Cross-Entropy ponderada para lidar com o desbalanceamento.
- Exibição: cinza (solo), verde (cultura) e vermelho (daninha). Na aplicação local, recortes são reunidos na resolução original e podem ser salvos em PNG.

## Estrutura atual do repositório

```text
AgroSeletivo/
├── README.md
├── .gitignore
├── requirements.txt               # dependências mínimas, sem versões fixadas
├── dados/                         # mantidos localmente, não publicados
├── docs/
│   └── GUIA_ESTUDO_APRESENTACAO.md
├── modelos/                       # checkpoint local, não publicado
├── notebooks/                     # notebook Colab ainda pendente
├── resultados/
└── src/
    ├── verificar_amostra.py
    ├── visualizar_amostra.py
    ├── verificar_dataset.py
    ├── testar_rotulos.py
    ├── dataset_agroseletivo.py
    ├── analisar_distribuicao.py
    ├── analisar_recortes.py
    ├── modelo_agroseletivo.py
    └── testar_treinamento.py
```

## Executar os testes locais

Os scripts em `src/` pressupõem que o dataset esteja disponível na estrutura de pastas esperada por eles (veja `dataset_agroseletivo.py`). Em Windows, com Python instalado:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/verificar_dataset.py
python src/dataset_agroseletivo.py
python src/modelo_agroseletivo.py
```

Para rodar **a aplicação final**, depois de trazer o `app.py` validado para a raiz e colocar o checkpoint em `modelos/unet_completo_ponderado_melhor.pt`:

```powershell
python -m streamlit run app.py
```

A aplicação carrega o checkpoint no CPU e faz inferência, **não retreina** o modelo. A documentação mais extensa explica seleção, treino no Colab e métricas.

## Guia completo para entender e apresentar

**[Dossiê técnico, passo a passo, métricas, perguntas da banca e checklist](docs/GUIA_ESTUDO_APRESENTACAO.md).**

## Próximos itens para fechar o projeto

- [ ] Copiar a versão final e **testada** de `app.py` do VS Code para o repositório.
- [ ] Exportar o notebook de treinamento do Colab para `notebooks/` após revisão dos outputs e caminhos.
- [ ] Registrar versões exatas dos pacotes; o `requirements.txt` é uma lista inicial, não um ambiente totalmente fixado.
- [ ] Guardar configuração, sementes, logs e lista de nomes das 50 e das 722 imagens para auditoria da avaliação.
- [ ] Testar a reprodução da inferência em instalação limpa, com checkpoint obtido por mecanismo próprio.

## Licença e referência dos dados

O PhenoBench é de autoria de Weyler e colaboradores (IEEE TPAMI, 2024); referência: [DOI 10.1109/TPAMI.2024.3419548](https://doi.org/10.1109/TPAMI.2024.3419548). O [site oficial](https://www.phenobench.org/dataset.html) declara **CC BY-SA 4.0**; é prudente também conferir o arquivo de licença na cópia utilizada antes de redistribuir imagens ou máscaras. O dataset original e os checkpoints não estão incluídos neste repositório.

**Uso educacional:** os resultados são experimentais, não um sistema validado para decisões agronômicas.
