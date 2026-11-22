# AgroSeletivo — Dossiê técnico e guia de estudo para apresentação

**Projeto acadêmico:** prova de conceito de segmentação semântica agrícola  
**Disciplina:** Inteligência Artificial (SIN 323), Sistemas de Informação  
**Data de referência:** 21/11/2026 · **Apresentação prevista:** 04/12/2026  
**Situação:** modelo treinado e avaliado; aplicação Streamlit executada localmente; publicação de alguns artefatos ainda pendente.

> **Como interpretar este arquivo.** O dossiê combina (a) fatos verificáveis nos scripts publicados em `src/`; (b) resultados registrados durante os experimentos em Colab e discutidos no desenvolvimento; e (c) explicações teóricas para estudo. O notebook final, os checkpoints e o app local ainda precisam ser confrontados com a versão efetivamente usada antes de se declarar reprodução integral pelo GitHub.

## 1. Resumo executivo

O **AgroSeletivo** recebe uma fotografia aérea/superior de lavoura e estima uma **classe por pixel**: **0: solo/fundo**, **1: planta cultivada**, **2: planta daninha**. A saída é uma máscara de segmentação e uma imagem com sobreposição colorida. A proposta é demonstrar o uso de visão computacional para distinguir regiões de cultura e plantas daninhas, com possibilidades futuras de apoio ao manejo localizado.

O foco experimental é o **PhenoBench**, que inclui cultivo de beterraba-açucareira e vegetação espontânea. O modelo **não identifica espécie de planta**, **não quantifica produtividade**, **não recomenda herbicida** e **não foi validado em diferentes fazendas, culturas ou dispositivos de captura**. A ferramenta é acadêmica, não um produto agrícola validado.

**Modelo final:** U-Net com encoder MobileNetV2, inicialização ImageNet no treinamento, três canais de saída, normalização ImageNet, previsão por recortes de 512×512 pixels, Cross-Entropy ponderada por classe.  
**Avaliação reservada (722 fotografias do PhenoBench):** IoU de solo **99,09%**; cultura **93,10%**; daninhas **55,32%**; precisão de daninhas **63,09%**; recall de daninhas **81,78%**. São métricas **do conjunto reservado**, não garantia em fotos externas.

## 2. Problema, motivação e escopo

### Problema
O reconhecimento visual de plantas daninhas entre plantas cultivadas exige identificar sua localização e contorno. A cor verde, sozinha, não diferencia confiavelmente uma planta cultivada de uma daninha. Um algoritmo baseado apenas em cor pode detectar vegetação, mas não necessariamente classificá-la em duas categorias botânicas/agronômicas.

### Pergunta do projeto
Uma rede neural de segmentação pode aprender padrões visuais que permitam distinguir solo, planta cultivada e planta daninha em fotografias semelhantes às utilizadas em seu treinamento?

### Entrada e saída
- **Entrada:** imagem RGB; fotos de treino originalmente 1024×1024. No aplicativo, outras resoluções são divididas em blocos sem redimensionar a foto inteira.
- **Saída:** matriz H×W de rótulos inteiros 0/1/2; mapa de cores; sobreposição das áreas de plantas sobre a fotografia; arquivos PNG para download.
- **Unidade da decisão:** pixel, não imagem inteira. Uma foto pode conter simultaneamente as três classes.

### O que a PoC demonstra
Treinamento próprio, ajuste dos dados, inferência real e uma avaliação quantitativa separada da escolha do melhor modelo.

### O que ela NÃO demonstra
Generalização para qualquer lavoura; acurácia validada fora do PhenoBench; identificação de espécies; estimativa de área física sem calibração; decisão agronômica automatizada.

## 3. Linha do tempo e decisões técnicas

1. **Escolha do problema:** segmentar culturas e plantas daninhas em fotos agrícolas; preocupação com relevância prática e com apresentar uma IA efetivamente treinada pela equipe.
2. **Escolha do dataset:** PhenoBench, por disponibilizar fotos RGB e máscaras de referência pixel a pixel.
3. **Organização do projeto:** GitHub com pastas `dados/`, `src/`, `modelos/`, `notebooks/`, `resultados/` e `docs/`; ambiente Python no VS Code.
4. **Inspeção inicial:** conferência das dimensões, correspondência de nomes das fotos/máscaras, valores dos rótulos e visualizações.
5. **Transformação de classes:** 5 rótulos originais agrupados em 3 classes úteis ao objetivo.
6. **Primeiros testes:** um experimento técnico da rede, para confirmar forward/backpropagation; exploração de uma ERFNet oficial em amostra única, sem confundir isso com resultado do modelo final.
7. **Diagnóstico do desbalanceamento:** solo domina a maioria dos pixels; daninhas representam menos de 1%.
8. **Experimentos piloto:** treinos menores (200 imagens de treino, 50 de validação) para verificar viabilidade e ajustar abordagem.
9. **Treinamento completo:** 1.407 fotos de treino; validação de 50 fotos; U-Net MobileNetV2; classe rara com peso maior.
10. **Seleção do checkpoint:** melhor resultado registrado na **época 3**, de um treino planejado de **5 épocas**.
11. **Avaliação reservada:** 722 fotos restantes do conjunto de validação oficial, sem sobreposição de nomes com as 50 utilizadas na seleção do modelo.
12. **Aplicativo local:** Streamlit com upload, predição por recortes, máscara, sobreposição e download; interface refinada.
13. **Estado atual:** documentação e versionamento da aplicação/notebook ainda devem ser fechados antes da apresentação.

> Importante: as 50 e as 722 imagens fazem parte do conjunto de **validação disponibilizado pelo PhenoBench**. A separação interna feita pelo projeto cria uma validação de desenvolvimento e uma avaliação reservada; não constitui um teste completamente independente de local, safra ou condições de captura.

## 4. Dataset PhenoBench — o que entender

**Fonte oficial:** https://www.phenobench.org/dataset.html  
**Artigo:** Weyler et al., *PhenoBench — A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain*, IEEE TPAMI, 2024, DOI 10.1109/TPAMI.2024.3419548.

### Divisões e formato
- Conjunto oficial de treinamento: **1.407** imagens.
- Conjunto oficial de validação: **772** imagens.
- Fotos de trabalho do pipeline: **1024×1024**, RGB.
- Máscaras semânticas: PNG de um único canal com valores de classe (os arquivos originais podem estar em 16 bits, por exemplo modo `I;16` na Pillow).
- Arquivos de fotografia e máscara devem corresponder **pelo nome** e **pelas dimensões**.

### Classes originais e conversão aplicada

| Rótulo de origem | Significado operacional | Classe final |
|---|---|---|
| 0 | fundo/solo | 0 — solo |
| 1 | cultura | 1 — cultura |
| 2 | daninha | 2 — daninha |
| 3 | cultura parcialmente visível | 1 — cultura |
| 4 | daninha parcialmente visível | 2 — daninha |

O agrupamento `3→1` e `4→2` foi aplicado porque a questão do projeto é **o tipo de região** (cultura/daninha), e não o grau de visibilidade. A máscara final tem somente 0, 1 e 2. Não se deve converter para imagem RGB antes de ler os valores numéricos dos rótulos: a cor é apenas representação visual.

### Distribuição de pixels registrada no treino
- Solo/fundo: **87,657%**
- Planta cultivada: **11,846%**
- Planta daninha: **0,497%**

Isso é **desbalanceamento de classes**. Um modelo que escolha solo quase sempre pode parecer bom em métricas globais e falhar justamente na daninha, que é o alvo relevante. Por isso o projeto dá atenção especial à IoU, precisão e recall **por classe**.

## 5. Pipeline dos dados — passo a passo

1. **Descoberta e conferência:** listar arquivos de `dados/phenobench/train/imagens`, `mascaras`, e equivalentes de `val`; confirmar pares válidos.
2. **Leitura:** Pillow abre imagem como RGB; NumPy lê máscara como matriz de inteiros.
3. **Validação:** confirmar dimensão 1024×1024, máscara 2D, presença apenas de rótulos esperados e correspondência entre nomes.
4. **Remapeamento:** classes 3 e 4 se transformam nas classes 1 e 2.
5. **Recortes:** no treinamento, o script local `PhenoBenchDataset` seleciona **um recorte aleatório de 512×512 por foto e acesso**; na validação, usa **quatro quadrantes fixos**, preservando cobertura de 1024×1024.
6. **Conversão:** foto H×W×3 → tensor 3×H×W, float, valores base de 0 a 1; máscara H×W → tensor inteiro `int64`.
7. **Normalização:** cada canal de imagem usa `(valor - média) / desvio` com média `[0.485,0.456,0.406]` e desvio `[0.229,0.224,0.225]`, conforme convenção ImageNet.
8. **Lote:** `DataLoader` combina exemplos para alimentar a rede.

**Pergunta recorrente:** por que 512×512? Reduz memória e custo por passo, mantém detalhe de plantas pequenas e oferece recortes compatíveis com o modelo. **Consequência:** a IA vê contexto limitado; bordas de recortes podem gerar diferenças nas previsões.

**Atenção de reprodutibilidade:** o script publicado em `src/dataset_agroseletivo.py` descreve a preparação local. O notebook final do Colab ainda precisa ser versionado e conferido para garantir que o procedimento completo de treinamento corresponde às configurações registradas.

## 6. Modelo de IA — arquitetura e conceitos

### Por que segmentação semântica?
Classificação comum: atribui um rótulo à imagem inteira. Detecção: cria caixas ao redor de objetos. **Segmentação semântica:** atribui classe a cada pixel, permitindo mostrar a extensão das regiões de solo, cultura e daninha. Ela **não** distingue instâncias individuais da mesma classe.

### U-Net e MobileNetV2
A **U-Net** usa caminho de contração (**encoder**) para extrair características em diferentes escalas e caminho de expansão (**decoder**) para construir um mapa espacial de classes. Conexões de salto (**skip connections**) recuperam detalhes úteis à localização precisa. **MobileNetV2** é o backbone do encoder, escolhido por ser relativamente leve.

### Transfer learning
Na criação para **treinamento**, o encoder foi inicializado com pesos **ImageNet** (`encoder_weights="imagenet"`). Isso não significa que o modelo final esteja pronto sem treino: seus parâmetros foram ajustados com dados do PhenoBench. Ao **recarregar** um checkpoint treinado, o app recria a arquitetura com `encoder_weights=None` e aplica os pesos guardados em `state_dict`; não precisa baixar novamente pesos ImageNet.

### Formatos principais
- Um recorte normalizado tem forma `[3,512,512]` (canais, altura, largura).
- Lote com duas fotos: `[2,3,512,512]`.
- Saída bruta: `[2,3,512,512]`: três **logits** por pixel, um por classe.
- `argmax(dim=1)` escolhe a classe de maior valor para cada pixel.
- Máscara prevista resultante: `[2,512,512]`.

**Logit não é probabilidade:** `activation=None` fornece pontuações brutas; `CrossEntropyLoss` trabalha diretamente com elas. `softmax` pode transformá-las em probabilidades, mas não é obrigatório para `argmax`.

## 7. Treinamento — configurações registradas

| Elemento | Configuração registrada |
|---|---|
| Arquitetura | `smp.Unet` com `mobilenet_v2` |
| Encoder no início do treino | pré-treinado em ImageNet |
| Classes | 3 |
| Treino completo | 1.407 fotografias |
| Validação de desenvolvimento | 50 fotografias |
| Recorte | 512 × 512 |
| Lote (`batch_size`) | 2 |
| Otimizador | AdamW |
| Taxa de aprendizado | 0,0001 |
| Função de perda | Cross-Entropy ponderada |
| Pesos registrados para classes [solo,cultura,daninhas] | [1,00; 2,72; 13,28] |
| Épocas planejadas | 5 |
| Checkpoint selecionado | época 3 |
| Ambiente | Google Colab com GPU Tesla T4 |
| Arquivo do melhor modelo | `unet_completo_ponderado_melhor.pt` |

**Checar no notebook original:** logs, sementes, curvas, cálculos dos pesos, execução exata e salvamento do checkpoint. A tabela é um **registro do experimento**, não uma transcrição auditada do notebook publicado (ele não consta no GitHub na data deste documento).

### Como um passo de treinamento funciona
1. `modelo.train()` ativa comportamento de treino.
2. O `DataLoader` fornece foto e máscara.
3. O modelo gera três logits por pixel.
4. A função de perda compara logits com os rótulos reais e penaliza erros; o peso maior para daninhas busca compensar raridade.
5. `optimizer.zero_grad()` limpa gradientes anteriores.
6. `loss.backward()` calcula gradientes por retropropagação.
7. `optimizer.step()` ajusta os pesos.
8. Ao fim da época, a validação estima desempenho para selecionar o checkpoint sem usar as 722 imagens reservadas.

**Diferença essencial:** perda baixa mede aderência ao objetivo definido, mas não garante boa IoU da classe rara. Por isso monitoramos métricas por classe, não apenas a perda.

## 8. Evolução e resultados quantitativos

### 8.1. Experimentos anteriores
- **Teste técnico do modelo local:** verificou formato, passagem adiante e atualização de parâmetros. Não demonstra capacidade de segmentar plantas corretamente.
- **ERFNet oficial:** avaliação exploratória de **uma única imagem**, identificada como `05-15_00028_P0030852.png`; IoU de daninhas **60,62%**, precisão **61,52%**, recall **97,65%**. É uma referência exploratória, **não** comparação experimental justa com os resultados de centenas de fotos do modelo final.
- **Piloto próprio:** 200 imagens de treinamento e 50 de validação, usado para identificar limitações e orientar treinamento final.

### 8.2. Melhor checkpoint — 50 fotos de validação

| Classe / medida | Resultado |
|---|---:|
| IoU solo | 99,12% |
| IoU cultura | 92,81% |
| IoU daninhas | 56,12% |
| Precisão daninhas | 63,84% |
| Recall daninhas | 82,28% |

Melhor época registrada: **3**. Esses números ajudaram na seleção do modelo e, portanto, **não devem ser apresentados como avaliação final independente**.

### 8.3. Avaliação reservada — 722 fotos, 2.888 recortes

| Classe / medida | Resultado |
|---|---:|
| IoU solo | **99,09%** |
| IoU cultura | **93,10%** |
| IoU daninhas | **55,32%** |
| Precisão daninhas | **63,09%** |
| Recall daninhas | **81,78%** |
| Perda média registrada | 0,0292 |

**Interpretação:** o desempenho se manteve próximo ao observado nas 50 fotos, o que é um sinal positivo **dentro do domínio do PhenoBench**. A classe daninha continua significativamente mais difícil; 55,32% de IoU é desempenho parcial, não detecção perfeita. O recall relativamente alto indica que o modelo encontra muitos pixels reais de daninhas, mas a precisão mais baixa significa que há falsos positivos.

**Diferença metodológica:** 722 fotos reservadas são independentes das 50 por **nome de arquivo**, mas não necessariamente por fazenda, dia de coleta ou contexto de campo. Não chamar essa avaliação de teste de generalização externa.

## 9. Métricas — compreender e explicar com números

Para a classe 'daninha', tratada contra o restante:
- **VP (verdadeiro positivo):** pixel real de daninha previsto como daninha.
- **FP (falso positivo):** solo ou cultura previsto incorretamente como daninha.
- **FN (falso negativo):** daninha real prevista como outra classe.
- **Precisão:** VP / (VP + FP). Pergunta: 'dos pixels marcados como daninha, quantos eram realmente daninha?'
- **Recall / sensibilidade:** VP / (VP + FN). Pergunta: 'dos pixels que eram daninha, quantos a rede encontrou?'
- **IoU:** VP / (VP + FP + FN). Mede interseção sobre união, penalizando tanto confusão positiva quanto omissões.

**Exemplo didático (hipotético, não resultado do projeto):** se VP = 60, FP = 30 e FN = 10, a precisão é 60/90 = 66,7%; recall é 60/70 = 85,7%; IoU é 60/100 = 60%.

**Por que não avaliar só pela acurácia global?** Como quase 88% dos pixels são fundo, acertar muito solo pode mascarar falhas com daninhas. **Por que IoU de daninhas é menor que de cultura?** Essa classe é rara, diversa em aparência, pode ser pequena e se confundir visualmente com o cultivo.

## 10. Aplicativo Streamlit — fluxo real

### Arquivos e dependências
O aplicativo local utiliza `app.py`, `streamlit`, `torch`, `segmentation_models_pytorch`, `numpy` e `Pillow`, com checkpoint local em `modelos/unet_completo_ponderado_melhor.pt`.

### Estados da interface
1. **Sem fotografia:** upload disponível; botão de análise desabilitado.
2. **Foto selecionada:** nome e miniatura; opção de trocar; botão ativo.
3. **Processamento:** modelo carregado com `@st.cache_resource`; progresso indicado por spinner; roda em CPU no computador local.
4. **Concluído:** botão de analisar desaparece; status 'Análise concluída'; opção 'Trocar imagem'; três visões: original, máscara, sobreposição.
5. **Download:** máscara e sobreposição em PNG; podem ser salvas sem novo treinamento.

### Por que o app usa recortes?
Aceita fotos de resoluções diversas. Para cada posição (x,y), extrai trecho de até 512×512; **preenche bordas** com `np.pad(mode="edge")` quando um trecho é menor; normaliza e envia à U-Net; remonta a previsão na posição original. A foto não é encolhida para 512×512. Podem surgir **artefatos nas costuras** entre blocos.

### Como cores e sobreposição funcionam?
- Solo 0 → cinza `[64,64,64]`
- Cultura 1 → verde `[32,182,90]`
- Daninhas 2 → vermelho `[240,68,68]`
- Sobreposição nas plantas: aproximadamente **55%** foto RGB + **45%** cor da classe; solo preserva aparência da foto original.

### Segurança de estado
O app identifica fotos por hash **SHA-256** dos bytes para evitar mostrar o resultado antigo de um arquivo diferente com mesmo nome. Usa `st.session_state` para manter foto/resultados entre reruns e `st.rerun()` para atualizar controles. Limite da aplicação registrado: **12 milhões de pixels**, não necessariamente o limite total de upload do servidor Streamlit.

### Testes concluídos e limites
- Rodou localmente em Streamlit; aplicação exibiu foto original, máscara e sobreposição reais do modelo.
- Uma foto de 1024×1024 do conjunto PhenoBench foi testada.
- Foto sintética externa de 1254×1254 foi usada para verificar aceitação e remontagem; a classificação nessa foto **não** é evidência de acurácia, pois não foi avaliada com uma máscara real confiável.
- Ainda é necessário demonstrar robustez e medir resultados com dados externos de referência se desejarmos afirmar generalização.

## 11. Inventário dos arquivos — GitHub versus ambiente local

### Arquivos já publicados em `src/`
| Arquivo | Função no projeto |
|---|---|
| `verificar_amostra.py` | inspecionar um par inicial foto/máscara |
| `visualizar_amostra.py` | construir painel visual de imagem e máscara real |
| `verificar_dataset.py` | verificar quantidade, nomes, tamanhos, classes |
| `testar_rotulos.py` | confirmar agrupamento 3→1 e 4→2 |
| `dataset_agroseletivo.py` | Dataset PyTorch: leitura, recortes, normalização e tensores |
| `analisar_distribuicao.py` | contar pixels por classe e presença por foto |
| `analisar_recortes.py` | amostrar recortes e verificar presença de daninhas |
| `modelo_agroseletivo.py` | criar U-Net MobileNetV2 e testar passagem adiante |
| `testar_treinamento.py` | executar uma atualização dos pesos no CPU |

### Itens locais que devem ser versionados após verificação
- `app.py` — **não constava no GitHub durante a revisão**; copiar a versão final, já testada, do VS Code.
- `notebooks/AgroSeletivo_Treinamento.ipynb` — **não constava no GitHub**; exportar do Colab, revisar dados privados/outputs pesados e adicionar.
- `requirements.txt` — lista de pacotes necessários, criada com esta atualização documental; instalar em ambiente próprio.
- Possíveis exemplos pequenos de resultado, após verificar direitos/licença e tamanho.

### Arquivos que NÃO devem ser enviados ao GitHub público
- `dados/phenobench/` (dataset original volumoso e sujeito à licença/atribuição).
- `.venv/`, senhas, tokens, `.env`.
- `modelos/*.pt` e checkpoints grandes; a configuração `.gitignore` atual os ignora. Se precisarem disponibilizar pesos, definir método de distribuição separado e documentá-lo.

## 12. Executar e explicar o ambiente

### No VS Code (Windows)
1. Abrir `C:\Projetos\AgroSeletivo`.
2. Ativar a `.venv` já criada (PowerShell: `.\.venv\Scripts\Activate.ps1`, de acordo com permissões locais).
3. Instalar dependências em ambiente novo: `python -m pip install -r requirements.txt`. **Versões do notebook ainda precisam ser registradas**; não tratar a lista de pacotes sem versões fixas como reprodutibilidade perfeita.
4. Garantir `modelos/unet_completo_ponderado_melhor.pt` no caminho esperado.
5. Executar `python -m streamlit run app.py` e abrir endereço localhost exibido no terminal (geralmente 8501; pode ser 8502 se a porta estiver ocupada).

### No Google Colab
- Conectar GPU, instalar bibliotecas e montar Drive quando necessário.
- Verificar `torch.cuda.is_available()`.
- Preparar arquivos do dataset com caminhos corretos.
- Treinar, validar, salvar o melhor checkpoint e baixá-lo para a pasta `modelos/` local.
- O treinamento **não é refeito toda vez que o app roda**. O app só carrega os pesos finais e faz inferência.

### Reprodutibilidade
Para reprodução científica, registrar **versões exatas das dependências**, versão do dataset, sementes, seleção das 50 fotos de validação, configurações finais, GPU, logs e hash do checkpoint. Há elementos registrados em conversas e no notebook, mas nem todos estão publicados.

## 13. Dificuldades e como a equipe tomou decisões

**Dados desbalanceados:** daninhas são aproximadamente 0,5% dos pixels → loss ponderada e métricas por classe. **Recursos computacionais:** CPU local serve a pequenos testes, enquanto GPU Colab permite treinamento. **Diferentes resoluções:** uso de recortes na inferência, em vez de redimensionar a foto toda. **Pré-treinamento:** encoder ImageNet acelera aprendizado; treino no PhenoBench especializa o modelo. **Avaliação:** separar 50 fotos usadas na escolha do melhor checkpoint de 722 reservadas para mensuração adicional. **Qualidade externa:** teste visual com imagem sintética mostra limitações; não é prova de acerto.

## 14. Riscos, limitações e honestidade científica

1. **Domínio:** dados oficiais representam tipos específicos de cultivo/captura; outras regiões e culturas podem diferir.
2. **Métrica por classe:** IoU de daninhas 55,32% indica acertos e erros relevantes.
3. **Falsos positivos:** precisão 63,09% significa que alguns pixels marcados como daninhas não são daninhas.
4. **Falsos negativos:** recall 81,78% significa que uma parcela de pixels reais de daninhas não foi identificada.
5. **Costuras de recortes:** podem interromper objetos próximos à borda.
6. **Validação reservada interna:** imagens de mesmo banco podem ter correlações; não há garantia de independência de dia/local.
7. **Dataset e pesos não publicados diretamente:** terceiros precisam obtê-los conforme a licença e configurar ambiente.
8. **Uma cor na máscara não equivale a área física de campo:** conversão em m² exigiria escala e calibração.
9. **Rótulos:** as classes 3 e 4 foram agrupadas, reduzindo distinção de visibilidade parcial.
10. **Reprodutibilidade incompleta:** notebook e arquivo final do app precisam ser versionados.

**Frase segura para apresentação:** “Treinamos e avaliamos uma U-Net para segmentar três classes em imagens do PhenoBench. Conseguimos 55,32% de IoU para daninhas na avaliação reservada do mesmo dataset, mas ainda não validamos o uso em outras lavouras.”

## 15. Respostas para perguntas prováveis da banca

1. **O que a IA faz?** Classifica cada pixel como solo, cultura ou daninha.
2. **Por que isso é IA e não regra por cor?** Aprende relações de textura, forma e contexto a partir de exemplos rotulados; a cor sozinha não distingue necessariamente cultura e daninha.
3. **A rede foi treinada por vocês?** Sim, ajustamos pesos com imagens rotuladas do PhenoBench; o encoder partiu de pesos ImageNet, o que deve ser mencionado.
4. **A arquitetura foi inventada pela equipe?** Não; usamos uma U-Net conhecida com MobileNetV2. A contribuição é adaptar, treinar, avaliar e integrar.
5. **O que são máscaras reais?** Rótulos feitos/disponibilizados pelo dataset, usados como referência para treinamento e avaliação.
6. **Por que três classes se o dataset tem cinco?** Fundimos classes parcialmente visíveis ao tipo principal de planta.
7. **Por que 512×512?** Equilíbrio entre memória, detalhes e processamento.
8. **Por que normalizar?** Adequar distribuição de entradas à esperada pelo encoder e facilitar a otimização.
9. **O que são logits?** Pontuações por classe antes de `softmax`; `argmax` escolhe a classe prevista.
10. **Como a rede aprende?** Previsão → perda → gradientes por backpropagation → atualização AdamW.
11. **O que é um epoch?** Uma passagem pelo conjunto de treinamento de acordo com a política de amostragem do loader.
12. **Por que Cross-Entropy ponderada?** Para tornar mais custosos os erros em classes menos representadas.
13. **Por que peso 13,28 para daninhas?** Para compensar sua raridade relativa; confirmar cálculo/critério no notebook.
14. **Por que IoU e não apenas acurácia?** IoU mede qualidade espacial por classe, evitando que abundância de solo esconda erros.
15. **O que significa IoU 55,32%?** Interseção entre pixels previstos e reais de daninhas representa 55,32% da união; não é “55% das plantas identificadas”.
16. **O que significa recall 81,78%?** O modelo encontrou cerca de 81,78% dos pixels de daninhas anotados no conjunto avaliado.
17. **E precisão 63,09%?** Cerca de 63,09% dos pixels que o modelo marcou como daninhas eram daninhas na referência.
18. **O que é um falso positivo?** Solo/cultura marcado como daninha.
19. **O que é um falso negativo?** Daninha real que o modelo marcou como solo/cultura.
20. **Por que a época 3?** Foi o checkpoint com melhor critério de validação registrado; conferir critério exato no notebook.
21. **Por que 50 + 722?** 50 para desenvolvimento/seleção; 722 reservadas para avaliação posterior, somando 772 fotos do conjunto oficial.
22. **Essas 722 são independentes?** São disjuntas por nome, mas pertencem ao mesmo dataset; não são validação externa.
23. **Por que o teste em outra foto pode falhar?** Mudanças de cultura, iluminação, solo, escala e distribuição dos dados.
24. **Qual a diferença entre treino e inferência?** Treino atualiza pesos; inferência usa pesos fixos para prever.
25. **Por que o app roda na CPU?** Os pesos já estão treinados; a CPU consegue fazer previsões, embora mais lentamente.
26. **O que `torch.inference_mode()` faz?** Evita rastreamento de gradientes na inferência, economizando recursos.
27. **Por que `encoder_weights=None` no app?** Porque o checkpoint completo já contém os pesos ajustados.
28. **O que `st.session_state` faz?** Mantém resultado e imagem entre execuções do script na interface.
29. **Como lidamos com imagem maior que 512?** Dividimos em blocos, preenchendo bordas e recompondo máscara.
30. **Próximo passo científico?** Avaliar em imagens externas rotuladas, melhorar classe daninha e comparação com baseline com protocolo controlado.

## 16. Roteiro sugerido para a apresentação (cerca de 10 minutos)

| Tempo | Tópico | Mensagem central |
|---|---|---|
| 0:00–1:00 | Problema real e objetivo | distinguir áreas cultivadas, daninhas e solo |
| 1:00–2:00 | Dataset e rótulos | imagens/máscaras PhenoBench; 5 → 3 classes |
| 2:00–3:15 | Preparação e desbalanceamento | recortes, normalização, classe rara |
| 3:15–4:45 | Modelo e treinamento | U-Net/MobileNetV2, transfer learning, loss ponderada |
| 4:45–6:15 | Métricas e resultados | separar 50 de validação e 722 de avaliação |
| 6:15–8:15 | Demonstração do Streamlit | subir foto, executar, explicar 3 resultados |
| 8:15–9:15 | Limitações | IoU parcial de daninhas, sem validação externa |
| 9:15–10:00 | Conclusão e próximos passos | prova de conceito funcional, caminhos para evoluir |

**Demonstração:** usar foto do PhenoBench já testada, abrir o app antes, ter checkpoint acessível, baixar um PNG, e manter capturas dos resultados como plano B em caso de falha de ambiente.

## 17. Plano de estudo e divisão de responsabilidades

**Bloco A — Dados:** explicar fonte, fotos, máscaras, cinco classes originais, mapeamento para três, verificação e recortes.  
**Bloco B — Treinamento:** explicar U-Net/MobileNetV2, encoder/decoder, ImageNet, logits, loss, AdamW, época e checkpoint.  
**Bloco C — Avaliação:** decorar a diferença entre 50 e 722; explicar fórmulas IoU, precisão, recall, acertos/erros da classe daninha e limitações.  
**Bloco D — App e demonstração:** explicar upload, hash, modelo carregado, tiling, máscara/sobreposição, downloads e limites.

### Rotina recomendada até 04/12/2026
- **1ª passagem:** cada integrante entende o fluxo completo de ponta a ponta.
- **2ª passagem:** cada pessoa explica sua parte sem ler, usando exemplos e valores.
- **3ª passagem:** trocar papéis e simular as 30 perguntas acima.
- **Ensaio técnico:** executar app com duas imagens; confirmar checkpoint, portas, conexão e plano B.
- **Ensaio final:** cronometrar o roteiro e reduzir slides redundantes.

## 18. Checklist antes de apresentar

- [ ] O `app.py` local final e testado está versionado no GitHub.
- [ ] O `AgroSeletivo_Treinamento.ipynb` foi exportado/revisado e versionado.
- [ ] O notebook contém logs exatos e permite conferir os resultados da época 3.
- [ ] Os 50 nomes de desenvolvimento e os 722 de avaliação estão documentados/reproduzíveis.
- [ ] Dependências com versões exatas foram registradas para reprodução.
- [ ] O checkpoint local foi testado com duas fotografias.
- [ ] Os downloads de máscara e sobreposição foram testados.
- [ ] Há uma explicação clara para 55,32% IoU e a diferença entre precisão e recall.
- [ ] Ninguém afirma generalização a qualquer cultura, fazenda ou condição.
- [ ] Os integrantes conseguem explicar o que cada arquivo principal faz.
- [ ] Referência e licença do PhenoBench são atribuídas.
- [ ] Há imagens de resultado salvas para uma demonstração de contingência.
- [ ] Os slides apresentam limites honestos e melhorias futuras.
- [ ] A equipe ensaiou perguntas e respostas com cronômetro.

## 19. Melhorias futuras (não implementadas)

- Comparação justa com um baseline simples e/ou ERFNet no **mesmo conjunto** de imagens.
- Aumento de dados/estratégias amostrais mais focadas em daninhas pequenas.
- Recortes sobrepostos com fusão de previsões para reduzir costuras.
- Avaliação externa com máscaras de referência de novas áreas e condições.
- Monitoramento de tempo de inferência por resolução.
- Avaliação de desempenho por tamanho de daninha, iluminação e estágio de crescimento.
- Registro completo de configuração e versões para treinamento reproduzível.
- Opcionalmente um relatório numérico por imagem, sempre evitando confundir proporção de pixels com área de campo.

## 20. Fontes e rastreabilidade

**Evidências do próprio projeto:** repositório `https://github.com/IzabelleImamura/AgroSeletivo`, scripts `src/`, resultados do treinamento/avaliação registrados durante o desenvolvimento, aplicativo local e notebook Colab ainda a versionar.

**Fontes técnicas oficiais:**
- PhenoBench, dataset, formato, números e licença: https://www.phenobench.org/dataset.html
- PhenoBench, página principal: https://www.phenobench.org/
- Weyler et al. (2024), DOI: https://doi.org/10.1109/TPAMI.2024.3419548
- Segmentation Models PyTorch, U-Net/encoders: https://segmentation-models-pytorch.readthedocs.io/en/latest/models.html
- Documentação PyTorch: https://pytorch.org/docs/stable/
- Documentação Streamlit: https://docs.streamlit.io/

**Ponto de atenção:** este material não substitui uma conferência final das saídas e dos hiperparâmetros dentro do notebook efetivo. Onde se lê “registrado”, trata-se de informação recuperada do histórico dos experimentos, não de um artefato Colab auditado no GitHub.


## 21. Auditoria do notebook recebido em 21/11/2026

Foi inspecionado o arquivo original `AgroSeletivo_Treinamento.ipynb` baixado do Google Colab (cerca de 1,9 MB), **antes de sua publicação no GitHub**. Ele possui **41 células** (20 Markdown e 21 de código), com resultados de execução armazenados. A inspeção confirma:

- Ambiente registrado: GPU **Tesla T4**, PyTorch **2.14.1+cu130** e `segmentation_models_pytorch` **0.6.0** nas saídas do notebook.
- Experimento piloto: 200 fotografias de treino, 50 de validação, primeira época de Cross-Entropy sem pesos com IoU de daninhas **0,0115**.
- Continuação piloto sem pesos: IoU de daninhas na época 5 de **0,0290**.
- Continuação piloto ponderada: IoU de daninhas na época 5 de **0,4150**. **Não comparar isso com o conjunto completo como experimento controlado**, pois os dados diferem.
- Treinamento completo: 1.407 fotografias, 50 de validação de desenvolvimento, 5 épocas, lote 2, Cross-Entropy ponderada com pesos **[1,00; 2,7202413; 13,2805204]** para solo/cultura/daninhas. `historico_completo_ponderado.json` foi escrito no Google Drive.
- Melhor checkpoint escolhido pela maior **IoU de daninhas**, não pela menor perda global: **época 3**, IoU **0,5612** no conjunto de 50 fotografias. A época 5 apresentou IoU **0,5572**, abaixo da época 3, embora a perda de validação tenha sido menor.
- Avaliação reservada: 722 fotografias / 2.888 recortes, resultado de **0,5532** de IoU de daninhas, **0,6309** de precisão e **0,8178** de recall, confirmados pelas saídas e pela matriz de confusão.
- O notebook registra verificação posterior de **zero nomes repetidos** entre as 50 fotografias de validação e as 722 da avaliação.
- **Ponto de correção:** a célula de preparação do conjunto reservado (etapa 8.1) contém uma saída de `FileNotFoundError` ao tentar reabrir o ZIP `AgroSeletivo_dados_piloto.zip` no Drive. A extração das 722 fotos já tinha ocorrido e uma célula independente posterior confirmou a separação dos grupos. A falha não invalida, por si só, as métricas posteriores, mas deve ser documentada ou corrigida em versão de publicação.
- **Ponto de reprodutibilidade:** a célula que carrega o conjunto completo aparece visualmente antes das células de experimentos piloto, porém seu número de execução é posterior. `Executar tudo` em sequência linear pode assim alterar o experimento piloto. Recomenda-se separar `notebooks/piloto.ipynb` e `notebooks/treinamento_completo.ipynb`, ou reorganizar os blocos e documentar dependências antes de afirmar reprodução integral.
- A saída da última célula de análise visual está oculta no arquivo exportado; as outras células retêm imagens, logs e gráficos.
- Uma inspeção automatizada de padrões comuns de tokens, chaves e endereços de e-mail não encontrou correspondências; isso **não é uma auditoria integral de privacidade** e o conteúdo deve ser revisado manualmente antes da publicação.

**Próxima ação:** versionar o notebook original/revisado separadamente do modelo `.pt` e dos dados volumosos. Preservar uma cópia intacta do original fora do GitHub. Os resultados registrados no notebook se referem ao ambiente Colab da execução; um reprocessamento atual pode variar conforme versões e ordem de células.
