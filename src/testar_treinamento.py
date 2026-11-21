
import random

import numpy as np
import torch

from dataset_agroseletivo import PhenoBenchDataset
from modelo_agroseletivo import criar_modelo


# Reproduzir a escolha dos dados do teste
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("AGROSELETIVO - TESTE DE APRENDIZADO")

# Carregar um exemplo real do treinamento
dataset = PhenoBenchDataset("train")

imagem, mascara = dataset[0]

# Usar um recorte menor apenas neste teste
# para reduzir o processamento na CPU
imagem = imagem[:, 128:384, 128:384].unsqueeze(0)
mascara = mascara[128:384, 128:384].unsqueeze(0)

print("\nDADOS")
print("Imagem:", imagem.shape)
print("Mascara:", mascara.shape)
print("Classes:", torch.unique(mascara).tolist())

# Criar a rede neural
modelo = criar_modelo()
modelo.train()

# Definir a funcao que mede o erro
criterio = torch.nn.CrossEntropyLoss()

# Definir como os pesos serao atualizados
otimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=0.0001
)

# Guardar os pesos de uma camada antes do treino
parametro = next(modelo.parameters())
pesos_antes = parametro.detach().clone()

# Limpar gradientes anteriores
otimizador.zero_grad()

# Fazer uma previsao
saida = modelo(imagem)

# Comparar previsao com a mascara real
perda = criterio(saida, mascara)

print("\nANTES DA ATUALIZACAO")
print("Perda:", perda.item())

# Verificar o valor da perda
if not torch.isfinite(perda).item():
    raise ValueError("A perda apresentou valor invalido.")

# Calcular gradientes
perda.backward()

# Atualizar os pesos
otimizador.step()

# Verificar se os pesos realmente mudaram
pesos_depois = parametro.detach()

mudanca = (
    pesos_depois - pesos_antes
).abs().max().item()

print("\nDEPOIS DA ATUALIZACAO")
print("Maior alteracao no parametro:", mudanca)
print("Pesos atualizados:", mudanca > 0)

print("\nTeste de treinamento finalizado!")
