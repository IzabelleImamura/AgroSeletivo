
import torch
import segmentation_models_pytorch as smp

from dataset_agroseletivo import PhenoBenchDataset


def criar_modelo():
    """
    Cria a U-Net do AgroSeletivo.

    Entrada: imagem RGB com 3 canais.
    Saida: 3 canais, um para cada classe.
    """

    modelo = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation=None
    )

    return modelo


if __name__ == "__main__":

    print("AGROSELETIVO - TESTE DA REDE NEURAL")

    # Carregar um exemplo do conjunto de validacao
    dataset = PhenoBenchDataset("val")

    imagem, mascara = dataset[0]

    print("\nDADOS DE ENTRADA")
    print("Formato da imagem:", imagem.shape)
    print("Formato da mascara:", mascara.shape)

    # Construir a U-Net
    print("\nCriando a U-Net com MobileNetV2...")

    modelo = criar_modelo()

    # Colocar a rede em modo de avaliacao
    modelo.eval()

    # Adicionar a dimensao do lote
    entrada = imagem.unsqueeze(0)

    print("Entrada da rede:", entrada.shape)

    # Executar uma passagem pela rede
    # sem calcular gradientes de treinamento
    with torch.inference_mode():
        saida = modelo(entrada)

    print("\nRESULTADO DA REDE")
    print("Formato da saida:", saida.shape)

    # Verificar se todos os valores sao validos
    valores_validos = torch.isfinite(saida).all().item()

    print("Valores numericos validos:", valores_validos)

    # Quantidade de parametros da rede
    parametros = sum(
        p.numel() for p in modelo.parameters()
    )

    print("Quantidade de parametros:", parametros)

    # Verificar o formato esperado
    formato_esperado = (1, 3, 512, 512)

    if saida.shape == formato_esperado and valores_validos:
        print("\nRede neural criada e testada com sucesso!")
    else:
        print("\nATENCAO: resultado inesperado!")

    print("\nIMPORTANTE:")
    print("A rede ainda nao foi treinada para identificar plantas.")
