# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2
from PIL import Image


def get_imagens(endereco, files, shape, cor = True, corte = (0,0)):

    num_photos = len(files)

    width,height = shape

    if cor:
        imagens = np.zeros((num_photos,width,height,3))
    else:
        imagens = np.zeros((num_photos,width,height))

    imagens = imagens.astype(np.uint8)

    x1, y1 = corte
    x2 = x1 + width
    y2 = y1 + height

    for i, file in enumerate(files):
        
        imagem = Image.open(f'{endereco}\\{file}')
        if not cor: imagem = imagem.convert('L')
        imagem_numpy = np.array(imagem)

        imagem_numpy = imagem_numpy[x1:x2,y1:y2]    # Corte da imagem original

        imagens[i] = imagem_numpy                       # Adição da imagem i ao array

    return imagens

def proporcao(tamanho):
    x = tamanho**(1/2)
    x = round(x)
    while(tamanho % x != 0):
        x -= 1
    y = int(tamanho/x)
    return x, y

def show_photos(imagens, cmap ='gray'):

    tamanho = imagens.shape[0]

    dim_x, dim_y = proporcao(tamanho)

    # Configurar a plotagem para mostrar imagens
    fig, axs = plt.subplots(dim_y, dim_x)

    # Iterar sobre as imagens e exibi-las em sequência
    for i in range(dim_y):
        for j in range(dim_x):
            axs[i,j].imshow(imagens[i*dim_x+j], cmap=cmap)
            axs[i,j].axis('off')  # Desativar eixos para uma melhor visualização

    # Exibir a sequência de imagens
    plt.show()

def create_photo(imagens, lin, col, cor = True):


    altura = imagens.shape[1]
    largura = imagens.shape[2]

    nova_altura = lin * altura
    nova_largura = col * largura

    if cor:
        imagem = np.zeros((nova_altura, nova_largura, 3))
        imagem = imagem.astype(np.uint8)
    else:
        imagem = np.zeros((nova_altura, nova_largura))


    for i in range(lin):
        for j in range(col):

            index = i*col+j

            foto_selecionada = imagens[index]

            if cor:
                imagem[i * altura : (i + 1) * altura, j * largura : (j + 1) * largura, :] = foto_selecionada
            else:
                imagem[i * altura : (i + 1) * altura, j * largura : (j + 1) * largura] = foto_selecionada

    return imagem

def images_vector(imagens):
    num_imagens, altura, largura = imagens.shape

    tam_col = largura * altura

    vectors = np.zeros((tam_col, num_imagens))

    for i in range(num_imagens):
        vectors[:,i] = imagens[i].reshape(tam_col)

    return vectors, altura, largura

def vector_image(vector, altura, largura):

    return vector.reshape(altura,largura)

def show_avarege_image(avarege_face, altura, largura):
    avarege_image = avarege_face.reshape(altura,largura)

    plt.axis('off')
    plt.imshow(avarege_image, cmap="gray")
    plt.show()



endereco = "photos"
lista_de_fotos = os.listdir(endereco)
imagens_0 = get_imagens(endereco, lista_de_fotos, (775, 775),cor = False, corte = (340,25))
#show_photos(imagens_0)


endereco = ".\\archive"
lista_de_fotos = os.listdir(endereco)
imagens = get_imagens(endereco, lista_de_fotos[2:], (243,320), cor = False)
#show_photos(imagens)

def treino(imagens, num_treino):
    imagens_vetor, altura, largura = images_vector(imagens)
    imagens_treino = imagens_vetor[:,num_treino:]

    avarege_face = np.mean(imagens_treino, axis=1)
    show_avarege_image(avarege_face, altura, largura)

    X = (imagens_treino.transpose() - avarege_face).transpose()

    U, S, V = np.linalg.svd(X, full_matrices=False)

treino(imagens, 10)
treino(imagens_0, 5)


print("FIM")

# foto = create_photo(imagens, 15, 11, cor = False)
# plt.axis('off')
# plt.imshow(foto, cmap="gray")
# plt.show()
#
#
# foto = create_photo(imagens_0, 8, 4, cor = False)
# img = plt.imshow(foto, cmap="gray")
# plt.axis('off')
# plt.show()
