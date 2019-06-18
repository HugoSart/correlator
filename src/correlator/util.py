import numpy as np
import cv2


# 2, 2, 2 ; 3, 3, 3 ; 5, 5, 5
def stoa(x):
    """
    Converte uma string no formato 'n,n,n;n,n,n;n,n,n' para uma matriz 3x3.
    :param x: a string a ser convertida.
    :return: a matriz 3x3 resultante.
    """
    ret = []
    rows = x.split(';')
    for row in rows:
        nums = row.split(',')
        ret += nums
    ret = np.array(list(map(float, ret)), float)
    ret.shape = (3, 3)
    return ret


def sub3x3(matrix, x, y):
    """
    Extrai uma sub matriz 3x3 de uma matrix.
    :param matrix: a matriz a ser extraída a sub matriz.
    :param x: a coluna da matriz a ser utilizado como centro da sub matriz.
    :param y: a linha da matriz a ser utilizado como centro da sub mtraiz.
    :return: uma matriz 3x3.
    """
    ret = np.array([], dtype='float64')
    for ay in range(-1, 2):
        for ax in range(-1, 2):
            cy = (y + ay) % len(matrix)
            cx = (x + ax) % len(matrix[0])
            ret = np.append(ret, matrix[cy][cx])
    ret.shape = (3, 3)
    return ret


def border(img, width=1, rgb=0):
    """
    Envolve a imagem com uma borda constante.
    :param img: a imagem a ser envolvida com uma borda.
    :param width: o tamanho da borda em pixel.
    :param rgb: a cor da borda ([0-255], [0-255], [0-255]).
    :return: a imagem com a borda.
    """
    bimg = np.full((img.shape[0] + 2 * width, img.shape[1] + 2 * width), rgb, 'float64')
    bimg[width:-width, width:-width] = img
    return bimg


def normalize(img):
    """
    Normaliza uma imagem GRAYSCALE.
    :param img:
    :return:
    """
    img += np.abs(np.amin(img))
    img *= (1.0 / np.amax(img))
    img *= 255.0
    return img


def show(*img_defs):
    """
    Mostra imagens na tela, onde cada tupla (titulo, imagem) é uma imagem a ser mostrada.
    :param img_defs: a tupla das imagens no formato (titulo, imagem).
    """
    for img_def in img_defs:
        cv2.imshow(img_def[0], img_def[1])


def wait():
    """
    Bloqueia o programa e aguarda uma interação do usuáio no teclado.
    """
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zeropad(img):
    """
    Faz zero padding na imagem 
    coluna da esquerda, coluna da direita, linha de baixo, linha de cima
    """
    row = len(img)
    col = len(img[0])
    first = 0
    img = np.insert(img, first, 0, axis=1)
    img = np.insert(img, col+1, 0, axis=1)
    img = np.insert(img, first, 0, axis=0)
    img = np.insert(img, row+1, 0, axis=0)
    return img


def padremove(img):
    """
    Tira os zero paddings colocado na imagem
    """
    row = len(img)
    col = len(img[0])
    first = 0
    img = np.delete(img, first, axis=0)
    img = np.delete(img, first, axis=1)
    row = len(img)
    col = len(img[0])
    img = np.delete(img, row-1, axis=0)
    img = np.delete(img, col-1, axis=1)
    return img


def mask_weights(mask):
    """
    Cria o conjunto da máscara, com as posições de deslocamento e peso
    """
    row = len(mask)
    col = len(mask[0])
    weights = []
    for i in range(row):
        for j in range(col):
            posx = -abs(i)+1
            posy = -abs(j)+1
            coordinate = [posx, posy]
            value = mask[i][j]
            weights.append([coordinate, value])
    return weights
