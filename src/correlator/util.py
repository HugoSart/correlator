import numpy as np
import cv2


def nmult(n):
    """
    Separa um número em três números onde a multiplicação destes três números resulta no número original.

    :param n: o número a ser fatorado em três números.
    :return: um array de três posições com os números fatorados.
    """
    # Verificação de segurança para n menor do que 1
    if n <= 1:
        return [1, 1, 1]

    c = n
    nums = []

    # Fatora o número n de forma convensional
    for i in range(2, n + 1):
        while c % i == 0:
            nums.append(i)
            c /= i

    if len(nums) == 1:
        nums = [nums[0], 1, 1]
    elif len(nums) == 2:
        nums = [nums[0], nums[1], 1]
    elif len(nums) > 3:
        # Se o número de elementos da fatoração for maior do que três, comprime o array em um de três posições fazendo
        # a multiplicação do último elemento do array com o menor elemento do array desconsiderando a si mesmo
        while len(nums) > 3:
            i1 = np.argmin(nums)
            n1 = nums[i1]
            nums.pop(i1)
            i2 = np.argmin(nums)
            nums[i2] *= n1

    return nums


def aprox(img, palheta):
    palheta = np.array(palheta).reshape(-1, 3)
    distancia = np.linalg.norm(img[:, :, None] - palheta[None, None, :], axis=3)
    indices_palheta = np.argmin(distancia, axis=2)
    return palheta[indices_palheta]


def argmedian(array, column):
    return np.argsort(array[:, column])[len(array) // 2]


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.math.log10(PIXEL_MAX / np.math.sqrt(mse))


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
    ret = np.array([], float)
    for ay in range(-1, 2):
        for ax in range(-1, 2):
            cy = (y + ay) % len(matrix)
            cx = (x + ax) % len(matrix[0])
            ret = np.append(ret, matrix[cy][cx])
    ret.shape = (3, 3, 3)
    return ret


def border(img, size=1, rgb=(0, 0, 0)):
    """
    Envolve a imagem com uma borda de tamanho variavel.
    :param img: a imagem a ser envolvida com uma borda.
    :param size: o tamanho da borda em pixel.
    :param rgb: a cor da borda ([0-255], [0-255], [0-255]).
    :return: a imagem com a borda.
    """
    bimg = np.full((img.shape[0] + 2 * size, img.shape[1] + 2 * size, img.shape[2]), rgb, float)
    bimg[size:-size, size:-size] = img
    return bimg


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