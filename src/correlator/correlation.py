import abc
import numpy as np

from correlator import util


class Correlator:

    def __init__(self, img):
        self.img = img

    @abc.abstractmethod
    def correlate(self, mask):
        pass


class SimpleCorrelator(Correlator):

    def correlate(self, mask):
        img = np.array(np.copy(self.img))
        bimg = util.border(img)
        for y in range(len(img)):
            for x in range(len(img[y])):
                sub = util.sub3x3(bimg, x + 1, y + 1)
                new = np.array([])
                for c in range(3):
                    band = sub[:, c]
                    band *= mask
                    new = np.append(new, np.sum(band))
                img[y][x] = new
        return img


class TranslatingCorrelator(Correlator):

    def correlate(self, mask):
            img = np.copy(self.img)
            img = util.zeropad(img)
            correlation = []

            # print(mask)

            # gera o conjunto de pesos da máscara e direções de translação
            weights = util.mask_weights(mask)

            """
            Quebra a imagem original em 9 imagens transladadas (mascara 3x3)
            Cada uma das imagens transladadas é multiplicada pelo peso da posição dela na mascara
            """

            for i in range(len(weights)):
                copy = np.copy(img)
                copy = np.roll(copy, weights[i][0][0], axis=0)
                copy = np.roll(copy, weights[i][0][1], axis=1)
                copy = util.padremove(copy)
                # copy *= weights[i][1]
                np.multiply(copy, weights[i][1], out=copy, casting='unsafe')
                correlation.append(copy)

            # soma as matrizes transladadas
            final = sum(correlation)
            img = final

            return img
