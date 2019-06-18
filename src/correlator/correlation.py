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
                sub = util.sub3x3(bimg, x, y)
                img[y][x] = np.sum(sub)
        return img


class TranslatingCorrelator(Correlator):

    def correlate(self, mask):
            img = np.copy(self.img)
            img = util.zeropad(img)
            correlation = []

            # gera o conjunto de pesos da máscara e direções de translação
            weights = util.mask_weights(mask)

            # gera as matrizes transladadas
            for i in range(len(weights)):
                copy = np.copy(img)
                copy = np.roll(copy, weights[i][0][0], axis=0)
                copy = np.roll(copy, weights[i][0][1], axis=1)
                copy = util.padremove(copy)
                copy *= weights[i][1]
                correlation.append(copy)

            # soma as matrizes transladadas
            final = sum(correlation)
            print(final)

            return img
