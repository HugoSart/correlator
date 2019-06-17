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
        img = np.array(np.copy(self.img), dtype=float)
        bimg = util.border(img)
        for y in range(len(img)):
            for x in range(len(img[y])):
                sub = util.sub3x3(bimg, x + 1, y + 1)
                new = np.array([], float)
                for c in range(3):
                    band = sub[:, c]
                    band *= mask
                    new = np.append(new, np.sum(band))
                img[y][x] = new
        return img + np.amin(img)


class TranslatingCorrelator(Correlator):

    def correlate(self, mask):
        pass
