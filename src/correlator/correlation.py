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
        img = np.copy(self.img)
        m = np.amin(self.img)
        for y in range(len(img)):
            for x in range(len(img[y])):
                sub = util.sub3x3(img, y, x)
                new = []
                for c in range(3):
                    band = sub[:, c]
                    band *= -mask
                    new.append(np.sum(band))
                img[y][x] = new + np.abs(m)
        print(img)
        return img.astype('uint8')


class TranslatingCorrelator(Correlator):

    def correlate(self, mask):
        pass
