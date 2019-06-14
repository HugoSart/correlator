import cv2 as cv
import argparse

import correlator.correlation as corr
from correlator import util


def define_args():
    parser = argparse.ArgumentParser(description='Aplica um algorítmo de correlação a uma dada imagem.')
    parser.add_argument('-i', dest='input', type=str, default='./input.png',
                        help='O caminho até a imagem a sofrer correlação.')
    parser.add_argument('-o', dest='output', type=str,
                        help='O caminho onde a imagem será escrita.')
    parser.add_argument('-a', dest='algorithm', type=str, default='simple',
                        help='O algorítmo a ser utilizado para a correlação.')
    parser.add_argument('-m', dest='mask', type=str, default='[[1, 1, 1], [1, 1, 1], [1, 1, 1]]',
                        help='A mascara 3x3 a ser aplicada na imagem.')
    return parser.parse_args()


def main():
    args = define_args()

    print('Aplicando quantização em \"%s\" usando o algorítmo %s.' % (args.input, args.algorithm))

    img = cv.imread(args.input)

    # Escolhe o algorítmo que será utilizado na quantização
    alg = None
    if args.algorithm == 'simple':
        alg = corr.SimpleCorrelator(img)
    elif args.algorithm == 'translate':
        alg = corr.TranslatingCorrelator(img)
        raise RuntimeError('O método de correlação %s não é um método válido.' % args.algorithm)

    # Aplica quantização
    mask = util.stoa(args.mask)
    corr_img = alg.correlate(mask)

    # Grava a imagem em um arquivo de saída se necessário
    if args.output is not None:
        cv.imwrite(args.output, corr_img)

    util.show(('Input', img), ('Output', corr_img))
    util.wait()


if __name__ == "__main__":
    main()
