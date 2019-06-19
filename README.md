# Correlator
Programa simples escrito em Python 3 para a aplicação de correlação em imagens.

## Métodos
Este programa possúi três métodos implementados para a quantização de cores:
- Correlação simples;
- Correlação por translação;

Cada uma destas implementações foi escrita separadamente em classes: _SimpleCorrelator e TranslatingCorrelator_ respectivamente.
<br><br>
Para o uso destes métodos, basta istanciar a respectiva classe, passando para o construtor a imagem a ser correlatada, e em seguida utilizar o método _correlate_, passando como parâmetro a mascara a ser utilizada.

## Execução
Antes de se executar o programa, é necessário instalar as seguintes dependências:
- _numpy_
- _opencv-python_

Com as dependências instaladas, realize os seguintes passos:
1. Navegue até a pasta _"correlator/src"_;
2. Abra o terminal de comandos e execute: "_python3 -m correlator.app -i IE -o IS -a ME -m MS -s SC_", onde:
   - IE: caminho da imagem a ser correlatada;
   - IS: caminho da imagem correlatada a ser salva;
   - ME: método de correlação (_simple_ ou _translate_);
   - MS: mascara no formato "x,x,x;x,x,x;x,x,x";
   - SC: fator de multiplicação da mascara;
   
   
## Exemplo
Quantização utilizando o método _SimpleQuantizer_:
```python
import correlator.correlation as cor
import numpy
import cv2

# Lê imagem arbitrária
img = cv2.imread('input_image.png')

# Escolhe método de quantização
method = cor.SimpleCorrelator(img)

# Realiza a correlação identidade
reduced = method.correlate(numpy.array([[0,0,0],[0,1,0],[0,0,0]]))

# Mostra imagem em janela e aguarda ser fechada ou tecla pressionada
cv2.imshow('output', reduced)
cv2.waitKey()
```