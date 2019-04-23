# -*- coding: utf-8 -*-

import numpy as np

iterations      = 10000
learningRate    = 0.1

#w1 . x1 + w2 . x2 + b = y

def prediction(x, w, b):
    y = b
    for i in range(len(x) - 1):
        y += w[i] * x[i]
    return y
    

def train(data, w, b):
    for i in range(iterations):
        sumError = 0.0
        for x in data:
            
            #"testando" a predição de um vídeo que já está na base
            y           = prediction(x, w, b)
            
            #cálculo de erros
            error       = x[2] - y    #erro da linha
            sumError    += error ** 2 #erro da iteração total
            
            #atualização de bias
            b = b + learningRate * error
            
            #atualiza w1 e w2
            for i in range(len(x) - 1):
                w[i] = w[i] + learningRate * error * x[i]
    
    print()
    print("O erro foi: " + str(sumError))
    print()
    return w, b
            

def main():
    #dados de treinamento
    data = np.array((
            #             x1                   x2                              y
            #   duração (em min.)  ($$$ investido em mil reais )   (total de visualizações em milhões)
            [         11.0 / 100  ,         1.0 / 100           ,          5.0 / 100     ],
            [         23.0 / 100  ,         9.0 / 100           ,          4.0 / 100     ],
            [         26.0 / 100  ,         1.0 / 100           ,          1.0 / 100     ]),
        dtype = float)
    
    #pesos e bias
         #w1  #w2
    w = [0.0, 0.0]
    b = [0.0]

    #treina a rede neural    
    w, b = train(data, w, b)

    #testa cada linha pra ver se a rede neural está calibrada
    print("Calibrando a rede neural")
    print("------------------------")
    yTeste = prediction(data[0], w, b)
    print(yTeste, data[0][2])
    yTeste = prediction(data[1], w, b)
    print(yTeste, data[1][2])
    yTeste = prediction(data[2], w, b)
    print(yTeste, data[2][2])
    print()

    #Exemplo Carlos Eduardo
    print("--------------------------------------")
    yTeste = prediction([ 5.0 / 100, 2.0 / 100, 0], w, b)
    print("Exemplo Carlos Eduardo..: " + str(abs(yTeste)))
    #Exemplo Fernanda
    yTeste = prediction([10.0 / 100, 1.5 / 100, 0], w, b)
    print("Exemplo Fernanda........: " + str(abs(yTeste)))
    #Exemplo Mestre de obra
    yTeste = prediction([15.0 / 100, 4.0 / 100, 0], w, b)
    print("Exemplo Mestre de obra..: " + str(abs(yTeste)))
    print()

if __name__ == "__main__":
    main()