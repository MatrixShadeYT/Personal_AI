from random import randint
import numpy as np

# SM = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
# ReLu = np.maximum(0,x)
# x = np.dot(inputs,weights)+biases

layers = []

def output(inputs):
    inputs = inputs
    for i in range(len(layers)):
        print(inputs)
        weights = layers[i][2] # [[],[]]
        print(weights)
        biases = layers[i][1] # []
        print(f'{biases}\n')
        x = np.dot(inputs,weights)+biases
        if layers[i][0] == 'SM':
            inputs = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
        elif layers[i][0] == 'ReLu':
            inputs = np.maximum(0,x)
        else:
            inputs = x
    return inputs

def Layer_Dense(activation,neurons,inputs):
    return [
        activation,np.zeros((neurons,1))[0],
        np.random.randn(neurons,inputs)
    ]