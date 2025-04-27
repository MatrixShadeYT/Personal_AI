from random import randint
import numpy as np

# SM = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
# ReLu = np.maximum(0,x)
# x = np.dot(inputs,weights)+biases

layers = []

def output(inputs):
    inputs = inputs
    for i in range(len(layers)):
        print(f'Inputs: {inputs}')
        print(f'Activation: {layers[i][0]}')
        print(f'Biases: \n{layers[i][1]}')
        print(f'Weights: \n{layers[i][2]}\n')
        inputs = Layer_Output(i,inputs)
        print(f'Output: {inputs}')
    return inputs

def Layer_Dense(activation,neurons,inputs):
    return [
        activation,
        np.transpose(np.zeros((neurons,1))),
        np.transpose(np.random.randn(neurons,inputs))
    ]

def Layer_Output(num,inputs):
    act, biases, weights = layers[num]
    x = np.dot(inputs,weights)+biases
    if act == 'SM':
        x = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
    elif act == 'ReLu':
        x = np.maximum(0,x)
    return x