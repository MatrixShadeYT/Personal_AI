from random import randint
import numpy as np

# SM = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
# ReLu = np.maximum(0,x)
# x = np.dot(inputs,weights)+biases

layers = []

def output(inputs):
    inputs = inputs
    for i in range(len(layers)):
        weights = layers[i][2] # [[],[]]
        biases = layers[i][1] # []
        x = np.dot(inputs,weights)+biases
        # SM
        if layers[i][0] == 'SM':
            inputs = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
        elif layers[]


def Layer_Dense(activation,inputs,outputs):
    return [
        activation,np.zeros((inputs,1))[0],
        np.random.randn(outputs,inputs)
    ]