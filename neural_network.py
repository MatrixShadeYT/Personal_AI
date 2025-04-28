from random import randint
import numpy as np

layers = []

def Layer_Dense(act,neurons,inputs):
    return [
        act,
        np.transpose(np.zeros((neurons,1))),
        np.transpose(np.random.randn(neurons,inputs))
    ]

class Model:
    def __init__(self,layers):
        self.layers = layers
    def output(self,inputs):
        inputs = inputs
        for i in range(len(self.layers)):
            inputs = self.layer_output(i,inputs)
        return inputs[0]
    def layer_output(num,inputs):
        act, biases, weights = self.layers[num]
        x = np.dot(inputs,weights)+biases
        if act == 'SM':
            x = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
        elif act == 'ReLu':
            x = np.maximum(0,x)
        return x