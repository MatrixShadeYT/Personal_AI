from random import randint
import numpy as np

def Layer_Dense(activation,inputs,outputs):
    return [activation,np.zeros((1,inputs))[0],np.random.randn(inputs,outputs)]

# Model Class - [Layer]
class model:
    def __init__(self,layers):
        self.layers = layers
    def output(self,inputs):
        inputs = inputs
        for i in range(len(self.layers)):
            print(f'\ninputs: {inputs}')
            print(f'biases: {self.layers[i][1]}')
            print(f'weights: {self.layers[i][2]}')
            x = np.dot([inputs],self.layers[i][2])+self.layers[i][1]
            if self.layers[i][0] == 'ReLu':
                inputs = np.maximum(0,x)
            elif self.layers[i][0] == 'SM':
                inputs = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
            else:
                inputs = x
        return inputs