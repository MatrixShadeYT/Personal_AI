from random import randint
import numpy as np

def Layer_Dense(activation,inputs,outputs):
    return [
        activation,np.zeros((inputs,1))[0],
        np.random.randn(outputs,inputs)
    ]

# Model Class - [Layer]
class model:
    def __init__(self,layers):
        self.layers = layers
    def output(self,inputs):
        inputs = inputs
        for i in range(len(self.layers)):
            print(f'inputs: {inputs}')
            x = np.transpose(self.layers[i][2])[0]
            for y in range(len(x)):
                print(f'Weights[{y}]: {x[y]}')
            print(f'biases: {self.layers[i][1]}\n')
            x = np.dot(inputs,x)
            x += self.layers[i][1]
            '''if self.layers[i][0] == 'ReLu':
                inputs = np.maximum(0,x)
            elif self.layers[i][0] == 'SM':
                inputs = np.exp(x-np.max(x,axis=1,keepdims=True)) / np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
            else:
                inputs = x'''
        return inputs