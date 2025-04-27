import neural_network as nn

inputed = 1
outputed = 2
nn.layers = [
    nn.Layer_Dense(0,5,inputed),
    nn.Layer_Dense('SM',5,outputed)
]

value = model.output([1])
print(value)
value = model.output([0])
print(value)