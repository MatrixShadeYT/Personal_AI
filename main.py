import neural_network as nn

inputed = 1
outputed = 2
nn.layers = [
    nn.Layer_Dense(0,5,inputed),
    nn.Layer_Dense('SM',outputed,5)
]

value = nn.output([1])
for i in range(len(value)):
    print(f'N{i+1}: {value[i]}')