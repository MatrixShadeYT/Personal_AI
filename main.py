import neural_network as nn
import string

tokenizer = list(' ,.!?*:'+string.ascii_letters)
chat = input('Shade: ')
inputs = []
for i in list(chat):
    inputs.append(tokenizer.index(i))
nn.layers = [
    nn.Layer_Dense(0,5,len(inputs)),
    nn.Layer_Dense('SM',5,5)
]

value = nn.output(inputs)
for i in range(len(value)):
    x = int(value[i])
    x = tokenizer[round(x,0)]
    print(f'N{i+1}: {value[i]}')