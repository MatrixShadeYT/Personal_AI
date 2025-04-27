import neural_network as nn
import string

tokenizer = list(',.!?*:'+string.ascii_letters)
chat = input('Shade: ')
inputs = []
for i in list(chat):
    inputs.append(tokenizer.index(i))
nn.layers = [
    nn.Layer_Dense('ReLu',5,len(inputs)),
    nn.Layer_Dense('SM',5,5)
]

value = nn.output(inputs)
for i in range(len(value)):
    print(f'N{i+1}: "{tokenizer[round(int(value[i]),0)]}" {value[i]}')