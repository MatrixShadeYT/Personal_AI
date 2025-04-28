import neural_network as nn
import string

tokenizer = list(' ,.!?*:'+string.ascii_letters)
chat = input('USR: ')
inputs = []
for i in list(chat):
    inputs.append(tokenizer.index(i))
nn.layers = [
    nn.Layer_Dense('SM',5,len(inputs))
]

value = nn.output(inputs)
print('Bot: "'+''.join([tokenizer[round(int(value[i]),0)] for i in range(len(value))])+'"')