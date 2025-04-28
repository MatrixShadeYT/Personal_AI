import neural_network as nn
import string

tokenizer = list(' ,.!?*:'+string.ascii_letters)
chat = input('USR: ')
inputs = 5
nn.layers = [
    nn.Layer_Dense('SM',5,inputs)
]

value = nn.output(inputs)
print('Output: '+''.join(value))