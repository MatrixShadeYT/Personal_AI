import neural_network as nn
import string

tokenizer = list(' ,.!?*:'+string.ascii_letters)
inputs = 5
nn.layers = [
    nn.Layer_Dense('SM',5,inputs)
]

value = nn.output(inputs)
print('Output: '+''.join([f'{value[0]}', f', {value[i+1]}' for i in range(len(value)-1)]))