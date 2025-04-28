import neural_network as nn
import string

tokenizer = list(' ,.!?*:'+string.ascii_letters)
inputs = 2
outputs = 5
model = nn.Model([
    nn.Layer_Dense(act='SM',neurons=outputs,inputs=inputs)
])

value = model.output([1,0])
print('Output: '+''.join([f'{value[i]} ' for i in range(len(value))]))