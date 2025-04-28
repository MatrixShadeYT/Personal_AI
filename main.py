import neural_network as nn
import string

tokenizer = list(' ,.!?*:'+string.ascii_letters)
model = nn.Model([
    nn.Layer_Dense(act='SM',neurons=2,inputs=2)
])

value = model.output([1,0])
print('Output: '+''.join(
    [f'{int(round(100*value[i],0))}% ' for i in range(len(value))]
))