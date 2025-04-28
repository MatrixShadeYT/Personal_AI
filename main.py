import neural_network as nn

inputs = [4,3]
x = nn.Layer_Dense(act="SM",neurons=2,inputs=2)
model = nn.Model([x])
value = model.output(inputs)
print('SM: '+''.join(
    [f'{int(round(100*value[i],0))}% ' for i in range(len(value))]
))
x = ['ReLu',x[1],x[2]]
model = nn.Model([x])
value = model.output(inputs)
print('ReLu: '+''.join(
    [f'{value[i]} ' for i in range(len(value))]
))
x = [0,x[1],x[2]]
model = nn.Model([x])
value = model.output(inputs)
print('Output: '+''.join(
    [f'{value[i]} ' for i in range(len(value))]
))