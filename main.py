import neural_network as nn

inputs, outputs = 2, 3
model = nn.Model([nn.Layer_Dense("SM",outputs,inputs)])

value = model.output([1,0])
print(value)