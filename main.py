import neural_network as nn

inputs = [2,3]
model = nn.Model([nn.Layer_Dense("SM",inputs[1],inputs[0])])

model.output([])