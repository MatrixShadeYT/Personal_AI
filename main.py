import neural_network as nn

inputs, outputs = 2, 3
model = nn.Model([nn.Layer_Dense("SM",outputs,inputs)])
data = model.model() # [[act,biases,[weights]]]

class genetic:
  generations = []
  def __init__(self,model):
    self.model = model
    self.base = model
  def generation(self,size,rate):
    hive = []
    for i in range(size):
      x = self.model

AI = genetic(model)