from neural_network import model, Layer_Dense

inputed = 1
outputed = 2
model = model([
    Layer_Dense(0,inputed,5),
    Layer_Dense('SM',5,outputed)
])

value = model.output([1])
print(value)
value = model.output([0])
print(value)