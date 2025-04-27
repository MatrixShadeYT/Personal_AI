from neural_network import model, Layer_Dense

inputed = 3
outputed = 2
model = model([
    Layer_Dense(0,5,inputed),
    Layer_Dense('SM',outputed,5)
])