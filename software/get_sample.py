import torch
from torchvision import datasets

dataset = datasets.MNIST(root='./data', train=False, download=True)

img, label = dataset[0]

img.save("pure_mnist_7.png")
print(f"success label: {label}")