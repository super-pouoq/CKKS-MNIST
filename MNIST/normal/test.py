import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from model import CNN

def test():
    device = torch.device("cpu")
    print("Loading...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = CNN().to(device)
    
    # 查找并加载刚刚训练好的模型权重
    model_path = "model/fhe_friendly_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Error {model_path} run train.py")
        return
        
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Testing over")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test()