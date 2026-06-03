import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleMNIST


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    model = SimpleMNIST().to(device)
    model.load_state_dict(torch.load("./checkpoints/mnist_mlp.pth", map_location=device,weights_only=True))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()