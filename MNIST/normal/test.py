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

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    model = CNN().to(device)

    model_path = "model/fhe_friendly_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    wrong_total = 0

    # 用来记录全局样本编号
    seen = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)

            # 等价于 torch.max(outputs, 1)，但更简洁
            predicted = outputs.argmax(dim=1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 找出当前 batch 中预测错的位置
            wrong_mask = predicted != target
            wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]

            if wrong_indices.numel() > 0:
                print(f"\nBatch {batch_idx} has {wrong_indices.numel()} wrong samples:")

                for i in wrong_indices:
                    i = i.item()

                    global_idx = seen + i
                    pred_label = predicted[i].item()
                    true_label = target[i].item()

                    print(
                        f"  Global index: {global_idx:5d}, "
                        f"Batch index: {i:4d}, "
                        f"Predicted: {pred_label}, "
                        f"True: {true_label}"
                    )

                wrong_total += wrong_indices.numel()

            seen += target.size(0)

    accuracy = 100 * correct / total

    print("\nTesting over")
    print(f"Total samples: {total}")
    print(f"Correct samples: {correct}")
    print(f"Wrong samples: {wrong_total}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test()