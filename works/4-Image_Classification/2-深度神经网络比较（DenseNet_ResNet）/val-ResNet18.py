import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "/sdb/2024_computer_vision/xjtu_cvpr_008/CVPR-works/4/"
val_dir = os.path.join(data_dir, 'val')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(val_dataset.classes))
model = model.to(device)
model.load_state_dict(torch.load('/sdb/2024_computer_vision/xjtu_cvpr_008/resnet_model.pth', map_location=device), strict=False)
model.eval()

criterion = nn.CrossEntropyLoss()

def validate_model(model, val_loader, criterion):
    val_losses = []
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_loss = sum(val_losses) / len(val_losses)
    avg_accuracy = 100 * val_correct / val_total
    print(f'Validation Loss: {avg_loss:.4f}')
    print(f'Validation Accuracy: {avg_accuracy:.2f}%')

    return avg_loss, avg_accuracy

val_loss, val_acc = validate_model(model, val_loader, criterion)