# 加载ResNet模型并移动到GPU
#model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # 替换最后的全连接层  下面的 classifier. 要换成 fc. 
#model = model.to(device)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.densenet import DenseNet169_Weights
import os
import matplotlib.pyplot as plt

# 检查 CUDA 是否可用，并选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

# 设置数据集路径
data_dir = "/sdb/2024_computer_vision/xjtu_cvpr_008/CVPR-works/4/"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载DenseNet模型并移动到GPU
model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(train_dataset.classes))  #classifier为DenseNet的全连接层
model = model.to(device)

# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False

# 仅对最后一层的参数进行梯度更新
for param in model.classifier.parameters():
    param.requires_grad = True

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# 训练模型
def train_model(model, criterion, optimizer, num_epochs=40):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_test_acc = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * train_correct / train_total)
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Train Accuracy: {train_accs[-1]}%')

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_losses.append(running_loss / len(test_loader))
        test_accs.append(100 * test_correct / test_total)
        print(f'Epoch {epoch+1}, Test Loss: {test_losses[-1]}, Test Accuracy: {test_accs[-1]}%')

        # 保存最佳模型
        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]
            best_model_state = model.state_dict()

    # 绘制损失和准确率图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # 定义保存最佳模型权重的路径
    best_model_path = "/sdb/2024_computer_vision/xjtu_cvpr_008/CVPR-works/4/best_densenet_model.pth"

    # 保存最佳模型权重
    torch.save(best_model_state, best_model_path)

    # 保存训练和测试结果
    with open('train_test_results.txt', 'w') as f:
        f.write(f'Train Losses: {train_losses}\n')
        f.write(f'Test Losses: {test_losses}\n')
        f.write(f'Train Accuracies: {train_accs}\n')
        f.write(f'Test Accuracies: {test_accs}\n')
        f.write(f'Best Test Accuracy: {best_test_acc}%\n')

    return best_test_acc

best_accuracy = train_model(model, criterion, optimizer, num_epochs=40)
print(f'Best Test Accuracy: {best_accuracy}%')