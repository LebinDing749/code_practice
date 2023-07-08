# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle
import numpy as np
import torch
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
matplotlib.use('TkAgg')


class Cifar10Dataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.data, self.labels = self.load_data(root_dir, train)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)

        return image, label

    def load_data(self, root_dir, train):
        data = []
        labels = []
        if train:
            for batch_id in range(1, 6):
                # load data_batch_(1-5),not include 6
                file_path = f"{root_dir}/data_batch_{batch_id}"
                # read binary
                with open(file_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    data.append(batch[b'data'])
                    labels += batch[b'labels']
        else:
            file_path = f"{root_dir}/test_batch"
            # read binary
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data.append(batch[b'data'])
                labels += batch[b'labels']

        # concatenate list "data" in axis 0
        data = np.concatenate(data, axis=0)
        # [] -> (batch_size,3,32,32) -> (batch_size,32,32,3)
        data = np.transpose(data.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
        labels = np.array(labels)

        return data, labels


# 定义一个简化版的残差块，没有残差连接
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# 定义 ResNet 的基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

def visualize(train_losses, train_acc):
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(train_acc, label="Train Acc")
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()

    plt.show()


def train(num_epochs):
    # record loss change
    train_losses = []
    train_acc = []
    # record average loss per epoch
    epoch_loss = 0.0
    total_batches = 0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 清除之前的梯度
            optimizer.zero_grad()

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            # record_loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            epoch_loss += loss.item()
            total_batches += 1

            # backward
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            train_acc.append(accuracy)
            avg_loss = epoch_loss / total_batches

            print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%, Average_Loss: {avg_loss:.4f}")

    # show loss change by batches
    visualize(train_losses, train_acc)


if __name__ := "__main__":
    # define data_loader
    transform = transforms.ToTensor()
    dataset_path = 'C:/Users/lenovo/Downloads/cifar-10-python/cifar-10-batches-py'
    train_dataset = Cifar10Dataset(root_dir=dataset_path, train=True, transform=transform)
    test_dataset = Cifar10Dataset(root_dir=dataset_path, train=False, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # define model
    model = ResNet(num_classes=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # define Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train(num_epochs)
