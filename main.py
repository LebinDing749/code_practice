# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
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


def show_image():
    image_index = 0
    image, label = train_dataset[image_index]
    image = image.numpy().transpose((1, 2, 0))

    plt.imshow(image)
    plt.title(f"Label:{label}")
    plt.axis('off')
    plt.show(block=True)


class BinNet(nn.Module):
    def __init__(self):
        super(BinNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def show_loss(train_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend
    plt.show()


def train(num_epochs):

    # record loss change
    train_losses = []
    # record average loss per epoch
    epoch_loss = 0.0
    total_batches = 0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            # record_loss
            loss_value = loss.item()
            train_losses.append(loss_value)

            epoch_loss += loss.item()
            total_batches += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            avg_loss = epoch_loss / total_batches

            print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%, Average_Loss: {avg_loss:.4f}")

    # show loss change by batches
    # show_loss(train_losses)


if __name__ := "__main__":

    # define data_loader
    transform = transforms.ToTensor()
    dataset_path = 'C:/Users/lenovo/Downloads/cifar-10-python/cifar-10-batches-py'
    train_dataset = Cifar10Dataset(root_dir=dataset_path, train=True, transform=transform)
    test_dataset = Cifar10Dataset(root_dir=dataset_path, train=False, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # show an image
    # show_image()

    # define model
    model = BinNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # define Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    train(num_epochs)

