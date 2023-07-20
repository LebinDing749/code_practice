
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.encoder_fc1 = nn.Linear(32 * 28 * 28, 256)
        self.encoder_mean = nn.Linear(256 + 10, 20)
        self.encoder_logvar = nn.Linear(256 + 10, 20)

        # Decoder
        self.decoder_fc1 = nn.Linear(20 + 10, 256)
        self.decoder_fc2 = nn.Linear(256, 32 * 28 * 28)
        self.decoder_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x, labels):
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.encoder_fc1(x))
        x = torch.cat([x, labels], dim=1)
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar

    def decode(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        x = x.view(x.size(0), 32, 28, 28)
        x = F.relu(self.decoder_conv1(x))
        x = torch.sigmoid(self.decoder_conv2(x))
        return x

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x, labels):
        mean, logvar = self.encode(x, labels)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mean, logvar


def loss_function(recon_x, x, mean, logvar):
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    loss = (10 * reconstruction_loss).pow(6.0) + kl_divergence_loss
    return loss


def train(epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, label in train_loader:
            data = data.to(device)  # [64, 1, 28, 28]
            label = F.one_hot(label, 10).to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data, label)

            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")


if __name__ == '__main__':

    # dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='../', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='../', train=False, download=False, transform=transform)

    # dataloader
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 设备
    device = torch.device("cuda")

    # 定义模型参数
    num_epochs = 10
    learning_rate = 0.005

    # model
    model = CVAE().to(device)

    # loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train
    train(num_epochs)

    # save model
    torch.save(model.state_dict(), 'model.pth')

