import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.utils import save_image
matplotlib.use("TKAgg")
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


device = torch.device("cuda")
model = CVAE().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 生成图像
with torch.no_grad():
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            sample = torch.randn(1, 20).to(device)
            c = F.one_hot(torch.tensor(i), 10).to(device)
            c = torch.unsqueeze(c, dim=0)
            image = model.decode(sample, c)
            image = image.cpu().numpy().reshape(28, 28)
            # save_image(sample, F'img_gen/{i}.png')
            axs[i, j].imshow(image, cmap='gray')
            axs[i, j].axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('img_gen/generated_images_mean_pow(6).png')
plt.show()
