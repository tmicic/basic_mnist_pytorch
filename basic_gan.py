import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from reproducibility import ensure_reproducibility

ensure_reproducibility(6000, debug_only=False)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50

discriminator_model = Discriminator(image_dim).to(device)
generator_model = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="data/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=lr)
generator_optimizer = optim.Adam(generator_model.parameters(), lr=lr)
criterion = nn.BCELoss()


for epoch in range(num_epochs):

    print(f'Epoch: {epoch+1} of {num_epochs}:')

    for batch_idx, (real_images, _) in enumerate(loader):

        real_images = real_images.view(-1, 784).to(device)
        
        batch_size = real_images.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator_model(noise)
        disc_real = discriminator_model(real_images).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator_model(fake_images).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        discriminator_model.zero_grad()
        lossD.backward(retain_graph=True)
        discriminator_optimizer.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = discriminator_model(fake_images).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        generator_model.zero_grad()
        lossG.backward()
        generator_optimizer.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake_images = generator_model(fixed_noise).reshape(-1, 1, 28, 28)[0:6]
                real_images_resized = real_images.reshape(-1, 1, 28, 28)[0:6]
                plt.imshow(torch.cat([fake_images.view(-1,28),real_images_resized.view(-1,28)], dim=1).detach().cpu())
                plt.show(block=False)
                plt.pause(0.001)

