import torch
import torch.nn as nn
import os
import torch.optim as optim
import torchvision.utils as vutils

# Constants
IMG_CHANNELS = 3
IMG_SIZE = 28
Z_DIM = 100
NUM_CLASSES = 8
EMBED_DIM = 50

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.model = nn.Sequential(
            nn.Linear(Z_DIM + EMBED_DIM, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, IMG_CHANNELS * IMG_SIZE * IMG_SIZE),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        label_input = label_input.squeeze(1)  # From shape [128, 1, 50] → [128, 50]
        #print(noise.shape)
        #print(label_input.shape)
        x = torch.cat((noise, label_input), dim=1)
        img = self.model(x)
        img = img.view(img.size(0), IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)

        self.model = nn.Sequential(
            nn.Linear(IMG_CHANNELS * IMG_SIZE * IMG_SIZE + IMG_SIZE * IMG_SIZE, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels).view(labels.size(0), 1, IMG_SIZE, IMG_SIZE)
        label_input = label_input.repeat(1, 1, 1, 1)  # Extend to match image size
        label_input = label_input.view(label_input.size(0), -1)
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, label_input), dim=1)
        validity = self.model(x)
        return validity

class ConditionalGAN(nn.Module):
    def __init__(self, generator, discriminator, dataloader, img_channels=3, img_size=28, z_dim=100, num_classes=7, embed_dim=50):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.dataloader = dataloader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, num_epochs, learning_rate, save_interval, output_dir='cgan_output'):
        os.makedirs(output_dir, exist_ok=True)
        criterion = nn.BCELoss()
        optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        for epoch in range(num_epochs):
            for i, (real_imgs, labels) in enumerate(self.dataloader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)

                # Adversarial ground truths
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)

                # ------------------
                # Train Generator
                # ------------------
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                #print(z.shape)
                #print(labels.shape)
                gen_imgs = self.generator(z, labels)
                g_loss = criterion(self.discriminator(gen_imgs, labels), valid)
                g_loss.backward()
                optimizer_G.step()

                # ------------------
                # Train Discriminator
                # ------------------
                optimizer_D.zero_grad()
                real_loss = criterion(self.discriminator(real_imgs, labels), valid)
                fake_loss = criterion(self.discriminator(gen_imgs.detach(), labels), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                if i % 100 == 0:
                    print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(self.dataloader)}] "
                          f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            # Save sample images
            if (epoch + 1) % save_interval == 0:
                self.save_samples(epoch + 1, output_dir)

    def save_samples(self, epoch, output_dir):
        with torch.no_grad():
            z = torch.randn(self.num_classes, self.z_dim, device=self.device)
            labels = torch.arange(self.num_classes, device=self.device)
            gen_imgs = self.generator(z, labels)
            gen_imgs = (gen_imgs + 1) / 2  # Rescale from [-1, 1] to [0, 1]
            filename = os.path.join(output_dir, f"epoch_{epoch}.png")
            vutils.save_image(gen_imgs, filename, nrow=self.num_classes, normalize=True)
            print(f"Saved samples to {filename}")