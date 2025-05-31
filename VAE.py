import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, latent_dim=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 3x28x28 → 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x14x14 → 64x7x7
            nn.ReLU(),
            nn.Flatten()
        )
        self.flatten_dim = 64 * (28 // 4) * (28 // 4)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        #self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x7x7 → 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 32x14x14 → 3x28x28
            #nn.ReLU(),
            #nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # 32x14x14 → 3x28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        #print("end of encoder shape: ", x.shape)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = z.to(device)
        #print("z shape = ", z.shape)
        #x = self.fc_decode(z).view(-1, 128, 7, 7)
        x = self.fc_decode(z).view(-1, 64, 7, 7)
        #print(x.shape)
        #print(self.decode(z).shape)
        return self.decoder(x)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        #print("test = ", z.shape)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def evaluate(self, loss_function):
        #self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                recon, mu, logvar = self.forward(inputs)
                loss = loss_function(recon, inputs, mu, logvar)
                total_loss += loss.item()
        return total_loss / len(dataloader.dataset)

    def train(self, num_epochs, loss_function, optimizer):
        train_loss = [0] * num_epochs
        valid_loss = [0] * num_epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self.train_loader:
                inputs, _ = batch

                inputs = inputs.to(device)
                
                recon, mu, logvar = self.forward(inputs)
                recon, mu, logvar = recon.to(device), mu.to(device), logvar.to(device)
                #print("recon shape = ", recon.shape)
                #print("inputs shape = ", inputs.shape)
                loss = loss_function(recon, inputs, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            #return total_loss / len(dataloader.dataset)
            train_loss[epoch] = total_loss / len(self.train_loader.dataset)
            print('Training loss(epoch %d) = %.3f' % (epoch + 1, train_loss[epoch]))
            total_loss = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    inputs, _ = batch

                    inputs = inputs.to(device)

                    recon, mu, logvar = self.forward(inputs)
                    recon, mu, logvar = recon.to(device), mu.to(device), logvar.to(device)
                    loss = loss_function(recon, inputs, mu, logvar)
                    total_loss += loss.item()
            #return total_loss / len(self.val_loader.dataset)
            valid_loss[epoch] = total_loss
            print('Validation loss(epoch %d) = %.3f' % (epoch + 1, valid_loss[epoch]))
        return train_loss, valid_loss

