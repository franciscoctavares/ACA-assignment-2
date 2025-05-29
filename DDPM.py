import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# ========================
# DDPM Components
# ========================

def get_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

# Noise schedule
T = 1000
betas = get_beta_schedule(T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# ========================
# U-Net like Model
# ========================

class SimpleDDPM(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.time_embed = nn.Embedding(T, 128)
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t).view(-1, 1, 1, 128).repeat(1, 1, x.shape[2], x.shape[3])
        t_channel = t_emb[:, :, :, 0].unsqueeze(1)  # Just one channel of the embedding
        x_in = torch.cat([x, t_channel], dim=1)
        return self.net(x_in)

# ========================
# Forward Process
# ========================

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

# ========================
# Training Loop (Dummy Data)
# ========================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleDDPM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy dataset: 3x28x28 images
def get_fake_data(batch_size=64):
    return torch.rand(batch_size, 3, 28, 28) * 2 - 1

# Training loop (very short for demo)
for epoch in range(5):
    model.train()
    for _ in range(100):  # batches
        x0 = get_fake_data(64).to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device).long()
        noise = torch.randn_like(x0)
        x_noisy = q_sample(x0, t, noise)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ========================
# Sampling
# ========================

@torch.no_grad()
def sample(model, num_samples=16):
    model.eval()
    x = torch.randn(num_samples, 3, 28, 28).to(device)
    for t in reversed(range(T)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        z = torch.randn_like(x) if t > 0 else 0
        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]

        pred_noise = model(x, t_batch)
        x = (1 / alpha.sqrt()) * (
            x - ((1 - alpha) / (1 - alpha_cumprod).sqrt()) * pred_noise
        ) + beta.sqrt() * z
    return x

samples = sample(model, num_samples=9).cpu()
grid = torchvision.utils.make_grid(samples, nrow=3, normalize=True, value_range=(-1, 1))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title("Generated 3×28×28 Images")
plt.show()