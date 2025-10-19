# gan_cifar.py
import torch, torch.nn as nn, torch.optim as optim, os, datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- CONFIG ----------------
latent_dim = int(os.getenv("LATENT_DIM", 100))
batch_size = int(os.getenv("BATCH_SIZE", 256))
lr_g = float(os.getenv("LR_G", 0.0002))
lr_d = float(os.getenv("LR_D", 0.0002))
epochs = int(os.getenv("EPOCHS", 50))
arch_variant = os.getenv("ARCH", "base")
exp_name = os.getenv("EXP_NAME", None)

# unique results folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = exp_name or f"cifar_lrG{lr_g}_lrD{lr_d}_z{latent_dim}_bs{batch_size}_{arch_variant}_{timestamp}"
output_dir = os.path.join("results_cifar", exp_name)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CIFAR] Using device: {device}, arch={arch_variant}, saving to {output_dir}")

# --------------- MODELS -----------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        channels = [128,128,64]
        self.model = nn.Sequential(
            nn.Linear(latent_dim, channels[0]*8*8),
            nn.ReLU(True),
            nn.Unflatten(1, (channels[0],8,8)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], 3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self,z): return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(64,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*4*4,1),
            nn.Sigmoid()
        )
    def forward(self,x): return self.model(x)

# --------------- DATA -------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
data = datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
loader = DataLoader(data,batch_size=batch_size,shuffle=True,num_workers=4)

# --------------- TRAIN ------------------
G, D = Generator(latent_dim).to(device), Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5,0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5,0.999))

g_losses, d_losses = [], []

fixed_z = torch.randn(32, latent_dim, device=device)  # fixed noise to track evolution

for epoch in range(epochs):
    for real,_ in loader:
        bs = real.size(0)
        real, valid, fake = real.to(device), torch.ones(bs,1,device=device), torch.zeros(bs,1,device=device)
        # --- D ---
        opt_D.zero_grad()
        d_real = criterion(D(real), valid)
        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z)
        d_fake = criterion(D(fake_imgs.detach()), fake)
        d_loss = d_real + d_fake
        d_loss.backward(); opt_D.step()
        # --- G ---
        opt_G.zero_grad()
        g_loss = criterion(D(fake_imgs), valid)
        g_loss.backward(); opt_G.step()

    g_losses.append(g_loss.item()); d_losses.append(d_loss.item())
    print(f"Epoch {epoch+1}/{epochs} | D {d_loss.item():.3f} | G {g_loss.item():.3f}")

    # save intermediate fake images and plots
    if (epoch+1)%5==0 or epoch==0:
        with torch.no_grad():
            samples = G(fixed_z)
            vutils.save_image(samples, f"{output_dir}/fake_epoch{epoch+1:03d}.png", normalize=True, nrow=8)

        plt.figure(figsize=(8,5))
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.title(f"Losses up to epoch {epoch+1}")
        plt.legend()
        plt.savefig(f"{output_dir}/loss_plot_epoch{epoch+1:03d}.png")
        plt.close()

        pd.DataFrame({'G_loss':g_losses,'D_loss':d_losses}).to_csv(f"{output_dir}/losses_epoch{epoch+1:03d}.csv",index=False)

        torch.save({
            'epoch': epoch+1,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'optG': opt_G.state_dict(),
            'optD': opt_D.state_dict()
        }, f"{output_dir}/checkpoint_epoch{epoch+1:03d}.pt")
