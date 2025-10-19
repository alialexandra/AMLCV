# gan_mnist_fixed.py
import torch, torch.nn as nn, torch.optim as optim, os, datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import pandas as pd

latent_dim = int(os.getenv("LATENT_DIM", 100))
batch_size = int(os.getenv("BATCH_SIZE", 256))
lr_g = float(os.getenv("LR_G", 0.0002))
lr_d = float(os.getenv("LR_D", 0.0002))
epochs = int(os.getenv("EPOCHS", 50))
arch_variant = os.getenv("ARCH", "base")
exp_name = os.getenv("EXP_NAME", None)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = exp_name or f"mnist_lrG{lr_g}_lrD{lr_d}_z{latent_dim}_bs{batch_size}_{arch_variant}_{timestamp}"
output_dir = os.path.join("results_mnist_arch", exp_name)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MNIST] Using device: {device}, arch={arch_variant}, saving to {output_dir}")


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        ch = [256, 128]
        self.model = nn.Sequential(
            nn.Linear(latent_dim, ch[0] * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (ch[0], 7, 7)),
            nn.Upsample(scale_factor=2),  # 7x7 -> 14x14

            # FIXED: Remove the duplicate layers that had wrong input channels
            nn.Conv2d(ch[0], ch[1], 3, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 14x14 -> 28x28

            nn.Conv2d(ch[1], 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z): return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        m = 2
        self.model = nn.Sequential(
            # Input: 1x28x28 -> 32*mx14x14
            nn.Conv2d(1, 32 * m, 3, 2, 1),
            nn.LeakyReLU(0.2),

            # 32*mx14x14 -> 64*mx7x7
            nn.Conv2d(32 * m, 64 * m, 3, 2, 1),
            nn.BatchNorm2d(64 * m),
            nn.LeakyReLU(0.2),

            # FIXED: Remove the duplicate layers that had wrong input channels
            nn.Flatten(),
            nn.Linear(64 * m * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x): return self.model(x)


# Test models before training
def verify_models():
    print("Verifying model architectures...")

    G_test = Generator(latent_dim).to(device)
    z_test = torch.randn(4, latent_dim, device=device)
    out_G = G_test(z_test)
    print(f"Generator: input {z_test.shape} -> output {out_G.shape}")

    D_test = Discriminator().to(device)
    img_test = torch.randn(4, 1, 28, 28, device=device)
    out_D = D_test(img_test)
    print(f"Discriminator: input {img_test.shape} -> output {out_D.shape}")

    print("✅ Models verified successfully!")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
data = datasets.MNIST(root="./data_mnist", train=True, transform=transform, download=True)
loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

print(f"Dataset loaded: {len(data)} images")

# Verify models before training
G, D = Generator(latent_dim).to(device), Discriminator().to(device)
verify_models()

print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
g_losses, d_losses = [], []
fixed_z = torch.randn(32, latent_dim, device=device)

print(f"Starting training for {epochs} epochs...")

for epoch in range(epochs):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    num_batches = 0

    for real, _ in loader:
        bs = real.size(0)
        real, valid, fake = real.to(device), torch.ones(bs, 1, device=device), torch.zeros(bs, 1, device=device)
        opt_D.zero_grad()
        d_real = criterion(D(real), valid)
        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z)
        d_fake = criterion(D(fake_imgs.detach()), fake)
        d_loss = d_real + d_fake
        d_loss.backward();
        opt_D.step()
        opt_G.zero_grad()
        g_loss = criterion(D(fake_imgs), valid)
        g_loss.backward();
        opt_G.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        num_batches += 1

    avg_g_loss = epoch_g_loss / num_batches
    avg_d_loss = epoch_d_loss / num_batches
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)

    print(f"Epoch {epoch + 1}/{epochs} | D {avg_d_loss:.3f} | G {avg_g_loss:.3f}")

    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            samples = G(fixed_z)
            vutils.save_image(samples, f"{output_dir}/fake_epoch{epoch + 1:03d}.png", normalize=True, nrow=8)

        plt.figure(figsize=(8, 5))
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.title(f"Losses up to epoch {epoch + 1}")
        plt.legend()
        plt.savefig(f"{output_dir}/loss_plot_epoch{epoch + 1:03d}.png")
        plt.close()

        pd.DataFrame({'G_loss': g_losses, 'D_loss': d_losses}).to_csv(f"{output_dir}/losses_epoch{epoch + 1:03d}.csv",
                                                                      index=False)

        torch.save({
            'epoch': epoch + 1,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'optG': opt_G.state_dict(),
            'optD': opt_D.state_dict()
        }, f"{output_dir}/checkpoint_epoch{epoch + 1:03d}.pt")

print(f"✅ Training completed! Results saved to {output_dir}")