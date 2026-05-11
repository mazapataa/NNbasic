"""
Toy Variational Autoencoder (VAE) from Semi-Scratch
====================================================

We use PyTorch only for:
  - Tensor operations & autograd (automatic differentiation)
  - Simple neural network layers

We implement ourselves:
  - The VAE model architecture
  - The reparameterization trick
  - The ELBO loss computation (reconstruction + KL divergence)
  - The training loop & visualization

This is intended as a LEARNING tool — not production code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Set randomness for reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# 2. Hyperparameters
# -----------------------------
BATCH_SIZE = 128
LATENT_DIM = 2          # Keep it 2D so we can easily visualise the latent space
EPOCHS = 20
LEARNING_RATE = 1e-3

# -----------------------------
# 3. Load a simple dataset: MNIST
#    (grayscale 28×28 handwritten digits)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten to 784-dim vector
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# For quick visualisation after training
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =====================================================================
# 4. VAE Model
# =====================================================================
# Intuition (read this carefully!):
#
# A standard autoencoder learns a deterministic mapping:
#     input  -->  encoder  -->  latent code z  -->  decoder  -->  reconstruction
#
# This is useful but has problems:
#   (a) The latent space is often irregular — gaps between clusters mean
#       interpolating gives garbage.
#   (b) You can't *generate* new data easily because you don't know the
#       distribution of z.
#
# A VAE fixes this by making the encoder output a *probability distribution*
# (typically a Gaussian) instead of a single point.
#
#   Encoder outputs:  μ (mean)  and  log(σ²)  [log-variance]
#
# Then we *sample* z from this distribution:   z ~ N(μ, σ²)
#
# But sampling is a stochastic operation — you can't backprop through it
# directly.  The **reparameterization trick** fixes this:
#
#   Instead of:   z = sample_from(N(μ, σ²))
#   We write:     z = μ + σ * ε    where  ε ~ N(0, 1)
#
# Now the randomness comes from an independent source (ε), and the
# gradient can flow through μ and σ.
#
# The loss function is the **ELBO** (Evidence Lower BOund):
#
#   ELBO = Reconstruction loss  -  KL divergence
#
#   - Reconstruction loss: how well can we reconstruct the input from z?
#     (binary cross-entropy for MNIST)
#
#   - KL divergence: how close is our learned distribution N(μ, σ²)
#     to the prior N(0, 1)?
#     This acts as a regulariser — it encourages the latent space to
#     be smooth and continuous.
#
# Maximising ELBO = minimising  -ELBO (which is what we do below).
# =====================================================================


class VAE(nn.Module):
    """
    A simple VAE with one hidden layer in both encoder and decoder.

    Architecture:
        Input (784) -> Hidden (400) -> μ (2), logσ² (2)
        z (2)       -> Hidden (400) -> Output (784)
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super().__init__()

        # ---- Encoder ----
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # log-variance

        # ---- Decoder ----
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    # -----------------------------
    # Encoder: returns μ and log(σ²)
    # -----------------------------
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ------------------------------------------------
    # Reparameterization trick
    # ------------------------------------------------
    def reparameterize(self, mu, logvar):
        """
        z = μ + σ ⊙ ε

        where  σ = exp(logvar / 2)   (standard deviation)
               ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)  # exp(logvar/2) = sqrt(var) = std
        eps = torch.randn_like(std)    # same shape as std, sampled from N(0,1)
        z = mu + eps * std
        return z

    # -----------------------------
    # Decoder: reconstruct from z
    # -----------------------------
    def decode(self, z):
        h = F.relu(self.fc3(z))
        recon = torch.sigmoid(self.fc4(h))  # sigmoid to squash to [0, 1]
        return recon

    # -----------------------------
    # Forward pass: end-to-end
    # -----------------------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# =====================================================================
# 5. Loss Function: the Negative ELBO
# =====================================================================
# We minimise:
#
#   Loss = -ELBO =  BCE(x, x̂)  +  β * KL(N(μ, σ²) || N(0, 1))
#
# where BCE is binary cross-entropy (reconstruction loss) and KL is
# the Kullback-Leibler divergence between the learned posterior and
# the standard normal prior.
#
# For two Gaussians, the KL divergence has a closed form:
#
#   KL(N(μ, σ²) || N(0, 1))
#       = -½ Σ (1 + log(σ²) - μ² - σ²)
#
# Derivation (for the curious):
#   KL(q || p) = ∫ q(x) log(q(x)/p(x)) dx
#   For Gaussians:
#     = -½ Σ ( 1 + log(σ_q²/σ_p²) - (μ_q - μ_p)² / σ_p² - σ_q²/σ_p² )
#
#   With σ_p = 1 and μ_p = 0:
#     = -½ Σ ( 1 + log(σ²) - μ² - σ² )
# =====================================================================

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Args:
        recon_x: reconstructed input (batch, 784)
        x:       original input      (batch, 784)
        mu:      mean of latent z    (batch, latent_dim)
        logvar:  log-variance of z   (batch, latent_dim)
        beta:    weight of KL term (β-VAE; β=1 is standard VAE)
    """
    # ---- Reconstruction loss ----
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # ---- KL divergence (closed form) ----
    # KL = -½ Σ (1 + log(σ²) - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD


# =====================================================================
# 6. Training
# =====================================================================

def train_vae(model, train_loader, epochs=20, lr=1e-3, beta=1.0):
    """Train the VAE and return the loss history."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_bce = 0
        total_kld = 0
        num_batches = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            # data shape: (batch, 784)
            data = data.view(data.size(0), -1)

            # ---- Forward ----
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar, beta)

            # ---- Backward ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Stats ----
            total_loss += loss.item()
            # Compute BCE and KLD separately for logging
            bce = F.binary_cross_entropy(recon_batch, data, reduction='sum').item()
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
            total_bce += bce
            total_kld += kld
            num_batches += 1

        # Average loss per data point (across all batches)
        avg_loss = total_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)

        history.append(avg_loss)

        print(f'Epoch {epoch:3d}/{epochs}  '
              f'Loss: {avg_loss:.4f}  '
              f'BCE: {avg_bce:.4f}  '
              f'KLD: {avg_kld:.4f}')

    return history


# =====================================================================
# 7. Visualisation
# =====================================================================

def plot_reconstructions(model, test_loader, num_examples=10):
    """
    Plot original digits alongside their VAE reconstructions.
    """
    model.eval()
    data, _ = next(iter(test_loader))
    data = data.view(data.size(0), -1)[:num_examples]

    with torch.no_grad():
        recon, mu, logvar = model(data)

    fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 1.5, 3))

    for i in range(num_examples):
        # Original
        axes[0, i].imshow(data[i].view(28, 28).numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstruction
        axes[1, i].imshow(recon[i].view(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.suptitle('VAE: Original vs Reconstructed Digits', fontsize=13)
    plt.tight_layout()
    plt.savefig('vae_reconstructions.png', dpi=150)
    plt.show()
    print("Saved: vae_reconstructions.png")


def plot_latent_space(model, test_loader):
    """
    Plot the 2D latent space, colour-coded by digit class.
    This shows how the VAE organises digits in the latent space.
    """
    model.eval()
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1)
            mu, _ = model.encode(data)   # just use the mean
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    all_mu = np.concatenate(all_mu, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1],
                          c=all_labels, cmap='tab10', alpha=0.7, s=8)
    plt.colorbar(scatter, label='Digit class')
    plt.title('VAE Latent Space (coloured by digit)', fontsize=13)
    plt.xlabel('z₁', fontsize=11)
    plt.ylabel('z₂', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('vae_latent_space.png', dpi=150)
    plt.show()
    print("Saved: vae_latent_space.png")


def generate_from_prior(model, num_examples=20):
    """
    Sample random points from the prior N(0, 1) and decode them.
    This is how a VAE generates *new* data!
    """
    model.eval()
    with torch.no_grad():
        # Sample from N(0, 1)
        z = torch.randn(num_examples, LATENT_DIM)
        samples = model.decode(z).view(-1, 28, 28).numpy()

    fig, axes = plt.subplots(2, num_examples // 2, figsize=(12, 3))
    axes = axes.ravel()
    for i in range(num_examples):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].axis('off')
    plt.suptitle('Generated Digits (sampled from prior N(0,1))', fontsize=13)
    plt.tight_layout()
    plt.savefig('vae_generated_samples.png', dpi=150)
    plt.show()
    print("Saved: vae_generated_samples.png")


def interpolate_latent(model, num_steps=10):
    """
    Interpolate between two random latent vectors to see smooth transitions.
    """
    model.eval()
    with torch.no_grad():
        z1 = torch.randn(1, LATENT_DIM)
        z2 = torch.randn(1, LATENT_DIM)

        alphas = np.linspace(0, 1, num_steps)
        zs = torch.tensor([
            (1 - alpha) * z1.numpy() + alpha * z2.numpy()
            for alpha in alphas
        ]).float()

        # Flatten the zs tensor properly
        zs = zs.view(num_steps, LATENT_DIM)
        samples = model.decode(zs).view(-1, 28, 28).numpy()

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
    for i in range(num_steps):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].axis('off')
    plt.suptitle('Latent Space Interpolation', fontsize=12)
    plt.tight_layout()
    plt.savefig('vae_interpolation.png', dpi=150)
    plt.show()
    print("Saved: vae_interpolation.png")


# =====================================================================
# 8. Run everything
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TOY VAE — Visualising the Latent Space")
    print("=" * 60)
    print(f"\nDataset: MNIST (784-dim input)")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # ---- Create model ----
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=LATENT_DIM)
    print(f"Model architecture:\n{model}\n")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # ---- Train ----
    history = train_vae(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    # ---- Quick loss plot ----
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Negative ELBO (per data point)')
    plt.title('Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig('vae_training_loss.png', dpi=150)
    plt.show()
    print("\nSaved: vae_training_loss.png")

    # ---- Visualise ----
    print("\n" + "=" * 60)
    print("Visualisations")
    print("=" * 60)
    plot_reconstructions(model, test_loader)
    plot_latent_space(model, test_loader)
    generate_from_prior(model, num_examples=10)
    interpolate_latent(model, num_steps=12)

    print("\n" + "=" * 60)
    print("DONE! All plots saved as PNG files.")
    print("=" * 60)
