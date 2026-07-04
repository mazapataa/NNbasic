from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn

#  Constants & priors


LAMBDA_P  = 20          # fixed parent count
MU_LO     = 3.0         # prior lower bound on μ
MU_HI     = 25.0        # prior upper bound on μ
SIG_LO    = 0.02        # prior lower bound on σ
SIG_HI    = 0.12        # prior upper bound on σ
GRID_SIZE = 32          # pixelisation resolution (G × G)
PK_BINS   = 10          # number of P(k) wavenumber bins



#  1. Thomas process simulator


def simulate_thomas(mu: float, sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Generate one Thomas-process galaxy catalog in [0, 1)^2.

    Parameters
    ----------
    mu    : mean offspring per cluster
    sigma : Gaussian scatter of offspring (box units)
    rng   : numpy Generator for reproducibility

    Returns
    -------
    positions : (N, 2) float32 array of galaxy positions
    """
    parents = rng.uniform(0.0, 1.0, (LAMBDA_P, 2))
    n_per   = rng.poisson(mu, LAMBDA_P)
    total   = int(n_per.sum())
    if total == 0:                              # edge case
        return rng.uniform(0, 1, (5, 2)).astype(np.float32)
    parent_rep = np.repeat(parents, n_per, axis=0)
    offsets    = rng.normal(0.0, sigma, (total, 2))
    positions  = np.mod(parent_rep + offsets, 1.0)
    return positions.astype(np.float32)


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw *n* parameter pairs (μ, σ) from the uniform prior."""
    mu    = rng.uniform(MU_LO,  MU_HI,  n)
    sigma = rng.uniform(SIG_LO, SIG_HI, n)
    return np.column_stack([mu, sigma]).astype(np.float32)


def analytical_pk(k: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Closed-form Thomas-process power spectrum (Eq. 2 of the report).
    """
    n_bar = LAMBDA_P * mu
    return 1.0 / n_bar + (mu**2 / LAMBDA_P) * np.exp(-k**2 * sigma**2)


# ═══════════════════════════════════════════════════════════════════════
#  2. Field representation — pixelisation
# ═══════════════════════════════════════════════════════════════════════

def pixelise(catalog: np.ndarray,
             grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Histogram galaxy positions onto a (G × G) grid and log-compress.

    The output has shape (1, G, G), ready for a Conv2d input channel.
    Permutation-invariant by construction.
    """
    H, _, _ = np.histogram2d(
        catalog[:, 0], catalog[:, 1],
        bins=grid_size, range=[[0.0, 1.0], [0.0, 1.0]],
    )
    return np.log1p(H).astype(np.float32)[np.newaxis]     # (1, G, G)


# ═══════════════════════════════════════════════════════════════════════
#  3. CNN embedding network
# ═══════════════════════════════════════════════════════════════════════

class CNNEmbedding(nn.Module):
    """
    Three-block ConvNet that compresses a (1, 32, 32) density field
    into a context vector of dimension *out_dim*.
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 32, 32) → (B, out_dim)."""
        h = self.conv(x)
        return self.fc(h.view(h.size(0), -1))



#  4. Normalizing flow — Real-NVP
class _CouplingNet(nn.Module):
    """Scale-and-shift MLP for one coupling layer."""

    def __init__(self, in_dim: int, ctx_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + ctx_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),           nn.Tanh(),
            nn.Linear(hidden, in_dim * 2),
        )

    def forward(self, x_fixed, ctx):
        out = self.net(torch.cat([x_fixed, ctx], dim=-1))
        s, t = out.chunk(2, dim=-1)
        return torch.tanh(s) * 2.0, t          # clamp scale


class RealNVP(nn.Module):
    """
    Real-NVP normalizing flow for 2-D parameters.

    
    """

    def __init__(self, ctx_dim: int = 32, hidden: int = 64,
                 n_layers: int = 5):
        super().__init__()
        self.n_layers = n_layers
        self.nets   = nn.ModuleList([
            _CouplingNet(1, ctx_dim, hidden) for _ in range(n_layers)
        ])
        self._fixed  = [i % 2 for i in range(n_layers)]
        self._transf = [(i + 1) % 2 for i in range(n_layers)]

    # ── forward: θ → z ───────────────────────────────────────────────
    def forward(self, theta: torch.Tensor,
                ctx: torch.Tensor):
        z   = theta.clone()
        ldj = torch.zeros(theta.shape[0], device=theta.device)
        for i, net in enumerate(self.nets):
            fd, td = self._fixed[i], self._transf[i]
            s, t   = net(z[:, fd:fd+1], ctx)
            z_new  = z[:, td:td+1] * torch.exp(s) + t
            ldj   += s.squeeze(-1)
            z = z.clone()
            z[:, td] = z_new.squeeze(-1)
        return z, ldj

    # ── inverse: z → θ ───────────────────────────────────────────────
    @torch.no_grad()
    def inverse(self, z: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        theta = z.clone()
        if ctx.shape[0] == 1:
            ctx = ctx.expand(z.shape[0], -1)
        for i in reversed(range(self.n_layers)):
            fd, td = self._fixed[i], self._transf[i]
            s, t   = self.nets[i](theta[:, fd:fd+1], ctx)
            th_new = (theta[:, td:td+1] - t) * torch.exp(-s)
            theta  = theta.clone()
            theta[:, td] = th_new.squeeze(-1)
        return theta

    def log_prob(self, theta, ctx):
        z, ldj = self.forward(theta, ctx)
        log_base = -0.5 * (z ** 2).sum(-1) - np.log(2 * np.pi)
        return log_base + ldj

    def sample(self, ctx, n_samples: int = 1000):
        z = torch.randn(n_samples, 2, device=ctx.device)
        return self.inverse(z, ctx.expand(n_samples, -1))



#  5. Combined model — CNN + Flow (end-to-end)

class FieldNPE(nn.Module):
    """
    Full field-level NPE model.


    """

    def __init__(self, ctx_dim: int = 32, hidden: int = 64,
                 n_layers: int = 5):
        super().__init__()
        self.cnn  = CNNEmbedding(out_dim=ctx_dim)
        self.flow = RealNVP(ctx_dim=ctx_dim, hidden=hidden,
                            n_layers=n_layers)

    def log_prob(self, theta: torch.Tensor,
                 field: torch.Tensor) -> torch.Tensor:
        """theta (B,2), field (B,1,G,G) → (B,) log-prob."""
        return self.flow.log_prob(theta, self.cnn(field))

    @torch.no_grad()
    def sample(self, field: torch.Tensor,
               n_samples: int = 5000) -> torch.Tensor:
        """field (1,1,G,G) → (n_samples, 2) normalised samples."""
        return self.flow.sample(self.cnn(field), n_samples)


#  6. Parameter normalisation  


def normalise(theta: np.ndarray) -> np.ndarray:
    """Map (μ, σ) from prior range to [-1, 1]²."""
    out = theta.copy().astype(np.float32)
    out[:, 0] = 2.0 * (theta[:, 0] - MU_LO)  / (MU_HI  - MU_LO)  - 1.0
    out[:, 1] = 2.0 * (theta[:, 1] - SIG_LO) / (SIG_HI - SIG_LO) - 1.0
    return out


def unnormalise(theta_n: np.ndarray) -> np.ndarray:
    """Map [-1, 1]² back to physical (μ, σ)."""
    out = theta_n.copy().astype(np.float32)
    out[:, 0] = (theta_n[:, 0] + 1.0) / 2.0 * (MU_HI  - MU_LO)  + MU_LO
    out[:, 1] = (theta_n[:, 1] + 1.0) / 2.0 * (SIG_HI - SIG_LO) + SIG_LO
    return out



#  7. Training utilities 

def generate_training_data(
    n_sim: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate *n_sim* Thomas catalogs and pixelise them.

    Returns
    -------
    theta_n : (n_sim, 2)            normalised parameters in [-1, 1]²
    fields  : (n_sim, 1, G, G)     log-count density fields
    """
    theta_raw = sample_prior(n_sim, rng)
    theta_n   = normalise(theta_raw)
    fields    = np.zeros((n_sim, 1, GRID_SIZE, GRID_SIZE),
                         dtype=np.float32)
    for i in range(n_sim):
        cat = simulate_thomas(float(theta_raw[i, 0]),
                              float(theta_raw[i, 1]), rng)
        fields[i] = pixelise(cat)
    return theta_n, fields


def train_model(
    model: FieldNPE,
    theta_n: np.ndarray,
    fields: np.ndarray,
    *,
    n_epochs: int   = 80,
    batch_size: int = 256,
    lr: float       = 5e-4,
    device: str     = "cpu",
    print_every: int = 20,
) -> list[float]:
    """
    Train the FieldNPE by minimising −log p(θ | field).

    Returns the per-epoch mean loss.
    """
    model.to(device).train()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=1e-5,
    )
    N     = len(theta_n)
    rng_t = np.random.default_rng(0)
    losses: list[float] = []

    for ep in range(n_epochs):
        idx = rng_t.permutation(N)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, N, batch_size):
            bi = idx[start : start + batch_size]
            if len(bi) < 4:
                continue
            th = torch.from_numpy(theta_n[bi]).to(device)
            fl = torch.from_numpy(fields[bi]).to(device)

            loss = -model.log_prob(th, fl).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            ep_loss   += loss.item()
            n_batches += 1

        sched.step()
        ml = ep_loss / max(n_batches, 1)
        losses.append(ml)

        if print_every and (ep + 1) % print_every == 0:
            print(f"  Epoch {ep+1:4d}/{n_epochs}:  loss = {ml:.4f}")

    return losses


#  8. Power-spectrum estimator (for the MCMC branch)


def compute_pk(catalog: np.ndarray,
               grid_size: int = GRID_SIZE,
               L: float = 1.0) -> np.ndarray:
    """
    Isotropic P(k) via 2-D FFT of the pixelised density field.

    Both the observed and model P(k) **must** pass through this same
    function so that window-function effects and normalisation cancel
    in the likelihood.
    """
    H, _, _ = np.histogram2d(
        catalog[:, 0], catalog[:, 1],
        bins=grid_size, range=[[0, L], [0, L]],
    )
    n_bar = H.mean()
    if n_bar == 0:
        return np.zeros(PK_BINS)

    delta = (H - n_bar) / n_bar
    ft    = np.fft.rfft2(delta)
    pk2d  = np.abs(ft) ** 2 / grid_size ** 4

    kx = np.fft.fftfreq(grid_size) * 2 * np.pi * grid_size
    ky = np.fft.rfftfreq(grid_size) * 2 * np.pi * grid_size
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    k_max   = np.pi * grid_size
    k_edges = np.linspace(2 * np.pi, k_max * 0.8, PK_BINS + 1)
    pk = np.zeros(PK_BINS)
    for i in range(PK_BINS):
        mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if mask.sum() > 0:
            pk[i] = pk2d[mask].mean()
    return pk


#  9. MCMC — simulation-based P(k) likelihood


def run_mcmc(
    x_obs: np.ndarray,
    *,
    n_walkers: int = 24,
    n_steps: int   = 1200,
    discard: int   = 300,
    n_avg: int     = 10,
    n_cov_sims: int = 500,
    fiducial: tuple[float, float] = (12.0, 0.05),
    verbose: bool  = True,
) -> np.ndarray:
    
    import emcee

    pk_obs = compute_pk(x_obs)

    # ── covariance at fiducial ───────────────────────────────────────
    if verbose:
        print(f"  Estimating P(k) covariance at fiducial "
              f"μ={fiducial[0]}, σ={fiducial[1]} …")
    rng_c   = np.random.default_rng(77)
    pk_sims = np.array([
        compute_pk(simulate_thomas(fiducial[0], fiducial[1], rng_c))
        for _ in range(n_cov_sims)
    ])
    cov     = np.cov(pk_sims, rowvar=False)
    hartlap = (n_cov_sims - PK_BINS - 2) / (n_cov_sims - 1)
    inv_cov = hartlap * np.linalg.inv(
        cov + 1e-12 * np.eye(PK_BINS)
    )

    # ── likelihood ───────────────────────────────────────────────────
    def log_like(theta):
        mu, sigma = theta
        if not (MU_LO <= mu <= MU_HI and SIG_LO <= sigma <= SIG_HI):
            return -np.inf
        rng = np.random.default_rng()
        pk_model = np.mean(
            [compute_pk(simulate_thomas(mu, sigma, rng))
             for _ in range(n_avg)],
            axis=0,
        )
        diff = pk_obs - pk_model
        return -0.5 * diff @ inv_cov @ diff

    # ── run sampler ──────────────────────────────────────────────────
    p0 = np.column_stack([
        np.random.uniform(MU_LO + 1, MU_HI - 1, n_walkers),
        np.random.uniform(SIG_LO + 0.005, SIG_HI - 0.005, n_walkers),
    ])
    if verbose:
        print(f"  Running emcee ({n_walkers}×{n_steps}, "
              f"{n_avg} sims/eval) …")

    sampler = emcee.EnsembleSampler(n_walkers, 2, log_like)
    sampler.run_mcmc(p0, n_steps, progress=verbose)
    chain = sampler.get_chain(discard=discard, flat=True)

    if verbose:
        print(f"  MCMC: {len(chain)} samples")
    return chain


# 10. NPE posterior sampling


@torch.no_grad()
def get_npe_samples(
    model: FieldNPE,
    x_obs: np.ndarray,
    n_samples: int = 5000,
    device: str    = "cpu",
) -> np.ndarray:
    """
    Draw posterior samples from the trained FieldNPE.

    Returns physical-space samples (μ, σ) that lie within the prior.
    """
    model.eval()
    field    = pixelise(x_obs)
    field_t  = torch.from_numpy(field[np.newaxis]).to(device)   # (1,1,G,G)
    samp_n   = model.sample(field_t, n_samples).cpu().numpy()   # normalised
    samp     = unnormalise(samp_n)                               # physical

    in_prior = (
        (samp[:, 0] >= MU_LO) & (samp[:, 0] <= MU_HI) &
        (samp[:, 1] >= SIG_LO) & (samp[:, 1] <= SIG_HI)
    )
    return samp[in_prior]