"""
thomas_npe.py  –  Beginner-Friendly Neural Posterior Estimation
================================================================
A self-contained, single-file implementation of the NPE pipeline from
Cranmer et al. (2020) applied to the Thomas Cluster Process.

WHAT THIS DOES (in plain English)
-----------------------------------
1. Simulate galaxy clusters using the Thomas point process.
   - A Thomas process has two parameters we want to recover:
       mu    : average number of galaxies per cluster  (e.g. 5–30)
       sigma : spatial spread of galaxies around each cluster centre (e.g. 0.02–0.12)

2. Summarise each simulated catalog with a small set of numbers
   (summary statistics: galaxy count, mean nearest-neighbor distance, …).

3. Train a normalizing flow (a neural network that learns probability
   distributions) to approximate  p(mu, sigma | summary_stats).

4. Validate the learned posterior with:
   - Posterior predictive check  (does the posterior explain the data?)
   - Simulation-Based Calibration (SBC) — are the error bars correct?)
   - Coverage test

HOW TO RUN
----------
    pip install numpy scipy matplotlib torch
    python thomas_npe.py

All outputs are saved to ./npe_outputs/.
Expected runtime: ~3 min on a laptop CPU.

REFERENCES
----------
Cranmer, Brehmer & Louppe (2020)  "The frontier of simulation-based inference"
    PNAS 117(48) 30055–30062   https://arxiv.org/abs/1911.01429

Thomas (1949) "A generalization of Poisson's binomial limit for use in ecology"
    Biometrika 36(1-2) 18–25

Talts et al. (2018) "Validating Bayesian inference algorithms with SBC"
    https://arxiv.org/abs/1804.06788
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, time
from pathlib import Path

# ── Scientific stack ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
from scipy.stats import chi2 as chi2_dist

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path("npe_outputs")
OUT.mkdir(exist_ok=True)

# =============================================================================
# SECTION 1 — THE THOMAS CLUSTER PROCESS
# =============================================================================
#
# A Thomas process generates galaxy positions in two steps:
#
#   Step 1. Scatter N_parent "cluster centres" uniformly in the unit square.
#           N_parent ~ Poisson(lambda_p), with lambda_p fixed (we don't
#           infer it — it's a nuisance we marginalise by fixing).
#
#   Step 2. Around each centre j, generate N_j galaxies,
#           where N_j ~ Poisson(mu) and each galaxy position is
#               x_ij = centre_j + Normal(0, sigma^2 * I)
#
# Parameters we INFER:
#   mu    in [MU_LO,  MU_HI]    mean galaxies per cluster
#   sigma in [SIG_LO, SIG_HI]   cluster spread (in [0,1] box units)
#
# Fixed nuisance:
#   lambda_p = 20  (number of cluster centres)
# =============================================================================

LAMBDA_P = 20       # fixed number of cluster parents
MU_LO,  MU_HI  = 3.0,  25.0   # prior on mu
SIG_LO, SIG_HI = 0.02,  0.12  # prior on sigma


def simulate_thomas(mu: float, sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Generate one Thomas-process galaxy catalog.

    Parameters
    ----------
    mu    : mean galaxies per cluster
    sigma : spatial spread of galaxies (box units)
    rng   : numpy random generator

    Returns
    -------
    catalog : (N, 2) array of galaxy positions in [0, 1]^2
              Returns at least 5 points (padded with uniform noise if needed).
    """
    # Cluster centres
    parents = rng.uniform(0.0, 1.0, (LAMBDA_P, 2))

    # Galaxy counts per cluster
    n_per_cluster = rng.poisson(mu, size=LAMBDA_P)
    total = int(n_per_cluster.sum())

    if total == 0:
        return rng.uniform(0.0, 1.0, (5, 2)).astype(np.float32)

    # Scatter galaxies around their parent
    parent_repeated = np.repeat(parents, n_per_cluster, axis=0)  # (total, 2)
    offsets = rng.normal(0.0, sigma, size=(total, 2))
    positions = parent_repeated + offsets

    # Wrap to [0, 1] (periodic boundary)
    positions = np.mod(positions, 1.0)

    return positions.astype(np.float32)


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n parameter pairs (mu, sigma) from the uniform prior.

    Returns
    -------
    theta : (n, 2) array, columns = [mu, sigma]
    """
    mu    = rng.uniform(MU_LO,  MU_HI,  size=n)
    sigma = rng.uniform(SIG_LO, SIG_HI, size=n)
    return np.column_stack([mu, sigma]).astype(np.float32)


# =============================================================================
# SECTION 2 — SUMMARY STATISTICS
# =============================================================================
#
# The NPE does NOT receive the raw point cloud.  Instead we compress each
# catalog into a small fixed-length vector of summary statistics.
#
# We use 5 statistics:
#   s0 : log(1 + N)                            — galaxy count (log-scaled)
#   s1 : mean nearest-neighbour distance        — small-scale clustering
#   s2 : std  nearest-neighbour distance        — clustering homogeneity
#   s3 : fraction of galaxies within r=0.05     — cluster compactness
#   s4 : Ripley's L(r) - r  at r=0.06          — excess clustering relative
#                                                  to Poisson baseline
#
# Why these?  Each statistic captures a different aspect of clustering that
# depends on (mu, sigma) in a different way, helping break degeneracies.
# =============================================================================

# Pre-computed random reference catalog for Ripley's L estimator
_REF_CAT = RNG.uniform(0.0, 1.0, (2000, 2))
_REF_TREE = cKDTree(_REF_CAT, boxsize=1.0)

RIPLEY_R   = 0.06    # radius for Ripley's L
COMPACT_R  = 0.05    # radius for fraction-within test
N_STATS    = 5       # total dimension of the summary vector


def compute_summary(catalog: np.ndarray) -> np.ndarray:
    """
    Compress a galaxy catalog into a 5-dimensional summary vector.

    Parameters
    ----------
    catalog : (N, 2) float array, positions in [0, 1)^2

    Returns
    -------
    stats : (5,) float32 array
    """
    N = len(catalog)

    # ── s0: log-galaxy count ─────────────────────────────────────────────────
    s0 = np.log1p(N) / 5.0       # normalise to roughly [0, 1]

    if N < 4:
        return np.array([s0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # ── s1, s2: nearest-neighbour distances ──────────────────────────────────
    tree = cKDTree(catalog, boxsize=1.0)
    # k=2 because k=1 returns the point itself (distance 0)
    nn_dist, _ = tree.query(catalog, k=2, workers=-1)
    nn_dist = nn_dist[:, 1]       # first *other* neighbour
    s1 = float(nn_dist.mean()) * 10.0   # scale up from ~0.01–0.10
    s2 = float(nn_dist.std())  * 10.0

    # ── s3: fraction within compact radius ───────────────────────────────────
    counts = tree.query_ball_point(catalog, COMPACT_R, return_length=True)
    s3 = float(np.mean(counts - 1)) / 20.0    # subtract self, normalise

    # ── s4: Ripley's L(r) - r (excess over Poisson) ──────────────────────────
    # K(r) = (area / N^2) * #{pairs within r}
    # L(r) = sqrt(K(r) / pi)
    pairs = tree.query_ball_point(catalog, RIPLEY_R, return_length=True)
    n_pairs = int(np.sum(pairs - 1))  # subtract self-pairs
    K = (1.0 / (N * (N - 1))) * n_pairs if N > 1 else 0.0
    L = np.sqrt(K / np.pi + 1e-12) - RIPLEY_R
    s4 = float(L) * 5.0

    return np.array([s0, s1, s2, s3, s4], dtype=np.float32)


# =============================================================================
# SECTION 3 — NORMALIZING FLOW (the neural network)
# =============================================================================
#
# A normalizing flow is a neural network f_phi that maps a simple base
# distribution (standard Normal) to a complex target distribution
# (our posterior) via a sequence of invertible transformations.
#
# We use a SIMPLE 3-layer Real-NVP (Real-valued Non-Volume Preserving):
#   - Each layer splits theta = (theta_0, theta_1)
#   - One part is transformed as:
#       z_1 = theta_1 * exp(s(theta_0, context)) + t(theta_0, context)
#     where s, t are small neural networks conditioned on the context
#     (the summary statistics).
#   - The log-determinant of the Jacobian is simply sum(s).
#
# WHY A FLOW?  Because we can compute log p(theta | x) in one forward pass
# and sample from p(theta | x) by inverting the transformation.
# =============================================================================

class CouplingNet(nn.Module):
    """
    Small MLP that computes (scale, shift) for one coupling layer.
    Input:  half of theta + summary statistics context
    Output: scale (s) and translation (t) for the other half of theta
    """
    def __init__(self, in_dim: int, context_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, in_dim * 2),   # outputs: [scale | shift]
        )

    def forward(self, x_fixed: torch.Tensor,
                context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x_fixed, context], dim=-1)
        out = self.net(inp)
        s, t = out.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0   # clamp scale to avoid explosions
        return s, t


class RealNVP(nn.Module):
    """
    3-layer Real-NVP normalizing flow for 2D parameters.

    The flow alternates which dimension is "fixed" vs "transformed":
      Layer 0: fix dim 0, transform dim 1
      Layer 1: fix dim 1, transform dim 0
      Layer 2: fix dim 0, transform dim 1

    This ensures both parameters are updated in every two layers.

    Parameters
    ----------
    context_dim : dimension of the summary statistics vector
    hidden      : hidden layer width in coupling networks
    n_layers    : number of coupling layers
    """
    def __init__(self, context_dim: int = N_STATS,
                 hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers
        # Each layer: one coupling net for each "fixed" dimension choice
        self.nets = nn.ModuleList([
            CouplingNet(1, context_dim, hidden)
            for _ in range(n_layers)
        ])
        # Which dim is fixed in each layer: alternates 0, 1, 0, 1, …
        self.fixed_dims  = [i % 2 for i in range(n_layers)]
        self.transf_dims = [(i + 1) % 2 for i in range(n_layers)]

    def forward(self, theta: torch.Tensor,
                context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: theta → latent z, compute log |det J|.

        Parameters
        ----------
        theta   : (B, 2) parameter tensor (normalised to [-1, 1]^2)
        context : (B, C) summary statistics

        Returns
        -------
        z       : (B, 2) latent variable
        log_det : (B,)   log |det J| = sum of log-scales
        """
        z = theta.clone()
        log_det = torch.zeros(theta.shape[0], device=theta.device)

        for i, net in enumerate(self.nets):
            fd = self.fixed_dims[i]
            td = self.transf_dims[i]

            x_fixed = z[:, fd:fd+1]             # (B, 1)
            x_transf = z[:, td:td+1]            # (B, 1)

            s, t = net(x_fixed, context)
            z_transf = x_transf * torch.exp(s) + t
            log_det += s.squeeze(-1)

            z = z.clone()
            z[:, td] = z_transf.squeeze(-1)

        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: latent z → theta (for sampling).

        Parameters
        ----------
        z       : (B, 2) latent samples from Normal(0, I)
        context : (B, C) or (1, C) summary statistics

        Returns
        -------
        theta : (B, 2) parameter samples
        """
        theta = z.clone()
        if context.shape[0] == 1:
            context = context.expand(z.shape[0], -1)

        # Layers in reverse order
        for i in reversed(range(self.n_layers)):
            net = self.nets[i]
            fd = self.fixed_dims[i]
            td = self.transf_dims[i]

            x_fixed  = theta[:, fd:fd+1]
            x_transf = theta[:, td:td+1]

            s, t = net(x_fixed, context)
            theta_transf = (x_transf - t) * torch.exp(-s)

            theta = theta.clone()
            theta[:, td] = theta_transf.squeeze(-1)

        return theta

    def log_prob(self, theta: torch.Tensor,
                 context: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(theta | context).

        Uses the change-of-variables formula:
            log p(theta) = log p_base(z) + log |det J|
        where p_base = Normal(0, I) and z = f(theta).

        Parameters
        ----------
        theta   : (B, 2) parameters (normalised)
        context : (B, C) summary statistics

        Returns
        -------
        log_prob : (B,) log-posterior values
        """
        z, log_det = self.forward(theta, context)
        # log Normal(0, I): -0.5 * ||z||^2 - D/2 * log(2pi)
        log_base = -0.5 * (z ** 2).sum(dim=-1) - np.log(2 * np.pi)
        return log_base + log_det

    @torch.no_grad()
    def sample(self, context: torch.Tensor,
               n_samples: int = 1000) -> torch.Tensor:
        """
        Sample theta ~ p(theta | context).

        Parameters
        ----------
        context  : (1, C) summary statistics for ONE observation
        n_samples: number of samples to draw

        Returns
        -------
        theta_samples : (n_samples, 2) parameter samples (normalised)
        """
        ctx = context.expand(n_samples, -1)
        z   = torch.randn(n_samples, 2, device=context.device)
        return self.inverse(z, ctx)


# =============================================================================
# SECTION 4 — PARAMETER NORMALISATION
# =============================================================================
# The flow works best when parameters live in [-1, 1].
# We linearly map:   mu    in [MU_LO,  MU_HI]  →  [-1, 1]
#                    sigma in [SIG_LO, SIG_HI] →  [-1, 1]

def normalise(theta: np.ndarray) -> np.ndarray:
    """Map (mu, sigma) from prior range to [-1, 1]^2."""
    out = theta.copy().astype(np.float32)
    out[:, 0] = 2.0 * (theta[:, 0] - MU_LO)  / (MU_HI  - MU_LO)  - 1.0
    out[:, 1] = 2.0 * (theta[:, 1] - SIG_LO) / (SIG_HI - SIG_LO) - 1.0
    return out


def unnormalise(theta_n: np.ndarray) -> np.ndarray:
    """Inverse of normalise: [-1, 1]^2 → (mu, sigma) prior range."""
    out = theta_n.copy().astype(np.float32)
    out[:, 0] = (theta_n[:, 0] + 1.0) / 2.0 * (MU_HI  - MU_LO)  + MU_LO
    out[:, 1] = (theta_n[:, 1] + 1.0) / 2.0 * (SIG_HI - SIG_LO) + SIG_LO
    return out


# =============================================================================
# SECTION 5 — TRAINING
# =============================================================================

def generate_training_data(n_sim: int,
                           rng: np.random.Generator
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n_sim (theta, summary_stats) pairs for training.

    Returns
    -------
    theta : (n_sim, 2) normalised parameter array
    stats : (n_sim, N_STATS) summary statistics array
    """
    theta_raw = sample_prior(n_sim, rng)          # (n_sim, 2)
    theta_n   = normalise(theta_raw)               # in [-1, 1]^2

    stats_list = []
    for i in range(n_sim):
        mu_i, sig_i = float(theta_raw[i, 0]), float(theta_raw[i, 1])
        cat  = simulate_thomas(mu_i, sig_i, rng)
        s    = compute_summary(cat)
        stats_list.append(s)

    stats = np.stack(stats_list)
    return theta_n, stats


def train_flow(flow: RealNVP,
               theta_n: np.ndarray,
               stats:   np.ndarray,
               n_epochs: int = 200,
               batch_size: int = 128,
               lr: float = 3e-4,
               device: str = "cpu") -> list[float]:
    """
    Train the normalizing flow by maximising log p(theta | stats).

    The loss is the NEGATIVE mean log-probability (we minimise it):
        L = -1/B * sum_i log p_phi(theta_i | stats_i)

    Parameters
    ----------
    flow       : RealNVP model
    theta_n    : (N, 2) normalised parameters
    stats      : (N, N_STATS) summary statistics
    n_epochs   : training epochs
    batch_size : mini-batch size
    lr         : learning rate
    device     : 'cpu' or 'cuda'

    Returns
    -------
    loss_history : list of per-epoch mean losses
    """
    flow.to(device).train()
    opt   = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs,
                                                         eta_min=1e-5)
    N = len(theta_n)
    rng_train = np.random.default_rng(0)
    losses = []

    for epoch in range(n_epochs):
        idx = rng_train.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            bi = idx[start: start + batch_size]
            if len(bi) < 4:
                continue

            th = torch.from_numpy(theta_n[bi]).to(device)
            st = torch.from_numpy(stats[bi]).to(device)

            log_p = flow.log_prob(th, st)       # (B,)
            loss  = -log_p.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=2.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()
        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)

        if (epoch + 1) % 50 == 0:
            lr_now = sched.get_last_lr()[0]
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: "
                  f"loss = {mean_loss:.4f}   lr = {lr_now:.2e}")

    return losses


# =============================================================================
# SECTION 6 — POSTERIOR INFERENCE
# =============================================================================

@torch.no_grad()
def get_posterior_samples(flow: RealNVP,
                          stats_obs: np.ndarray,
                          n_samples: int = 3000,
                          device: str = "cpu") -> np.ndarray:
    """
    Draw samples from the approximate posterior p(mu, sigma | x_obs).

    Parameters
    ----------
    flow      : trained RealNVP
    stats_obs : (N_STATS,) summary statistics of the observed catalog
    n_samples : number of posterior samples to draw

    Returns
    -------
    samples : (n_valid, 2) array with columns [mu, sigma] in physical units
    """
    flow.eval()
    st_t = torch.from_numpy(stats_obs[None]).to(device)   # (1, N_STATS)

    # Sample from the flow
    samples_n = flow.sample(st_t, n_samples=n_samples)    # (n_samples, 2)
    samples   = unnormalise(samples_n.cpu().numpy())       # back to (mu, sigma)

    # Keep only samples within the prior (discard out-of-bounds)
    in_prior = (
        (samples[:, 0] >= MU_LO)  & (samples[:, 0] <= MU_HI)  &
        (samples[:, 1] >= SIG_LO) & (samples[:, 1] <= SIG_HI)
    )
    return samples[in_prior]


# =============================================================================
# SECTION 7 — VALIDATION
# =============================================================================

def sbc_validation(flow: RealNVP,
                   n_tests: int = 200,
                   n_posterior: int = 300,
                   rng: np.random.Generator = None,
                   device: str = "cpu") -> dict:
    """
    Simulation-Based Calibration (Talts et al. 2018).

    IDEA: If the posterior is perfectly calibrated, then the rank of
    theta_true in a sorted list of posterior samples is uniformly
    distributed on {0, 1, …, L}.

    Algorithm for each test:
      1. Draw theta_true ~ prior
      2. Simulate catalog x ~ Thomas(theta_true)
      3. Compute posterior samples theta_1, …, theta_L ~ q(theta | x)
      4. Record rank = #{i : theta_i < theta_true}

    If ranks are uniform: well-calibrated.
    If ranks cluster low: posterior is overestimating (too wide).
    If ranks cluster high: posterior is underestimating (too narrow).

    Returns
    -------
    dict with 'ranks' (n_tests, 2) and 'p_values' (2,)
    """
    if rng is None:
        rng = np.random.default_rng(99)

    flow.eval()
    ranks = np.zeros((n_tests, 2), dtype=int)
    theta_true_all = sample_prior(n_tests, rng)

    print(f"  Running SBC with {n_tests} tests (L={n_posterior})…")
    for i in range(n_tests):
        mu_t, sig_t = float(theta_true_all[i, 0]), float(theta_true_all[i, 1])
        cat     = simulate_thomas(mu_t, sig_t, rng)
        stats   = compute_summary(cat)
        samples = get_posterior_samples(flow, stats, n_posterior, device)

        if len(samples) < 10:
            continue

        ranks[i, 0] = int((samples[:, 0] < mu_t).sum())
        ranks[i, 1] = int((samples[:, 1] < sig_t).sum())

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_tests}")

    # Chi-squared test for uniformity
    n_bins = 10
    p_vals = []
    for d in range(2):
        counts, _ = np.histogram(ranks[:, d], bins=n_bins,
                                  range=(0, n_posterior))
        expected  = n_tests / n_bins
        chi2_stat = float(((counts - expected) ** 2 / expected).sum())
        p_vals.append(chi2_dist.sf(chi2_stat, n_bins - 1))

    return {"ranks": ranks, "p_values": np.array(p_vals), "n_posterior": n_posterior}


def coverage_test(flow: RealNVP,
                  n_tests: int = 200,
                  n_posterior: int = 300,
                  alpha_levels: tuple = (0.5, 0.68, 0.9, 0.95),
                  rng: np.random.Generator = None,
                  device: str = "cpu") -> dict:
    """
    Expected coverage test (Hermans et al. 2022).

    For each nominal level alpha, check that the true parameter falls
    inside the alpha-credible interval in fraction alpha of trials.

    Returns
    -------
    dict with 'alpha_levels' and 'empirical_coverage' (n_alpha, 2)
    """
    if rng is None:
        rng = np.random.default_rng(77)

    flow.eval()
    alpha_arr = np.array(alpha_levels)
    in_ci     = np.zeros((n_tests, len(alpha_arr), 2), dtype=bool)
    theta_true_all = sample_prior(n_tests, rng)

    print(f"  Running coverage test with {n_tests} tests…")
    for i in range(n_tests):
        mu_t, sig_t = float(theta_true_all[i, 0]), float(theta_true_all[i, 1])
        cat     = simulate_thomas(mu_t, sig_t, rng)
        stats   = compute_summary(cat)
        samples = get_posterior_samples(flow, stats, n_posterior, device)

        if len(samples) < 10:
            continue

        for j, alpha in enumerate(alpha_arr):
            half = (1.0 - alpha) / 2.0
            lo_mu,  hi_mu  = np.quantile(samples[:, 0], [half, 1-half])
            lo_sig, hi_sig = np.quantile(samples[:, 1], [half, 1-half])
            in_ci[i, j, 0] = lo_mu  <= mu_t  <= hi_mu
            in_ci[i, j, 1] = lo_sig <= sig_t <= hi_sig

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_tests}")

    return {
        "alpha_levels":       alpha_arr,
        "empirical_coverage": in_ci.mean(axis=0),  # (n_alpha, 2)
    }


# =============================================================================
# SECTION 8 — PLOTTING
# =============================================================================

def plot_example_catalogs(rng):
    """Show what Thomas-process catalogs look like for different parameters."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Thomas Cluster Process: example realizations",
                 fontsize=14, fontweight="bold")

    configs = [
        (5,  0.03, "μ=5,  σ=0.03\n(few tight clusters)"),
        (15, 0.03, "μ=15, σ=0.03\n(many tight clusters)"),
        (5,  0.09, "μ=5,  σ=0.09\n(few loose clusters)"),
        (15, 0.09, "μ=15, σ=0.09\n(many loose clusters)"),
    ]

    for col, (mu, sigma, title) in enumerate(configs):
        for row in range(2):
            cat = simulate_thomas(mu, sigma, rng)
            ax  = axes[row, col]
            ax.scatter(cat[:, 0], cat[:, 1], s=4, alpha=0.6,
                       c="#1B4965")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.95, f"N={len(cat)}", transform=ax.transAxes,
                    fontsize=8, va="top",
                    bbox=dict(fc="white", alpha=0.7, boxstyle="round"))
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "1_example_catalogs.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 1_example_catalogs.png")


def plot_training_loss(losses: list):
    """Plot the NPE training loss curve."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(losses, lw=1.5, c="#1B4965", alpha=0.8, label="Training loss")

    # Smoothed curve
    window = max(5, len(losses) // 15)
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        x_smooth = np.arange(len(smooth)) + window // 2
        ax.plot(x_smooth, smooth,
                lw=2.5, c="#C1121F", label=f"Moving average ({window})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("−log p(θ | x)  [lower is better]", fontsize=12)
    ax.set_title("NPE Training: Loss over Epochs", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "2_training_loss.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 2_training_loss.png")


def plot_posterior(samples: np.ndarray,
                   mu_true: float, sigma_true: float):
    """
    Plot the inferred posterior for (mu, sigma).
    Shows both marginals and the joint 2D posterior.
    """
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.4, figure=fig)

    colors = {"npe": "#1B4965", "true": "#C1121F"}

    # ── Marginal: mu ──────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.hist(samples[:, 0], bins=40, density=True,
             color=colors["npe"], alpha=0.7, edgecolor="white", lw=0.3)
    ax0.axvline(mu_true, c=colors["true"], lw=2.5, ls="--",
                label=f"True μ = {mu_true:.1f}")
    ax0.set_xlabel("μ  (mean galaxies / cluster)", fontsize=12)
    ax0.set_ylabel("Posterior density", fontsize=11)
    ax0.set_title("Marginal posterior: μ", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.3)

    mu_lo, mu_hi = np.quantile(samples[:, 0], [0.025, 0.975])
    ax0.axvspan(mu_lo, mu_hi, alpha=0.15, color=colors["npe"],
                label="95% CI")
    ax0.text(0.05, 0.92,
             f"Mean: {samples[:,0].mean():.2f}\n"
             f"Std:  {samples[:,0].std():.2f}",
             transform=ax0.transAxes, fontsize=9,
             bbox=dict(fc="lightyellow", ec="gold", alpha=0.9))

    # ── Marginal: sigma ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.hist(samples[:, 1], bins=40, density=True,
             color="#4CAF50", alpha=0.7, edgecolor="white", lw=0.3)
    ax1.axvline(sigma_true, c=colors["true"], lw=2.5, ls="--",
                label=f"True σ = {sigma_true:.3f}")
    ax1.set_xlabel("σ  (cluster spread, box units)", fontsize=12)
    ax1.set_title("Marginal posterior: σ", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    sig_lo, sig_hi = np.quantile(samples[:, 1], [0.025, 0.975])
    ax1.axvspan(sig_lo, sig_hi, alpha=0.15, color="#4CAF50")
    ax1.text(0.05, 0.92,
             f"Mean: {samples[:,1].mean():.4f}\n"
             f"Std:  {samples[:,1].std():.4f}",
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(fc="lightyellow", ec="gold", alpha=0.9))

    # ── Joint 2D posterior ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.scatter(samples[:, 0], samples[:, 1],
                s=2, alpha=0.2, c=colors["npe"], rasterized=True)
    ax2.plot(mu_true, sigma_true, "*", ms=14, c=colors["true"],
             zorder=5, label="True parameters", markeredgecolor="white",
             markeredgewidth=0.5)
    ax2.set_xlabel("μ  (mean galaxies / cluster)", fontsize=12)
    ax2.set_ylabel("σ  (cluster spread)", fontsize=12)
    ax2.set_title("Joint posterior p(μ, σ | x)", fontsize=12,
                  fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_xlim(MU_LO, MU_HI)
    ax2.set_ylim(SIG_LO, SIG_HI)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(
        f"NPE Posterior  |  True: μ={mu_true:.1f}, σ={sigma_true:.3f}",
        fontsize=13, fontweight="bold"
    )
    plt.savefig(OUT / "3_posterior.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 3_posterior.png")


def plot_sbc(sbc_result: dict):
    """
    Plot SBC rank histograms.
    A uniform histogram = well-calibrated posterior.
    """
    ranks      = sbc_result["ranks"]
    p_vals     = sbc_result["p_values"]
    n_post     = sbc_result["n_posterior"]
    n_tests    = ranks.shape[0]
    n_bins     = 10
    expected   = n_tests / n_bins

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Simulation-Based Calibration (SBC)",
                 fontsize=14, fontweight="bold")

    param_names = ["μ  (mean galaxies/cluster)",
                   "σ  (cluster spread)"]
    colors      = ["#6A4C93", "#1982C4"]

    for d, (ax, name, col) in enumerate(zip(axes, param_names, colors)):
        counts, edges = np.histogram(ranks[:, d], bins=n_bins,
                                      range=(0, n_post))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(bin_centers, counts,
               width=(edges[1] - edges[0]) * 0.85,
               color=col, alpha=0.75, edgecolor="black", lw=0.8)

        # Expected uniform level + 2-sigma Poisson band
        ax.axhline(expected, c="red", ls="--", lw=2,
                   label=f"Uniform ({expected:.0f})")
        ax.axhspan(expected - 2 * np.sqrt(expected),
                   expected + 2 * np.sqrt(expected),
                   alpha=0.2, color="red")

        p = p_vals[d]
        verdict = "✓ PASS" if p > 0.05 else "✗ FAIL"
        ax.set_title(f"{name}\nχ² p-value = {p:.3f}   {verdict}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Rank of θ_true in posterior samples", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT / "4_sbc.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 4_sbc.png")


def plot_coverage(cov_result: dict):
    """
    Plot expected vs. empirical coverage.
    Perfect calibration = points lie on the diagonal y = x.
    """
    alphas = cov_result["alpha_levels"]
    em_cov = cov_result["empirical_coverage"]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")
    ax.fill_between([0, 1], [-0.05, -0.05], [0.05, 0.05],
                    alpha=0.0)   # placeholder
    ax.fill_between(alphas,
                    alphas - 0.05, alphas + 0.05,
                    alpha=0.15, color="gray", label="±5% tolerance")

    ax.plot(alphas, em_cov[:, 0], "o-", lw=2.5, ms=9,
            c="#6A4C93", label="μ  (mean galaxies/cluster)")
    ax.plot(alphas, em_cov[:, 1], "s-", lw=2.5, ms=9,
            c="#1982C4", label="σ  (cluster spread)")

    ax.set_xlabel("Nominal coverage level α", fontsize=13)
    ax.set_ylabel("Empirical coverage", fontsize=13)
    ax.set_title("Expected Coverage Test", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.3, 1.1)

    plt.tight_layout()
    plt.savefig(OUT / "5_coverage.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 5_coverage.png")


def plot_posterior_predictive(flow: RealNVP,
                              x_obs: np.ndarray,
                              stats_obs: np.ndarray,
                              mu_true: float,
                              sigma_true: float,
                              rng: np.random.Generator,
                              device: str = "cpu"):
    """
    Posterior predictive check:
    Draw (mu, sigma) from the posterior, simulate a new catalog, and
    compare its summary statistics to those of the observed catalog.

    If the posterior is correct, the simulated catalogs should look
    statistically similar to x_obs.
    """
    samples = get_posterior_samples(flow, stats_obs, 500, device)
    if len(samples) < 20:
        print("  Not enough posterior samples for PPC. Skipping.")
        return

    # Simulate new catalogs from posterior draws
    sim_stats = []
    for mu_s, sig_s in samples[:200]:
        cat = simulate_thomas(float(mu_s), float(sig_s), rng)
        sim_stats.append(compute_summary(cat))
    sim_stats = np.array(sim_stats)

    stat_names = [
        "log(1+N) / 5\n(galaxy count)",
        "Mean NN dist × 10\n(clustering scale)",
        "Std NN dist × 10\n(clustering variability)",
        "Cluster fraction / 20\n(compactness)",
        "Ripley L - r × 5\n(excess clustering)",
    ]

    fig, axes = plt.subplots(1, N_STATS, figsize=(18, 4))
    fig.suptitle("Posterior Predictive Check\n"
                 "(simulated catalogs should match observed stats)",
                 fontsize=13, fontweight="bold")

    for i, (ax, name) in enumerate(zip(axes, stat_names)):
        ax.hist(sim_stats[:, i], bins=25, density=True,
                color="#1B4965", alpha=0.6, label="Posterior predictive")
        ax.axvline(stats_obs[i], c="#C1121F", lw=2.5, ls="--",
                   label=f"Observed: {stats_obs[i]:.3f}")
        ax.set_title(name, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUT / "6_posterior_predictive.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("Saved: 6_posterior_predictive.png")


# =============================================================================
# SECTION 9 — MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print(" Neural Posterior Estimation for the Thomas Cluster Process")
    print(" Based on: Cranmer, Brehmer & Louppe (2020), PNAS")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Device: {device}")
    print(f" Output directory: {OUT.resolve()}\n")

    t_start = time.time()

    # ─── Step 1: Show example catalogs ───────────────────────────────────────
    print("─" * 65)
    print("STEP 1: Visualise the Thomas cluster process")
    print("─" * 65)
    rng = np.random.default_rng(SEED)
    plot_example_catalogs(rng)

    # ─── Step 2: Generate training data ──────────────────────────────────────
    N_TRAIN = 3_000    # number of (theta, summary) pairs
    print(f"\n{'─'*65}")
    print(f"STEP 2: Generating {N_TRAIN} training simulations…")
    print("─" * 65)
    t0 = time.time()
    theta_train, stats_train = generate_training_data(N_TRAIN, rng)
    print(f" Done in {time.time()-t0:.1f}s  "
          f"| theta shape: {theta_train.shape}, "
          f"stats shape: {stats_train.shape}")

    # ─── Step 3: Build and train the flow ────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 3: Training the normalizing flow (RealNVP)…")
    print("─" * 65)
    flow = RealNVP(context_dim=N_STATS, hidden=64, n_layers=3)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f" Model parameters: {n_params:,}")

    losses = train_flow(flow, theta_train, stats_train,
                        n_epochs=300, batch_size=128, lr=3e-4,
                        device=device)
    plot_training_loss(losses)

    # ─── Step 4: Generate a mock observation ─────────────────────────────────
    MU_TRUE    = 12.0
    SIGMA_TRUE = 0.05

    print(f"\n{'─'*65}")
    print(f"STEP 4: Generating mock observation")
    print(f" True parameters: μ = {MU_TRUE}, σ = {SIGMA_TRUE}")
    print("─" * 65)

    rng_obs = np.random.default_rng(123)
    x_obs   = simulate_thomas(MU_TRUE, SIGMA_TRUE, rng_obs)
    stats_obs = compute_summary(x_obs)
    print(f" Observed catalog: N = {len(x_obs)} galaxies")
    print(f" Summary stats: {stats_obs}")

    # ─── Step 5: Infer the posterior ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 5: Posterior inference")
    print("─" * 65)
    samples = get_posterior_samples(flow, stats_obs,
                                    n_samples=5000, device=device)
    print(f" Posterior samples (in prior): {len(samples)}")
    print(f" μ  posterior: {samples[:,0].mean():.2f} ± {samples[:,0].std():.2f}"
          f"  (true: {MU_TRUE})")
    print(f" σ  posterior: {samples[:,1].mean():.4f} ± {samples[:,1].std():.4f}"
          f"  (true: {SIGMA_TRUE})")

    plot_posterior(samples, MU_TRUE, SIGMA_TRUE)

    # ─── Step 6: Posterior predictive check ──────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 6: Posterior predictive check")
    print("─" * 65)
    plot_posterior_predictive(flow, x_obs, stats_obs,
                              MU_TRUE, SIGMA_TRUE, rng, device)

    # ─── Step 7: SBC ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 7: Simulation-Based Calibration (SBC)")
    print("─" * 65)
    rng_val = np.random.default_rng(55)
    sbc_res = sbc_validation(flow, n_tests=200, n_posterior=300,
                              rng=rng_val, device=device)
    plot_sbc(sbc_res)
    print(f" SBC p-values:  μ = {sbc_res['p_values'][0]:.4f}, "
          f"σ = {sbc_res['p_values'][1]:.4f}")
    sbc_ok = (sbc_res["p_values"] > 0.05).all()
    print(f" Calibration: {'PASSED ✓' if sbc_ok else 'FAILED ✗'}")

    # ─── Step 8: Coverage test ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 8: Expected coverage test")
    print("─" * 65)
    rng_cov = np.random.default_rng(77)
    cov_res = coverage_test(flow, n_tests=200, n_posterior=300,
                             rng=rng_cov, device=device)
    plot_coverage(cov_res)

    # ─── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(" PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f" Total runtime: {elapsed/60:.1f} min")
    print(f" True parameters: μ = {MU_TRUE},  σ = {SIGMA_TRUE}")
    print(f" NPE estimate:    μ = {samples[:,0].mean():.2f} ± {samples[:,0].std():.2f}")
    print(f"                  σ = {samples[:,1].mean():.4f} ± {samples[:,1].std():.4f}")
    print(f" SBC: {'PASS ✓' if sbc_ok else 'FAIL ✗'}  "
          f"(p_mu={sbc_res['p_values'][0]:.3f}, "
          f"p_sigma={sbc_res['p_values'][1]:.3f})")
    print(f"\n All figures saved to: {OUT.resolve()}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()