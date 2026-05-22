"""
poc_core_course.py
==================
Simplified Neural Posterior Estimation for Galaxy Clustering
Course Project Implementation

Based on: Cranmer et al. (2020) "The frontier of simulation-based inference"
          Proceedings of the National Academy of Sciences, 117(48), 30055-30062

This module implements a minimal but complete simulation-based inference pipeline
to demonstrate that neural networks can constrain parameters invisible to 
traditional two-point correlation function analysis.

Components:
-----------
1. Thomas (Neyman-Scott) cluster process simulator
2. Landy-Szalay two-point correlation function estimator  
3. Conditional normalizing flow (simplified RealNVP, 3 layers)
4. Simulation-Based Calibration (SBC) diagnostic

Author: Course Project
Date: 2026
"""

from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.stats import chi2
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Parameter ranges (in log10 space)
THETA_LO = np.array([1.0, 0.7], dtype=np.float32)   # [log10(λ_p), log10(μ)]
THETA_HI = np.array([2.0, 1.5], dtype=np.float32)

# Nuisance parameter (survey completeness)
ETA_LO, ETA_HI = 0.55, 1.0

# Fixed cluster scale (not inferred)
SIGMA_C = 0.03

# Minimum points per realization
N_MIN = 25

# Radial bins for correlation function
R_EDGES = np.linspace(0.0, 0.22, 9)

# Pre-computed random catalog for Landy-Szalay
_RAND_RNG = np.random.default_rng(1)
_RAND = _RAND_RNG.uniform(0, 1, size=(1400, 2))
_RTREE = cKDTree(_RAND, boxsize=1.0)
_RR_CUM = _RTREE.count_neighbors(_RTREE, R_EDGES[1:])


# ============================================================================
# 1. THOMAS PROCESS SIMULATOR
# ============================================================================

def simulate_thomas(
    theta: npt.NDArray[np.float32],
    eta: float,
    rng: np.random.Generator,
    misspecified: bool = False
) -> npt.NDArray[np.float32]:
    """
    Simulate a 2D periodic Thomas (Neyman-Scott) cluster process.
    
    This is a doubly-stochastic point process where:
    1. Parent points form a homogeneous Poisson process (intensity λ_p)
    2. Each parent generates Poisson(μ) offspring
    3. Offspring scatter with Gaussian dispersion (scale σ_c)
    
    Parameters
    ----------
    theta : array of shape (2,)
        Parameters [log10(λ_p), log10(μ)]
    eta : float
        Survey completeness fraction in [0, 1]
    rng : numpy.random.Generator
        Random number generator for reproducibility
    misspecified : bool, default=False
        If True, add interloper population for testing model misspecification
        
    Returns
    -------
    points : array of shape (N, 2)
        Point coordinates in unit square [0, 1)²
        
    Notes
    -----
    The two-point correlation function ξ(r) depends only on λ_p and σ_c,
    but NOT on μ. This makes μ an ideal test case for field-level inference.
    
    References
    ----------
    Thomas, M. (1949). "A generalization of Poisson's binomial limit"
    Neyman, J., & Scott, E. L. (1958). "Statistical approach to problems of cosmology"
    """
    # Convert from log10 space
    lam_p = 10.0 ** theta[0]  # Parent intensity (clusters per unit area)
    mu = 10.0 ** theta[1]      # Mean offspring per cluster
    
    # Attempt to generate a valid realization (up to 50 tries)
    for attempt in range(50):
        # Sample number of parent points from Poisson
        n_parents = rng.poisson(lam_p)
        if n_parents == 0:
            continue
            
        # Uniformly distribute parent points in [0,1)²
        parents = rng.uniform(0.0, 1.0, size=(n_parents, 2))
        
        # Sample offspring counts for each parent
        n_offspring = rng.poisson(mu, size=n_parents)
        if n_offspring.sum() < 5:
            continue
            
        # Generate offspring around parents with Gaussian scatter
        points = np.repeat(parents, n_offspring, axis=0)
        points = np.mod(
            points + rng.normal(0.0, SIGMA_C, size=points.shape), 
            1.0  # Periodic boundary conditions
        )
        
        # Apply survey completeness (random thinning)
        if eta < 1.0:
            mask = rng.random(len(points)) < eta
            points = points[mask]
            
        # Optional: add misspecified interloper population
        if misspecified:
            n_big = max(rng.poisson(4), 1)
            big_parents = rng.uniform(0, 1, size=(n_big, 2))
            n_big_offspring = rng.poisson(0.8 * len(points) / n_big, size=n_big)
            if n_big_offspring.sum() > 0:
                interlopers = np.repeat(big_parents, n_big_offspring, axis=0)
                interlopers = np.mod(
                    interlopers + rng.normal(0, 0.13, interlopers.shape),
                    1.0
                )
                points = np.vstack([points, interlopers])
                
        # Check if we have enough points
        if len(points) >= N_MIN:
            return points.astype(np.float32)
            
    # Fallback: uniform random distribution if generation fails
    return rng.uniform(0, 1, size=(N_MIN, 2)).astype(np.float32)


def sample_prior(
    n: int,
    rng: np.random.Generator
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Sample parameters from uniform prior.
    
    Parameters
    ----------
    n : int
        Number of samples
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    theta : array of shape (n, 2)
        Physical parameters
    eta : array of shape (n,)
        Completeness values
    """
    theta = rng.uniform(THETA_LO, THETA_HI, size=(n, 2)).astype(np.float32)
    eta = rng.uniform(ETA_LO, ETA_HI, size=n).astype(np.float32)
    return theta, eta


def generate_batch(
    batch_size: int,
    rng: np.random.Generator,
    misspecified: bool = False
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generate a batch of point clouds with parameters.
    
    Parameters
    ----------
    batch_size : int
        Number of realizations
    rng : numpy.random.Generator
        Random number generator
    misspecified : bool
        Whether to include misspecified interlopers
        
    Returns
    -------
    clouds : list of Tensor
        Point clouds, each of shape (N_i, 2)
    theta : Tensor of shape (batch_size, 2)
        Parameters
    eta : Tensor of shape (batch_size,)
        Completeness values
    """
    theta, eta = sample_prior(batch_size, rng)
    clouds = []
    for i in range(batch_size):
        pts = simulate_thomas(theta[i], eta[i], rng, misspecified)
        clouds.append(torch.from_numpy(pts))
    return clouds, torch.from_numpy(theta), torch.from_numpy(eta)


# ============================================================================
# 2. LANDY-SZALAY TWO-POINT CORRELATION FUNCTION
# ============================================================================

def landy_szalay(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """
    Compute two-point correlation function using Landy-Szalay estimator.
    
    The estimator is:
        ξ(r) = (DD - 2DR + RR) / RR
    
    where DD, DR, RR are normalized pair counts for data-data, data-random,
    and random-random pairs respectively.
    
    Parameters
    ----------
    points : array of shape (N, 2)
        Point coordinates in [0,1)²
        
    Returns
    -------
    xi : array of shape (len(R_EDGES)-1,)
        Correlation function in radial bins
        
    Notes
    -----
    Uses periodic boundary conditions via scipy.spatial.cKDTree with boxsize=1.0.
    The random catalog is pre-computed globally for efficiency.
    
    References
    ----------
    Landy, S. D., & Szalay, A. S. (1993). ApJ, 412, 64-71.
    """
    nD, nR = points.shape[0], _RAND.shape[0]
    dt = cKDTree(points, boxsize=1.0)
    
    # Count pairs in cumulative bins, then take differences
    dd = np.diff(np.concatenate([[0], dt.count_neighbors(dt, R_EDGES[1:])])).astype(float)
    dr = np.diff(np.concatenate([[0], dt.count_neighbors(_RTREE, R_EDGES[1:])])).astype(float)
    rr = np.diff(np.concatenate([[0], _RR_CUM])).astype(float)
    
    # Remove self-pairs from first bin
    dd[0] = max(dd[0] - nD, 0)
    rr[0] = max(rr[0] - nR, 0)
    
    # Normalize by total number of pairs
    dd /= nD * (nD - 1) / 2.0
    rr /= nR * (nR - 1) / 2.0
    dr /= nD * nR
    
    # Landy-Szalay estimator
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.where(rr > 0, (dd - 2*dr + rr) / rr, 0.0)
    
    return np.nan_to_num(xi)


# ============================================================================
# 3. SIMPLIFIED NEURAL ARCHITECTURE
# ============================================================================

class SimpleDeepSet(nn.Module):
    """
    Simplified permutation-invariant encoder for point clouds.
    
    Architecture:
    1. Per-point MLP: R² → R⁶⁴
    2. Sum pooling (permutation-invariant aggregation)
    3. Context MLP: R⁶⁴⁺³ → R⁶⁴
    
    Parameters
    ----------
    point_dim : int
        Dimension of each point (default: 2)
    cond_dim : int
        Dimension of conditioning (default: 3 for θ₀, θ₁, η)
    hidden_dim : int
        Hidden layer dimension (default: 64)
        
    References
    ----------
    Zaheer et al. (2017). "Deep Sets." NeurIPS.
    """
    
    def __init__(
        self,
        point_dim: int = 2,
        cond_dim: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        # Per-point feature extraction
        self.point_net = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Context network (combines pooled features with conditioning)
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        x: List[torch.Tensor],
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode point clouds into fixed-dimensional representations.
        
        Parameters
        ----------
        x : list of Tensor
            Point clouds, each of shape (N_i, 2)
        condition : Tensor of shape (B, 3)
            Conditioning variables [θ₀, θ₁, η]
            
        Returns
        -------
        context : Tensor of shape (B, hidden_dim)
            Encoded representations
        """
        B = len(x)
        contexts = []
        
        for i in range(B):
            if x[i].numel() == 0:
                # Handle empty point clouds
                contexts.append(
                    torch.zeros(1, self.context_net[-1].out_features,
                               device=condition.device)
                )
                continue
                
            # Process points independently
            point_features = self.point_net(x[i].to(condition.device))
            
            # Sum pooling (permutation-invariant)
            pooled = point_features.sum(dim=0, keepdim=True)
            
            # Combine with conditioning
            combined = torch.cat([pooled, condition[i:i+1]], dim=1)
            context = self.context_net(combined)
            contexts.append(context)
            
        return torch.cat(contexts, dim=0)


class AffineCoupling(nn.Module):
    """
    Single affine coupling layer for RealNVP.
    
    Transformation:
        y_masked = x_masked
        y_free = x_free * exp(s(x_masked, c)) + t(x_masked, c)
        
    where s and t are neural networks, c is context, and the mask
    determines which dimensions are frozen vs transformed.
    
    Parameters
    ----------
    mask : Tensor
        Binary mask (0s and 1s)
    context_dim : int
        Dimension of context vector
    hidden_dim : int
        Hidden layer dimension
        
    References
    ----------
    Dinh et al. (2016). "Density estimation using Real NVP." arXiv:1605.08803.
    """
    
    def __init__(
        self,
        mask: torch.Tensor,
        context_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        
        # Scale network
        self.scale_net = nn.Sequential(
            nn.Linear(2 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Translation network
        self.translate_net = nn.Sequential(
            nn.Linear(2 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coupling transformation.
        
        Parameters
        ----------
        x : Tensor of shape (N, 2)
            Input points
        context : Tensor of shape (N, context_dim) or (1, context_dim)
            Context vector
        reverse : bool
            If True, apply inverse transformation
            
        Returns
        -------
        y : Tensor of shape (N, 2)
            Transformed points
        log_det : Tensor of shape (N,)
            Log-determinant of Jacobian
        """
        # Broadcast context if needed
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(x.shape[0], -1)
            
        # Masked part (frozen)
        x_masked = x * self.mask
        
        # Network input: frozen part + context
        net_input = torch.cat([x_masked, context], dim=1)
        
        # Compute scale and translation
        s = self.scale_net(net_input)
        t = self.translate_net(net_input)
        
        # Apply transformation
        if not reverse:
            # Forward: x → y
            y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = ((1 - self.mask) * s).sum(dim=1)
        else:
            # Inverse: y → x
            y = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -((1 - self.mask) * s).sum(dim=1)
            
        return y, log_det


class SimplifiedFlow(nn.Module):
    """
    Simplified conditional normalizing flow for point clouds.
    
    Architecture:
    - DeepSet encoder for context extraction
    - 3 alternating affine coupling layers
    - Base distribution: standard Gaussian
    
    This is a pedagogical implementation with fewer layers than production code.
    
    Parameters
    ----------
    point_dim : int
        Point dimensionality (default: 2)
    param_dim : int
        Parameter dimensionality (default: 3 for θ₀, θ₁, η)
    hidden_dim : int
        Hidden dimension (default: 64)
        
    References
    ----------
    Cranmer et al. (2020). "The frontier of simulation-based inference." PNAS.
    Papamakarios et al. (2021). "Normalizing flows for probabilistic modeling." JMLR.
    """
    
    def __init__(
        self,
        point_dim: int = 2,
        param_dim: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        # Context encoder
        self.encoder = SimpleDeepSet(point_dim, param_dim, hidden_dim)
        
        # 3 coupling layers with alternating masks
        self.coupling_layers = nn.ModuleList()
        for i in range(3):
            mask = torch.zeros(2)
            mask[i % 2] = 1.0  # Alternate which dimension is frozen
            self.coupling_layers.append(
                AffineCoupling(mask, context_dim=hidden_dim, hidden_dim=hidden_dim)
            )
    
    def forward(
        self,
        x: List[torch.Tensor],
        condition: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward or inverse pass through the flow.
        
        Parameters
        ----------
        x : list of Tensor
            Point clouds
        condition : Tensor of shape (B, param_dim)
            Physical parameters
        reverse : bool
            If False: compute log p(x|θ) (data → latent)
            If True: sample x ~ p(x|θ) (latent → data)
            
        Returns
        -------
        If reverse=False:
            log_prob : Tensor of shape (B,)
            latents : list of Tensor
        If reverse=True:
            samples : list of Tensor
        """
        # Extract context
        context = self.encoder(x, condition)
        B = len(x)
        
        if not reverse:
            # Forward: compute log-likelihood
            log_probs = torch.zeros(B, device=context.device)
            latents = []
            
            for i in range(B):
                xi = x[i].to(context.device)
                if xi.numel() == 0:
                    latents.append(xi)
                    continue
                    
                ctx = context[i]
                z = xi
                log_det_sum = 0.0
                
                # Apply coupling layers
                for layer in self.coupling_layers:
                    z, log_det = layer(z, ctx, reverse=False)
                    log_det_sum += log_det.sum()
                    
                # Base distribution log-probability (standard Gaussian)
                base_log_prob = -0.5 * (z**2).sum() - 0.5 * np.log(2*np.pi) * z.numel()
                
                # Total log-probability (per point)
                log_probs[i] = (base_log_prob + log_det_sum) / len(xi)
                latents.append(z)
                
            return log_probs, latents
            
        else:
            # Reverse: generate samples
            samples = []
            for i in range(B):
                xi = x[i].to(context.device)
                if xi.numel() == 0:
                    samples.append(xi)
                    continue
                    
                ctx = context[i]
                z = xi
                
                # Apply inverse coupling layers (in reverse order)
                for layer in reversed(self.coupling_layers):
                    z, _ = layer(z, ctx, reverse=True)
                    
                samples.append(z)
                
            return samples
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_points: int,
        device: torch.device = torch.device('cpu')
    ) -> List[torch.Tensor]:
        """
        Generate point clouds from the learned distribution.
        
        Parameters
        ----------
        condition : Tensor of shape (B, param_dim)
            Physical parameters
        n_points : int
            Number of points per cloud
        device : torch.device
            Computation device
            
        Returns
        -------
        samples : list of Tensor
            Generated point clouds
        """
        B = condition.shape[0]
        
        # Sample from base distribution
        z = [torch.randn(n_points, 2, device=device) for _ in range(B)]
        
        # Transform to data space
        return self.forward(z, condition.to(device), reverse=True)


# ============================================================================
# 4. TRAINING ROUTINE
# ============================================================================

def train_flow(
    flow: SimplifiedFlow,
    rng: np.random.Generator,
    n_iter: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> List[float]:
    """
    Train the conditional flow model.
    
    Training objective: maximize log p(x|θ,η) on simulated data
    
    Parameters
    ----------
    flow : SimplifiedFlow
        Model to train
    rng : numpy.random.Generator
        Random number generator for data
    n_iter : int
        Number of training iterations
    batch_size : int
        Batch size
    lr : float
        Learning rate
    device : torch.device
        Training device
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    losses : list of float
        Training loss history
    """
    flow.to(device)
    flow.train()
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    
    losses = []
    for step in range(n_iter):
        # Generate training batch
        clouds, theta, eta = generate_batch(batch_size, rng)
        condition = torch.cat([theta, eta.unsqueeze(1)], dim=1).to(device)
        clouds = [c.to(device) for c in clouds]
        
        # Forward pass
        log_prob, _ = flow(clouds, condition, reverse=False)
        loss = -log_prob.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and step % 200 == 0:
            print(f"  Iteration {step:4d}/{n_iter}: loss = {loss.item():.3f}")
            
    return losses


# ============================================================================
# 5. SIMULATION-BASED CALIBRATION (SBC)
# ============================================================================

def compute_sbc_ranks(
    flow: SimplifiedFlow,
    rng: np.random.Generator,
    n_samples: int = 100,
    n_posterior: int = 200,
    device: torch.device = torch.device('cpu')
) -> npt.NDArray[np.int32]:
    """
    Compute ranks for Simulation-Based Calibration.
    
    For each test:
    1. Sample θ ~ prior
    2. Generate x ~ p(x|θ)
    3. Approximate posterior via ABC-like rejection sampling
    4. Count how many posterior samples fall below true θ
    
    A well-calibrated model produces uniform ranks.
    
    Parameters
    ----------
    flow : SimplifiedFlow
        Trained flow model
    rng : numpy.random.Generator
        Random number generator
    n_samples : int
        Number of SBC tests
    n_posterior : int
        Number of posterior samples per test
    device : torch.device
        Computation device
        
    Returns
    -------
    ranks : array of shape (n_samples, 2)
        Rank of true parameter in posterior for each test
        
    References
    ----------
    Talts et al. (2018). "Validating Bayesian inference algorithms with 
    simulation-based calibration." arXiv:1804.06788.
    """
    flow.eval()
    
    theta_prior, _ = sample_prior(n_samples, rng)
    ranks = []
    
    for i in range(n_samples):
        # 1. Sample true parameters
        theta_true = theta_prior[i]
        eta_true = rng.uniform(ETA_LO, ETA_HI)
        
        # 2. Generate observed data
        x_obs = simulate_thomas(theta_true, eta_true, rng)
        
        # 3. Approximate posterior via rejection sampling
        # (In practice, would use a trained inference network here)
        # For simplicity, we use grid-based approximation
        theta_samples, eta_samples = sample_prior(n_posterior * 10, rng)
        log_probs = []
        
        with torch.no_grad():
            # Compute likelihood for each candidate
            for j in range(n_posterior * 10):
                cond = torch.tensor(
                    [[theta_samples[j, 0], theta_samples[j, 1], eta_samples[j]]],
                    device=device
                )
                x_list = [torch.from_numpy(x_obs).to(device)]
                lp, _ = flow(x_list, cond, reverse=False)
                log_probs.append(lp.item())
                
        # 4. Sample from approximate posterior (importance sampling)
        log_probs = np.array(log_probs)
        weights = np.exp(log_probs - log_probs.max())
        weights /= weights.sum()
        
        indices = rng.choice(
            len(weights), 
            size=n_posterior, 
            replace=True, 
            p=weights
        )
        posterior_samples = theta_samples[indices]
        
        # 5. Compute ranks
        rank0 = (posterior_samples[:, 0] < theta_true[0]).sum()
        rank1 = (posterior_samples[:, 1] < theta_true[1]).sum()
        ranks.append([rank0, rank1])
        
    return np.array(ranks, dtype=np.int32)


def sbc_uniformity_test(
    ranks: npt.NDArray[np.int32],
    n_posterior: int,
    nbins: int = 10
) -> npt.NDArray[np.float64]:
    """
    Test rank uniformity using chi-squared goodness-of-fit.
    
    Parameters
    ----------
    ranks : array of shape (n_samples, n_params)
        Ranks from compute_sbc_ranks
    n_posterior : int
        Number of posterior samples used
    nbins : int
        Number of histogram bins
        
    Returns
    -------
    p_values : array of shape (n_params,)
        Chi-squared test p-value for each parameter
        
    Notes
    -----
    p > 0.05 indicates no evidence against calibration (good!)
    p < 0.05 suggests miscalibration (bad!)
    """
    p_vals = []
    
    for d in range(ranks.shape[1]):
        # Histogram of ranks
        counts, _ = np.histogram(ranks[:, d], bins=nbins, range=(0, n_posterior))
        
        # Expected count under uniformity
        expected = len(ranks) / nbins
        
        # Chi-squared statistic
        chi2_stat = ((counts - expected)**2 / expected).sum()
        
        # p-value from chi-squared distribution
        p_vals.append(chi2.sf(chi2_stat, nbins - 1))
        
    return np.array(p_vals)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_xi_sensitivity(
    rng: np.random.Generator,
    n_realizations: int = 20
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Demonstrate that μ is invisible to ξ(r).
    
    Generate realizations with same λ_p but different μ, and show that
    their correlation functions are statistically identical.
    
    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator
    n_realizations : int
        Number of realizations per configuration
        
    Returns
    -------
    xi_low_mu : array of shape (n_realizations, n_bins)
        ξ(r) for low μ
    xi_high_mu : array of shape (n_realizations, n_bins)
        ξ(r) for high μ
    """
    # Same λ_p, different μ
    theta_low_mu = np.array([1.5, 0.8], dtype=np.float32)   # μ ≈ 6
    theta_high_mu = np.array([1.5, 1.3], dtype=np.float32)  # μ ≈ 20
    
    xi_low = []
    xi_high = []
    
    for _ in range(n_realizations):
        # Low μ
        pts_low = simulate_thomas(theta_low_mu, 0.85, rng)
        xi_low.append(landy_szalay(pts_low))
        
        # High μ
        pts_high = simulate_thomas(theta_high_mu, 0.85, rng)
        xi_high.append(landy_szalay(pts_high))
        
    return np.array(xi_low), np.array(xi_high)


# ============================================================================
# END OF MODULE
# ============================================================================