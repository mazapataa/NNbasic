"""
poc_core.py
===========
A lightweight generative pipeline for galaxy clustering analysis.

This module implements:
- 2D Thomas (Neyman-Scott) cluster process simulator
- Classical Landy-Szalay two-point correlation function estimator
- Permutation-invariant conditional normalizing flow (RealNVP)
- Amortised posterior inference network (DeepSet + MDN)
- Single-realisation self-calibration (jackknife + conformal prediction)
- Diagnostic tools: SBC, coverage analysis, posterior-predictive checks

Author: Anonymous
Date: May 2026
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
# CONSTANTS & PRIOR RANGES
# ============================================================================

# Parameter ranges: [log10(λ_p), log10(μ)]
THETA_LO = np.array([1.0, 0.7], dtype=np.float32)
THETA_HI = np.array([2.0, 1.5], dtype=np.float32)

# Completeness range
ETA_LO, ETA_HI = 0.55, 1.0

# Fixed cluster scale parameter
SIGMA_C = 0.03

# Minimum galaxy count for valid realizations
N_MIN = 25

# Radial bin edges for correlation function [0, 0.22] in 8 bins
R_EDGES = np.linspace(0.0, 0.22, 9)

# Pre-compute random catalog for Landy-Szalay estimator
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
    
    The Thomas process is a doubly-stochastic point process where parent points
    are distributed as a homogeneous Poisson process, and each parent generates
    a Poisson-distributed number of offspring scattered with Gaussian dispersion.
    
    Parameters
    ----------
    theta : ndarray of shape (2,)
        Physical parameters: [log10(λ_p), log10(μ)] where
        - λ_p is parent intensity (clusters per unit area)
        - μ is mean offspring per cluster
    eta : float
        Survey completeness fraction in [0, 1]
    rng : numpy.random.Generator
        Random number generator for reproducibility
    misspecified : bool, optional
        If True, adds a second population of large-scale clusters
        to test model misspecification detection
        
    Returns
    -------
    pts : ndarray of shape (N, 2)
        Point coordinates in [0, 1)² with periodic boundaries
        
    Notes
    -----
    The cluster scale σ_c is fixed at 0.03 (see SIGMA_C constant).
    Up to 50 attempts are made to generate a valid realization with
    at least N_MIN points.
    """
    lam_p = 10.0 ** theta[0]
    mu = 10.0 ** theta[1]
    
    # Attempt to generate a valid realization
    for _ in range(50):
        # Sample parent points from Poisson process
        n_parents = rng.poisson(lam_p)
        if n_parents == 0:
            continue
            
        parents = rng.uniform(0.0, 1.0, size=(n_parents, 2))
        
        # Sample offspring counts
        n_off = rng.poisson(mu, size=n_parents)
        if n_off.sum() < 5:
            continue
            
        # Generate offspring with Gaussian scatter around parents
        pts = np.repeat(parents, n_off, axis=0)
        pts = np.mod(pts + rng.normal(0.0, SIGMA_C, size=pts.shape), 1.0)
        
        # Apply survey completeness
        if eta < 1.0:
            pts = pts[rng.random(len(pts)) < eta]
            
        # Add misspecified interloper population if requested
        if misspecified:
            n_big = max(rng.poisson(4), 1)
            bp = rng.uniform(0, 1, size=(n_big, 2))
            nb = rng.poisson(0.8 * len(pts) / n_big, size=n_big)
            if nb.sum() > 0:
                extra = np.repeat(bp, nb, axis=0)
                extra = np.mod(extra + rng.normal(0, 0.13, extra.shape), 1.0)
                pts = np.vstack([pts, extra])
                
        if len(pts) >= N_MIN:
            return pts.astype(np.float32)
            
    # Fallback: uniform random distribution
    return rng.uniform(0, 1, size=(N_MIN, 2)).astype(np.float32)


def sample_prior(
    n: int,
    rng: np.random.Generator
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Sample parameters from the uniform prior.
    
    Parameters
    ----------
    n : int
        Number of samples to generate
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    theta : ndarray of shape (n, 2)
        Physical parameters sampled uniformly from [THETA_LO, THETA_HI]
    eta : ndarray of shape (n,)
        Completeness values sampled uniformly from [ETA_LO, ETA_HI]
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
    Generate a batch of point clouds with their parameters.
    
    Parameters
    ----------
    batch_size : int
        Number of point clouds to generate
    rng : numpy.random.Generator
        Random number generator
    misspecified : bool, optional
        Whether to generate misspecified realizations
        
    Returns
    -------
    clouds : list of Tensor
        List of point clouds, each of shape (N_i, 2)
    theta : Tensor of shape (batch_size, 2)
        Physical parameters
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
# 2. LANDY-SZALAY ESTIMATOR
# ============================================================================

def landy_szalay(pts: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """
    Compute two-point correlation function using Landy-Szalay estimator.
    
    The estimator is given by:
        ξ(r) = (DD - 2DR + RR) / RR
    where DD, DR, RR are normalized pair counts for data-data, data-random,
    and random-random pairs respectively.
    
    Parameters
    ----------
    pts : ndarray of shape (N, 2)
        Point coordinates in [0, 1)²
        
    Returns
    -------
    xi : ndarray of shape (len(R_EDGES)-1,)
        Correlation function values in radial bins
        
    Notes
    -----
    Uses periodic boundary conditions and a pre-computed random catalog.
    """
    nD, nR = pts.shape[0], _RAND.shape[0]
    dt = cKDTree(pts, boxsize=1.0)
    
    # Count pairs in cumulative bins
    dd = np.diff(np.concatenate([[0], dt.count_neighbors(dt, R_EDGES[1:])])).astype(float)
    dr = np.diff(np.concatenate([[0], dt.count_neighbors(_RTREE, R_EDGES[1:])])).astype(float)
    rr = np.diff(np.concatenate([[0], _RR_CUM])).astype(float)
    
    # Remove self-pairs from first bin
    dd[0] = max(dd[0] - nD, 0)
    rr[0] = max(rr[0] - nR, 0)
    
    # Normalize by number of pairs
    dd /= nD * (nD - 1) / 2.0
    rr /= nR * (nR - 1) / 2.0
    dr /= nD * nR
    
    # Compute Landy-Szalay estimator
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.where(rr > 0, (dd - 2*dr + rr) / rr, 0.0)
    
    return np.nan_to_num(xi)


# ============================================================================
# 3. DEEPSET ENCODER (Permutation-Invariant)
# ============================================================================

class DeepSetEncoder(nn.Module):
    """
    Permutation-invariant encoder for point clouds.
    
    Processes each point independently, aggregates via summation,
    and combines with conditional information.
    
    Parameters
    ----------
    point_dim : int
        Dimensionality of each point (default: 2)
    cond_dim : int
        Dimensionality of conditional input (default: 3)
    hid_dim : int
        Hidden layer dimension (default: 128)
    out_dim : int
        Output dimension (default: 128)
    """
    
    def __init__(
        self,
        point_dim: int = 2,
        cond_dim: int = 3,
        hid_dim: int = 128,
        out_dim: int = 128
    ):
        super().__init__()
        
        # Per-point feature extraction
        self.point_net = nn.Sequential(
            nn.Linear(point_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        
        # Conditional feature processing
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
    
    def forward(
        self,
        x: List[torch.Tensor],
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : list of Tensor
            Point clouds, each of shape (N_i, point_dim)
        condition : Tensor of shape (B, cond_dim)
            Conditional information for each cloud
            
        Returns
        -------
        global_features : Tensor of shape (B, out_dim)
            Aggregated features for each cloud
        """
        B = len(x)
        global_feats = []
        
        for i in range(B):
            if x[i].numel() == 0:
                # Handle empty clouds
                global_feats.append(
                    torch.zeros(1, self.cond_net[-1].out_features,
                               device=condition.device)
                )
                continue
                
            # Process points and aggregate via summation
            per_point = self.point_net(x[i].to(condition.device))
            pooled = per_point.sum(dim=0, keepdim=True)
            
            # Combine with conditional features
            cond_feat = self.cond_net(condition[i:i+1])
            global_feats.append(pooled + cond_feat)
            
        return torch.cat(global_feats, dim=0)


# ============================================================================
# 4. CONDITIONAL NORMALIZING FLOW (RealNVP)
# ============================================================================

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP normalizing flow.
    
    Applies an affine transformation to masked dimensions:
        y = x_masked + (1 - mask) * (x * exp(s(x_masked, c)) + t(x_masked, c))
    
    Parameters
    ----------
    mask : Tensor
        Binary mask indicating which dimensions to keep fixed
    context_dim : int
        Dimensionality of context vector
    hid_dim : int
        Hidden layer dimension (default: 64)
    """
    
    def __init__(
        self,
        mask: torch.Tensor,
        context_dim: int,
        hid_dim: int = 64
    ):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        
        # Scale network s(·)
        self.scale_net = nn.Sequential(
            nn.Linear(2 + context_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2)
        )
        
        # Translation network t(·)
        self.translate_net = nn.Sequential(
            nn.Linear(2 + context_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coupling layer.
        
        Parameters
        ----------
        x : Tensor of shape (N, 2)
            Input points
        context : Tensor of shape (1, context_dim) or (N, context_dim)
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
        # Broadcast context if necessary
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(x.shape[0], -1)
            
        # Apply mask and construct network input
        x_masked = x * self.mask
        net_in = torch.cat([x_masked, context], dim=1)
        
        # Compute scale and translation

        s = self.scale_net(net_in)
        t = self.translate_net(net_in)
        
        # Apply transformation
        if not reverse:
            y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = ((1 - self.mask) * s).sum(dim=1)
        else:
            y = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -((1 - self.mask) * s).sum(dim=1)
            
        return y, log_det


class ConditionalPointFlow(nn.Module):
    """
    Conditional RealNVP normalizing flow for point clouds.
    
    Learns to transform between a standard Gaussian base distribution
    and the conditional distribution of point positions.
    
    Parameters
    ----------
    point_dim : int
        Point dimensionality (default: 2)
    param_dim : int
        Parameter dimensionality (default: 3)
    hid_dim : int
        Hidden dimension for encoder and coupling layers (default: 128)
    n_coupling : int
        Number of coupling layers (default: 6)
    """
    
    def __init__(
        self,
        point_dim: int = 2,
        param_dim: int = 3,
        hid_dim: int = 128,
        n_coupling: int = 6
    ):
        super().__init__()
        
        # Context encoder
        self.encoder = DeepSetEncoder(
            point_dim, param_dim, hid_dim, out_dim=hid_dim
        )
        
        # Alternating coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(n_coupling):
            mask = torch.zeros(2)
            mask[i % 2] = 1.0
            self.coupling_layers.append(
                AffineCouplingLayer(mask, context_dim=hid_dim, hid_dim=hid_dim)
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
            Point clouds, each of shape (N_i, 2)
        condition : Tensor of shape (B, param_dim)
            Physical parameters and completeness
        reverse : bool
            If True, generate samples (latent → data)
            If False, compute log-probability (data → latent)
            
        Returns
        -------
        If reverse=False:
            log_prob : Tensor of shape (B,)
                Log-probability for each cloud
            z : list of Tensor
                Latent representations
        If reverse=True:
            x : list of Tensor
                Generated point clouds
        """
        # Extract context from encoder
        context = self.encoder(x, condition)
        B = len(x)
        
        if not reverse:
            # Forward: compute log-probability
            total_log_prob = torch.zeros(B, device=context.device)
            outs = []
            
            for i in range(B):
                xi = x[i].to(context.device)
                if xi.numel() == 0:
                    outs.append(xi)
                    continue
                    
                ctx = context[i]
                z = xi
                log_det_sum = 0.0
                
                # Apply coupling layers
                for layer in self.coupling_layers:
                    z, log_det = layer(z, ctx, reverse=False)
                    log_det_sum += log_det.sum()
                    
                # Base distribution log-probability
                base_log_prob = -0.5 * (z**2).sum(dim=1) - 0.5*np.log(2*np.pi)*2
                total_log_prob[i] = (base_log_prob.sum() + log_det_sum) / len(xi)
                outs.append(z)
                
            return total_log_prob, outs
            
        else:
            # Reverse: generate samples
            outs = []
            for i in range(B):
                xi = x[i].to(context.device)
                if xi.numel() == 0:
                    outs.append(xi)
                    continue
                    
                ctx = context[i]
                z = xi
                
                # Apply inverse coupling layers
                for layer in reversed(self.coupling_layers):
                    z, _ = layer(z, ctx, reverse=True)
                    
                outs.append(z)
                
            return outs
    
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
            Physical parameters and completeness
        n_points : int
            Number of points per cloud
        device : torch.device
            Device for computation
            
        Returns
        -------
        samples : list of Tensor
            Generated point clouds, each of shape (n_points, 2)
        """
        B = condition.shape[0]
        
        # Sample from base distribution
        z = [torch.randn(n_points, 2, device=device) for _ in range(B)]
        
        # Transform to data space
        samples = self.forward(z, condition.to(device), reverse=True)
        
        # Clamp to valid range and check for NaNs/infs
        for i in range(B):
            s = samples[i]
            if not torch.isfinite(s).all():
                # Fallback: uniform random points if flow produced invalid values
                samples[i] = torch.rand(n_points, 2, device=device)
            else:
                samples[i] = s.clamp(0.0, 1.0)
        
        return samples


# ============================================================================
# 5. INFERENCE NETWORK (DeepSet + Mixture Density Network)
# ============================================================================

class MDNHead(nn.Module):
    """
    Mixture Density Network head for multi-modal posteriors.
    
    Outputs a Gaussian mixture model with learnable component weights,
    means, and log-standard deviations.
    
    Parameters
    ----------
    in_dim : int
        Input feature dimension
    out_dim : int
        Output parameter dimension (default: 2)
    n_components : int
        Number of Gaussian components (default: 6)
    hid_dim : int
        Hidden dimension (not currently used, for API compatibility)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        n_components: int = 6,
        hid_dim: int = 64
    ):
        super().__init__()
        self.n_comp = n_components
        self.out_dim = out_dim
        
        # Mixture weights (logits)
        self.logit = nn.Linear(in_dim, n_components)
        
        # Component means
        self.mu = nn.Linear(in_dim, n_components * out_dim)
        
        # Component log-standard deviations
        self.logs = nn.Linear(in_dim, n_components * out_dim)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mixture parameters.
        
        Parameters
        ----------
        z : Tensor of shape (B, in_dim)
            Input features
            
        Returns
        -------
        logits : Tensor of shape (B, n_components)
            Unnormalized mixture weights
        mus : Tensor of shape (B, n_components, out_dim)
            Component means
        logs : Tensor of shape (B, n_components, out_dim)
            Component log-standard deviations (clamped to [-2.5, 2.5])
        """
        logits = self.logit(z)
        mus = self.mu(z).view(-1, self.n_comp, self.out_dim)
        logs = self.logs(z).view(-1, self.n_comp, self.out_dim).clamp(-2.5, 2.5)
        return logits, mus, logs


class InferenceNetwork(nn.Module):
    """
    Neural posterior estimator for amortised inference.
    
    Maps point clouds directly to posterior distributions over
    physical parameters using a DeepSet encoder and MDN head.
    
    Parameters
    ----------
    point_dim : int
        Point dimensionality (default: 2)
    n_params : int
        Number of parameters to infer (default: 2)
    hidden : int
        Hidden dimension (default: 128)
    """
    
    def __init__(
        self,
        point_dim: int = 2,
        n_params: int = 2,
        hidden: int = 128
    ):
        super().__init__()
        
        # Encoder: dummy condition since we don't observe completeness
        self.encoder = DeepSetEncoder(
            point_dim,
            cond_dim=n_params,  # dummy condition
            hid_dim=hidden,
            out_dim=hidden
        )
        
        # Mixture density network head
        self.mdn = MDNHead(
            hidden,
            out_dim=n_params,
            n_components=6,
            hid_dim=hidden
        )
    
    def forward(
        self,
        x: List[torch.Tensor],
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mixture parameters.
        
        Parameters
        ----------
        x : list of Tensor
            Point clouds
        condition : Tensor
            Dummy condition (zeros)
            
        Returns
        -------
        logits, mus, logs : Tensors
            Mixture parameters from MDN head
        """
        z = self.encoder(x, condition)
        return self.mdn(z)
    
    def nll(
        self,
        x: List[torch.Tensor],
        theta_true: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for training.
        
        Parameters
        ----------
        x : list of Tensor
            Point clouds
        theta_true : Tensor of shape (B, n_params)
            True parameter values
        condition : Tensor
            Dummy condition
            
        Returns
        -------
        nll : Tensor (scalar)
            Mean negative log-likelihood over batch
        """
        logits, mus, logs = self(x, condition)
        
        # Compute component log-probabilities
        theta_true = theta_true.unsqueeze(1)  # (B, 1, n_params)
        comp = -0.5 * (((theta_true - mus) / logs.exp())**2) - logs - 0.5*np.log(2*np.pi)
        log_prob_comp = comp.sum(dim=-1)  # (B, n_components)
        
        # Mixture log-probability
        lw = F.log_softmax(logits, dim=-1)
        return -torch.logsumexp(lw + log_prob_comp, dim=-1).mean()
    
    @torch.no_grad()
    def sample_posterior(
        self,
        x: List[torch.Tensor],
        condition: torch.Tensor,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """
        Draw samples from the approximate posterior.
        
        Parameters
        ----------
        x : list of Tensor
            Point clouds
        condition : Tensor
            Dummy condition
        n_samples : int
            Number of samples per cloud
            
        Returns
        -------
        samples : Tensor of shape (B, n_samples, n_params)
            Posterior samples
        """
        logits, mus, logs = self(x, condition)
        B = len(x) if isinstance(x, list) else x.shape[0]
        
        # Sample mixture components
        w = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(w, n_samples, replacement=True)
        
        # Gather corresponding means and stds
        bi = torch.arange(B).unsqueeze(1).expand(-1, n_samples)
        mu_samp = mus[bi, idx]
        logs_samp = logs[bi, idx]
        
        # Sample from Gaussians
        return mu_samp + logs_samp.exp() * torch.randn_like(mu_samp)


# ============================================================================
# 6. TRAINING ROUTINES
# ============================================================================

def train_generative_model(
    flow: ConditionalPointFlow,
    rng: np.random.Generator,
    n_iter: int = 3000,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu')
) -> List[float]:
    """
    Train the generative flow model.
    
    Parameters
    ----------
    flow : ConditionalPointFlow
        Flow model to train
    rng : numpy.random.Generator
        Random number generator for data generation
    n_iter : int
        Number of training iterations
    batch_size : int
        Batch size
    lr : float
        Learning rate
    device : torch.device
        Training device
        
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
        
        # Compute loss
        log_prob, _ = flow(clouds, condition, reverse=False)
        loss = -log_prob.mean()
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 500 == 0:
            print(f"Gen [{step:5d}/{n_iter}] loss: {loss.item():.3f}")
            
    return losses


def train_inference_network(
    infer_net: InferenceNetwork,
    flow: ConditionalPointFlow,
    rng: np.random.Generator,
    n_iter: int = 2000,
    batch_size: int = 64,
    lr: float = 2e-4,
    device: torch.device = torch.device('cpu')
) -> List[float]:
    """
    Train the inference network using synthetic data from the flow.
    
    Parameters
    ----------
    infer_net : InferenceNetwork
        Inference network to train
    flow : ConditionalPointFlow
        Pre-trained generative flow (frozen)
    rng : numpy.random.Generator
        Random number generator
    n_iter : int
        Number of training iterations
    batch_size : int
        Batch size
    lr : float
        Learning rate
    device : torch.device
        Training device
        
    Returns
    -------
    losses : list of float
        Training loss history
    """
    flow.eval()  # Freeze generative model
    infer_net.to(device)
    infer_net.train()
    optimizer = torch.optim.Adam(infer_net.parameters(), lr=lr, weight_decay=1e-5)
    
    # Dummy condition for inference network
    dummy_cond = torch.zeros(1, 2).to(device)
    
    losses = []
    for step in range(n_iter):
        # Sample parameters
        theta, eta = sample_prior(batch_size, rng)
        theta_t = torch.from_numpy(theta).to(device)
        eta_t = torch.from_numpy(eta).to(device)
        cond = torch.cat([theta_t, eta_t.unsqueeze(1)], dim=1)
        
        # Generate synthetic clouds
        clouds_syn = flow.sample(cond, n_points=60, device=device)
        
        # Compute inference loss
        dummy_batch = dummy_cond.expand(batch_size, -1)
        loss = infer_net.nll(clouds_syn, theta_t, dummy_batch)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 500 == 0:
            print(f"Inf [{step:5d}/{n_iter}] loss: {loss.item():.3f}")
            
    return losses


# ============================================================================
# 7. SINGLE-REALIZATION CALIBRATION (Jackknife + Conformal)
# ============================================================================

def jackknife_calibration(
    infer_net: InferenceNetwork,
    obs_cloud: torch.Tensor,
    alpha: float = 0.1,
    K: int = 10,
    n_samp: int = 500,
    device: torch.device = torch.device('cpu')
) -> Tuple[float, npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Perform single-realization calibration using spatial jackknife and
    conformal prediction.
    
    Splits the survey into K spatial regions, computes posteriors for
    each masked region, and uses the distribution of nonconformity scores
    to inflate credible intervals to achieve nominal coverage.
    
    Parameters
    ----------
    infer_net : InferenceNetwork
        Trained inference network
    obs_cloud : Tensor of shape (N, 2)
        Observed point cloud
    alpha : float
        Miscoverage level (default: 0.1 for 90% coverage)
    K : int
        Number of jackknife splits
    n_samp : int
        Number of posterior samples
    device : torch.device
        Computation device
        
    Returns
    -------
    q_hat : float
        Conformal quantile for interval inflation
    full_mean : ndarray of shape (n_params,)
        Posterior mean from full dataset
    full_cov : ndarray of shape (n_params, n_params)
        Posterior covariance from full dataset
        
    Notes
    -----
    The nonconformity score is the Mahalanobis distance between the
    jackknife posterior mean and the full posterior mean.
    """
    infer_net.eval()
    dummy_cond = torch.zeros(1, 2).to(device)
    
    # Full posterior
    with torch.no_grad():
        full_samp = infer_net.sample_posterior(
            [obs_cloud.to(device)],
            dummy_cond,
            n_samples=n_samp
        ).squeeze(0)
        
    full_mean = full_samp.mean(dim=0)
    full_cov = torch.cov(full_samp.T)
    
    # Jackknife by x-coordinate splits
    x_coords = obs_cloud[:, 0]
    quantiles = np.linspace(0, 1, K+1)
    scores = []
    
    for k in range(K):
        lo, hi = quantiles[k], quantiles[k+1]
        
        # Create mask: keep all points outside [lo, hi)
        mask = (x_coords < lo) | (x_coords >= hi)
        masked_cloud = obs_cloud[mask]
        
        if len(masked_cloud) < 5:
            continue
            
        # Compute jackknife posterior
        with torch.no_grad():
            samp_k = infer_net.sample_posterior(
                [masked_cloud.to(device)],
                dummy_cond,
                n_samples=n_samp//2
            ).squeeze(0)
            
        mean_k = samp_k.mean(dim=0)
        diff = mean_k - full_mean
        
        # Mahalanobis distance (nonconformity score)
        # FIXED: Removed deprecated .T for 1D tensor
        score = torch.sqrt(
            (diff @ torch.linalg.inv(full_cov) @ diff).clamp(min=0)
        )
        scores.append(score.item())
    
    # Conformal quantile
    q_hat = np.quantile(scores, 1 - alpha) if scores else 1.0
    
    return q_hat, full_mean.cpu().numpy(), full_cov.cpu().numpy()


# ============================================================================
# 8. DIAGNOSTICS
# ============================================================================

def compute_sbc_ranks(
    infer_net: InferenceNetwork,
    rng: np.random.Generator,
    n_samples: int = 500,
    n_posterior: int = 400,
    device: torch.device = torch.device('cpu')
) -> npt.NDArray[np.int32]:
    """
    Compute ranks for Simulation-Based Calibration (SBC).
    
    For each parameter draw from the prior:
    1. Simulate data given that parameter
    2. Compute approximate posterior
    3. Count how many posterior samples are below the true value
    
    A well-calibrated posterior should produce uniform ranks.
    
    Parameters
    ----------
    infer_net : InferenceNetwork
        Trained inference network
    rng : numpy.random.Generator
        Random number generator
    n_samples : int
        Number of SBC samples
    n_posterior : int
        Number of posterior samples per SBC sample
    device : torch.device
        Computation device
        
    Returns
    -------
    ranks : ndarray of shape (n_samples, n_params)
        Rank of true parameter in posterior for each sample
    """
    infer_net.eval()
    dummy_cond = torch.zeros(1, 2).to(device)
    
    theta_prior, _ = sample_prior(n_samples, rng)
    ranks = []
    
    for i in range(n_samples):
        # Simulate data
        pts = simulate_thomas(theta_prior[i], rng.uniform(ETA_LO, ETA_HI), rng)
        cloud = torch.from_numpy(pts).to(device)
        
        # Compute posterior
        with torch.no_grad():
            post = infer_net.sample_posterior(
                [cloud], dummy_cond, n_samples=n_posterior
            ).squeeze(0)
            
        # Compute ranks
        true = torch.from_numpy(theta_prior[i]).to(device)
        rank0 = (post[:, 0] < true[0]).sum().item()
        rank1 = (post[:, 1] < true[1]).sum().item()
        ranks.append([rank0, rank1])
        
    return np.array(ranks, dtype=np.int32)


def sbc_uniformity_pvalues(
    ranks: npt.NDArray[np.int32],
    n_posterior: int,
    nbins: int = 10
) -> npt.NDArray[np.float64]:
    """
    Test rank uniformity using chi-squared goodness-of-fit.
    
    Parameters
    ----------
    ranks : ndarray of shape (n_samples, n_params)
        SBC ranks from compute_sbc_ranks
    n_posterior : int
        Number of posterior samples used
    nbins : int
        Number of histogram bins
        
    Returns
    -------
    p_values : ndarray of shape (n_params,)
        Chi-squared test p-value for each parameter
        
    Notes
    -----
    p-values > 0.05 indicate no evidence against calibration.
    """
    p_vals = []
    
    for d in range(ranks.shape[1]):
        counts, _ = np.histogram(ranks[:, d], bins=nbins, range=(0, n_posterior))
        expected = len(ranks) / nbins
        chi2_stat = ((counts - expected)**2 / expected).sum()
        p_vals.append(chi2.sf(chi2_stat, nbins - 1))
        
    return np.array(p_vals)


def compute_coverage(
    post_samples: npt.NDArray[np.float32],
    true_theta: npt.NDArray[np.float32],
    levels: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Compute empirical coverage at various credible levels.
    
    Parameters
    ----------
    post_samples : ndarray of shape (n_test, n_post, n_param)
        Posterior samples for test cases
    true_theta : ndarray of shape (n_test, n_param)
        True parameter values
    levels : ndarray of shape (n_levels,)
        Credible levels to evaluate (e.g., [0.68, 0.95])
        
    Returns
    -------
    coverage : ndarray of shape (n_param, n_levels)
        Empirical coverage fraction for each parameter and level
        
    Notes
    -----
    Well-calibrated posteriors should have coverage close to the
    nominal credible level.
    """
    n_test, n_post, n_param = post_samples.shape
    cov = np.zeros((n_param, len(levels)))
    
    for i in range(n_test):
        for d in range(n_param):
            for j, lv in enumerate(levels):
                # Compute symmetric credible interval
                lo = np.quantile(post_samples[i, :, d], (1 - lv) / 2)
                hi = np.quantile(post_samples[i, :, d], 1 - (1 - lv) / 2)
                
                # Check if true value is contained
                if lo <= true_theta[i, d] <= hi:
                    cov[d, j] += 1
                    
    return cov / n_test


def posterior_predictive_discrepancy(
    infer_net: InferenceNetwork,
    flow: ConditionalPointFlow,
    cloud: torch.Tensor,
    n_draws: int = 60,
    device: torch.device = torch.device('cpu')
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute posterior-predictive check discrepancy.
    
    Measures how well the model can reproduce the observed two-point
    correlation function when parameters are drawn from the posterior.
    
    Parameters
    ----------
    infer_net : InferenceNetwork
        Trained inference network
    flow : ConditionalPointFlow
        Trained generative flow
    cloud : Tensor of shape (N, 2)
        Observed point cloud
    n_draws : int
        Number of posterior draws
    device : torch.device
        Computation device
        
    Returns
    -------
    discrepancy : float
        Mean standardized squared residual
    xi_obs : ndarray
        Observed correlation function
    xi_preds : ndarray of shape (n_draws, n_bins)
        Predicted correlation functions from posterior
        
    Notes
    -----
    High discrepancy suggests model misspecification.
    """
    infer_net.eval()
    flow.eval()
    
    dummy_cond = torch.zeros(1, 2).to(device)
    
    # Draw from approximate posterior
    with torch.no_grad():
        post_th = infer_net.sample_posterior(
            [cloud.to(device)],
            dummy_cond,
            n_samples=n_draws
        ).squeeze(0)
    
    # Generate predictions
    xi_preds = []
    rng = np.random.default_rng()
    
    for i in range(n_draws):
        th = post_th[i].cpu().numpy()
        eta = rng.uniform(ETA_LO, ETA_HI)
        cond = torch.tensor([[th[0], th[1], eta]], device=device)
        
        # Generate synthetic cloud
        syn_cloud = flow.sample(cond, n_points=len(cloud), device=device)[0]
        xi_preds.append(landy_szalay(syn_cloud.cpu().numpy()))
    
    xi_preds = np.array(xi_preds)
    xi_obs = landy_szalay(cloud.cpu().numpy())
    
    # Compute discrepancy
    pred_mean = xi_preds.mean(axis=0)
    pred_std = xi_preds.std(axis=0) + 1e-6
    disc = np.mean(((xi_obs - pred_mean) / pred_std) ** 2)
    
    return disc, xi_obs, xi_preds


# ============================================================================
# END OF MODULE
# ============================================================================