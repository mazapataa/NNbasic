"""
poc_core.py
===========
Core tools for the light‑weight generative point‑process pipeline:
- 2‑D Thomas process simulator
- classical Landy‑Szalay ξ(r)
- permutation‑invariant conditional normalising flow
- amortised posterior estimator (DeepSet + MDN)
- training routines
- diagnostics: SBC, coverage, jackknife conformal calibration, PPC
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ #
#  constants & priors
# ------------------------------------------------------------------ #
THETA_LO = np.array([1.0, 0.7], dtype=np.float32)   # log10 λ_p, log10 μ
THETA_HI = np.array([2.0, 1.5], dtype=np.float32)
ETA_LO, ETA_HI = 0.55, 1.0
SIGMA_C = 0.03          # cluster scale (fixed)
N_MIN = 25              # minimum galaxy count

# bin edges for the correlation function
R_EDGES = np.linspace(0.0, 0.22, 9)

# pre‑compute the random–random pairs (static box, periodic)
_RAND_RNG = np.random.default_rng(1)
_RAND = _RAND_RNG.uniform(0, 1, size=(1400, 2))
_RTREE = cKDTree(_RAND, boxsize=1.0)
_RR_CUM = _RTREE.count_neighbors(_RTREE, R_EDGES[1:])


# ------------------------------------------------------------------ #
#  1. simulator
# ------------------------------------------------------------------ #
def simulate_thomas(theta, eta, rng, misspecified=False):
    """
    Periodic 2‑D Thomas process.

    Parameters
    ----------
    theta : array-like [log10(λ_p), log10(μ)]
    eta   : float      survey completeness
    rng   : numpy Generator
    misspecified : bool  add a second large‑scale cluster population

    Returns
    -------
    pts : (N,2) float32 array
    """
    lam_p = 10.0 ** theta[0]
    mu    = 10.0 ** theta[1]
    for _ in range(50):
        n_parents = rng.poisson(lam_p)
        if n_parents == 0:
            continue
        parents = rng.uniform(0.0, 1.0, size=(n_parents, 2))
        n_off = rng.poisson(mu, size=n_parents)
        if n_off.sum() < 5:
            continue
        pts = np.repeat(parents, n_off, axis=0)
        pts = np.mod(pts + rng.normal(0.0, SIGMA_C, size=pts.shape), 1.0)
        if eta < 1.0:
            pts = pts[rng.random(len(pts)) < eta]
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
    return rng.uniform(0, 1, size=(N_MIN, 2)).astype(np.float32)


def sample_prior(n, rng):
    """Return (theta, eta) arrays of length n."""
    theta = rng.uniform(THETA_LO, THETA_HI, size=(n, 2)).astype(np.float32)
    eta   = rng.uniform(ETA_LO, ETA_HI, size=n).astype(np.float32)
    return theta, eta


def generate_batch(batch_size, rng, misspecified=False):
    """Draw a batch of point clouds together with their parameters."""
    theta, eta = sample_prior(batch_size, rng)
    clouds = []
    for i in range(batch_size):
        pts = simulate_thomas(theta[i], eta[i], rng, misspecified)
        clouds.append(torch.from_numpy(pts))
    return clouds, torch.from_numpy(theta), torch.from_numpy(eta)


# ------------------------------------------------------------------ #
#  2. classical Landy‑Szalay ξ(r)
# ------------------------------------------------------------------ #
def landy_szalay(pts):
    """Landy–Szalay estimator for periodic box."""
    nD, nR = pts.shape[0], _RAND.shape[0]
    dt = cKDTree(pts, boxsize=1.0)


    dd = np.diff(np.concatenate([[0], dt.count_neighbors(dt, R_EDGES[1:])])).astype(float)
    dr = np.diff(np.concatenate([[0], dt.count_neighbors(_RTREE, R_EDGES[1:])])).astype(float)
    rr = np.diff(np.concatenate([[0], _RR_CUM])).astype(float)
    dd[0] = max(dd[0] - nD, 0)
    rr[0] = max(rr[0] - nR, 0)
    dd /= nD * (nD - 1) / 2.0
    rr /= nR * (nR - 1) / 2.0
    dr /= nD * nR
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.where(rr > 0, (dd - 2*dr + rr) / rr, 0.0)
    return np.nan_to_num(xi)


# ------------------------------------------------------------------ #
#  3. permutation‑invariant encoder (DeepSet)
# ------------------------------------------------------------------ #
class DeepSetEncoder(nn.Module):
    def __init__(self, point_dim=2, cond_dim=3, hid_dim=128, out_dim=128):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(point_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x, condition):
        """
        x: list of (N_i, point_dim) tensors
        condition: (B, cond_dim)
        """
        B = len(x)
        global_feats = []
        for i in range(B):
            if x[i].numel() == 0:
                global_feats.append(torch.zeros(1, self.cond_net[-1].out_features))
                continue
            per_point = self.point_net(x[i].to(condition.device))
            pooled = per_point.sum(dim=0, keepdim=True)    # (1, hid_dim)
            cond_feat = self.cond_net(condition[i:i+1])     # (1, out_dim)
            global_feats.append(pooled + cond_feat)         # combine
        return torch.cat(global_feats, dim=0)               # (B, out_dim)


# ------------------------------------------------------------------ #
#  4. conditional normalising flow (RealNVP)
# ------------------------------------------------------------------ #
class AffineCouplingLayer(nn.Module):
    def __init__(self, mask, context_dim, hid_dim=64):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.scale_net = nn.Sequential(
            nn.Linear(2 + context_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, 2)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(2 + context_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, 2)
        )

    def forward(self, x, context, reverse=False):
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(x.shape[0], -1)
        x_masked = x * self.mask
        net_in = torch.cat([x_masked, context], dim=1)
        s = self.scale_net(net_in)
        t = self.translate_net(net_in)
        if not reverse:
            y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = ((1 - self.mask) * s).sum(dim=1)
        else:
            y = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -((1 - self.mask) * s).sum(dim=1)
        return y, log_det


class ConditionalPointFlow(nn.Module):
    def __init__(self, point_dim=2, param_dim=3, hid_dim=128, n_coupling=6):
        super().__init__()
        self.encoder = DeepSetEncoder(point_dim, param_dim, hid_dim, out_dim=hid_dim)
        self.coupling_layers = nn.ModuleList()
        for i in range(n_coupling):
            mask = torch.zeros(2)
            mask[i % 2] = 1.0
            self.coupling_layers.append(
                AffineCouplingLayer(mask, context_dim=hid_dim, hid_dim=hid_dim))

    def forward(self, x, condition, reverse=False):
        context = self.encoder(x, condition)
        B = len(x)
        total_log_prob = torch.zeros(B, device=context.device)
        outs = []
        for i in range(B):
            xi = x[i].to(context.device)
            if xi.numel() == 0:
                outs.append(xi)
                continue
            ctx = context[i]
            if not reverse:
                z = xi
                log_det_sum = 0.0
                for layer in self.coupling_layers:
                    z, log_det = layer(z, ctx, reverse=False)
                    log_det_sum += log_det
                base_log_prob = -0.5 * (z**2).sum(dim=1) - 0.5*np.log(2*np.pi)*2
                total_log_prob[i] = (base_log_prob + log_det_sum).mean()
                outs.append(z)
            else:
                z = xi
                for layer in reversed(self.coupling_layers):
                    z, _ = layer(z, ctx, reverse=True)
                outs.append(z)
        if not reverse:
            return total_log_prob, outs
        else:
            return outs

    def sample(self, condition, n_points, device=torch.device('cpu')):
        B = condition.shape[0]
        z = [torch.randn(n_points, 2, device=device) for _ in range(B)]
        return self.forward(z, condition.to(device), reverse=True)


# ------------------------------------------------------------------ #
#  5. inference network  (DeepSet + Mixture Density Network)
# ------------------------------------------------------------------ #
class MDNHead(nn.Module):
    def __init__(self, in_dim, out_dim=2, n_components=6, hid_dim=64):
        super().__init__()
        self.n_comp = n_components
        self.out_dim = out_dim
        self.logit = nn.Linear(in_dim, n_components)
        self.mu    = nn.Linear(in_dim, n_components * out_dim)
        self.logs  = nn.Linear(in_dim, n_components * out_dim)

    def forward(self, z):
        logits = self.logit(z)
        mus   = self.mu(z).view(-1, self.n_comp, self.out_dim)
        logs  = self.logs(z).view(-1, self.n_comp, self.out_dim).clamp(-2.5, 2.5)
        return logits, mus, logs


class InferenceNetwork(nn.Module):
    def __init__(self, point_dim=2, n_params=2, hidden=128):
        super().__init__()
        # The encoder receives a dummy condition (zeros) because it doesn't know η
        self.encoder = DeepSetEncoder(point_dim, cond_dim=n_params,
                                      hid_dim=hidden, out_dim=hidden)
        self.mdn = MDNHead(hidden, out_dim=n_params, n_components=6, hid_dim=hidden)

    def forward(self, x, condition):
        z = self.encoder(x, condition)
        return self.mdn(z)

    def nll(self, x, theta_true, condition):
        logits, mus, logs = self(x, condition)
        theta_true = theta_true.unsqueeze(1)
        comp = -0.5*(((theta_true - mus)/logs.exp())**2) - logs - 0.5*np.log(2*np.pi)
        log_prob_comp = comp.sum(dim=-1)
        lw = F.log_softmax(logits, dim=-1)
        return -torch.logsumexp(lw + log_prob_comp, dim=-1).mean()

    @torch.no_grad()
    def sample_posterior(self, x, condition, n_samples=1000):
        logits, mus, logs = self(x, condition)
        B = len(x) if isinstance(x, list) else x.shape[0]
        w = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(w, n_samples, replacement=True)
        bi = torch.arange(B).unsqueeze(1).expand(-1, n_samples)
        mu_samp = mus[bi, idx]
        logs_samp = logs[bi, idx]
        return mu_samp + logs_samp.exp() * torch.randn_like(mu_samp)


# ------------------------------------------------------------------ #
#  6. training routines
# ------------------------------------------------------------------ #
def train_generative_model(flow, rng, n_iter=3000, batch_size=32, lr=1e-3,
                           device=torch.device('cpu')):
    flow.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    losses = []
    for step in range(n_iter):
        clouds, theta, eta = generate_batch(batch_size, rng)
        condition = torch.cat([theta, eta.unsqueeze(1)], dim=1).to(device)
        clouds = [c.to(device) for c in clouds]
        log_prob, _ = flow(clouds, condition, reverse=False)
        loss = -log_prob.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"Gen [{step:5d}/{n_iter}] loss: {loss.item():.3f}")
    return losses


def train_inference_network(infer_net, flow, rng, n_iter=2000, batch_size=64, lr=2e-4,
                            device=torch.device('cpu')):
    flow.eval()
    infer_net.to(device)
    opt = torch.optim.Adam(infer_net.parameters(), lr=lr, weight_decay=1e-5)
    dummy_cond = torch.zeros(1, 2).to(device)
    losses = []
    for step in range(n_iter):
        theta, eta = sample_prior(batch_size, rng)
        theta_t = torch.from_numpy(theta).to(device)
        eta_t   = torch.from_numpy(eta).to(device)
        cond = torch.cat([theta_t, eta_t.unsqueeze(1)], dim=1)
        clouds_syn = flow.sample(cond, n_points=60, device=device)
        dummy_batch = dummy_cond.expand(batch_size, -1)
        loss = infer_net.nll(clouds_syn, theta_t, dummy_batch)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"Inf [{step:5d}/{n_iter}] loss: {loss.item():.3f}")
    return losses


# ------------------------------------------------------------------ #
#  7. single‑realisation calibration (jackknife + conformal)
# ------------------------------------------------------------------ #
def jackknife_calibration(infer_net, obs_cloud, alpha=0.1, K=10, n_samp=500,
                          device=torch.device('cpu')):
    infer_net.eval()
    dummy_cond = torch.zeros(1, 2).to(device)
    # full posterior
    with torch.no_grad():
        full_samp = infer_net.sample_posterior([obs_cloud.to(device)],
                                               dummy_cond, n_samples=n_samp).squeeze(0)
    full_mean = full_samp.mean(dim=0)
    full_cov  = torch.cov(full_samp.T)

    # jackknife by x‑coordinate splits
    x_coords = obs_cloud[:, 0]
    quantiles = np.linspace(0, 1, K+1)
    scores = []
    for k in range(K):
        lo, hi = quantiles[k], quantiles[k+1]
        mask = (x_coords < lo) | (x_coords >= hi)
        masked_cloud = obs_cloud[mask]
        if len(masked_cloud) < 5:
            continue
        with torch.no_grad():
            samp_k = infer_net.sample_posterior([masked_cloud.to(device)],
                                                dummy_cond, n_samples=n_samp//2).squeeze(0)
        mean_k = samp_k.mean(dim=0)
        diff = mean_k - full_mean
        score = torch.sqrt((diff @ torch.linalg.inv(full_cov) @ diff.T).clamp(min=0))
        scores.append(score.item())
    q_hat = np.quantile(scores, 1 - alpha) if scores else 1.0
    return q_hat, full_mean.cpu().numpy(), full_cov.cpu().numpy()


# ------------------------------------------------------------------ #
#  8. diagnostics (SBC, coverage, PPC)
# ------------------------------------------------------------------ #
def compute_sbc_ranks(infer_net, rng, n_samples=500, n_posterior=400,
                      device=torch.device('cpu')):
    infer_net.eval()
    dummy_cond = torch.zeros(1, 2).to(device)
    theta_prior, _ = sample_prior(n_samples, rng)
    ranks = []
    for i in range(n_samples):
        pts = simulate_thomas(theta_prior[i], rng.uniform(ETA_LO, ETA_HI), rng)
        cloud = torch.from_numpy(pts).to(device)
        with torch.no_grad():
            post = infer_net.sample_posterior([cloud], dummy_cond,
                                              n_samples=n_posterior).squeeze(0)
        true = torch.from_numpy(theta_prior[i]).to(device)
        rank0 = (post[:, 0] < true[0]).sum().item()
        rank1 = (post[:, 1] < true[1]).sum().item()
        ranks.append([rank0, rank1])
    return np.array(ranks)


def sbc_uniformity_pvalues(ranks, n_posterior, nbins=10):
    p_vals = []
    for d in range(ranks.shape[1]):
        counts, _ = np.histogram(ranks[:, d], bins=nbins, range=(0, n_posterior))
        expected = len(ranks) / nbins
        chi2_stat = ((counts - expected)**2 / expected).sum()
        p_vals.append(chi2.sf(chi2_stat, nbins - 1))
    return np.array(p_vals)


def compute_coverage(post_samples, true_theta, levels):
    """post_samples: (n_test, n_post, n_param)"""
    n_test, n_post, n_param = post_samples.shape
    cov = np.zeros((n_param, len(levels)))
    for i in range(n_test):
        for d in range(n_param):
            for j, lv in enumerate(levels):
                lo = np.quantile(post_samples[i, :, d], (1 - lv) / 2)
                hi = np.quantile(post_samples[i, :, d], 1 - (1 - lv) / 2)
                if lo <= true_theta[i, d] <= hi:
                    cov[d, j] += 1
    return cov / n_test


def posterior_predictive_discrepancy(infer_net, flow, cloud, n_draws=60,
                                     device=torch.device('cpu')):
    infer_net.eval(); flow.eval()
    dummy_cond = torch.zeros(1, 2).to(device)
    with torch.no_grad():
        post_th = infer_net.sample_posterior([cloud.to(device)],
                                             dummy_cond, n_samples=n_draws).squeeze(0)
    xi_preds = []
    rng = np.random.default_rng()
    for i in range(n_draws):
        th = post_th[i].cpu().numpy()
        eta = rng.uniform(ETA_LO, ETA_HI)
        cond = torch.tensor([[th[0], th[1], eta]], device=device)
        syn_cloud = flow.sample(cond, n_points=len(cloud), device=device)[0].cpu().numpy()
        xi_preds.append(landy_szalay(syn_cloud))
    xi_preds = np.array(xi_preds)
    xi_obs = landy_szalay(cloud.cpu().numpy())
    pred_mean = xi_preds.mean(axis=0)
    pred_std  = xi_preds.std(axis=0) + 1e-6
    disc = np.mean(((xi_obs - pred_mean) / pred_std) ** 2)
    return disc, xi_obs, xi_preds