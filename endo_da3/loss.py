"""
DA3-style training loss for Endo-DA3.

Paper: "Depth Anything 3" (arXiv:2511.10647), Section on Training Objectives.

Full loss (per batch):
    L = LD + LM + LP + α·Lgrad  (+β·LC when camera decoder is active)

    LD  — confidence-weighted L1 depth loss with log-barrier on confidence Dc
    LM  — L1 ray-map loss (predicted vs GT ray map M)
    LP  — L1 point-map loss: predicted 3D points (D̂⊙d + t) vs GT point map P
    Lgrad — depth gradient loss (preserves sharp edges)
    LC  — optional camera-pose loss (β=1, not used in Stage 1)

Normalisation: all GT signals are divided by the mean ℓ2 norm of valid GT point
vectors P before any loss computation (stabilises scale across scenes).

Ray format (B, S, Hp, Wp, 6):
    dims 0–2 : world-space ray direction per patch  — NOT normalised.
               d = R @ K⁻¹[u,v,1]; z-component = 1 in camera frame, so
               D(u,v) × d correctly lifts z-depth to a 3D point (§3.1).
               Preserving |d| encodes the camera projection scale.
    dims 3–5 : camera centre in world space (same for every patch in a frame)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


# ── GT ray / point-map builders ───────────────────────────────────────────────

def compute_gt_ray_and_pointmap(
    gt_depth: torch.Tensor,     # (B, S, H, W)
    c2w: torch.Tensor,          # (B, S, 4, 4)
    K: torch.Tensor,            # (B, 3, 3) pixel-space intrinsics at img_size
    ray_hw: tuple[int, int],    # (Hp, Wp) — spatial size of the ray head output
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        ray_gt    : (B, S, Hp, Wp, 6)  GT ray map M
        point_gt  : (B, S, Hp, Wp, 3)  GT point map P  (= depth_down ⊙ dir + trans)
        depth_down: (B, S, Hp, Wp)     GT depth at ray head resolution
        scale     : scalar             normalisation factor (mean ‖P‖₂ of valid pts)
    """
    B, S, H, W = gt_depth.shape
    Hp, Wp = ray_hw
    device = gt_depth.device

    # ── step 1: GT ray directions at ray-head resolution ─────────────────────
    # patch-centre pixel coordinates in the full H×W image
    # ray head has Hp×Wp spatial positions; each covers (H/Hp) × (W/Wp) pixels
    dx = W / Wp
    dy = H / Hp
    js = (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) * dx
    is_ = (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) * dy
    grid_y, grid_x = torch.meshgrid(is_, js, indexing="ij")    # (Hp, Wp)
    ones = torch.ones_like(grid_x)
    coords = torch.stack([grid_x, grid_y, ones], dim=-1)        # (Hp, Wp, 3)
    coords_flat = coords.reshape(-1, 3).T                        # (3, N), N=Hp*Wp

    # K⁻¹ per batch item
    K_inv = torch.linalg.inv(K.float())                         # (B, 3, 3)
    # camera-space ray directions: (B, 3, N)
    cam_dirs = torch.einsum("bij,jn->bin", K_inv, coords_flat.to(K_inv.dtype))

    # rotate to world space: (B, S, 3, N)
    R = c2w[:, :, :3, :3].float()                              # (B, S, 3, 3)
    t = c2w[:, :, :3,  3].float()                              # (B, S, 3)
    cam_dirs_bs = cam_dirs.unsqueeze(1).expand(-1, S, -1, -1)  # (B, S, 3, N)
    world_dirs = torch.einsum("bsij,bsjn->bsin", R, cam_dirs_bs)
    # do NOT normalise: |d| preserves the projection scale (paper §3.1)
    world_dirs = world_dirs.permute(0, 1, 3, 2).reshape(B, S, Hp, Wp, 3)

    # translation broadcast to all patches
    t_expanded = t.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, Hp, Wp, -1)  # (B, S, Hp, Wp, 3)

    ray_gt = torch.cat([world_dirs, t_expanded], dim=-1)        # (B, S, Hp, Wp, 6)

    # ── step 2: GT depth at ray-head resolution ───────────────────────────────
    # average-pool depth to (Hp, Wp); mask=0 regions contribute 0 (ignore)
    depth_flat = gt_depth.reshape(B * S, 1, H, W)
    depth_down = F.adaptive_avg_pool2d(depth_flat, (Hp, Wp)).reshape(B, S, Hp, Wp)

    # ── step 3: GT point map P = depth_down ⊙ dir + trans ────────────────────
    point_gt = depth_down.unsqueeze(-1) * world_dirs + t_expanded  # (B, S, Hp, Wp, 3)

    # ── step 4: normalisation scale = mean ‖P‖₂ of valid points ─────────────
    valid = depth_down > 0                                      # (B, S, Hp, Wp)
    norms = point_gt.norm(dim=-1)                               # (B, S, Hp, Wp)
    scale = norms[valid].mean().clamp(min=1e-6) if valid.any() else norms.mean().clamp(min=1e-6)

    return ray_gt, point_gt, depth_down, scale


# ── individual loss terms ─────────────────────────────────────────────────────

def _ld(pred_depth, gt_depth_norm, depth_conf, mask, lambda_c: float = 1.0):
    """
    Confidence-weighted L1 depth loss with log-barrier on confidence.

    LD = (1/Z) Σ_{p∈Ω} [ mask_p · Dc_p · |D̂_p − D_p| − λc · log(Dc_p) ]

    The log-barrier prevents the model from driving Dc → 0 to trivially
    minimise the weighted term.
    """
    Dc = depth_conf.float()
    if Dc.ndim == 5:           # (B,S,1,H,W) → (B,S,H,W)
        Dc = Dc.squeeze(2)
    Dc = Dc.clamp(min=1e-6)

    l1   = (pred_depth.float() - gt_depth_norm.float()).abs()  # (B,S,H,W)
    m    = mask.float()
    # Both terms summed over valid pixels Ω only (paper eq. for L_D)
    loss = m * (Dc * l1 - lambda_c * torch.log(Dc))

    Z = m.sum().clamp(min=1)
    return loss.sum() / Z


def _lm(pred_ray, ray_gt_norm):
    """L1 ray-map loss."""
    return F.l1_loss(pred_ray.float(), ray_gt_norm.float())


def _lp(pred_depth_down, pred_ray, point_gt_norm):
    """
    Point-map loss: predicted 3D point = depth ⊙ dir + trans vs GT.

    pred_depth_down : (B, S, Hp, Wp)
    pred_ray        : (B, S, Hp, Wp, 6)  dims 0–2 = dir, 3–5 = trans
    point_gt_norm   : (B, S, Hp, Wp, 3)  normalised GT point map
    """
    pred_dir   = pred_ray[..., :3].float()
    pred_trans = pred_ray[..., 3:].float()
    pred_pts   = pred_depth_down.float().unsqueeze(-1) * pred_dir + pred_trans
    return F.l1_loss(pred_pts, point_gt_norm.float())


def _lgrad(pred_depth, gt_depth_norm, mask):
    """Gradient loss on depth maps (L1 on finite differences)."""
    m = mask.float()
    pd = pred_depth.float()
    gd = gt_depth_norm.float()

    dx_pred = pd[..., :, 1:] - pd[..., :, :-1]
    dx_gt   = gd[..., :, 1:] - gd[..., :, :-1]
    mx      = m[..., :, 1:] * m[..., :, :-1]

    dy_pred = pd[..., 1:, :] - pd[..., :-1, :]
    dy_gt   = gd[..., 1:, :] - gd[..., :-1, :]
    my      = m[..., 1:, :] * m[..., :-1, :]

    loss = ((dx_pred - dx_gt).abs() * mx).mean() + \
           ((dy_pred - dy_gt).abs() * my).mean()
    return loss


# ── combined DA3 loss ─────────────────────────────────────────────────────────

def da3_loss(
    out: dict,
    gt_depth: torch.Tensor,    # (B, S, H, W)
    c2w: torch.Tensor,         # (B, S, 4, 4)
    K: torch.Tensor,           # (B, 3, 3)
    *,
    alpha: float = 1.0,        # Lgrad weight (paper: α=1)
    lambda_c: float = 1.0,     # log-barrier strength on depth confidence
) -> tuple[torch.Tensor, dict]:
    """
    Full DA3 training loss: LD + LM + LP + α·Lgrad.

    Args:
        out      : model output dict with keys 'depth', 'depth_conf', 'ray', 'ray_conf'
        gt_depth : (B, S, H, W) GT depth in metres
        c2w      : (B, S, 4, 4) GT camera-to-world matrices
        K        : (B, 3, 3) pixel-space intrinsics
        alpha    : Lgrad weight
        lambda_c : log-barrier weight on depth confidence

    Returns:
        total loss (scalar), dict of individual loss terms for logging
    """
    pred_depth = out["depth"]          # (B, S, H, W)
    depth_conf = out["depth_conf"]     # (B, S, H, W) or (B, S, 1, H, W)
    pred_ray   = out["ray"]            # (B, S, Hp, Wp, 6)
    H, W   = pred_depth.shape[-2:]
    Hp, Wp = pred_ray.shape[2], pred_ray.shape[3]

    # ── GT ray map + point map + depth at ray resolution ─────────────────────
    ray_gt, point_gt, _, scale = compute_gt_ray_and_pointmap(
        gt_depth, c2w, K, ray_hw=(Hp, Wp)
    )

    # ── normalise all GT signals by scale ────────────────────────────────────
    depth_norm      = gt_depth / scale                  # (B, S, H, W)
    point_gt_norm   = point_gt / scale                  # (B, S, Hp, Wp, 3)
    # ray directions are unit vectors — no scale needed; translations are scaled
    ray_gt_norm = ray_gt.clone()
    ray_gt_norm[..., 3:] = ray_gt_norm[..., 3:] / scale

    # Normalise predicted ray translations by the same scale so they're comparable
    pred_ray_norm = pred_ray.clone()
    pred_ray_norm[..., 3:] = pred_ray_norm[..., 3:] / scale

    # Downsample predicted depth to ray head resolution for LP
    B, S, H, W = pred_depth.shape
    pred_depth_down = F.adaptive_avg_pool2d(
        pred_depth.reshape(B * S, 1, H, W), (Hp, Wp)
    ).reshape(B, S, Hp, Wp)
    pred_depth_down_norm = pred_depth_down / scale

    # ── validity mask ─────────────────────────────────────────────────────────
    mask_full = depth_norm > 0                          # (B, S, H, W)

    # ── individual losses ─────────────────────────────────────────────────────
    ld    = _ld(pred_depth / scale, depth_norm, depth_conf, mask_full, lambda_c)
    lm    = _lm(pred_ray_norm, ray_gt_norm)
    lp    = _lp(pred_depth_down_norm, pred_ray_norm, point_gt_norm)
    lgrad = _lgrad(pred_depth / scale, depth_norm, mask_full)

    total = ld + lm + lp + alpha * lgrad

    return total, {"LD": ld.item(), "LM": lm.item(), "LP": lp.item(),
                   "Lgrad": lgrad.item(), "scale": scale.item()}
