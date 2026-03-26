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
    # masked average-pool: ignore invalid (zero) pixels so depth_down reflects
    # true depth rather than being diluted by invalid pixels
    depth_flat  = gt_depth.reshape(B * S, 1, H, W)
    valid_flat  = (depth_flat > 0).float()
    valid_pool  = F.adaptive_avg_pool2d(valid_flat, (Hp, Wp))          # (B*S,1,Hp,Wp)
    depth_sum   = F.adaptive_avg_pool2d(depth_flat * valid_flat, (Hp, Wp))
    depth_down  = (depth_sum / valid_pool.clamp(min=1e-6)).reshape(B, S, Hp, Wp)
    depth_down  = depth_down * (valid_pool > 0).reshape(B, S, Hp, Wp).float()

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
    Dc = Dc.clamp(min=1e-6, max=1e6)

    l1   = (pred_depth.float() - gt_depth_norm.float()).abs()  # (B,S,H,W)
    m    = mask.float()
    # Both terms summed over valid pixels Ω only (paper eq. for L_D)
    loss = m * (Dc * l1 - lambda_c * torch.log(Dc))

    Z = m.sum().clamp(min=1)
    return loss.sum() / Z


def _lm(pred_ray, ray_gt_norm):
    """L1 ray-map loss."""
    return F.l1_loss(pred_ray.float(), ray_gt_norm.float())


def _lp(pred_depth_down, pred_ray, point_gt_norm, mask_down=None):
    """
    Point-map loss: predicted 3D point = depth ⊙ dir + trans vs GT.

    pred_depth_down : (B, S, Hp, Wp)
    pred_ray        : (B, S, Hp, Wp, 6)  dims 0–2 = dir, 3–5 = trans
    point_gt_norm   : (B, S, Hp, Wp, 3)  normalised GT point map
    mask_down       : (B, S, Hp, Wp) bool, optional — restrict to valid depth
                      pixels (needed when some frames have no GT depth, e.g.
                      the right frame of a stereo pair)
    """
    pred_dir   = pred_ray[..., :3].float()
    pred_trans = pred_ray[..., 3:].float()
    pred_pts   = pred_depth_down.float().unsqueeze(-1) * pred_dir + pred_trans
    if mask_down is None:
        return F.l1_loss(pred_pts, point_gt_norm.float())
    diff = (pred_pts - point_gt_norm.float()).abs().mean(dim=-1)   # (B,S,Hp,Wp)
    Z    = mask_down.float().sum().clamp(min=1)
    return (diff * mask_down.float()).sum() / Z


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

    # Downsample mask to ray-head resolution for LP
    # (needed when some frames have no GT depth, e.g. stereo right frame)
    mask_down = F.adaptive_avg_pool2d(
        mask_full.float().reshape(B * S, 1, H, W), (Hp, Wp)
    ).reshape(B, S, Hp, Wp) > 0

    # ── individual losses ─────────────────────────────────────────────────────
    ld    = _ld(pred_depth / scale, depth_norm, depth_conf, mask_full, lambda_c)
    lm    = _lm(pred_ray_norm, ray_gt_norm)
    lp    = _lp(pred_depth_down_norm, pred_ray_norm, point_gt_norm, mask_down)
    lgrad = _lgrad(pred_depth / scale, depth_norm, mask_full)

    total = ld + lm + lp + alpha * lgrad

    return total, {"LD": ld.item(), "LM": lm.item(), "LP": lp.item(),
                   "Lgrad": lgrad.item(), "scale": scale.item()}


# ── photometric helpers ───────────────────────────────────────────────────────

def _warp(
    src_img: torch.Tensor,      # (B, 3, H, W)
    depth_tgt: torch.Tensor,    # (B, H, W)
    K: torch.Tensor,            # (B, 3, 3)
    T_src_from_tgt: torch.Tensor,  # (B, 4, 4)  world→src  ∘  tgt→world
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Warp src_img into the target view using predicted depth_tgt and relative pose.

    Returns:
        warped : (B, 3, H, W)  source image sampled at projected coordinates
        valid  : (B, H, W) bool  pixels that project inside the source image
    """
    B, H, W = depth_tgt.shape
    device  = depth_tgt.device

    # pixel grid for target frame (homogeneous)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones   = torch.ones_like(xs)
    coords = torch.stack([xs, ys, ones], dim=0)              # (3, H, W)
    coords_flat = coords.reshape(3, -1)                       # (3, H*W)

    # lift to target camera space:  p_cam = K⁻¹ · [u,v,1] · d
    K_inv    = torch.linalg.inv(K.float())                   # (B, 3, 3)
    cam_dirs = torch.einsum("bij,jn->bin", K_inv, coords_flat)  # (B, 3, H*W)
    cam_pts  = cam_dirs * depth_tgt.reshape(B, 1, -1)        # (B, 3, H*W)

    # transform to source camera space
    ones_hw  = torch.ones(B, 1, H * W, device=device)
    cam_pts_h = torch.cat([cam_pts, ones_hw], dim=1)         # (B, 4, H*W)
    src_pts  = torch.bmm(T_src_from_tgt[:, :3, :].float(), cam_pts_h)  # (B, 3, H*W)

    # project into source image
    src_2d   = torch.bmm(K.float(), src_pts)                 # (B, 3, H*W)
    z        = src_2d[:, 2:3, :].clamp(min=1e-6)
    src_2d   = src_2d[:, :2, :] / z                          # (B, 2, H*W)

    # normalise to [-1, 1] for grid_sample
    src_x = (src_2d[:, 0, :] / (W - 1)) * 2 - 1
    src_y = (src_2d[:, 1, :] / (H - 1)) * 2 - 1
    grid  = torch.stack([src_x, src_y], dim=-1).reshape(B, H, W, 2)

    valid = (
        (src_x > -1) & (src_x < 1) &
        (src_y > -1) & (src_y < 1) &
        (z.squeeze(1) > 0)
    ).reshape(B, H, W)

    warped = F.grid_sample(src_img.float(), grid,
                           mode="bilinear", padding_mode="zeros",
                           align_corners=True)
    return warped, valid


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Simplified SSIM via 3×3 average pooling (Monodepth2 style).
    Returns per-pixel map in [0, 1]; lower = more similar.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x  = F.avg_pool2d(x, 3, 1, 1)
    mu_y  = F.avg_pool2d(y, 3, 1, 1)
    sig_x  = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sig_y  = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sig_xy = F.avg_pool2d(x * y,  3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
    return (1 - num / den.clamp(min=1e-6)) / 2   # (B, C, H, W) in [0,1]


def _photo_err(tgt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Per-pixel photometric error: 0.85·SSIM + 0.15·L1, mean over channels."""
    return (0.85 * _ssim(tgt, src) + 0.15 * (tgt - src).abs()).mean(dim=1)  # (B,H,W)


def _smooth(pred_depth: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """
    Edge-aware depth smoothness.  Depth gradients are down-weighted where
    image gradients are large (edges), so the model can be smooth in textureless
    regions without being penalised for sharp depth discontinuities at edges.

    pred_depth : (B, S, H, W)
    images     : (B, S, 3, H, W)
    """
    # mean-normalise depth per frame so smoothness is scale-invariant
    d   = pred_depth / pred_depth.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    img = images.mean(dim=2)  # (B, S, H, W) — luminance proxy

    d_dx = (d[:, :, :, 1:] - d[:, :, :, :-1]).abs()
    d_dy = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs()
    i_dx = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
    i_dy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()

    return (d_dx * torch.exp(-i_dx)).mean() + (d_dy * torch.exp(-i_dy)).mean()


# ── Stage 2b combined loss ────────────────────────────────────────────────────

def stage2b_loss(
    out: dict,
    images: torch.Tensor,      # (B, S, 3, H, W)  S consecutive frames
    c2w: torch.Tensor,          # (B, S, 4, 4)  known camera-to-world poses
    K: torch.Tensor,            # (B, 3, 3)
    *,
    alpha: float = 0.1,        # edge-aware smoothness weight
    gamma: float = 1.0,        # photometric loss weight
    auto_mask: bool = True,    # exclude static pixels (Monodepth2 trick)
) -> tuple[torch.Tensor, dict]:
    """
    Self-supervised Stage 2b loss: photometric reprojection + smoothness.

    For each target frame, warp every source frame into the target view using
    predicted depth + known relative pose, then compute a SSIM+L1 photometric
    error.  Two masks are applied:

      1. warp_valid — excludes pixels that project outside the source image bounds
      2. auto_mask  — excludes pixels where the unwarped source has lower error
                      than the warped source (handles static regions, black lens
                      borders, and text-on-border artefacts — all of which are
                      consistent across frames and thus filtered by this check)

    Note: the black circular FOV border of capsule cameras is identical across
    frames, so its identity error ≈ 0 ≤ warped error, and it is automatically
    excluded by the auto-mask without needing an explicit FOV mask.

    L = γ · Lphoto + α · Lsmooth
    """
    pred_depth = out["depth"]          # (B, S, H, W)
    B, S, H, W = pred_depth.shape

    total_photo = pred_depth.new_zeros(())
    n_pairs     = 0

    for tgt_s in range(S):
        tgt_img   = images[:, tgt_s]           # (B, 3, H, W)
        depth_tgt = pred_depth[:, tgt_s]       # (B, H, W)
        c2w_tgt   = c2w[:, tgt_s]             # (B, 4, 4)

        src_errors  = []
        id_errors   = []
        valid_union = tgt_img.new_zeros(B, H, W, dtype=torch.bool)

        for src_s in range(S):
            if src_s == tgt_s:
                continue
            src_img = images[:, src_s]         # (B, 3, H, W)
            c2w_src = c2w[:, src_s]            # (B, 4, 4)

            w2c_src        = torch.linalg.inv(c2w_src.float())
            T_src_from_tgt = w2c_src @ c2w_tgt.float()

            warped, warp_valid = _warp(src_img, depth_tgt, K, T_src_from_tgt)
            valid_union = valid_union | warp_valid

            pe = _photo_err(tgt_img, warped) * warp_valid.float()
            src_errors.append(pe)

            if auto_mask:
                id_errors.append(_photo_err(tgt_img, src_img))

        if not src_errors:
            continue

        min_photo, _ = torch.stack(src_errors, dim=0).min(dim=0)  # (B, H, W)

        if auto_mask and id_errors:
            min_id, _  = torch.stack(id_errors, dim=0).min(dim=0)
            final_mask = (min_photo < min_id) & valid_union
        else:
            final_mask = valid_union

        denom = final_mask.float().sum().clamp(min=1)
        total_photo += (min_photo * final_mask.float()).sum() / denom
        n_pairs += 1

    lphoto  = total_photo / max(n_pairs, 1)
    lsmooth = _smooth(pred_depth, images)

    total = gamma * lphoto + alpha * lsmooth
    return total, {"Lphoto": lphoto.item(), "Lsmooth": lsmooth.item()}


def distillation_loss(
    student_out: dict,
    teacher_depth: torch.Tensor,   # (B, S, H, W)  teacher prediction, no grad
    images: torch.Tensor,          # (B, S, 3, H, W)
    *,
    beta: float = 1.0,             # distillation weight
    alpha: float = 0.1,            # edge-aware smoothness weight
) -> tuple[torch.Tensor, dict]:
    """
    Teacher-student distillation loss for Stage 2b on unlabelled real data.

    Scale-invariant L1: both student and teacher depths are normalised by
    their per-frame median before computing L1, so the loss is insensitive
    to the absolute scale difference between the two models.

    L = β · Ldistill + α · Lsmooth
    """
    student_depth = student_out["depth"]   # (B, S, H, W)
    B, S = student_depth.shape[:2]

    # Normalise both by the teacher's median — anchors student to teacher's scale.
    # Using independent medians would allow student scale to collapse (minimising
    # the loss while drifting to near-zero depth via median compensation).
    t_med = teacher_depth.flatten(2).median(dim=2).values.view(B, S, 1, 1).clamp(min=1e-6)

    s_norm = student_depth / t_med
    t_norm = teacher_depth / t_med

    ldistill = F.l1_loss(s_norm, t_norm)
    lsmooth  = _smooth(student_depth, images)

    total = beta * ldistill + alpha * lsmooth
    return total, {"Ldistill": ldistill.item(), "Lsmooth": lsmooth.item()}


def depth_only_loss(
    out: dict,
    gt_depth: torch.Tensor,    # (B, S, H, W)
    *,
    alpha: float = 1.0,
    lambda_c: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Simplified loss for Stage 2a (no GT poses available):
        L = LD + α·Lgrad

    Scale normalisation uses mean GT depth (no point map available).
    """
    pred_depth = out["depth"]
    depth_conf = out["depth_conf"]

    valid = gt_depth > 0
    scale = gt_depth[valid].mean().clamp(min=1e-6) if valid.any() else gt_depth.mean().clamp(min=1e-6)

    depth_norm = gt_depth / scale
    mask       = valid

    ld    = _ld(pred_depth / scale, depth_norm, depth_conf, mask, lambda_c)
    lgrad = _lgrad(pred_depth / scale, depth_norm, mask)
    total = ld + alpha * lgrad

    return total, {"LD": ld.item(), "Lgrad": lgrad.item(), "scale": scale.item()}
