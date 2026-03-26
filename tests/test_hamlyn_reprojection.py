"""
Reprojection test for Hamlyn daVinci stereo dataset.

Verifies camera intrinsics/extrinsics by:
1. Computing disparity (left→right) via OpenCV SGBM
2. Warping left image into right view using disparity + cam params
3. Comparing warped left vs actual right image (MAE, visualisation)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Camera parameters (from Hamlyn daVinci README / repo)
# ---------------------------------------------------------------------------
FX = FY = 373.47833252
CX = 182.91804504
CY = 113.72999573
BASELINE_M = 5.63117313e-3  # metres (|tx|)
W, H = 384, 192

DATA_ROOT = Path("/home/in4218/code/data/Hamlyn/daVinci/train")
OUT_PATH  = Path("/home/in4218/code/endo-da3/tests/hamlyn_reprojection.png")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stereo_pair(idx: int = 1):
    name = f"{idx:06d}.png"
    left  = cv2.imread(str(DATA_ROOT / "image_0" / name))
    right = cv2.imread(str(DATA_ROOT / "image_1" / name))
    assert left is not None and right is not None, f"Failed to load frame {name}"
    return left, right          # BGR, uint8, (H, W, 3)


def compute_disparity(left_bgr, right_bgr):
    """Semi-global block matching on grayscale images → left disparity map."""
    lg = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,     # must be divisible by 16
        blockSize=5,
        P1=8  * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(lg, rg).astype(np.float32) / 16.0  # SGBM returns fixed-point x16
    disp[disp <= 0] = np.nan
    return disp                 # pixels, left-camera disparity


def disparity_to_depth(disp):
    """depth (m) = fx * B / disparity"""
    with np.errstate(invalid='ignore', divide='ignore'):
        depth = np.where(np.isfinite(disp) & (disp > 0),
                         FX * BASELINE_M / disp,
                         np.nan)
    return depth


def warp_left_to_right(left_bgr, disp):
    """
    For rectified stereo, a left pixel (u, v) with disparity d maps to
    right pixel (u - d, v).  We forward-warp by building a remap.

    We use an inverse-warp instead: for each right pixel (u_r, v_r),
    sample the left image at (u_r + d(u_r, v_r), v_r).
    Since we only have left disparity, we approximate d at u_r ≈ d at u_r.
    """
    u_coords = np.arange(W, dtype=np.float32)
    v_coords = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)    # (H, W)

    # map_x: for right pixel (u_r, v_r), sample left at u_r + disp
    map_x = (uu + disp).astype(np.float32)
    map_y = vv.astype(np.float32)

    warped = cv2.remap(left_bgr, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def photometric_error(warped, right_bgr, disp):
    """MAE over valid (non-NaN disparity) pixels."""
    valid = np.isfinite(disp)                          # (H, W)
    diff  = np.abs(warped.astype(np.float32) -
                   right_bgr.astype(np.float32))       # (H, W, 3)
    mae   = diff[valid].mean()
    return mae, valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    frame_idx = 1000   # pick a mid-sequence frame

    print(f"Loading frame {frame_idx:06d} …")
    left, right = load_stereo_pair(frame_idx)

    print("Computing SGBM disparity …")
    disp  = compute_disparity(left, right)
    depth = disparity_to_depth(disp)

    print("Warping left → right …")
    warped = warp_left_to_right(left, disp)

    mae, valid_mask = photometric_error(warped, right, disp)
    valid_pct = valid_mask.mean() * 100
    depth_valid = depth[valid_mask]
    print(f"Valid pixels : {valid_pct:.1f}%")
    print(f"Depth range  : {np.nanmin(depth_valid)*1000:.1f} – {np.nanmax(depth_valid)*1000:.1f} mm")
    print(f"Depth median : {np.nanmedian(depth_valid)*1000:.1f} mm")
    print(f"Reprojection MAE : {mae:.2f} / 255")

    # -----------------------------------------------------------------------
    # Visualise
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.15)

    def show(ax, img_bgr, title):
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    show(fig.add_subplot(gs[0, 0]), left,   "Left (image_0)")
    show(fig.add_subplot(gs[0, 1]), right,  "Right (image_1)")
    show(fig.add_subplot(gs[0, 2]), warped, "Warped left → right")

    # Disparity
    ax_d = fig.add_subplot(gs[1, 0])
    im = ax_d.imshow(disp, cmap="plasma", vmin=0, vmax=np.nanpercentile(disp, 98))
    ax_d.set_title("Disparity (px)", fontsize=11)
    ax_d.axis("off")
    plt.colorbar(im, ax=ax_d, fraction=0.046)

    # Depth
    ax_z = fig.add_subplot(gs[1, 1])
    im2 = ax_z.imshow(depth * 1000, cmap="magma_r",
                      vmin=0, vmax=np.nanpercentile(depth_valid * 1000, 98))
    ax_z.set_title("Depth (mm)", fontsize=11)
    ax_z.axis("off")
    plt.colorbar(im2, ax=ax_z, fraction=0.046)

    # Difference image
    diff = np.abs(warped.astype(np.float32) - right.astype(np.float32)).mean(axis=2)
    diff[~valid_mask] = np.nan
    ax_e = fig.add_subplot(gs[1, 2])
    im3 = ax_e.imshow(diff, cmap="hot", vmin=0, vmax=30)
    ax_e.set_title(f"Reprojection error (MAE={mae:.1f})", fontsize=11)
    ax_e.axis("off")
    plt.colorbar(im3, ax=ax_e, fraction=0.046)

    fig.suptitle(f"Hamlyn daVinci — stereo reprojection test (frame {frame_idx:06d})",
                 fontsize=13, fontweight="bold")
    plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
