"""
DataLoader factories for each training stage.

Stage 1       : 4 synthetic datasets (SimCol3D, C3VD, EndoSLAM, PolypSense3D)
Stage 2a      : Real stereo datasets with RAFT pseudo-GT (Hamlyn, StereoMIS P1)
Stage 2b      : Capsule endoscopy ex-vivo (MiroCam, PillCam) — photometric loss only
Stage 2b-distill: Real colonoscopy (PolypSense3D clinical) — DA3-BASE distillation
Stage 3      : SCARED keyframes (structured-light GT depth, stereo)
"""

from __future__ import annotations

from torch.utils.data import ConcatDataset, DataLoader

from endo_da3.data.base import EndoDepthDataset
from endo_da3.data.c3vd import C3VDDataset
from endo_da3.data.endoslam import EndoSLAMSynthDataset
from endo_da3.data.hamlyn import HamlynDataset
from endo_da3.data.mirocam import MiroCamDataset
from endo_da3.data.pillcam import PillCamDataset
from endo_da3.data.polypsense3d import PolypSense3DVirtualDataset
from endo_da3.data.polypsense3d_clinical import PolypSense3DClinicalDataset
from endo_da3.data.simcol3d import SimCol3DDataset
from endo_da3.data.stereomis import StereoMISDataset
from endo_da3.data.scared import SCAREDDataset


def make_stage1_loaders(
    *,
    simcol_root: str,
    c3vd_root: str,
    endoslam_root: str,
    polypsense3d_root: str,
    img_size: int = 336,
    seq_len: int = 2,
    stride: int = 1,
    batch_size: int = 4,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and val DataLoaders for Stage 1 from all synthetic datasets.

    To add a dataset:
        1. Implement a subclass of EndoDepthDataset with scene-level splits
           documented in the class docstring.
        2. Append train/val instances to the lists below.

    Returns:
        train_loader, val_loader, dataset_names
    """
    shared = dict(img_size=img_size, seq_len=seq_len, stride=stride, with_pose=True)

    train_datasets: list[EndoDepthDataset] = [
        SimCol3DDataset(simcol_root, split="train", **shared),
        C3VDDataset(c3vd_root, split="train", **shared),
        EndoSLAMSynthDataset(endoslam_root, split="train", **shared),
        PolypSense3DVirtualDataset(polypsense3d_root, split="train", **shared),
    ]
    val_datasets: list[EndoDepthDataset] = [
        SimCol3DDataset(simcol_root, split="val", **shared),
        C3VDDataset(c3vd_root, split="val", **shared),
        EndoSLAMSynthDataset(endoslam_root, split="val", **shared),
        PolypSense3DVirtualDataset(polypsense3d_root, split="val", **shared),
    ]

    names = [type(ds).__name__ for ds in train_datasets]

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, names


def make_stage2a_loaders(
    *,
    hamlyn_root: str,
    stereomis_root: str,
    img_size: int = 336,
    batch_size: int = 5,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and val DataLoaders for Stage 2a.

    Train: Hamlyn daVinci (train split) + StereoMIS P1 + StereoMIS P2_0–P2_8 + P3
    Val  : Hamlyn daVinci (val split, last 10% of test)

    Returns:
        train_loader, val_loader, dataset_names
    """
    stereomis_seqs = ["P1", "P2_0", "P2_1", "P2_2", "P2_3", "P2_4",
                      "P2_5", "P2_6", "P2_7", "P2_8"]  # P3 held out as test
    train_datasets = [
        HamlynDataset(hamlyn_root, split="train", img_size=img_size),
        *[StereoMISDataset(stereomis_root, seq=seq, img_size=img_size)
          for seq in stereomis_seqs],
    ]
    val_ds = HamlynDataset(hamlyn_root, split="val", img_size=img_size)

    names = ["HamlynDataset", "StereoMISDataset (P1+P2, P3=test)"]

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, names


def make_stage2b_loaders(
    *,
    endoslam_root: str,
    img_size: int = 336,
    seq_len: int = 2,
    stride: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    pillcam_cameras: str = "both",
) -> tuple[DataLoader, list[str]]:
    """
    Build a train DataLoader for Stage 2b (self-supervised photometric loss).

    Datasets: MiroCam + PillCam from the EndoSLAM ex-vivo collection.
    No GT depth — train with stage2b_loss (photometric reprojection).

    The EndoSLAM root must contain extracted PillCam/ and MiroCam/ subdirectories
    (unzip Cameras/MiroCam.zip and Cameras/PillCam.zip inside the EndoSLAM root).

    No validation split: these datasets have no GT depth, so held-out val
    is evaluated qualitatively or on SERV-CT.

    Args:
        endoslam_root   : EndoSLAM root (contains MiroCam/ and PillCam/).
        img_size        : Square crop size (default 336).
        seq_len         : Frames per window (default 2).
        stride          : Sliding window step (default 1).
        batch_size      : Samples per batch (default 8).
        num_workers     : DataLoader workers (default 4).
        pillcam_cameras : "both" | "C1" | "C2" (default "both").

    Returns:
        train_loader, dataset_names
    """
    shared = dict(img_size=img_size, seq_len=seq_len, stride=stride)

    datasets = [
        MiroCamDataset(endoslam_root, **shared),
        PillCamDataset(endoslam_root, cameras=pillcam_cameras, **shared),
    ]
    names = [type(ds).__name__ for ds in datasets]

    train_loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    return train_loader, names


def make_stage3_loaders(
    *,
    scared_root: str,
    img_size: int = 336,
    batch_size: int = 4,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and val DataLoaders for Stage 3.

    Train: SCARED keyframes (25) + SCARED video frames (17,206)
    Val  : SCARED dataset_8–9 keyframes (10 pairs, structured-light GT)

    Returns:
        train_loader, val_loader, dataset_names
    """
    train_kf  = SCAREDDataset(scared_root, train=True, video=False, img_size=img_size)
    train_vid = SCAREDDataset(scared_root, train=True, video=True,  img_size=img_size)
    val_kf  = SCAREDDataset(scared_root, train=False, video=False, img_size=img_size)
    val_vid = SCAREDDataset(scared_root, train=False, video=True,  img_size=img_size)

    names = ["SCAREDDataset (keyframes)", "SCAREDDataset (video)"]

    train_loader = DataLoader(
        ConcatDataset([train_kf, train_vid]),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_kf_loader = DataLoader(
        val_kf,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    val_vid_loader = DataLoader(
        val_vid,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_kf_loader, val_vid_loader, names


def make_stage2b_distill_loaders(
    *,
    polypsense3d_root: str,
    img_size: int = 336,
    batch_size: int = 8,
    num_workers: int = 4,
    undistort: bool = True,
) -> tuple[DataLoader, list[str]]:
    """
    Build a train DataLoader for Stage 2b distillation.

    Dataset: PolypSense3D clinical (~11,645 real colonoscopy frames).
    No GT depth — train with distillation_loss (DA3-BASE teacher → Endo-DA3 student).

    Args:
        polypsense3d_root : PolypSense3D root (contains Clinical-Dataset-For-PolypSense3D/).
        img_size          : Square crop size (default 336).
        batch_size        : Samples per batch (default 8).
        num_workers       : DataLoader workers (default 4).
        undistort         : Apply barrel-distortion correction (default True).

    Returns:
        train_loader, dataset_names
    """
    ds = PolypSense3DClinicalDataset(
        polypsense3d_root, img_size=img_size, undistort=undistort,
    )
    names = [ds.name]

    train_loader = DataLoader(
        ds,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    return train_loader, names
