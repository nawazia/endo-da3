"""
Stage 1 DataLoader factory.

Add new synthetic datasets here by appending to _train_datasets / _val_datasets.
Splits must be defined at scene level inside each dataset class before use.
"""

from __future__ import annotations

from torch.utils.data import ConcatDataset, DataLoader

from endo_da3.data.base import EndoDepthDataset
from endo_da3.data.c3vd import C3VDDataset
from endo_da3.data.endoslam import EndoSLAMSynthDataset
from endo_da3.data.polypsense3d import PolypSense3DVirtualDataset
from endo_da3.data.simcol3d import SimCol3DDataset


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
