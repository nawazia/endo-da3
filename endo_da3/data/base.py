"""
Base class for all Endo-DA3 training datasets.

Every dataset subclass must implement:
    - __len__
    - __getitem__ → dict with keys: images (S,3,H,W), depths (S,H,W),
                                    K (3,3), c2w (S,4,4) [if with_pose]

Splits must be defined at scene/subject level before any training, and must
not change once training has started.  Document the split in a comment inside
each subclass so it is reproducible from the source alone.
"""

from __future__ import annotations
from torch.utils.data import Dataset


class EndoDepthDataset(Dataset):
    """Minimal contract for all Stage 1 depth datasets."""

    #: Human-readable name used in logging
    name: str = "unnamed"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """
        Returns at minimum:
            images : (S, 3, H, W)  float32, ImageNet-normalised
            depths : (S, H, W)     float32, metres
            K      : (3, 3)        float32, pixel-space intrinsics at img_size
        Optionally:
            c2w    : (S, 4, 4)     float32, camera-to-world (right-handed)
        """
        raise NotImplementedError
