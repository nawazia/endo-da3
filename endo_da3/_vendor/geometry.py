# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Source: depth_anything_3/utils/geometry.py (subset)

import numpy as np
import torch


def as_homogeneous(ext):
    """
    Accept (..., 3,4) or (..., 4,4) extrinsics, return (...,4,4) homogeneous matrix.
    Supports torch.Tensor or np.ndarray.
    """
    if isinstance(ext, torch.Tensor):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = torch.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")
    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)
