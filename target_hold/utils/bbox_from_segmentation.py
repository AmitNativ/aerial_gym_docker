"""Batched 2D bounding-box extraction from a segmentation image.

Consumes the segmentation mask that ``IsaacGymCameraSensor`` already writes to
``global_tensor_dict["segmentation_pixels"]`` when ``segmentation_camera=True``
on the camera config. For each env, finds the min/max pixel coordinates of
pixels whose id matches ``target_semantic_id`` and returns the resulting bbox
plus a per-env ``visible`` flag.

The seg pixels tensor has shape ``(num_envs, num_sensors, H, W)`` — we consume
the first sensor.
"""

from typing import Tuple

import torch


def bbox_from_segmentation(
    seg_pixels: torch.Tensor,
    target_semantic_id: int,
    image_width: int,
    image_height: int,
    return_normalized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract per-env axis-aligned 2D bboxes from a segmentation image.

    Args:
        seg_pixels: ``(num_envs, num_sensors, H, W)`` int tensor.
        target_semantic_id: semantic id assigned to the target asset.
        image_width: ``W`` — used to build a coord vector; also to normalize.
        image_height: ``H`` — same.
        return_normalized: if True, divides bbox by ``(W, H, W, H)``.

    Returns:
        bbox: ``(num_envs, 4)`` float tensor ``[x_min, y_min, x_max, y_max]``.
            Zeros for envs where the target isn't visible.
        visible: ``(num_envs,)`` bool tensor.
    """
    if seg_pixels.dim() != 4:
        raise ValueError(
            f"seg_pixels must be (N, S, H, W); got shape {tuple(seg_pixels.shape)}"
        )

    # Use the first sensor.
    seg = seg_pixels[:, 0]  # (N, H, W)
    device = seg.device
    H, W = seg.shape[-2], seg.shape[-1]
    if (H, W) != (image_height, image_width):
        raise ValueError(
            f"seg_pixels spatial dims {(H, W)} disagree with "
            f"config {(image_height, image_width)}"
        )

    mask = seg == target_semantic_id  # (N, H, W) bool
    visible = mask.any(dim=-1).any(dim=-1)  # (N,)

    # Per-env presence of each column / row.
    cols_any = mask.any(dim=1)  # (N, W)
    rows_any = mask.any(dim=2)  # (N, H)

    x_coords = torch.arange(W, device=device)
    y_coords = torch.arange(H, device=device)

    # For min: replace absent cols with a large value so they don't win the min.
    # For max: replace absent cols with -1 so they don't win the max.
    x_min = torch.where(cols_any, x_coords.unsqueeze(0).expand_as(cols_any),
                        torch.full_like(cols_any, W, dtype=torch.long))
    x_min = x_min.min(dim=1).values.to(torch.float32)

    x_max = torch.where(cols_any, x_coords.unsqueeze(0).expand_as(cols_any),
                        torch.full_like(cols_any, -1, dtype=torch.long))
    x_max = x_max.max(dim=1).values.to(torch.float32)

    y_min = torch.where(rows_any, y_coords.unsqueeze(0).expand_as(rows_any),
                        torch.full_like(rows_any, H, dtype=torch.long))
    y_min = y_min.min(dim=1).values.to(torch.float32)

    y_max = torch.where(rows_any, y_coords.unsqueeze(0).expand_as(rows_any),
                        torch.full_like(rows_any, -1, dtype=torch.long))
    y_max = y_max.max(dim=1).values.to(torch.float32)

    bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # (N, 4)
    bbox = torch.where(visible.unsqueeze(-1), bbox, torch.zeros_like(bbox))

    if return_normalized:
        denom = torch.tensor([W, H, W, H], device=device, dtype=bbox.dtype)
        bbox = bbox / denom

    return bbox, visible
