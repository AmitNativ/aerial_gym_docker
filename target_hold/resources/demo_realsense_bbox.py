"""Render the first generated target-box URDF with the aerial_gym RealSense
D455 camera and extract a 2D bbox via ``bbox_from_segmentation``.

The camera is placed 15 m from the box, pointing at the origin. Output:
  * ``demo_outputs/seg.png``  — colorized segmentation mask
  * ``demo_outputs/rgb.png``  — RGB render
  * ``demo_outputs/bbox.png`` — RGB with extracted bbox drawn on top

This script bypasses aerial_gym's env manager and drives Isaac Gym directly so
that PhysX memory stays small enough for a 4 GiB GPU (aerial_gym's upstream
PhysX settings are sized for >8k envs).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# isaacgym MUST be imported before torch
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401  (gymutil used for args)

import numpy as np
import torch

# RealSense D455 params from aerial_gym
from aerial_gym.config.sensor_config.camera_config.intel_realsense_d455_config import (
    IntelRealSenseD455Config,
)

from target_hold.configs.target_hold_asset_config import TARGET_BOX_SEMANTIC_ID
from target_hold.utils.bbox_from_segmentation import bbox_from_segmentation


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_BOX_DIR = Path(__file__).resolve().parent / "target_boxes"
OUTPUT_DIR = Path(__file__).resolve().parent / "demo_outputs"

# Scene layout:
#   Ground plane at z=0.
#   Target box sits on the ground (box center_z = lz/2).
#   Camera is 15 m behind the target along -X, +2 m in Y, and 5 m up in Z.
CAMERA_POS = (-15.0, 2.0, 5.0)


def build_sim():
    """Create a minimal Isaac Gym sim sized to fit on small GPUs."""
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False  # CPU tensors — avoids gpu root-tensor buffer

    sim_params.physx.solver_type = 1
    sim_params.physx.num_threads = 2
    sim_params.physx.num_position_iterations = 2
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = False  # CPU physics — we're not actually simulating
    # The following fields still need setting to avoid PhysX complaints, but
    # because use_gpu=False the GPU pools below are not allocated.
    sim_params.physx.max_gpu_contact_pairs = 2**14
    sim_params.physx.default_buffer_size_multiplier = 2
    sim_params.physx.contact_offset = 0.002
    sim_params.physx.rest_offset = 0.001

    # compute_device, graphics_device, physics type, params
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")
    return gym, sim


def add_ground(gym, sim):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.distance = 0.0  # ground at z = 0
    gym.add_ground(sim, plane_params)


def load_box_asset(gym, sim, urdf_path: Path):
    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = True
    asset_opts.collapse_fixed_joints = True
    asset_opts.disable_gravity = True
    asset_opts.use_mesh_materials = True
    return gym.load_asset(
        sim, str(urdf_path.parent), urdf_path.name, asset_opts
    )


def create_env_with_box(gym, sim, box_asset, box_center_z: float):
    env_lower = gymapi.Vec3(-20.0, -10.0, 0.0)
    env_upper = gymapi.Vec3(10.0, 10.0, 20.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Center the box at (0, 0, lz/2) so its underside rests on the z=0 plane.
    box_pose = gymapi.Transform()
    box_pose.p = gymapi.Vec3(0.0, 0.0, box_center_z)
    box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    box_handle = gym.create_actor(
        env,
        box_asset,
        box_pose,
        "target_box",
        0,  # collision_group
        0,  # collision_filter
        TARGET_BOX_SEMANTIC_ID,  # segmentation id
    )

    # Give it a visible color so the RGB render is not pure gray.
    gym.set_rigid_body_color(
        env, box_handle, 0, gymapi.MESH_VISUAL,
        gymapi.Vec3(0.85, 0.22, 0.22),
    )
    return env, box_handle


def create_realsense_camera(gym, env, target_center_xyz):
    cfg = IntelRealSenseD455Config
    cam_props = gymapi.CameraProperties()
    cam_props.width = cfg.width
    cam_props.height = cfg.height
    cam_props.horizontal_fov = cfg.horizontal_fov_deg
    cam_props.near_plane = max(cfg.min_range, 0.01)
    # Stretch far plane to comfortably include the target + some ground beyond.
    cam_to_target = math.sqrt(
        sum((c - t) ** 2 for c, t in zip(CAMERA_POS, target_center_xyz))
    )
    cam_props.far_plane = max(cfg.max_range, cam_to_target + 5.0, 25.0)
    cam_props.enable_tensors = True
    cam_props.use_collision_geometry = cfg.use_collision_geometry

    cam_handle = gym.create_camera_sensor(env, cam_props)
    if cam_handle == gymapi.INVALID_HANDLE:
        raise RuntimeError("Failed to create camera sensor")

    cam_pos = gymapi.Vec3(*CAMERA_POS)
    cam_target = gymapi.Vec3(*target_center_xyz)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    return cam_handle, cfg, cam_to_target


def fetch_tensors(gym, sim, env, cam_handle):
    """Render and return (rgb_hw4 uint8, seg_hw int32) as CPU numpy arrays."""
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    color_gpu = gym.get_camera_image_gpu_tensor(
        sim, env, cam_handle, gymapi.IMAGE_COLOR
    )
    seg_gpu = gym.get_camera_image_gpu_tensor(
        sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION
    )
    color = gymtorch.wrap_tensor(color_gpu).clone()  # (H, W, 4) uint8
    seg = gymtorch.wrap_tensor(seg_gpu).clone()  # (H, W) int32

    gym.end_access_image_tensors(sim)

    return color.cpu().numpy(), seg.cpu().numpy()


def save_outputs(rgb_hw4: np.ndarray, seg_hw: np.ndarray, bbox_xyxy, visible: bool,
                 cam_to_target_m: float):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # RGB is RGBA uint8 — strip alpha for display.
    rgb = rgb_hw4[..., :3]

    # Raw PNGs — no titles, no margins. These are the pixel tensors the
    # camera sensor produced, written one-to-one.
    Image.fromarray(rgb).save(OUTPUT_DIR / "rgb_raw.png")
    # Segmentation as a uint8 grayscale mask: target pixels -> 255, background -> 0.
    seg_mask_u8 = (seg_hw == TARGET_BOX_SEMANTIC_ID).astype(np.uint8) * 255
    Image.fromarray(seg_mask_u8, mode="L").save(OUTPUT_DIR / "seg_raw.png")

    # Colorized seg — map id->distinct color. Only 2 values here (0 and 100).
    seg_vis = np.zeros_like(rgb)
    seg_vis[seg_hw == TARGET_BOX_SEMANTIC_ID] = [220, 60, 60]
    # Background stays black for clarity.

    # 1) RGB alone
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.imshow(rgb)
    ax.set_title(
        f"RealSense D455 RGB (cam {CAMERA_POS}, target on ground, dist {cam_to_target_m:.2f} m)"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rgb.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2) Segmentation alone
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.imshow(seg_vis)
    ax.set_title(f"Segmentation mask (id={TARGET_BOX_SEMANTIC_ID})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "seg.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3) RGB + bbox overlay + seg side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB + extracted bbox")
    axes[0].axis("off")
    if visible:
        x0, y0, x1, y1 = bbox_xyxy
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].text(
            x0, max(y0 - 4, 0),
            f"[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]",
            color="lime", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
        )
    else:
        axes[0].text(
            10, 20, "target not visible", color="red", fontsize=11,
            bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"),
        )

    axes[1].imshow(seg_vis)
    axes[1].set_title(f"Segmentation mask (id={TARGET_BOX_SEMANTIC_ID})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bbox.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    urdfs = sorted(p for p in TARGET_BOX_DIR.glob("cube_*.urdf"))
    if not urdfs:
        raise FileNotFoundError(
            f"No cube_*.urdf files in {TARGET_BOX_DIR}. "
            "Run target_hold/resources/generate_target_boxes.py first."
        )
    urdf = urdfs[0]
    print(f"[demo] Using URDF: {urdf}")

    # Read the generated box dimensions so we can place the box resting on the
    # ground (bottom at z=0 -> center at z = lz/2).
    dims_map = json.loads((TARGET_BOX_DIR / "box_dims.json").read_text())
    box_size = dims_map[urdf.name]["size"]  # [lx, ly, lz]
    box_center_z = box_size[2] / 2.0
    target_center = (0.0, 0.0, box_center_z)
    print(
        f"[demo] Box size: {box_size}  -> center z = {box_center_z:.3f} m  "
        f"(box rests on z=0 ground)"
    )
    print(f"[demo] Camera position: {CAMERA_POS}  looking at {target_center}")

    gym, sim = build_sim()
    try:
        add_ground(gym, sim)
        box_asset = load_box_asset(gym, sim, urdf)
        env, _ = create_env_with_box(gym, sim, box_asset, box_center_z)
        cam_handle, cfg, cam_to_target = create_realsense_camera(gym, env, target_center)
        print(f"[demo] Camera-to-target distance: {cam_to_target:.3f} m")
        gym.prepare_sim(sim)

        rgb_np, seg_np = fetch_tensors(gym, sim, env, cam_handle)
        print(f"[demo] RGB shape: {rgb_np.shape}, seg shape: {seg_np.shape}")
        print(
            f"[demo] Seg ids present: {sorted(int(v) for v in np.unique(seg_np))}"
        )

        # Feed segmentation through bbox_from_segmentation exactly as the task does.
        seg_t = (
            torch.from_numpy(seg_np.astype(np.int32))
            .unsqueeze(0)  # add num_envs
            .unsqueeze(0)  # add num_sensors
        )  # (1, 1, H, W)
        bbox, visible = bbox_from_segmentation(
            seg_t,
            target_semantic_id=TARGET_BOX_SEMANTIC_ID,
            image_width=cfg.width,
            image_height=cfg.height,
        )
        bbox_xyxy = bbox[0].tolist()
        is_visible = bool(visible[0].item())
        print(f"[demo] Bbox (xyxy): {bbox_xyxy}  visible={is_visible}")

        save_outputs(rgb_np, seg_np, bbox_xyxy, is_visible, cam_to_target)
        print(f"[demo] Wrote images to {OUTPUT_DIR}")
    finally:
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
