"""Generate a library of target URDFs with smoothly varying sizes.

Dimensions interpolate continuously along a person → sedan → truck curve:
  t=0.0  person:  0.5 x 0.3 x 1.8 m  (width x depth x height)
  t=0.5  sedan:   4.5 x 1.8 x 1.5 m
  t=1.0  truck:   6.0 x 2.5 x 3.0 m

A single parameter t is sampled uniformly in [0, 1]. Dimensions are
piecewise-linearly interpolated between the keypoints, then jittered by
±variation_pct for extra variety. This produces a smooth, continuous
distribution of target sizes — no discrete clusters.

URDF origins are bottom-aligned (visual/collision shifted up by half_z)
so that placing base_link at z=0 puts the box on the ground.
"""

import argparse
import json
import random
from pathlib import Path


URDF_TEMPLATE = """<?xml version='1.0' encoding='UTF-8'?>
<robot name="target_box">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 {half_z}" rpy="0.0 0.0 0.0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual name="base_link_visual">
      <geometry>
        <box size="{lx} {ly} {lz}"/>
      </geometry>
      <origin xyz="0 0 {half_z}" rpy="0 0 0"/>
    </visual>
    <collision name="base_link_collision">
      <geometry>
        <box size="{lx} {ly} {lz}"/>
      </geometry>
      <origin xyz="0 0 {half_z}" rpy="0 0 0"/>
    </collision>
  </link>
</robot>
"""

# Keypoints: (t, (width_x, depth_y, height_z)) in meters.
KEYPOINTS = [
    (0.0, (0.5, 0.3, 1.8)),   # person
    (0.5, (4.5, 1.8, 1.5)),   # sedan
    (1.0, (6.0, 2.5, 3.0)),   # truck
]


def _interp(t: float) -> tuple:
    """Piecewise-linear interpolation along the keypoint curve."""
    for i in range(len(KEYPOINTS) - 1):
        t0, dims0 = KEYPOINTS[i]
        t1, dims1 = KEYPOINTS[i + 1]
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return tuple(d0 + s * (d1 - d0) for d0, d1 in zip(dims0, dims1))
    return KEYPOINTS[-1][1]


def _jitter(value: float, variation_pct: float, rng: random.Random) -> float:
    lo = value * (1.0 - variation_pct)
    hi = value * (1.0 + variation_pct)
    return rng.uniform(lo, hi)


def _category_label(t: float) -> str:
    if t < 0.25:
        return "person"
    elif t < 0.75:
        return "sedan"
    else:
        return "truck"


def generate(
    out_dir: Path,
    count: int,
    variation_pct: float,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    dims_map = {}

    for i in range(count):
        t = rng.random()  # uniform [0, 1]
        nom_x, nom_y, nom_z = _interp(t)
        lx = _jitter(nom_x, variation_pct, rng)
        ly = _jitter(nom_y, variation_pct, rng)
        lz = _jitter(nom_z, variation_pct, rng)
        filename = f"target_{i:04d}.urdf"
        urdf_path = out_dir / filename
        urdf_path.write_text(
            URDF_TEMPLATE.format(lx=lx, ly=ly, lz=lz, half_z=lz / 2.0)
        )
        dims_map[filename] = {
            "t": round(t, 4),
            "category": _category_label(t),
            "nominal": [round(nom_x, 3), round(nom_y, 3), round(nom_z, 3)],
            "size": [round(lx, 4), round(ly, 4), round(lz, 4)],
            "half_extents": [round(lx / 2.0, 4), round(ly / 2.0, 4), round(lz / 2.0, 4)],
        }

    (out_dir / "box_dims.json").write_text(json.dumps(dims_map, indent=2))
    print(f"Wrote {count} URDFs + box_dims.json to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).resolve().parent / "target_boxes"
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument(
        "--count", type=int, default=64,
        help="Total number of URDFs to generate",
    )
    parser.add_argument(
        "--variation-pct", type=float, default=0.15,
        help="Per-dimension jitter as fraction of interpolated value (default 0.15 = ±15%%)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.count, args.variation_pct, args.seed)


if __name__ == "__main__":
    main()
