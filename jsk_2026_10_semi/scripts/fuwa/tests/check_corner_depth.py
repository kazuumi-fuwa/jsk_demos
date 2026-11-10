#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np

base = Path.home() / "tablecloth_data"

depth = cv2.imread(
    str(base / "depth_01186.png"),
    cv2.IMREAD_UNCHANGED
)

corners = json.loads(
    (base / "corners.json").read_text()
)

if depth is None:
    raise RuntimeError("Depth image not found")

height, width = depth.shape

for name, points in corners.items():
    u, v = points["grasp_candidate"]

    # Diagnostic only: not yet registered to RGB
    x1, x2 = max(0, u - 3), min(width, u + 4)
    y1, y2 = max(0, v - 3), min(height, v + 4)

    patch = depth[y1:y2, x1:x2]
    valid = patch[patch > 0]

    if valid.size:
        print(
            f"{name}: pixel=({u}, {v}),"
            f"raw depth median={np.median(valid):.1f},"
            f"valid={valid.size} / {patch.size}"
        )
    else:
        print(f"{name}: no valid depth")