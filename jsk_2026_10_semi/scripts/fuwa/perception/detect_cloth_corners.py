#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--mask", required=True)
parser.add_argument("--output", default="corner_result.jpg")
args = parser.parse_args()

image = cv2.imread(args.image)
mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise RuntimeError("Could not load image or mask")

if image.shape[:2] != mask.shape:
    raise RuntimeError("Image and mask dimensions differ")

# 1. Convert to binary
binary = np.uint8(mask > 127) * 255

# 2. Extract the largest connected contour
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError("No fabric contour found")

contour = max(contours, key=cv2.contourArea)

# 3. Compute convex hull
hull = cv2.convexHull(contour)
perimeter = cv2.arcLength(hull, True)

# 4. Approximate a quadrilateral
corners = None

for ratio in np.linspace(0.01, 0.06, 51):
    approx = cv2.approxPolyDP(
        hull, ratio * perimeter, True
    )

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(float)
        break

if corners is None:
    print("Warning: using minimum-area rectangle")
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(float)

# Order corners by image position:
# top-left, top-right, bottom-right, bottom-left
by_y = corners[np.argsort(corners[:, 1])]
top = by_y[:2][np.argsort(by_y[:2, 0])]
bottom = by_y[2:][np.argsort(by_y[:2:, 0])]

corners = np.array([
    top[0], bottom[0],
    bottom[1], top[1]
])

names = [
    "top_left",
    "bottom_left",
    "bottom_right",
    "top_right",
]

# Create candidate grasp points slightly inward.
# These are geometric candidates, not validated grasps
center = np.mean(corners, axis=0)
grasp_candidates = corners * 0.93 + center * 0.07

result = image.copy()
data = {}

# Draw corner polygon
cv2.polylines(
    result,
    [np.round(corners).astype(np.int32)],
    True, (0, 255, 0), 2
)

for name, corner, grasp in zip(
    names, corners, grasp_candidates
):
    u, v = np.round(corner).astype(int)
    gu, gv = np.round(grasp).astype(int)

    # Red: geometric corner
    cv2.circle(result, (u, v), 6, (0, 0, 255), -1)

    # Blue: candidate slightly inside fabric
    cv2.circle(result, (gu, gv), 6, (255, 0, 0), -1)

    cv2.putText(
        result, name, (u + 8, v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 255, 255), 1 
    )

    data[name] = {
        "corner": [int(u), int(v)],
        "grasp_candidate": [int(gu), int(gv)]
    }

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output), result)

json_path = output.with_suffix(".json")
json_path.write_text(json.dumps(data, indent=2))