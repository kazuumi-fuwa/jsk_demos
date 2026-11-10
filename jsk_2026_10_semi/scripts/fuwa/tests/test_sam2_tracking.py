#!/usr/bin/env python3

from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--frames", required=True)
parser.add_argument("--mask", required=True)
parser.add_argument("--output", default="sam2_tracking")
args = parser.parse_args()

frames_dir = Path(args.frames)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

frame_files = sorted(frames_dir.glob("*.jpg"))
if not frame_files:
    raise RuntimeError("No JPEG frames found")

first_frame = cv2.imread(str(frame_files[0]))
initial_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

if initial_mask is None:
    raise RuntimeError("Cannot read initial mask")

if initial_mask.shape != first_frame.shape[:2]:
    raise RuntimeError("Mask and frame dimensions differ")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

predictor = SAM2VideoPredictor.from_pretrained(
    "facebook/sam2.1-hiera-tiny",
    device=device,
)

with torch.inference_mode():
    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=(device == "cpu"),
        offload_state_to_cpu=(device == "cpu"),
    )

    predictor.add_new_mask(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        mask=(initial_mask > 0),
    )

    for frame_idx, obj_ids, mask_logits in (
        predictor.propagate_in_video(state)
    ):
        frame = cv2.imread(str(frame_files[frame_idx]))

        idx = list(obj_ids).index(1)
        mask = (
            mask_logits[idx, 0] > 0
        ).cpu().numpy()

        overlay = frame.copy()
        overlay[mask] = (0, 255, 0)

        result = cv2.addWeighted(
            frame, 0.65, overlay, 0.35, 0
        )

        cv2.imwrite(
            str(output_dir / f"{frame_idx:05d}.png"),
            result
        )

        print(f"Processed frame {frame_idx}")

print("Tracking complete:", output_dir)