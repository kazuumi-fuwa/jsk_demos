#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image

BAG = str(Path.home()/ "semi_ws/tablecloth_close_sample")
OUT = Path.home() / "tablecloth_data"
OUT.mkdir(exist_ok=True)

TOPICS = {
    "rgb": "/camera/camera/color/image_raw",
    "depth": "/camera/camera/depth/image_rect_raw",
    "mask": "/grounded_sam2/mask",
}

def open_bag():
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=BAG, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    types = {
        t.name: get_message(t.type)
        for t in reader.get_all_topics_and_types()
    }
    return reader, type

def stamp_ns(msg):
    s = msg.header.stamp
    return s.sec * 1_000_000_000 + s.nanosec

# First pass: collect mask timestamps only
reader, types = open_bag()
mask_times = []

while reader.has_next():
    topic, raw, _ = reader.read_next()
    if topic == TOPICS["mask"]:
        msg = deserialize_message(raw, Image)
        mask_times.append(stamp_ns(msg))

if not mask_times:
    raise RuntimeError("No mask messages found")

target_ns = mask_times[len(mask_times) // 2]

# Second pass: select the closest message per topic.
reader, types = open_bag()
best = {}
errors = {}

while reader.has_next():
    topic, raw, _ = reader.read_next()

    if topic not in TOPICS.values():
        continue

    msg = deserialize_message(raw, Image)
    delta = abs(stamp_ns(msg) - target_ns)

    if topic not in errors or delta < errors[topic]:
        errors[topic] = delta
        best[topic] = msg

bridge = CvBridge()

for name, topic in TOPICS.items():
    msg = best[topic]
    print(
        f"{name}: {msg.width}x{msg.height}, "
        f"encoding={msg.encoding}, "
        f"frame={msg.header.frame_id}, "
        f"time offset={errors[topic] / 1e9:.4f}s"
    )

rgb = bridge.imgmsg_to_cv2(
    best[TOPICS["rgb"]], desired_encoding="bgr8"
)

depth = bridge.imgmsg_to_cv2(
    best[TOPICS["depth"]], desired_encoding="passthrough"
)

mask = bridge.imgmsg_to_cv2(
    best[TOPICS["mask"]], desired_encoding="mono8"
)

# Save original images without changing depth values.
cv2.imwrite(str(OUT / "rgb.png"), rgb)
cv2.imwrite(str(OUT / "depth.png"), depth)
cv2.imwrite(str(OUT / "mask.png"), mask)

# Visualize mask on the RGB image.
overlay = rgb.copy()
overlay[mask > 0] = (0, 255, 0)
visualization = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)

cv2.imwrite(str(OUT / "mask_overlay.png"), visualization)

valid = depth[depth > 0]
print("Unique mask values:", np.unique(mask))
print("Valid depth pixels:", len(valid))

if valid.size:
    print("Depth range (raw units):", valid.min(), valid.max())

print("Saved Images to :", OUT)