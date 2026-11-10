#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from sensor_msgs.msg import Image, CameraInfo
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge

BAG = str(Path.home() / "semi_ws/tablecloth_open_sample")
OUT = Path.home() / "tablecloth_data"
RGB_INDEX = 1186

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
DEPTH_INFO_TOPIC = "/camera/camera/depth/camera_info"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=BAG, storage_id="mcap"),
    rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
)

def stamp_ns(msg):
    return (
        msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    )

rgb_times = []
depth_messages = []
depth_info = None

while reader.has_next():
    topic, raw, _ = reader.read_next()

    if topic == RGB_TOPIC:
        msg = deserialize_message(raw, Image)
        rgb_times.append(stamp_ns(msg))

    elif topic == DEPTH_TOPIC:
        msg = deserialize_message(raw, Image)
        depth_messages.append((stamp_ns(msg), msg))

    elif topic == DEPTH_INFO_TOPIC and depth_info is None:
        depth_info = deserialize_message(raw, CameraInfo)

if RGB_INDEX >= len(rgb_times):
    raise RuntimeError("RGB index is out of range")

target_time = rgb_times[RGB_INDEX]

best_time, best_msg = min(
    depth_messages,
    key=lambda item: abs(item[0] - target_time)
)

dt = abs(best_time - target_time) / 1e9

print(f"RGB frame: {RGB_INDEX}")
print(f"Timestamp difference: {dt:.4f} seconds")
print(f"Depth frame: {best_msg.header.frame_id}")
print(f"Depth encoding: {best_msg.encoding}")

bridge = CvBridge()
depth = bridge.imgmsg_to_cv2(
    best_msg, desired_encoding="passthrough"
)

OUT.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT / "depth_01186.png"), depth)

valid = depth[depth > 0]

if valid.size:
    print(f"Minimum raw depth: {valid.min()}")
    print(f"Maximum raw depth: {valid.max()}")
    print(f"Median raw depth: {np.median(valid)}")

if depth_info is not None:
    print("Depth intrinsic matrix:", depth_info.k)

print("Saved:", OUT / "depth_01186.png")