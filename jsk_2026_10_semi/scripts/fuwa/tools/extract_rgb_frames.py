#!/usr/bin/env python3

from pathlib import Path

import cv2
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

BAG = str(Path.home() / "semi_ws/tablecloth_open_sample")
OUT = Path.home() / "tablecloth_data" / "open_frames"
OUT.mkdir(parents=True, exist_ok=True)

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=BAG, storage_id="mcap"),
    rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
)

bridge = CvBridge()
count = 0

while reader.has_next():
    topic, raw, _ = reader.read_next()

    if topic != "/camera/camera/color/image_raw":
        continue

    msg = deserialize_message(raw, Image)
    frame = bridge.imgmsg_to_cv2(
        msg, desired_encoding="bgr8"
    )

    cv2.imwrite(
        str(OUT / f"{count:05d}.jpg"),
        frame
    )

    count += 1

print(f"Extracted {count} frames to {OUT}")