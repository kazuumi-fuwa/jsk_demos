
#!/usr/bin/env python3
"""
RealSense -> Grounding DINO -> SAM 2 -> ROS 2 image topics

Output:
  /grounded_sam2/debug_image
  /grounded_sam2/mask
"""

import threading
import time

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_TOPIC = "/camera/camera/color/image_raw"

DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam2.1-hiera-tiny"

PROMPT = "a tablecloth. a piece of fabric. a cloth."

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# Reject huge bounding boxes covering most of the image.
MAX_BOX_AREA_RATIO = 0.60


class GroundedSAM2Node(Node):

    def __init__(self):
        super().__init__("grounded_sam2_node")

        self.bridge = CvBridge()

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.get_logger().info(
            f"Using device: {self.device}"
        )

        # Load models only once.
        self.get_logger().info("Loading Grounding DINO...")

        self.processor = AutoProcessor.from_pretrained(
            DINO_MODEL
        )

        self.dino = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(DINO_MODEL)
            .to(self.device)
            .eval()
        )

        self.get_logger().info("Loading SAM 2...")

        self.sam = SAM2ImagePredictor.from_pretrained(
            SAM_MODEL,
            device=self.device,
        )

        self.get_logger().info("Models loaded.")

        self.debug_pub = self.create_publisher(
            Image,
            "/grounded_sam2/debug_image",
            10,
        )

        self.mask_pub = self.create_publisher(
            Image,
            "/grounded_sam2/mask",
            10,
        )

        # Latest-frame-only buffer.
        self.lock = threading.Lock()
        self.latest_frame = None
        self.new_frame = threading.Event()
        self.running = True

        self.subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.worker = threading.Thread(
            target=self.process_loop,
            daemon=True,
        )
        self.worker.start()

        self.get_logger().info(
            f"Waiting for images on {IMAGE_TOPIC}"
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )

        # Keep newest frame, discard older frames.
        with self.lock:
            self.latest_frame = (frame.copy(), msg.header)

        self.new_frame.set()

    def process_loop(self):
        while self.running:
            if not self.new_frame.wait(timeout=0.5):
                continue

            with self.lock:
                item = self.latest_frame
                self.latest_frame = None
                self.new_frame.clear()

            if item is None:
                continue

            frame, header = item

            start = time.perf_counter()

            try:
                self.process_frame(frame, header)

                elapsed = time.perf_counter() - start
                self.get_logger().info(
                    f"Processing time: {elapsed:.2f}s"
                )

            except Exception as exc:
                self.get_logger().error(
                    f"Processing error: {exc}"
                )

    def detect(self, frame):
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = PILImage.fromarray(rgb)

        inputs = self.processor(
            images=pil_image,
            text=PROMPT,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.dino(**inputs)

        result = (
            self.processor
            .post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[(h, w)],
            )[0]
        )

        candidates = []

        for box, score, label in zip(
            result["boxes"],
            result["scores"],
            result["text_labels"],
        ):
            box = box.cpu().numpy()
            score = float(score)

            x1, y1, x2, y2 = box

            area = max(0, x2 - x1) * max(0, y2 - y1)
            area_ratio = area / (w * h)

            if area_ratio > MAX_BOX_AREA_RATIO:
                continue

            candidates.append(
                (box, score, str(label))
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return candidates

    def segment(self, frame, box):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        with torch.inference_mode():
            self.sam.set_image(rgb)

            masks, scores, _ = self.sam.predict(
                box=np.asarray(
                    box, dtype=np.float32
                ),
                multimask_output=True,
            )

        best_index = int(np.argmax(scores))

        return masks[best_index].astype(np.uint8) * 255

    def process_frame(self, frame, header):

        candidates = self.detect(frame)

        debug = frame.copy()
        mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8,
        )

        if candidates:
            box, score, label = candidates[0]

            self.get_logger().info(
                f"Selected: {label} ({score:.2f})"
            )

            mask = self.segment(frame, box)

            # Orange overlay on mask.
            overlay = debug.copy()
            overlay[mask > 0] = (0, 180, 255)

            debug = cv2.addWeighted(
                overlay, 0.40,
                debug, 0.60,
                0,
            )

            # Draw largest external contour.
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                largest = max(
                    contours,
                    key=cv2.contourArea,
                )

                cv2.drawContours(
                    debug,
                    [largest],
                    -1,
                    (0, 0, 255),
                    2,
                )

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                debug,
                f"{label}: {score:.2f}",
                (max(0, x1), max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:
            self.get_logger().info("No object detected")

            cv2.putText(
                debug,
                "No detection",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # Publish annotated image.
        debug_msg = self.bridge.cv2_to_imgmsg(
            debug,
            encoding="bgr8",
        )
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

        # Publish binary mask.
        mask_msg = self.bridge.cv2_to_imgmsg(
            mask,
            encoding="mono8",
        )
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)

    def stop(self):
        self.running = False
        self.new_frame.set()
        self.worker.join()


def main():
    rclpy.init()
    node = None

    try:
        node = GroundedSAM2Node()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.stop()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
