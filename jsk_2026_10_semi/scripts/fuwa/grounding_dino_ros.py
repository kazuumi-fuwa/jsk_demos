
#!/usr/bin/env python3

import threading

import cv2
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


MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# 検出したい物体を英語で指定
PROMPT = "a tablecloth. a piece of fabric. a cloth."

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25


class GroundingDinoNode(Node):
    def __init__(self):
        super().__init__("grounding_dino_node")

        self.bridge = CvBridge()
        self.latest_image = None
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.running = True

        self.get_logger().info("Loading Grounding DINO on CPU...")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(MODEL_ID)
            .to("cpu")
            .eval()
        )

        self.subscription = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.publisher = self.create_publisher(
            Image,
            "/grounding_dino/debug_image",
            10,
        )

        self.worker = threading.Thread(
            target=self.process_loop,
            daemon=True,
        )
        self.worker.start()

        self.get_logger().info("Ready. Waiting for camera images.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="bgr8"
        )

        # 古い画像を蓄積せず、最新画像だけ保持
        with self.lock:
            self.latest_image = (frame, msg.header)

        self.new_frame.set()

    def process_loop(self):
        while self.running:
            self.new_frame.wait(timeout=0.5)

            if not self.running:
                break

            with self.lock:
                if self.latest_image is None:
                    self.new_frame.clear()
                    continue

                frame, header = self.latest_image
                self.latest_image = None
                self.new_frame.clear()

            try:
                self.detect_and_publish(frame, header)
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}")

    def detect_and_publish(self, frame, header):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(rgb)

        inputs = self.processor(
            images=image,
            text=PROMPT,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]

        count = 0

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2,
            )

            cv2.putText(
                frame,
                f"{label}: {float(score):.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            count += 1

        self.get_logger().info(f"Detections: {count}")

        result_msg = self.bridge.cv2_to_imgmsg(
            frame, encoding="bgr8"
        )
        result_msg.header = header

        self.publisher.publish(result_msg)

    def stop(self):
        self.running = False
        self.new_frame.set()
        self.worker.join()


def main():
    rclpy.init()
    node = GroundingDinoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
