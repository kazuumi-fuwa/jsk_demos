#!/usr/bin/env python3
"""ROS2: DINO initializes cloth, SAM2VideoPredictor tracks short frame windows.

This is batched video tracking, NOT low-latency visual servoing.
Publish existing /grounded_sam2/mask and /grounded_sam2/debug_image topics.
Do not run together with grounded_sam2_ros.py (same publishers).
"""
import argparse
import contextlib
import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DINO_ID = 'IDEA-Research/grounding-dino-tiny'
SAM_ID = 'facebook/sam2.1-hiera-tiny'
PROMPT = 'a tablecloth. a piece of fabric. a cloth.'


class ClothTracker(Node):
    def __init__(self, args):
        super().__init__('grounded_sam2_tracking_node')
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bridge = CvBridge()
        self.frames = queue.Queue(maxsize=args.queue_size)
        self.anchor = None  # Last (image, header, mask); same frame seeds next window
        self.running = True
        self.dropped = 0
        self.get_logger().info(f'Loading DINO + SAM2 on {self.device}')
        self.processor = AutoProcessor.from_pretrained(DINO_ID)
        self.dino = (AutoModelForZeroShotObjectDetection
                     .from_pretrained(DINO_ID).to(self.device).eval())
        self.sam = SAM2VideoPredictor.from_pretrained(SAM_ID, device=self.device)
        self.mask_pub = self.create_publisher(Image, '/grounded_sam2/mask', 10)
        self.debug_pub = self.create_publisher(Image, '/grounded_sam2/debug_image', 10)
        self.sub = self.create_subscription(
            Image, args.image_topic, self.on_image, qos_profile_sensor_data)
        self.worker = threading.Thread(target=self.run_tracking, daemon=True)
        self.worker.start()
        self.get_logger().info(f'RGB: {args.image_topic} | window: {args.chunk_size}')

    def on_image(self, msg):
        try:
            item = (self.bridge.imgmsg_to_cv2(msg, 'bgr8').copy(), msg.header)
            if self.frames.full():
                try:
                    self.frames.get_nowait()
                    self.frames.task_done()
                    self.dropped += 1
                except queue.Empty:
                    pass
            self.frames.put_nowait(item)
        except queue.Full:
            self.dropped += 1
        except Exception as exc:
            self.get_logger().error(f'Bad RGB input: {exc}')

    def detect_box(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=PILImage.fromarray(rgb),
                                text=PROMPT, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.dino(**inputs)
        detections = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25,
            target_sizes=[(h, w)])[0]
        candidates = []
        for box, score in zip(detections['boxes'], detections['scores']):
            xyxy = box.detach().cpu().numpy().astype(np.float32)
            x1, y1, x2, y2 = xyxy
            fraction = max(0, x2-x1) * max(0, y2-y1) / (w*h)
            if 0.001 < fraction < 0.60:
                candidates.append((float(score), xyxy))
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    def publish(self, frame, header, mask):
        # This header stamp refers to the ORIGINAL camera frame, not processing time.
        msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        msg.header = header
        self.mask_pub.publish(msg)
        preview = frame.copy()
        if np.any(mask):
            overlay = preview.copy()
            overlay[mask > 0] = (0, 180, 255)
            preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)
            cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview, cs, -1, (0, 0, 255), 2)
        else:
            cv2.putText(preview, 'TRACK LOST - INVALID MASK', (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        dbg = self.bridge.cv2_to_imgmsg(preview, encoding='bgr8')
        dbg.header = header
        self.debug_pub.publish(dbg)

    @staticmethod
    def valid(mask):
        if mask is None:
            return False
        fraction = np.count_nonzero(mask) / mask.size
        if not 0.001 <= fraction <= 0.65:
            return False
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        return bool(cs) and cv2.contourArea(max(cs, key=cv2.contourArea)) >= 200

    def predict_window(self, window):
        """Run true SAM2 temporal propagation on a finite video window."""
        with tempfile.TemporaryDirectory(prefix='sam2_cloth_') as tmp:
            for i, (frame, _) in enumerate(window):
                if not cv2.imwrite(str(Path(tmp) / f'{i:05d}.jpg'), frame):
                    raise RuntimeError('Could not write a temporary video frame')
            with torch.inference_mode():
                amp = (torch.autocast('cuda', dtype=torch.bfloat16)
                       if self.device == 'cuda' else contextlib.nullcontext())
                with amp:
                    state = self.sam.init_state(
                        video_path=tmp,
                        offload_video_to_cpu=(self.device == 'cpu'),
                        offload_state_to_cpu=(self.device == 'cpu'))
                    try:
                        if self.anchor is None:
                            box = self.detect_box(window[0][0])
                            if box is None:
                                return None
                            self.get_logger().info('DINO initialized new track')
                            self.sam.add_new_points_or_box(
                                inference_state=state, frame_idx=0,
                                obj_id=1, box=box)
                        else:
                            self.sam.add_new_mask(
                                inference_state=state, frame_idx=0,
                                obj_id=1, mask=(self.anchor[2] > 0))
                        masks = {}
                        for index, ids, logits in self.sam.propagate_in_video(state):
                            if 1 in list(ids):
                                j = list(ids).index(1)
                                masks[int(index)] = ((logits[j, 0] > 0)
                                                     .detach().cpu().numpy()
                                                     .astype(np.uint8) * 255)
                        return masks
                    finally:
                        self.sam.reset_state(state)

    def run_tracking(self):
        while self.running and rclpy.ok():
            fresh = []
            try:
                while (len(fresh) < self.args.chunk_size and
                       self.running and rclpy.ok()):
                    try:
                        fresh.append(self.frames.get(timeout=0.2))
                        self.frames.task_done()
                    except queue.Empty:
                        pass
                if not fresh:
                    continue
                start = time.monotonic()
                # First frame of new window is IDENTICAL to last published frame.
                overlap = self.anchor is not None
                window = ([self.anchor[:2]] if overlap else []) + fresh
                masks = self.predict_window(window)
                last_good = None
                start_index = int(overlap)
                failed = masks is None
                for i in range(start_index, len(window)):
                    frame, header = window[i]
                    mask = masks.get(i) if masks is not None else None
                    if failed or not self.valid(mask):
                        failed = True
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    else:
                        last_good = (frame, header, mask)
                    self.publish(frame, header, mask)
                self.anchor = None if failed else last_good
                if failed:
                    self.get_logger().warning('Tracking invalid; DINO will restart next window')
                self.get_logger().info(
                    f'{len(fresh)} images processed in {time.monotonic()-start:.1f}s; '
                    f'queue drops={self.dropped}')
            except Exception as exc:
                self.get_logger().error(f'Tracking error: {exc}')
                self.anchor = None
                for frame, header in fresh:
                    self.publish(frame, header,
                                 np.zeros(frame.shape[:2], dtype=np.uint8))

    def close(self):
        self.running = False
        self.worker.join(timeout=2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-topic', default='/camera/camera/color/image_raw')
    p.add_argument('--chunk-size', type=int, default=4)
    p.add_argument('--queue-size', type=int, default=12)
    args = p.parse_args()
    if args.chunk_size < 2 or args.queue_size < args.chunk_size:
        p.error('Require 2 <= chunk-size <= queue-size')
    rclpy.init()
    node = None
    try:
        node = ClothTracker(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
