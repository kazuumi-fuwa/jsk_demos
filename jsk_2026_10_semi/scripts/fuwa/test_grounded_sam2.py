
#!/usr/bin/env python3
"""
Grounding DINO + SAM 2 single-frame test.

Examples:
  python test_grounded_sam2.py
  python test_grounded_sam2.py --image sam_test/input.jpg
  python test_grounded_sam2.py --image sam_test/input.jpg --box-index 1
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam2.1-hiera-tiny"

TOPIC = "/camera/camera/color/image_raw"
PROMPT = "a tablecloth. a piece of fabric. a cloth."

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25


def capture_camera(timeout=15.0):
    """Capture one ROS 2 image, then stop ROS."""
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("sam2_camera_capture")
    bridge = CvBridge()
    received = []

    def callback(msg):
        if not received:
            received.append(
                bridge.imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                ).copy()
            )

    sub = node.create_subscription(
        Image, TOPIC, callback, qos_profile_sensor_data
    )

    print(f"[Camera] Waiting for {TOPIC}", flush=True)

    import time
    deadline = time.monotonic() + timeout

    try:
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_subscription(sub)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not received:
        raise RuntimeError("Camera image not received")

    return received[0]


def detect(image_bgr, device):
    """Run Grounding DINO on a single image."""
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
    )

    print("[DINO] Loading model...", flush=True)

    processor = AutoProcessor.from_pretrained(DINO_ID)
    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(DINO_ID)
        .to(device)
        .eval()
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = PILImage.fromarray(image_rgb)

    inputs = processor(
        images=image,
        text=PROMPT,
        return_tensors="pt",
    ).to(device)

    print("[DINO] Detecting...", flush=True)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]

    candidates = []

    for box, score, label in zip(
        results["boxes"],
        results["scores"],
        results["text_labels"],
    ):
        candidates.append({
            "box": box.cpu().numpy().tolist(),
            "score": float(score),
            "label": str(label),
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates


def draw_detections(image, candidates):
    output = image.copy()

    for i, item in enumerate(candidates):
        x1, y1, x2, y2 = map(int, item["box"])

        cv2.rectangle(
            output, (x1, y1), (x2, y2),
            (0, 255, 0), 2,
        )

        label = f"#{i} {item['label']} {item['score']:.2f}"

        cv2.putText(
            output, label,
            (max(0, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2,
        )

    return output


def choose_candidate(candidates, width, height, box_index):
    if box_index is not None:
        if not 0 <= box_index < len(candidates):
            raise ValueError("Invalid box index")
        return box_index

    # Skip boxes covering most of the whole image.
    image_area = width * height

    for i, item in enumerate(candidates):
        x1, y1, x2, y2 = item["box"]
        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area / image_area < 0.60:
            return i

    # Fallback: use highest confidence.
    return 0


def segment(image_bgr, box, device):
    """Segment the selected bounding box with SAM 2."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("[SAM2] Loading model...", flush=True)

    predictor = SAM2ImagePredictor.from_pretrained(
        SAM_ID,
        device=device,
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("[SAM2] Segmenting...", flush=True)

    with torch.inference_mode():
        predictor.set_image(image_rgb)

        masks, scores, _ = predictor.predict(
            box=np.array(box, dtype=np.float32),
            multimask_output=True,
        )

    best = int(np.argmax(scores))
    mask = masks[best].astype(np.uint8) * 255

    return mask, float(scores[best])


def visualize(image_bgr, mask, box):
    """Overlay mask and contour on the image."""
    result = image_bgr.copy()

    colored = result.copy()
    colored[mask > 0] = (0, 180, 255)

    result = cv2.addWeighted(
        colored, 0.4, result, 0.6, 0
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)

        cv2.drawContours(
            result, [largest], -1,
            (0, 0, 255), 2,
        )

    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(
        result, (x1, y1), (x2, y2),
        (0, 255, 0), 2,
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--box-index", type=int, default=None)
    parser.add_argument("--output", default="sam_test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise RuntimeError(f"Cannot read {args.image}")
    else:
        image = capture_camera()

    h, w = image.shape[:2]
    print(f"[INFO] Image size: {w}x{h}", flush=True)

    cv2.imwrite(str(output_dir / "input.jpg"), image)

    candidates = detect(image, device)

    print(f"[DINO] Found {len(candidates)} candidates")

    for i, item in enumerate(candidates):
        print(
            f"  #{i}: {item['label']} "
            f"score={item['score']:.3f} "
            f"box={[round(v) for v in item['box']]}"
        )

    cv2.imwrite(
        str(output_dir / "detections.jpg"),
        draw_detections(image, candidates),
    )

    with open(output_dir / "detections.json", "w") as f:
        json.dump(candidates, f, indent=2)

    if not candidates:
        print("[WARN] No detections. Try another image.")
        return

    index = choose_candidate(
        candidates, w, h, args.box_index
    )

    selected = candidates[index]
    print(f"[INFO] Selected bounding box #{index}")

    mask, sam_score = segment(
        image, selected["box"], device
    )

    print(f"[SAM2] Mask score: {sam_score:.3f}")

    cv2.imwrite(
        str(output_dir / "mask.png"), mask
    )

    result = visualize(
        image, mask, selected["box"]
    )

    cv2.imwrite(
        str(output_dir / "result.jpg"), result
    )

    print(f"[DONE] Results saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
