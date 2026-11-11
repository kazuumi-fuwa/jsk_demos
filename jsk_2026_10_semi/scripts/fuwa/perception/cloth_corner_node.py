#!/usr/bin/env python3
"""SAM2 mask -> four ordered, inward-offset corner candidates (pixels).
Polygon order: top_left, bottom_left, bottom_right, top_right.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PolygonStamped, Point32
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def find_corners(mask):
    binary = np.uint8(mask > 127) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500:
        return None
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    corners = None
    for fraction in np.linspace(.01, .06, 51):
        approx = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(float)
            break
    if corners is None:
        corners = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
    by_y = corners[np.argsort(corners[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    ordered = np.array([top[0], bottom[0], bottom[1], top[1]])
    return ordered * .93 + ordered.mean(axis=0) * .07


class ClothCornerNode(Node):
    def __init__(self):
        super().__init__('cloth_corner_node')
        self.declare_parameter('mask_topic', '/grounded_sam2/mask')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(PolygonStamped, '/cloth/corners_2d', 10)
        self.debug_pub = self.create_publisher(Image, '/cloth/corners_debug', 10)
        topic = str(self.get_parameter('mask_topic').value)
        self.create_subscription(Image, topic, self.on_mask, qos_profile_sensor_data)
        self.get_logger().info(f'Subscribed to {topic}; publishing /cloth/corners_2d')

    def on_mask(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            corners = find_corners(mask)
            if corners is None:
                self.get_logger().warn('No reliable cloth contour')
                return
            result = PolygonStamped()
            result.header = msg.header  # original RGB capture time from SAM2
            for x, y in corners:
                p = Point32()
                p.x, p.y = float(x), float(y)
                result.polygon.points.append(p)
            self.publisher.publish(result)
            debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for name, (x, y) in zip(('TL', 'BL', 'BR', 'TR'), corners):
                xy = int(round(x)), int(round(y))
                cv2.circle(debug, xy, 5, (0, 0, 255), -1)
                cv2.putText(debug, name, (xy[0] + 6, xy[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            out = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            out.header = msg.header
            self.debug_pub.publish(out)
        except Exception as exc:
            self.get_logger().error(f'Corner detection: {exc}')


def main():
    rclpy.init()
    node = ClothCornerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
