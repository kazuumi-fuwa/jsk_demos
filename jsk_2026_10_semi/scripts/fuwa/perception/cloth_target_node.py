#!/usr/bin/env python3
"""Pixel cloth corners + depth + TF -> PointStamped grasp candidate in pelvis.

Safety: publishes observations only; never sends robot motion. Requires a verified
positive depth scale, timestamp-matched depth, and timestamp-specific TF.
Assumes depth/image_rect_raw is rectified and color is raw with plumb_bob/rational distortion.
"""
from collections import deque
import math
import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import PolygonStamped, PointStamped
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener, TransformException

NAMES = ('top_left', 'bottom_left', 'bottom_right', 'top_right')


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def tf_matrix(transform):
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x*x+y*y+z*z+w*w
    if n < 1e-10:
        raise ValueError('Invalid TF rotation')
    s = 2.0/n
    rot = np.array([[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                    [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                    [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]])
    t = transform.transform.translation
    return rot, np.array((t.x, t.y, t.z))


class ClothTargetNode(Node):
    def __init__(self):
        super().__init__('cloth_target_node')
        for key, value in {
            'depth_topic': '/camera/camera/depth/image_rect_raw',
            'color_info_topic': '/camera/camera/color/camera_info',
            'depth_info_topic': '/camera/camera/depth/camera_info',
            'depth_scale_m_per_unit': 0.0,  # MUST verify driver/SDK; no default guess
            'pelvis_frame': 'pelvis',
            'corner': 'top_right',
            'max_time_diff_s': 0.07,
            'max_depth_range_m': 3.0,
            'radius_px': 4.0,
        }.items():
            self.declare_parameter(key, value)
        self.bridge = CvBridge()
        self.depth_msgs = deque(maxlen=150)
        self.color_info = None
        self.depth_info = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=60))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(PointStamped, '/cloth/grasp_target', 10)
        self.create_subscription(PolygonStamped, '/cloth/corners_2d', self.on_corners, 10)
        self.create_subscription(Image, str(self.param('depth_topic')), self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, str(self.param('color_info_topic')),
                                 self.on_color_info, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, str(self.param('depth_info_topic')),
                                 self.on_depth_info, qos_profile_sensor_data)
        self.get_logger().info('Waiting for 2D corners, depth, CameraInfo and timestamp-matched TF')

    def param(self, name):
        return self.get_parameter(name).value

    def on_depth(self, msg):
        self.depth_msgs.append(msg)

    def on_color_info(self, msg):
        self.color_info = msg

    def on_depth_info(self, msg):
        self.depth_info = msg

    def on_corners(self, msg):
        try:
            self.process(msg)
        except (ValueError, RuntimeError, TransformException) as exc:
            self.get_logger().warn(f'Skipping target: {exc}')
        except Exception as exc:
            self.get_logger().error(f'Target processing failed: {exc}')

    def process(self, corners):
        scale = float(self.param('depth_scale_m_per_unit'))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Set verified depth_scale_m_per_unit (ROS parameter)')
        if len(corners.polygon.points) != 4:
            raise ValueError('Expected exactly four corners')
        if not self.depth_msgs or self.depth_info is None or self.color_info is None:
            raise ValueError('Waiting for depth and CameraInfo')
        name = str(self.param('corner'))
        if name not in NAMES:
            raise ValueError('corner must be one of ' + ', '.join(NAMES))
        stamp = stamp_ns(corners.header.stamp)
        if stamp == 0:
            raise ValueError('Mask timestamp is zero; cannot synchronize')
        depth_msg = min(self.depth_msgs,
                        key=lambda m: abs(stamp_ns(m.header.stamp) - stamp))
        dt = abs(stamp_ns(depth_msg.header.stamp) - stamp) / 1e9
        if dt > float(self.param('max_time_diff_s')):
            raise ValueError(f'RGB-depth time difference too large: {dt:.3f}s')
        if self.color_info.header.frame_id != corners.header.frame_id:
            raise ValueError('Mask frame and RGB CameraInfo frame differ')
        if self.depth_info.header.frame_id != depth_msg.header.frame_id:
            raise ValueError('Depth and depth CameraInfo frame differ')
        ci, di = self.color_info, self.depth_info
        if ci.distortion_model not in ('plumb_bob', 'rational_polynomial'):
            raise ValueError('Unsupported RGB distortion model')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError('Expected 16UC1 depth; adapt code for 32FC1')
        if (depth.shape[1], depth.shape[0]) != (di.width, di.height):
            raise ValueError('Depth size differs from CameraInfo')
        if (ci.width, ci.height) != (di.width, di.height):
            # This is supported: explicit extrinsics and reprojection below.
            pass
        Kd = np.array(di.k, dtype=float).reshape(3, 3)
        Kc = np.array(ci.k, dtype=float).reshape(3, 3)
        if Kd[0, 0] <= 0 or Kd[1, 1] <= 0 or Kc[0, 0] <= 0:
            raise ValueError('CameraInfo intrinsics invalid')
        if di.distortion_model not in ('plumb_bob', 'rational_polynomial', ''):
            raise ValueError('Depth image is assumed rectified')
        # Depth -> color: the depth and color optical coordinate systems differ.
        time = Time.from_msg(corners.header.stamp)
        t_color_depth = self.tf_buffer.lookup_transform(
            corners.header.frame_id, depth_msg.header.frame_id, time,
            timeout=Duration(seconds=0.1))
        t_pelvis_color = self.tf_buffer.lookup_transform(
            str(self.param('pelvis_frame')), corners.header.frame_id, time,
            timeout=Duration(seconds=0.1))
        Rcd, tcd = tf_matrix(t_color_depth)
        Rpc, tpc = tf_matrix(t_pelvis_color)

        yy, xx = np.indices(depth.shape)
        valid = (depth > 0) & (depth.astype(float)*scale <= float(self.param('max_depth_range_m')))
        if not valid.any():
            raise ValueError('No valid depth data')
        z = depth[valid].astype(float) * scale
        ud, vd = xx[valid].astype(float), yy[valid].astype(float)
        points_d = np.stack(((ud-Kd[0, 2])*z/Kd[0, 0],
                             (vd-Kd[1, 2])*z/Kd[1, 1], z), axis=1)
        points_c = points_d @ Rcd.T + tcd
        points_c = points_c[points_c[:, 2] > 0]
        if len(points_c) < 5:
            raise ValueError('Not enough projected depth points')
        uv, _ = cv2.projectPoints(points_c, np.zeros(3), np.zeros(3),
                                  Kc, np.array(ci.d, dtype=float))
        uv = uv.reshape(-1, 2)
        point = corners.polygon.points[NAMES.index(name)]
        target_uv = np.array([point.x, point.y])
        near = np.sum((uv - target_uv)**2, axis=1) <= float(self.param('radius_px'))**2
        if near.sum() < 5:
            raise ValueError('Insufficient registered depth around corner')
        patch = points_c[near]
        med_z = np.median(patch[:, 2])
        patch = patch[np.abs(patch[:, 2]-med_z) < .025]
        if len(patch) < 5:
            raise ValueError('Depth discontinuity at corner')
        point_color = np.median(patch, axis=0)
        xyz = Rpc @ point_color + tpc
        if not np.all(np.isfinite(xyz)):
            raise ValueError('Invalid transformed XYZ')
        target = PointStamped()
        target.header.stamp = corners.header.stamp
        target.header.frame_id = str(self.param('pelvis_frame'))
        target.point.x, target.point.y, target.point.z = map(float, xyz)
        self.publisher.publish(target)
        self.get_logger().info(f'{name} in pelvis: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]m; dt={dt:.3f}s')


def main():
    rclpy.init()
    node = ClothTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
