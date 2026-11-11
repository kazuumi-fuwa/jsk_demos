#!/usr/bin/env python3
"""G1 cloth corner pixels -> depth-matched 3D candidates in pelvis (ROS 2 Jazzy).

READ ONLY: subscribes to corners, depth, CameraInfo and TF; never commands robot.
Output /cloth/corners_3d (PolygonStamped) only when ALL four are valid.
Each Point32 xyz is in pelvis (meters), ordered TL, BL, BR, TR.

Depth pixels are projected depth optical -> color optical -> RGB pixels; the RGB
corner is never treated as a raw depth pixel. RGB/Depth timestamps are matched.
"""
import collections
import math
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, PolygonStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def matrix_of(tf):
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if norm < 1e-8:
        raise ValueError('invalid TF quaternion')
    x,y,z,w = x/norm,y/norm,z/norm,w/norm
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])
    t = tf.transform.translation
    return R, np.array([t.x, t.y, t.z])


class Corner3D(Node):
    def __init__(self):
        super().__init__('cloth_corners_3d_node')
        self.declare_parameter('corners_topic', '/cloth/corners_2d')
        self.declare_parameter('depth_topic', '/head_camera/d435/depth/image_rect_raw')
        self.declare_parameter('color_info_topic', '/head_camera/d435/color/camera_info')
        self.declare_parameter('depth_info_topic', '/head_camera/d435/depth/camera_info')
        self.declare_parameter('target_frame', 'pelvis')
        self.declare_parameter('depth_scale_m_per_unit', 0.001)  # VERIFY on hardware for 16UC1
        self.declare_parameter('max_time_diff_s', 0.12)
        self.declare_parameter('radius_px', 5.0)
        self.declare_parameter('min_points', 5)
        self.declare_parameter('stride', 1)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.depth_frames = collections.deque(maxlen=100)
        self.color_info = None
        self.depth_info = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=20))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PolygonStamped, '/cloth/corners_3d', 10)
        self.create_subscription(Image, str(self.get_parameter('depth_topic').value), self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, str(self.get_parameter('color_info_topic').value), self.on_color_info,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, str(self.get_parameter('depth_info_topic').value), self.on_depth_info,
                                 qos_profile_sensor_data)
        self.create_subscription(PolygonStamped, str(self.get_parameter('corners_topic').value), self.on_corners, 10)
        self.get_logger().info('Read-only depth registration: /cloth/corners_2d -> /cloth/corners_3d')
        self.get_logger().warn('Check depth_scale_m_per_unit against camera depth units before trusting XYZ')

    def on_depth(self, msg):
        try:
            if msg.encoding not in ('16UC1','32FC1'):
                self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}', throttle_duration_sec=5)
                return
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').copy()
            with self.lock:
                self.depth_frames.append((stamp_ns(msg.header), msg.header.frame_id, msg.encoding, depth))
        except Exception as exc:
            self.get_logger().error(f'Depth decode: {exc}')

    def on_color_info(self, msg):
        with self.lock:
            self.color_info = msg

    def on_depth_info(self, msg):
        with self.lock:
            self.depth_info = msg

    def tf_lookup(self, target, source, time_ns):
        return self.tf_buffer.lookup_transform(target, source, Time(nanoseconds=time_ns),
                                               timeout=Duration(seconds=0.2))

    def on_corners(self, msg):
        if len(msg.polygon.points) != 4:
            self.get_logger().warn('Expected exactly 4 corners', throttle_duration_sec=3)
            return
        rgb_ns = stamp_ns(msg.header)
        with self.lock:
            ci = self.color_info
            di = self.depth_info
            candidates = list(self.depth_frames)
        if ci is None or di is None or not candidates:
            self.get_logger().warn(
                f"Waiting: color_info={ci is not None}, "
                f"depth_info={di is not None}, "
                f"depth_frames={len(candidates)}",
                throttle_duration_sec=3
            )
            return
        selected = min(candidates, key=lambda item: abs(item[0] - rgb_ns))
        depth_ns, depth_frame, encoding, depth = selected
        dt = abs(depth_ns - rgb_ns) / 1e9
        if dt > float(self.get_parameter('max_time_diff_s').value):
            self.get_logger().warn(f'No synchronized depth: nearest dt={dt:.3f}s', throttle_duration_sec=3)
            return
        if (depth.shape != (di.height, di.width) or not msg.header.frame_id or
                not depth_frame or not ci.header.frame_id):
            self.get_logger().warn('Missing or mismatched frame/size metadata', throttle_duration_sec=3)
            return
        if msg.header.frame_id != ci.header.frame_id:
            self.get_logger().warn('RGB corner frame and color CameraInfo frame mismatch', throttle_duration_sec=3)
            return
        try:
            # Depth -> RGB camera at depth capture; RGB -> pelvis at RGB capture.
            tf_cd = self.tf_lookup(ci.header.frame_id, depth_frame, depth_ns)
            tf_pc = self.tf_lookup(str(self.get_parameter('target_frame').value), ci.header.frame_id, rgb_ns)
        except TransformException as exc:
            self.get_logger().warn(f'TF unavailable at capture time: {exc}', throttle_duration_sec=3)
            return
        rc, tc = matrix_of(tf_cd)
        rp, tp = matrix_of(tf_pc)
        d = depth.astype(np.float64)
        scale = float(self.get_parameter('depth_scale_m_per_unit').value)
        if encoding == '16UC1':
            if scale <= 0:
                self.get_logger().error('Invalid depth scale')
                return
            d *= scale
        # ROS 32FC1 convention: values already in meters.
        stride = max(1, int(self.get_parameter('stride').value))
        yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        zz = d[::stride, ::stride]
        good = np.isfinite(zz) & (zz > 0.1) & (zz < 10.0)
        if not np.any(good):
            self.get_logger().warn('No valid depth samples', throttle_duration_sec=3)
            return
        k_d = np.asarray(di.k).reshape(3, 3)
        k_c = np.asarray(ci.k).reshape(3, 3)
        z = zz[good]
        u = xx[good].astype(np.float64)
        v = yy[good].astype(np.float64)
        xyz_d = np.stack(((u-k_d[0,2])*z/k_d[0,0], (v-k_d[1,2])*z/k_d[1,1], z), axis=-1)
        xyz_c = xyz_d @ rc.T + tc
        valid = xyz_c[:,2] > 0
        xyz_c = xyz_c[valid]
        if not len(xyz_c):
            return
        # Distortion is included for raw RGB. Uses CameraInfo D/K.
        dist = np.asarray(ci.d, dtype=np.float64)
        uv, _ = cv2.projectPoints(xyz_c, np.zeros(3), np.zeros(3), k_c, dist)
        uv = uv.reshape(-1, 2)
        radius2 = float(self.get_parameter('radius_px').value)**2
        min_points = int(self.get_parameter('min_points').value)
        result_xyz = []
        for index, p in enumerate(msg.polygon.points):
            dsq = (uv[:,0]-p.x)**2+(uv[:,1]-p.y)**2
            local = xyz_c[dsq <= radius2]
            if len(local) < min_points:
                self.get_logger().warn(f'Corner {index} has only {len(local)} depth points', throttle_duration_sec=3)
                return
            median_z = np.median(local[:,2])
            consistent = local[np.abs(local[:,2] - median_z) < 0.025]
            if len(consistent) < min_points:
                self.get_logger().warn(f'Corner {index} depth discontinuity', throttle_duration_sec=3)
                return
            point_color = np.median(consistent, axis=0)
            point_pelvis = rp @ point_color + tp
            if not np.all(np.isfinite(point_pelvis)):
                return
            result_xyz.append(point_pelvis)
        out = PolygonStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = str(self.get_parameter('target_frame').value)
        for xyz in result_xyz:
            out.polygon.points.append(Point32(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])))
        self.pub.publish(out)
        summary = ' '.join(f'{name}=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f})' for name,q in
                           zip(('TL','BL','BR','TR'), result_xyz))
        self.get_logger().info(f'3D corners in {out.header.frame_id} dt={dt:.3f}s: {summary}', throttle_duration_sec=1)


def main():
    rclpy.init()
    node = Corner3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

