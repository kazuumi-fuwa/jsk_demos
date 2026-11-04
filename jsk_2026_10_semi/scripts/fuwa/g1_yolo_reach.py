
#!/usr/bin/env python3
"""G1 YOLO 3D detection -> MoveIt pose plan -> RViz preview (NEVER executes).

Examples:
  # Start with a manually chosen, validated target:
  python3 g1_yolo_reach.py --mode manual --x 0.45 --y -0.20 --z -0.10

  # Test the rosbag YOLO->MoveIt connection using an existing object class:
  python3 g1_yolo_reach.py --mode yolo --class-name chair

  # Make the RViz trajectory preview 8x slower:
  python3 g1_yolo_reach.py --mode manual --x 0.45 --y -0.20 --z -0.10 --slowdown 8

WARNING: Plans are for demonstration only. Detected box centers are not grasp points.
Never use a person as an automatic reaching target.
"""

import argparse
from copy import deepcopy
import math
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, DisplayTrajectory, MoveItErrorCodes,
    OrientationConstraint, PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from yolo_msgs.msg import DetectionArray

GROUP = 'right_arm_with_waist'
TIP = 'right_wrist_yaw_link'
TOPIC = '/yolo/cam0/detections_3d'
DISPLAY_TOPIC = '/display_planned_path'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['manual', 'yolo'], required=True)
    p.add_argument('--x', type=float)
    p.add_argument('--y', type=float)
    p.add_argument('--z', type=float)
    p.add_argument('--frame', default='pelvis', help='Coordinate frame for manual mode')
    p.add_argument('--class-name', help='YOLO class to select, e.g. chair (required in YOLO mode)')
    p.add_argument('--min-score', type=float, default=0.7)
    p.add_argument('--timeout', type=float, default=20.0)
    p.add_argument('--qx', type=float, default=0.0)
    p.add_argument('--qy', type=float, default=0.0)
    p.add_argument('--qz', type=float, default=0.0)
    p.add_argument('--qw', type=float, default=1.0)
    p.add_argument('--slowdown', type=float, default=5.0,
                   help='Playback time multiplier for RViz ONLY')
    args = p.parse_args()
    if args.mode == 'manual' and any(v is None for v in (args.x, args.y, args.z)):
        p.error('manual mode requires --x, --y and --z')
    if args.mode == 'yolo' and not args.class_name:
        p.error('yolo mode requires --class-name (e.g. chair)')
    if args.class_name and args.class_name.lower() == 'person':
        p.error('person is intentionally blocked as a target class')
    if not 0 <= args.min_score <= 1:
        p.error('--min-score must be between 0 and 1')
    if args.slowdown < 1 or not math.isfinite(args.slowdown):
        p.error('--slowdown must be a finite number >= 1')
    if args.timeout <= 0:
        p.error('--timeout must be positive')
    if not all(math.isfinite(v) for v in (args.qx, args.qy, args.qz, args.qw)):
        p.error('quaternion must contain finite numbers')
    return args


class ReachNode(Node):
    def __init__(self):
        super().__init__('g1_yolo_reach')
        self.target = None
        self.sub = None
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.display_pub = self.create_publisher(DisplayTrajectory, DISPLAY_TOPIC, qos)
        self.move_client = ActionClient(self, MoveGroup, '/move_action')

    def listen_for_target(self, class_name, min_score, timeout):
        def on_detection(msg):
            matching = [d for d in msg.detections
                        if d.class_name.lower() == class_name.lower()
                        and d.class_name.lower() != 'person'
                        and d.score >= min_score
                        and d.bbox3d.frame_id.strip()
                        and all(math.isfinite(v) for v in (
                            d.bbox3d.center.position.x,
                            d.bbox3d.center.position.y,
                            d.bbox3d.center.position.z))
                        and all(v > 0 and math.isfinite(v) for v in (
                            d.bbox3d.size.x, d.bbox3d.size.y, d.bbox3d.size.z))]
            if matching and self.target is None:
                detection = max(matching, key=lambda d: d.score)
                pos = detection.bbox3d.center.position
                self.target = (pos.x, pos.y, pos.z,
                               detection.bbox3d.frame_id)
                self.get_logger().info(
                    f'Selected {detection.class_name} (score={detection.score:.2f}), '
                    f'3D bbox CENTER={self.target}. This is NOT a grasp point.'
                )

        self.sub = self.create_subscription(DetectionArray, TOPIC, on_detection, 10)
        self.get_logger().info(f'Waiting for {class_name!r} on {TOPIC} (timeout {timeout}s)')
        limit = time.monotonic() + timeout
        while rclpy.ok() and self.target is None and time.monotonic() < limit:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(self.sub)
        self.sub = None
        return self.target


def pose_goal(x, y, z, frame, args):
    norm = math.sqrt(sum(v*v for v in (args.qx, args.qy, args.qz, args.qw)))
    if norm < 1e-9:
        raise ValueError('Quaternion cannot be zero')
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.x = args.qx / norm
    pose.orientation.y = args.qy / norm
    pose.orientation.z = args.qz / norm
    pose.orientation.w = args.qw / norm

    pc = PositionConstraint()
    pc.header.frame_id = frame
    pc.link_name = TIP
    pc.weight = 1.0
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [0.01]  # 1 cm radius, planning tolerance
    pc.constraint_region.primitives.append(sphere)
    pc.constraint_region.primitive_poses.append(pose)

    oc = OrientationConstraint()
    oc.header.frame_id = frame
    oc.link_name = TIP
    oc.orientation = pose.orientation
    oc.absolute_x_axis_tolerance = 0.3
    oc.absolute_y_axis_tolerance = 0.3
    oc.absolute_z_axis_tolerance = 0.3
    oc.weight = 1.0

    constraints = Constraints()
    constraints.position_constraints.append(pc)
    constraints.orientation_constraints.append(oc)
    return constraints


def scaled_preview(robot_trajectory, slowdown):
    preview = deepcopy(robot_trajectory)
    # Scaling timestamps is for RViz visualization, not actual execution.
    for pt in preview.joint_trajectory.points:
        ns = pt.time_from_start.sec * 1_000_000_000 + pt.time_from_start.nanosec
        scaled_ns = int(round(ns * slowdown))
        pt.time_from_start.sec = scaled_ns // 1_000_000_000
        pt.time_from_start.nanosec = scaled_ns % 1_000_000_000
        pt.velocities = [v / slowdown for v in pt.velocities]
        pt.accelerations = [a / slowdown**2 for a in pt.accelerations]
    return preview


def plan_and_show(node, target, args):
    x, y, z, frame = target
    node.get_logger().info(f'Target in {frame}: ({x:.3f}, {y:.3f}, {z:.3f}) m')
    node.get_logger().warn('PLAN ONLY. No controllers will be commanded.')
    if not node.move_client.wait_for_server(timeout_sec=10.0):
        node.get_logger().error('/move_action unavailable; launch and source MoveIt first')
        return

    goal = MoveGroup.Goal()
    goal.request.group_name = GROUP
    goal.request.num_planning_attempts = 10
    goal.request.allowed_planning_time = 10.0
    goal.request.max_velocity_scaling_factor = 0.1
    goal.request.max_acceleration_scaling_factor = 0.1
    goal.request.start_state.is_diff = True
    goal.request.goal_constraints.append(pose_goal(x, y, z, frame, args))
    goal.planning_options.plan_only = True
    goal.planning_options.planning_scene_diff.is_diff = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True

    sent = node.move_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, sent)
    handle = sent.result()
    if handle is None or not handle.accepted:
        node.get_logger().error('MoveIt goal rejected')
        return
    finished = handle.get_result_async()
    rclpy.spin_until_future_complete(node, finished)
    response = finished.result()
    if response is None:
        node.get_logger().error('No action result')
        return
    result = response.result
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
        node.get_logger().error(f'Planning failed (MoveIt code {result.error_code.val})')
        return

    traj = result.planned_trajectory.joint_trajectory
    node.get_logger().info(f'Planning succeeded: {len(traj.points)} points')
    if traj.points:
        for name, first, last in zip(traj.joint_names,
                                     traj.points[0].positions,
                                     traj.points[-1].positions):
            node.get_logger().info(f'{name}: {first:+.3f} -> {last:+.3f} rad '
                                   f'(delta {last-first:+.3f})')

    display = DisplayTrajectory()
    display.model_id = 'g1'
    display.trajectory_start = result.trajectory_start
    display.trajectory.append(scaled_preview(result.planned_trajectory, args.slowdown))
    node.display_pub.publish(display)
    node.get_logger().info(f'Published {args.slowdown:g}x slower RViz preview to {DISPLAY_TOPIC}')

    # Briefly keep node alive for RViz discovery/receipt.
    deadline = time.monotonic() + 2.0
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def main():
    args = parse_args()
    rclpy.init()
    node = ReachNode()
    try:
        if args.mode == 'manual':
            target = (args.x, args.y, args.z, args.frame)
        else:
            target = node.listen_for_target(args.class_name, args.min_score, args.timeout)
            if target is None:
                node.get_logger().error('No qualifying detection received. No plan generated.')
                return
            if target[3] != 'pelvis':
                node.get_logger().warn(
                    f'Detection frame is {target[3]!r}, not pelvis. '
                    'MoveIt must have a valid TF transform for this frame.'
                )
        plan_and_show(node, target, args)
    except (ValueError, KeyboardInterrupt) as exc:
        node.get_logger().error(str(exc))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
