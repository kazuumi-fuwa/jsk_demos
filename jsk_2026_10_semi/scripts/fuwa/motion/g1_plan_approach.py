#!/usr/bin/env python3
"""MoveIt PLAN-ONLY chained pregrasp -> approach for Unitree G1 (ROS 2 Jazzy).

Does NOT execute motion or close fingers. Target coordinates are wrist-link poses,
NOT a calibrated finger contact pose. Floor/table collision objects must be present.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, DisplayTrajectory, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint)
from shape_msgs.msg import SolidPrimitive

GROUP = 'right_arm_with_waist'
TIP = 'right_wrist_yaw_link'
MOCK_POINT = [0.45, -0.20, -0.10]


def load_target(args):
    if args.mode == 'mock':
        return MOCK_POINT
    if not args.input:
        raise ValueError('--input is required in file mode')
    payload = json.loads(Path(args.input).read_text())
    target = payload.get('targets', {}).get(args.corner)
    if not target:
        raise ValueError(f'No target entry for {args.corner}')
    if target.get('status') != 'CANDIDATE_ONLY' or 'pelvis_xyz_m' not in target:
        raise ValueError('Target lacks a calibrated pelvis_xyz_m; refusing camera-frame coordinates')
    xyz = target['pelvis_xyz_m']
    if len(xyz) != 3 or not all(math.isfinite(float(n)) for n in xyz):
        raise ValueError('Invalid target XYZ')
    return [float(x) for x in xyz]


def constraints(xyz, quaternion):
    pos = PositionConstraint()
    pos.header.frame_id = 'pelvis'
    pos.link_name = TIP
    pos.weight = 1.0
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [0.01]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion
    pos.constraint_region.primitives = [sphere]
    pos.constraint_region.primitive_poses = [pose]
    rot = OrientationConstraint()
    rot.header.frame_id = 'pelvis'
    rot.link_name = TIP
    rot.orientation = pose.orientation
    rot.absolute_x_axis_tolerance = 0.3
    rot.absolute_y_axis_tolerance = 0.3
    rot.absolute_z_axis_tolerance = 0.3
    rot.weight = 1.0
    result = Constraints()
    result.position_constraints = [pos]
    result.orientation_constraints = [rot]
    return result


def chained_start(result):
    """Use prior plan's predicted END configuration, not current unchanged joints."""
    state = copy.deepcopy(result.trajectory_start)
    traj = result.planned_trajectory.joint_trajectory
    if not traj.points:
        raise RuntimeError('Previous trajectory contains no points')
    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)
    if len(positions) != len(names):
        raise RuntimeError('Unexpected MoveIt start-state joint dimensions')
    index = {name: j for j, name in enumerate(names)}
    for joint_name, end_value in zip(traj.joint_names, traj.points[-1].positions):
        if joint_name in index:
            positions[index[joint_name]] = end_value
        else:
            names.append(joint_name)
            positions.append(end_value)
    state.joint_state.name = names
    state.joint_state.position = positions
    # velocities/efforts may be empty or outdated, so omit them.
    state.joint_state.velocity = []
    state.joint_state.effort = []
    state.is_diff = False
    return state


class Planner(Node):
    def __init__(self):
        super().__init__('g1_plan_approach')
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.display = self.create_publisher(DisplayTrajectory, '/display_planned_path', qos)
        self.client = ActionClient(self, MoveGroup, '/move_action')

    def plan(self, name, xyz, quaternion, start_state=None):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.05
        req.max_acceleration_scaling_factor = 0.05
        if start_state is None:
            req.start_state.is_diff = True
        else:
            req.start_state = start_state
        req.goal_constraints.append(constraints(xyz, quaternion))
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self.get_logger().info(f'Planning {name}: pelvis XYZ = {tuple(round(x, 4) for x in xyz)}')
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f'{name}: MoveIt action goal rejected')
        future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, future)
        wrapped = future.result()
        if wrapped is None or wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if wrapped is None else wrapped.result.error_code.val
            raise RuntimeError(f'{name}: planning failed, error code {code}')
        result = wrapped.result
        self.get_logger().info(f'{name}: planned {len(result.planned_trajectory.joint_trajectory.points)} points')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['mock', 'file'], required=True)
    p.add_argument('--input', help='grasp_3d.json from calibrated reconstructor')
    p.add_argument('--corner', default='top_right', choices=['top_left', 'bottom_left', 'bottom_right', 'top_right'])
    p.add_argument('--pregrasp-lift', type=float, default=0.12)
    p.add_argument('--approach-clearance', type=float, default=0.06,
                   help='Keep wrist above target; NOT verified finger clearance')
    p.add_argument('--qx', type=float, default=0.0)
    p.add_argument('--qy', type=float, default=0.0)
    p.add_argument('--qz', type=float, default=0.0)
    p.add_argument('--qw', type=float, default=1.0)
    p.add_argument('--plan', action='store_true', help='Connect to MoveIt PLAN ONLY; otherwise print dry run')
    args = p.parse_args()
    if not (0.03 <= args.approach_clearance < args.pregrasp_lift <= 0.3):
        p.error('Require 0.03 <= approach-clearance < pregrasp-lift <= 0.3 meters')
    q = [args.qx, args.qy, args.qz, args.qw]
    norm = math.sqrt(sum(x*x for x in q))
    if norm < 1e-8: p.error('Invalid quaternion')
    q = [x/norm for x in q]
    try:
        target = load_target(args)
    except (ValueError, OSError, KeyError, TypeError) as ex:
        p.error(str(ex))
    pre = [target[0], target[1], target[2] + args.pregrasp_lift]
    approach = [target[0], target[1], target[2] + args.approach_clearance]
    print('NO robot motion or gripper actuation in this script.', flush=True)
    print('Wrist orientation is a PLACEHOLDER, not a calculated grasp orientation.', flush=True)
    print('Target is a wrist-link pose, not a fingertip pose. Collision geometry must be verified.', flush=True)
    print(f'PREGRASP: {pre}', flush=True)
    print(f'APPROACH : {approach}', flush=True)
    if not args.plan:
        print('DRY RUN complete. Add --plan after starting MoveIt.', flush=True)
        return
    rclpy.init()
    node = Planner()
    try:
        if not node.client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('MoveIt /move_action is unavailable')
        first = node.plan('PREGRASP', pre, q)
        second = node.plan('APPROACH', approach, q, chained_start(first))
        display = DisplayTrajectory()
        display.model_id = 'g1'
        display.trajectory_start = first.trajectory_start
        display.trajectory = [first.planned_trajectory, second.planned_trajectory]
        node.display.publish(display)
        node.get_logger().info('Published CHAINED pregrasp and approach to /display_planned_path')
        node.get_logger().info('PLAN-ONLY: neither trajectory is executed')
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as ex:
        print(f'ERROR: {ex}', file=sys.stderr)
        sys.exit(1)
