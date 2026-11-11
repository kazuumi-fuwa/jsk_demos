#!/usr/bin/env python3
"""Manual PLAN-ONLY bridge from /cloth/grasp_target to existing MoveIt Planner.

Does not execute hardware actions. Start explicitly with --plan-once.
This is a geometric integration check, not a verified grasp controller.
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from moveit_msgs.msg import DisplayTrajectory
from g1_plan_approach import Planner


class Coordinator(Planner):
    def __init__(self, max_age_s):
        super().__init__()
        self.latest = None
        self.max_age_s = max_age_s
        self.create_subscription(PointStamped, '/cloth/grasp_target', self.on_target, 10)

    def on_target(self, msg):
        if msg.header.frame_id != 'pelvis':
            self.get_logger().warn('Ignored target not in pelvis frame')
            return
        xyz = (msg.point.x, msg.point.y, msg.point.z)
        if not all(math.isfinite(v) for v in xyz):
            self.get_logger().warn('Ignored non-finite target')
            return
        self.latest = msg
        self.get_logger().info(f'Received 3D target: {xyz}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-once', action='store_true', help='Explicitly request one PLAN-ONLY MoveIt result')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--max-age', type=float, default=5.0,
                        help='Maximum age (seconds) of target when clocks share a time base')
    parser.add_argument('--pregrasp-lift', type=float, default=0.12)
    args = parser.parse_args()
    if not args.plan_once:
        parser.error('Specify --plan-once. The script never executes trajectories.')
    if args.timeout <= 0 or args.max_age <= 0:
        parser.error('timeout and max-age must be positive')
    if not 0.08 <= args.pregrasp_lift <= 0.30:
        parser.error('pregrasp-lift must be 0.08..0.30 meters')

    rclpy.init()
    node = Coordinator(args.max_age)
    try:
        deadline = time.monotonic() + args.timeout
        while rclpy.ok() and node.latest is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        target = node.latest
        if target is None:
            raise RuntimeError('No valid /cloth/grasp_target received')
        now = node.get_clock().now().nanoseconds
        stamp = target.header.stamp.sec * 1_000_000_000 + target.header.stamp.nanosec
        age = (now-stamp)/1e9
        if stamp == 0 or age > args.max_age or age < -0.5:
            raise RuntimeError(f'Target timestamp invalid/stale: age={age:.2f}s')
        xyz = (target.point.x, target.point.y, target.point.z + args.pregrasp_lift)
        node.get_logger().warn('PLAN ONLY. Uncalibrated wrist orientation, no grasp or execution.')
        if not node.client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('MoveIt /move_action unavailable')
        result = node.plan('PREGRASP FROM TOPIC', xyz, [0.0, 0.0, 0.0, 1.0])
        display = DisplayTrajectory()
        display.model_id = 'g1'
        display.trajectory_start = result.trajectory_start
        display.trajectory = [result.planned_trajectory]
        node.display.publish(display)
        node.get_logger().info('Published PLAN-ONLY trajectory to /display_planned_path')
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
