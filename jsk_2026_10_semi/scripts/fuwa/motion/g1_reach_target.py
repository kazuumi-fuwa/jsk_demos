
#!/usr/bin/env python3
"""
Plan a G1 right-hand pose with the arm and waist.

This script is PLAN-ONLY. It never executes the trajectory.

Example:
  python3 g1_reach_target.py \
      --x 0.45 --y -0.20 --z -0.10 \
      --frame pelvis

The default orientation is only for testing.
"""

import argparse
import math
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)


GROUP = "right_arm_with_waist"
TIP_LINK = "right_wrist_yaw_link"


def make_constraints(args):
    pose = Pose()
    pose.position.x = args.x
    pose.position.y = args.y
    pose.position.z = args.z

    q = [args.qx, args.qy, args.qz, args.qw]
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-9:
        raise ValueError("Quaternion cannot be zero")

    pose.orientation.x = args.qx / norm
    pose.orientation.y = args.qy / norm
    pose.orientation.z = args.qz / norm
    pose.orientation.w = args.qw / norm

    # Position target region: sphere of radius 1 cm
    position = PositionConstraint()
    position.header.frame_id = args.frame
    position.link_name = TIP_LINK
    position.weight = 1.0

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [0.01]

    position.constraint_region.primitives.append(sphere)
    position.constraint_region.primitive_poses.append(pose)

    # Orientation target
    orientation = OrientationConstraint()
    orientation.header.frame_id = args.frame
    orientation.link_name = TIP_LINK
    orientation.orientation = pose.orientation
    orientation.absolute_x_axis_tolerance = 0.3
    orientation.absolute_y_axis_tolerance = 0.3
    orientation.absolute_z_axis_tolerance = 0.3
    orientation.weight = 1.0

    constraints = Constraints()
    constraints.position_constraints.append(position)
    constraints.orientation_constraints.append(orientation)
    return constraints


def slow_down_trajectory(trajectory, factor=5.0):
    """Slow a trajectory for visualization only."""
    result = deepcopy(trajectory)

    for point in result.joint_trajectory.points:
        t = point.time_from_start.sec + \
            point.time_from_start.nanosec * 1e-9

        new_t_ns = round(t * factor * 1e9)
        point.time_from_start.sec = new_t_ns // 1_000_000_000
        point.time_from_start.nanosec = new_t_ns % 1_000_000_000

        point.velocities = [v / factor for v in point.velocities]
        point.accelerations = [
            a / (factor ** 2) for a in point.accelerations
        ]

    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--frame", default="pelvis")

    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)
    parser.add_argument("--qw", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node("g1_reach_target")

    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

    display_pub = node.create_publisher(
        DisplayTrajectory,
        "/display_planned_path",
        qos,
    )

    client = ActionClient(node, MoveGroup, "/move_action")

    try:
        if not client.wait_for_server(timeout_sec=10.0):
            node.get_logger().error(
                "/move_action unavailable. Launch MoveIt first."
            )
            return

        goal = MoveGroup.Goal()
        request = goal.request

        request.group_name = GROUP
        request.num_planning_attempts = 10
        request.allowed_planning_time = 10.0
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1
        request.start_state.is_diff = True

        request.goal_constraints.append(make_constraints(args))

        # Do not execute on mock or real hardware
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        node.get_logger().info(
            f"Target: ({args.x:.3f}, {args.y:.3f}, "
            f"{args.z:.3f}) in {args.frame}"
        )
        node.get_logger().info(
            f"Group: {GROUP}, tip: {TIP_LINK}"
        )

        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future)
        handle = future.result()

        if handle is None or not handle.accepted:
            node.get_logger().error("MoveIt goal rejected")
            return

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        action_result = result_future.result()

        if action_result is None:
            node.get_logger().error("No result received")
            return

        result = action_result.result

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            node.get_logger().error(
                f"Planning failed: {result.error_code.val}"
            )
            return

        trajectory = result.planned_trajectory.joint_trajectory

        node.get_logger().info(
            f"Planning succeeded: {len(trajectory.points)} points"
        )

        if trajectory.points:
            start = trajectory.points[0].positions
            end = trajectory.points[-1].positions

            node.get_logger().info("Joint movements:")
            for name, q0, q1 in zip(
                trajectory.joint_names, start, end
            ):
                node.get_logger().info(
                    f"  {name}: {q0:.3f} -> {q1:.3f} "
                    f"(delta {q1 - q0:+.3f} rad)"
                )

        # Publish the trajectory for RViz visualization
        display = DisplayTrajectory()
        display.model_id = "g1"
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)

        display_pub.publish(display)
        node.get_logger().info(
            "Published trajectory to /display_planned_path"
        )

        # Allow discovery and delivery to RViz
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
