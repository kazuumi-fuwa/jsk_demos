#!/usr/bin/env python3
"""
Minimal sample: raise the right arm and put it back with upper_body_controller.

Examples
--------
  ros2 launch g1_moveit_config demo.launch.py   # mock hardware, no robot needed

  ros2 topic echo --once --no-lost-messages /joint_states | python3 -c "import sys,yaml; s=sys.stdin.read(); m=yaml.safe_load(s[s.index('header:'):].split('---')[0]); print(dict(zip(m['name'], [round(p, 2) for p in m['position']])))"

  ros2 run g1_bringup upper_body_move_sample
  # for real robot
  ros2 run g1_bringup upper_body_move_sample --ros-args -p controller:=gear_sonic_controller
"""

from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

# upper_body_controller expects all of its joints in every goal
JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def make_point(current, positions_by_name, sec):
    point = JointTrajectoryPoint()
    point.positions = [positions_by_name.get(name, current[name]) for name in JOINT_NAMES]
    point.time_from_start = Duration(seconds=sec).to_msg()  # sec may be a float
    return point


def main(args=None):
    rclpy.init(args=args)
    node = Node("upper_body_move_sample")
    # upper_body_controller (ros2_control) or gear_sonic_controller (SONIC)
    controller = node.declare_parameter("controller", "upper_body_controller").value
    client = ActionClient(node, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
    node.get_logger().info(f"Waiting for {controller} ...")    
    client.wait_for_server()

    # /joint_states has several publishers, so merge them; kept updated while moving
    latest = {}
    node.create_subscription(
        JointState, "/joint_states", lambda msg: latest.update(zip(msg.name, msg.position)), 10
    )
    node.get_logger().info("Reading current joint positions ...")
    while rclpy.ok() and not all(name in latest for name in JOINT_NAMES):
        rclpy.spin_once(node, timeout_sec=0.1)
        current = dict(latest)  # starting posture
 
    # [rad], unlisted joints keep their current positions
    target = {'L_index_proximal_joint': 0.0, 'L_middle_proximal_joint': 0.0, 'L_pinky_proximal_joint': 0.0, 'L_ring_proximal_joint': 0.0, 'L_thumb_proximal_pitch_joint': 0.0, 'L_thumb_proximal_yaw_joint': 0.0, 'R_index_proximal_joint': 0.0, 'R_middle_proximal_joint': 0.0, 'R_pinky_proximal_joint': 0.0, 'R_ring_proximal_joint': 0.0, 'R_thumb_proximal_pitch_joint': 0.0, 'R_thumb_proximal_yaw_joint': 0.0, 'left_elbow_joint': 0.35, 'left_shoulder_pitch_joint': 0.0, 'left_shoulder_roll_joint': -0.0, 'left_shoulder_yaw_joint': -0.12, 'left_wrist_pitch_joint': 0.0, 'left_wrist_roll_joint': -0.0, 'left_wrist_yaw_joint': -0.17, 'right_elbow_joint': 0.34, 'right_shoulder_pitch_joint': 0.01, 'right_shoulder_roll_joint': 0.01, 'right_shoulder_yaw_joint': 0.13, 'right_wrist_pitch_joint': -0.0, 'right_wrist_roll_joint': 0.0, 'right_wrist_yaw_joint': 0.18, 'waist_pitch_joint': 0.52, 'waist_roll_joint': -0.0, 'waist_yaw_joint': -0.0}
    for name, position in target.items():
        node.get_logger().info("%s: %.2f -> %.2f" % (name, current[name], position))

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    goal.trajectory.points = [
        make_point(current, {}, 0.5),
        make_point(current, target, 5),
        make_point(current, {}, 10),  # back to the starting posture
    ]

    node.get_logger().info("Sending goal")
    goal_handle = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, goal_handle)
    if not goal_handle.result().accepted:
        node.get_logger().error("Goal rejected")
        node.destroy_node()
        rclpy.shutdown()
        return
    result = goal_handle.result().get_result_async()
    # rclpy.spin_until_future_complete(node, result)
    # print the current posture of the moving joints every 0.5 s until done
    next_print = node.get_clock().now()
    while rclpy.ok() and not result.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now() >= next_print:
            node.get_logger().info(
                "now: " + ", ".join("%s %.2f" % (name, latest[name]) for name in target)
            )
            next_print += Duration(seconds=0.5)
    node.get_logger().info(
        "Done: error_code=%d %s"
        % (result.result().result.error_code, result.result().result.error_string)
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
