
#!/usr/bin/env python3
"""One-shot G1 Inspire RH56DFX hand controller."""

import argparse
import math
import sys

RIGHT_JOINTS = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]

LEFT_JOINTS = [
    name.replace("R_", "L_", 1)
    for name in RIGHT_JOINTS
]

OPEN_POSITIONS = (0.0,) * 6
CLOSED_POSITIONS = (0.3, 0.6, 1.7, 1.7, 1.7, 1.7)


def describe(hand, action, duration):
    if hand not in ("right", "left"):
        raise ValueError("Invalid hand")
    if action not in ("open", "close"):
        raise ValueError("Invalid action")
    if not math.isfinite(duration) or not 1.0 <= duration <= 15.0:
        raise ValueError("Invalid duration")

    return {
        "action_name": f"/{hand}_hand_controller/follow_joint_trajectory",
        "joints": RIGHT_JOINTS if hand == "right" else LEFT_JOINTS,
        "positions": OPEN_POSITIONS if action == "open" else CLOSED_POSITIONS,
    }


def send_hand_goal(node, hand, action, duration=3.0):
    import rclpy
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    details = describe(hand, action, duration)

    client = ActionClient(
        node,
        FollowJointTrajectory,
        details["action_name"],
    )

    if not client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError(
            f"Action unavailable: {details['action_name']}"
        )

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = details["joints"]

    point = JointTrajectoryPoint()
    point.positions = list(details["positions"])

    point.time_from_start.sec = int(duration)
    point.time_from_start.nanosec = round(
        (duration - int(duration)) * 1e9
    )

    if point.time_from_start.nanosec >= 1_000_000_000:
        point.time_from_start.sec += 1
        point.time_from_start.nanosec -= 1_000_000_000

    goal.trajectory.points = [point]

    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(
        node, future, timeout_sec=10.0
    )

    handle = future.result() if future.done() else None

    if handle is None or not handle.accepted:
        raise RuntimeError("Hand action was not accepted")

    result_future = handle.get_result_async()

    rclpy.spin_until_future_complete(
        node,
        result_future,
        timeout_sec=duration + 10.0,
    )

    wrapped = (
        result_future.result()
        if result_future.done()
        else None
    )

    if wrapped is None or wrapped.result.error_code != 0:
        raise RuntimeError("Hand command failed or timed out")

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
    )
    parser.add_argument(
        "--action",
        choices=["open", "close"],
        required=True,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    details = describe(
        args.hand,
        args.action,
        args.duration,
    )

    print(f"HAND: {args.hand} -> {args.action}")
    print("Controller:", details["action_name"])
    print("Joint targets:", list(details["positions"]))

    if not args.execute:
        print("DRY RUN. No ROS command sent.")
        return

    print("WARNING: This can move the physical hand.")
    print("Verify hand type, finger clearance and E-stop.")

    phrase = f"{args.action.upper()} {args.hand.upper()} HAND"

    if input(f"Type exactly '{phrase}': ").strip() != phrase:
        print("Cancelled.")
        return

    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("g1_hand_one_shot")

    try:
        send_hand_goal(
            node,
            args.hand,
            args.action,
            args.duration,
        )
        print("Controller reported completion.")
        print("Verify the actual hand state.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, KeyboardInterrupt) as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
