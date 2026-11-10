
#!/usr/bin/env python3
"""Plan G1 right-wrist pregrasp and optionally execute.

Default: PLAN ONLY.
Execution requires calibrated file input and confirmation.
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient

from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, DisplayTrajectory

from g1_plan_approach import Planner, load_target
from g1_hand_control import send_hand_goal


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["mock", "file"],
        required=True,
    )
    parser.add_argument("--input")
    parser.add_argument(
        "--corner",
        choices=[
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ],
        default="top_right",
    )
    parser.add_argument(
        "--pregrasp-lift",
        type=float,
        default=0.12,
    )

    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)
    parser.add_argument("--qw", type=float, default=1.0)

    parser.add_argument(
        "--execute",
        action="store_true",
    )
    parser.add_argument(
        "--open-hand-before-pregrasp",
        action="store_true",
    )
    parser.add_argument(
        "--hand-duration",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    if not 0.08 <= args.pregrasp_lift <= 0.30:
        parser.error("Invalid pregrasp lift")

    if args.execute and args.mode != "file":
        parser.error("Cannot execute a mock target")

    if args.open_hand_before_pregrasp and not args.execute:
        parser.error("Opening hand requires --execute")

    if not math.isfinite(args.hand_duration):
        parser.error("Invalid hand duration")

    if not 1.0 <= args.hand_duration <= 15.0:
        parser.error("Hand duration must be 1-15 seconds")

    q = [args.qx, args.qy, args.qz, args.qw]
    norm = math.sqrt(sum(v * v for v in q))

    if not math.isfinite(norm) or norm < 1e-8:
        parser.error("Invalid quaternion")

    q = [v / norm for v in q]

    target = load_target(args)

    pregrasp = [
        target[0],
        target[1],
        target[2] + args.pregrasp_lift,
    ]

    print("Target wrist PREGRASP:", pregrasp)
    print("WARNING: Wrist orientation is not calibrated.")
    print("Collision scene and fingertip offset must be checked.")

    rclpy.init()
    node = Planner()

    try:
        if not node.client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveIt unavailable")

        plan = node.plan(
            "PREGRASP ONLY",
            pregrasp,
            q,
        )

        trajectory = plan.planned_trajectory.joint_trajectory

        if not trajectory.points:
            raise RuntimeError("Empty trajectory")

        display = DisplayTrajectory()
        display.model_id = "g1"
        display.trajectory_start = plan.trajectory_start
        display.trajectory = [plan.planned_trajectory]

        node.display.publish(display)

        print(
            f"PLAN generated: {len(trajectory.points)} points"
        )

        if not args.execute:
            print("PLAN ONLY. No robot motion sent.")

            deadline = time.monotonic() + 2.0
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)

            return

        print()
        print("WARNING: REAL ROBOT EXECUTION")
        print("Only PREGRASP will be executed.")
        print("No approach, grasp or lift is performed.")
        print("Verify calibration, collision scene, balance,")
        print("current joint state, controllers and E-stop.")

        if args.open_hand_before_pregrasp:
            print("Right hand will also OPEN before arm motion.")

        phrase = "EXECUTE PREGRASP"

        if input(f"Type '{phrase}': ").strip() != phrase:
            print("Cancelled.")
            return

        if args.open_hand_before_pregrasp:
            print("Opening right hand...")

            send_hand_goal(
                node,
                "right",
                "open",
                args.hand_duration,
            )

            print("Hand controller reported completion.")

            if input("Type 'CONTINUE ARM': ").strip() != "CONTINUE ARM":
                print("Arm execution cancelled.")
                return

        executor = ActionClient(
            node,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

        if not executor.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("ExecuteTrajectory unavailable")

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan.planned_trajectory

        future = executor.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future)

        handle = future.result()

        if handle is None or not handle.accepted:
            raise RuntimeError("Execution goal rejected")

        print("Trajectory execution requested.")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            node,
            result_future,
        )

        wrapped = result_future.result()

        if (
            wrapped is None
            or wrapped.result.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            raise RuntimeError("Trajectory execution failed")

        print("MoveIt reported success.")
        print("Verify the physical robot state.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
