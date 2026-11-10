#!/usr/bin/env python3
"""Safe PLAN-ONLY adapter: grasp candidates -> existing G1 MoveIt reach script.

No camera or robot required for --mode mock --dry-run (default).
--plan calls g1_reach_target.py, which must itself remain PLAN-ONLY.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

MOCK_PELVIS_POINT = (0.45, -0.20, -0.10)  # Known demonstration point, not a measured grasp
VALID_STAGES = ('pregrasp', 'target')


def get_pelvis_point(mode, input_file, corner):
    if mode == 'mock':
        return MOCK_PELVIS_POINT
    if input_file is None:
        raise ValueError('--input is required for --mode file')
    data = json.loads(input_file.read_text())
    targets = data.get('targets', {})
    if corner not in targets:
        raise ValueError(f'Corner {corner!r} not found. Available: {list(targets)}')
    item = targets[corner]
    if item.get('status') != 'CANDIDATE_ONLY' or 'pelvis_xyz_m' not in item:
        raise ValueError('No calibrated pelvis_xyz_m for this corner. Do not use camera coordinates as pelvis coordinates.')
    pt = item['pelvis_xyz_m']
    if len(pt) != 3 or any(not math.isfinite(float(v)) for v in pt):
        raise ValueError('Invalid pelvis XYZ')
    return tuple(float(v) for v in pt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['mock', 'file'])
    parser.add_argument('--input', type=Path, help='JSON created by reconstruct_corners.py')
    parser.add_argument('--corner', default='top_right',
                        choices=['top_left', 'top_right', 'bottom_left', 'bottom_right'])
    parser.add_argument('--stage', choices=VALID_STAGES, default='pregrasp')
    parser.add_argument('--pregrasp-lift', type=float, default=0.10,
                        help='Offset along +pelvis Z, meters; illustrative, not a grasp approach')
    parser.add_argument('--reach-script', type=Path,
                        default=Path(__file__).resolve().parent / 'g1_reach_target.py')
    parser.add_argument('--plan', action='store_true',
                        help='Call the existing PLAN-ONLY MoveIt script. Default: just print.')
    args = parser.parse_args()

    if not 0 <= args.pregrasp_lift <= 0.3:
        parser.error('--pregrasp-lift must be between 0 and 0.3 meters')
    try:
        x, y, z = get_pelvis_point(args.mode, args.input, args.corner)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.stage == 'pregrasp':
        z += args.pregrasp_lift
    command = [sys.executable, str(args.reach_script), '--frame', 'pelvis',
               '--x', f'{x:.6f}', '--y', f'{y:.6f}', '--z', f'{z:.6f}',
               '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1']

    print(f'Mode: {args.mode}; corner: {args.corner}; stage: {args.stage}', flush=True)
    print(f'PLAN-ONLY target in pelvis: ({x:.3f}, {y:.3f}, {z:.3f}) m', flush=True)
    print('Orientation is a test placeholder; not a verified grasp pose.', flush=True)
    print('Command:', ' '.join(command), flush=True)
    if not args.plan:
        print('DRY RUN: no ROS/MoveIt call. Use --plan to request planning only.', flush=True)
        return
    if not args.reach_script.is_file():
        parser.error(f'Reach script not found: {args.reach_script}')
    if args.mode == 'file':
        print('CAUTION: verify calibration and grasp orientation before physical use.', flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
