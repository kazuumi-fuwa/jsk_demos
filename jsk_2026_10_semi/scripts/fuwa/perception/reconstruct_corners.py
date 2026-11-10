#!/usr/bin/env python3
"""Calibrated, offline, RGB-mask grasp candidates -> color/pelvis XYZ.

The depth frame is assumed rectified (depth/image_rect_raw); its K is used for
backprojection. RGB image is raw, so RGB K + distortion are used for projection.
T_color_depth maps points in the DEPTH optical frame into COLOR optical frame.
T_pelvis_color, if provided, maps COLOR optical points into G1 pelvis.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml


def matrix_from_config(value, name, required=False):
    if value is None:
        if required:
            raise ValueError(f"{name} is missing. Collect camera extrinsics in the laboratory.")
        return None
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.all(np.isfinite(t)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} must end with [0,0,0,1]")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-3) or np.linalg.det(t[:3, :3]) < 0.99:
        raise ValueError(f"{name} rotation is not orthonormal")
    return t


def camera_matrix(params):
    k = np.asarray(params["K"], dtype=np.float64)
    if k.size != 9:
        raise ValueError("K must have 9 entries")
    return k.reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb', required=True)
    ap.add_argument('--depth', required=True)
    ap.add_argument('--corners', required=True, help='JSON output from detect_cloth_corners.py')
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--output', default='grasp_3d.json')
    ap.add_argument('--radius-px', type=float, default=4.0, help='Max reprojection distance in RGB pixels')
    args = ap.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('Expected RGB image and 16UC1 depth PNG')
    cfg = yaml.safe_load(Path(args.calibration).read_text())
    if cfg.get('depth_scale_m_per_unit') is None or float(cfg['depth_scale_m_per_unit']) <= 0:
        raise ValueError('Set a verified depth_scale_m_per_unit in calibration YAML')
    scale = float(cfg['depth_scale_m_per_unit'])
    t_cd = matrix_from_config(cfg.get('T_color_depth'), 'T_color_depth', required=True)
    t_pc = matrix_from_config(cfg.get('T_pelvis_color'), 'T_pelvis_color')
    kd = camera_matrix(cfg['depth'])
    kc = camera_matrix(cfg['color'])
    dc = np.asarray(cfg['color'].get('D', [0,0,0,0,0]), dtype=np.float64)
    if cfg['color'].get('distortion_model') not in ('plumb_bob', 'rational_polynomial', None):
        raise ValueError('Unsupported color distortion model; adapt projection first')
    if cfg['depth'].get('rectified', True) is not True:
        raise ValueError('Depth backprojection assumes rectified image')
    if cfg['depth'].get('width') != depth.shape[1] or cfg['depth'].get('height') != depth.shape[0]:
        raise ValueError('Depth dimensions do not match calibration')
    if cfg['color'].get('width') != rgb.shape[1] or cfg['color'].get('height') != rgb.shape[0]:
        raise ValueError('RGB dimensions do not match calibration')

    # Backproject every valid depth pixel into depth optical coordinates (meters).
    yy, xx = np.indices(depth.shape)
    good = depth > 0
    z = depth[good].astype(np.float64) * scale
    u_d, v_d = xx[good].astype(np.float64), yy[good].astype(np.float64)
    pts_d = np.column_stack(((u_d - kd[0,2]) * z / kd[0,0],
                             (v_d - kd[1,2]) * z / kd[1,1], z))
    pts_c = pts_d @ t_cd[:3, :3].T + t_cd[:3, 3]
    in_front = pts_c[:, 2] > 0
    pts_c = pts_c[in_front]
    if not len(pts_c):
        raise ValueError('No positive projected depth points')
    # Projection includes color camera lens distortion for raw RGB images.
    projected, _ = cv2.projectPoints(pts_c, np.zeros(3), np.zeros(3), kc, dc)
    uv = projected.reshape(-1,2)
    h, w = rgb.shape[:2]
    visible = (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
    uv, pts_c = uv[visible], pts_c[visible]

    candidates = json.loads(Path(args.corners).read_text())
    output = {'coordinate_convention': 'optical: +x right, +y down, +z forward',
              'calibration_source': args.calibration, 'targets': {}}
    overlay = rgb.copy()
    for name, entry in candidates.items():
        target_uv = np.asarray(entry['grasp_candidate'], dtype=np.float64)
        # Local projected cloud around grasp pixel; use median to reduce depth noise.
        distance_sq = np.sum((uv - target_uv)**2, axis=1)
        nearby = distance_sq <= args.radius_px**2
        record = {'pixel_rgb': target_uv.tolist(), 'valid_projected_points': int(nearby.sum())}
        if nearby.sum() < 5:
            record['status'] = 'INSUFFICIENT_CORRESPONDENCE'
            record['warning'] = 'No reliable registered depth patch; do not command a robot'
        else:
            local = pts_c[nearby]
            # Reject background/foreground outliers using depth around the local median.
            medz = np.median(local[:,2])
            consistent = local[np.abs(local[:,2] - medz) < 0.025]
            if len(consistent) < 5:
                record['status'] = 'DEPTH_DISCONTINUITY'
            else:
                xyz_c = np.median(consistent, axis=0)
                record['status'] = 'CANDIDATE_ONLY'
                record['color_optical_xyz_m'] = np.round(xyz_c, 5).tolist()
                record['local_depth_spread_m'] = round(float(np.ptp(consistent[:,2])), 5)
                if t_pc is not None:
                    xyz_p = t_pc[:3, :3] @ xyz_c + t_pc[:3, 3]
                    record['pelvis_xyz_m'] = np.round(xyz_p, 5).tolist()
                cv2.circle(overlay, tuple(np.rint(target_uv).astype(int)), 6, (0,0,255), -1)
                cv2.putText(overlay, name, tuple(np.rint(target_uv).astype(int) + [8,-8]),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,255), 1)
        output['targets'][name] = record
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2))
    cv2.imwrite(str(destination.with_suffix('.jpg')), overlay)
    print(json.dumps(output, indent=2))
    print('This script estimates points, NOT gripper orientation, collision-free motion, or grasp success.')

if __name__ == '__main__':
    main()
