#!/usr/bin/env python3
"""At the laboratory: capture RGB/depth intrinsics and optical-frame TF.
Use --depth-scale value verified independently from the camera SDK / driver.
T_pelvis_color is OPTIONAL; only record if G1 robot TF is available at that time.
For a moving camera, use time-matched transforms for each future grasp, not this snapshot.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener
import yaml

def transform_matrix(t):
    q=t.transform.rotation
    x,y,z,w=q.x,q.y,q.z,q.w
    n=x*x+y*y+z*z+w*w
    if n < 1e-10: raise ValueError('Invalid TF quaternion')
    s=2/n
    R=np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
    o=t.transform.translation
    M=np.eye(4);M[:3,:3]=R;M[:3,3]=[o.x,o.y,o.z]
    return M.tolist()

class Capture(Node):
    def __init__(self, color_topic, depth_topic):
        super().__init__('camera_calibration_capture')
        self.color=None; self.depth=None
        self.create_subscription(CameraInfo,color_topic,lambda m:setattr(self,'color',m),10)
        self.create_subscription(CameraInfo,depth_topic,lambda m:setattr(self,'depth',m),10)
        self.tf_buffer=Buffer()
        self.tf_listener=TransformListener(self.tf_buffer,self)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--color-info',default='/camera/camera/color/camera_info')
    ap.add_argument('--depth-info',default='/camera/camera/depth/camera_info')
    ap.add_argument('--depth-scale',type=float,required=True,help='Meters per raw depth unit; verify from device')
    ap.add_argument('--pelvis-frame',default='pelvis')
    ap.add_argument('--output',default='g1_camera_calibration.yaml')
    ap.add_argument('--timeout',type=float,default=20)
    args=ap.parse_args()
    if args.depth_scale<=0: ap.error('depth-scale must be positive')
    rclpy.init()
    node=Capture(args.color_info,args.depth_info)
    try:
        deadline=time.monotonic()+args.timeout
        while time.monotonic()<deadline and (node.color is None or node.depth is None):
            rclpy.spin_once(node,timeout_sec=.2)
        if node.color is None or node.depth is None:
            raise RuntimeError('Missing CameraInfo; check topics and ROS_DOMAIN_ID')
        c,d=node.color,node.depth
        # Allow tf listener to receive /tf_static and /tf data.
        t_cd=None; t_pc=None
        while time.monotonic()<deadline and t_cd is None:
            rclpy.spin_once(node,timeout_sec=.2)
            try:
                # Latest transform for STATIC depth/color extrinsics.
                t_cd=node.tf_buffer.lookup_transform(c.header.frame_id,d.header.frame_id,rclpy.time.Time())
            except Exception: pass
        if t_cd is None:
            raise RuntimeError('No optical-frame extrinsics in /tf or /tf_static. Obtain via device SDK.')
        try:
            t_pc=node.tf_buffer.lookup_transform(args.pelvis_frame,c.header.frame_id,rclpy.time.Time())
        except Exception:
            print('Pelvis TF unavailable. Calibration still captured; G1 transform later.')
        # K for depth rectified image (if relevant, use P projection matrix instead).
        output={
            'color':{'width':c.width,'height':c.height,'distortion_model':c.distortion_model,'K':list(c.k),'D':list(c.d)},
            'depth':{'width':d.width,'height':d.height,'rectified':True,'K':list(d.k)},
            'depth_scale_m_per_unit':args.depth_scale,
            'T_color_depth':transform_matrix(t_cd),
            'T_pelvis_color':transform_matrix(t_pc) if t_pc else None,
        }
        Path(args.output).write_text(yaml.safe_dump(output,sort_keys=False))
        print('Saved',args.output)
        print('WARNING: T_pelvis_color is a snapshot, not a live timestamp-matched head pose.')
    finally:
        node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
