
"""PyBullet 虚拟 RGB-D 相机模块。"""

import cv2
import numpy as np
import pybullet as p

from sensor_msgs.msg import CameraInfo


# ========== 相机参数 ==========

CAM_W, CAM_H = 640, 480
CAM_FOV = 60.0
CAM_NEAR, CAM_FAR = 0.05, 2.0

CAM_EYE = [0.48, 0.0, 1.12]
CAM_TARGET = [0.48, 0.0, 0.30]
CAM_UP = [0, 1, 0]

FRAME_ID = 'camera_optical_frame'


class SimRGBDCamera:
    """PyBullet RGB-D 相机。"""

    def __init__(self):

        # 1. 相机观察矩阵
        self.view = p.computeViewMatrix(
            cameraEyePosition=CAM_EYE,
            cameraTargetPosition=CAM_TARGET,
            cameraUpVector=CAM_UP
        )

        # 2. 相机投影矩阵
        self.proj = p.computeProjectionMatrixFOV(
            fov=CAM_FOV,
            aspect=CAM_W / CAM_H,
            nearVal=CAM_NEAR,
            farVal=CAM_FAR
        )

        # 3. 计算相机内参
        projection = np.array(
            self.proj, dtype=np.float64
        ).reshape((4, 4), order='F')

        fx = float(projection[0, 0] * CAM_W / 2)
        fy = float(projection[1, 1] * CAM_H / 2)
        cx = CAM_W / 2
        cy = CAM_H / 2

        # 4. 构建 ROS 2 CameraInfo
        self.info = CameraInfo()

        self.info.width = CAM_W
        self.info.height = CAM_H
        self.info.distortion_model = 'plumb_bob'
        self.info.d = [0.0] * 5

        self.info.k = [
            fx, 0., cx,
            0., fy, cy,
            0., 0., 1.
        ]

        self.info.r = [
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ]

        self.info.p = [
            fx, 0., cx, 0.,
            0., fy, cy, 0.,
            0., 0., 1., 0.
        ]

    def capture(self):
        """采集 RGB-D 图像。

        返回:
            rgb_bgr: BGR 图像，uint8
            depth_m: 深度图，float32，单位米
        """

        # 1. PyBullet 渲染图像
        image = p.getCameraImage(
            CAM_W,
            CAM_H,
            viewMatrix=self.view,
            projectionMatrix=self.proj,
            renderer=p.ER_TINY_RENDERER
        )

        # 2. RGBA 转 BGR
        rgba = np.asarray(
            image[2], dtype=np.uint8
        ).reshape(CAM_H, CAM_W, 4)

        rgb_bgr = cv2.cvtColor(
            rgba,
            cv2.COLOR_RGBA2BGR
        )

        # 3. 获取深度缓冲
        depth_buffer = np.asarray(
            image[3], dtype=np.float32
        ).reshape(CAM_H, CAM_W)

        # 4. 深度缓冲转换为实际深度（米）
        depth_m = (
            CAM_FAR * CAM_NEAR /
            (
                CAM_FAR -
                (CAM_FAR - CAM_NEAR) * depth_buffer
            )
        ).astype(np.float32)

        return rgb_bgr, depth_m

    def to_messages(self, bridge, rgb_bgr, depth_m, stamp):
        """将已采集的相机坐标系 RGB-D 用同一时间戳生成 ROS 消息。"""
        rgb_msg = bridge.cv2_to_imgmsg(rgb_bgr, encoding='bgr8')
        depth_msg = bridge.cv2_to_imgmsg(depth_m, encoding='32FC1')
        rgb_msg.header.stamp = stamp
        depth_msg.header.stamp = stamp
        rgb_msg.header.frame_id = FRAME_ID
        depth_msg.header.frame_id = FRAME_ID
        self.info.header.stamp = stamp
        self.info.header.frame_id = FRAME_ID
        return rgb_msg, depth_msg, self.info
