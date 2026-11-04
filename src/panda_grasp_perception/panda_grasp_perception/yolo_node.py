"""Synchronized RGB + depth + CameraInfo -> YOLOv8 segmentation -> real XYZ.

The coordinates are computed only from rendered camera pixels and calibration;
no mock target XYZ or PyBullet object ground truth enters the estimation.
"""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

from panda_grasp_perception.rgbd_geometry import locate_cube
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener, TransformException


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_cube_detector')
        default_weights = str(Path.home() / 'panda_grasp_ws' / 'models' / 'cube_best.pt')
        self.declare_parameter('weights', default_weights)
        self.declare_parameter('confidence', 0.20)
        self.declare_parameter('camera_eye', [0.48, 0.0, 1.12])
        self.declare_parameter('camera_target', [0.48, 0.0, 0.30])
        self.declare_parameter('camera_up', [0.0, 1.0, 0.0])
        self.declare_parameter('extrinsic_mode', 'groundtruth')
        self.declare_parameter('table_z', 0.30)
        self.declare_parameter('cube_half', 0.025)

        weights = Path(str(self.get_parameter('weights').value)).expanduser()
        self.confidence = float(self.get_parameter('confidence').value)
        self.camera_eye = list(self.get_parameter('camera_eye').value)
        self.camera_target = list(self.get_parameter('camera_target').value)
        self.camera_up = list(self.get_parameter('camera_up').value)
        self.extrinsic_mode = str(self.get_parameter('extrinsic_mode').value)
        if self.extrinsic_mode not in ('groundtruth', 'calibrated'):
            raise ValueError(f'Unknown extrinsic_mode: {self.extrinsic_mode}')
        self.table_z = float(self.get_parameter('table_z').value)
        self.cube_half = float(self.get_parameter('cube_half').value)
        if not weights.is_file():
            raise FileNotFoundError(f'Missing trained YOLO weights: {weights}')

        self.model = YOLO(str(weights))
        self.bridge = CvBridge()

        # 创建 TF2 缓冲区
        self.tf_buffer = Buffer()

        # 订阅 TF 坐标关系
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # Use exactly equal timestamps, produced by sim_camera for its 3 topics.
        # Buffer a few stamps because callback delivery order is not guaranteed.
        self._pending = {name: OrderedDict() for name in ('rgb', 'depth', 'info')}
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            lambda msg: self.on_frame('rgb', msg), 5)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            lambda msg: self.on_frame('depth', msg), 5)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            lambda msg: self.on_frame('info', msg), 5)

        self.annot_pub = self.create_publisher(Image, '/yolo/annotated_image', 1)
        self.mask_pub = self.create_publisher(Image, '/yolo/cube_mask', 1)
        # Coordinates of the cube center (not the camera image pixel center).
        self.target_pub = self.create_publisher(PointStamped, '/grasp/target_point', 5)
        self.target_camera_pub = self.create_publisher(
            PointStamped, '/grasp/target_point_camera', 5)
        self.get_logger().info(
            f'RGB-D YOLOv8 ready: {weights}; waiting for aligned RGB/Depth/CameraInfo')

    @staticmethod
    def stamp_key(msg):
        return (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def on_frame(self, source, msg):
        key = self.stamp_key(msg)
        pending = self._pending[source]
        pending[key] = msg
        # Avoid stale, unbounded frames if any topic is missing or delayed.
        while len(pending) > 6:
            pending.popitem(last=False)

        if not all(key in self._pending[name]
                   for name in ('rgb', 'depth', 'info')):
            return
        rgb_msg = self._pending['rgb'].pop(key)
        depth_msg = self._pending['depth'].pop(key)
        info_msg = self._pending['info'].pop(key)
        self.process_frame(rgb_msg, depth_msg, info_msg)

    def camera_geometry_for_frame(self, rgb_msg):
        """标定模式从 TF2 获取相机姿态，避免定位计算混入手工真值。"""
        if self.extrinsic_mode == 'groundtruth':
            return self.camera_eye, self.camera_target, self.camera_up
        frame = rgb_msg.header.frame_id or 'camera_optical_frame'
        try:
            tf = self.tf_buffer.lookup_transform(
                'world', frame, Time.from_msg(rgb_msg.header.stamp)
            )
        except TransformException as exc:
            raise ValueError(f'Calibrated camera TF unavailable: {exc}') from exc
        t, q = tf.transform.translation, tf.transform.rotation
        quat = np.asarray([q.x, q.y, q.z, q.w], dtype=np.float64)
        length = np.linalg.norm(quat)
        if not np.isfinite(length) or length < 1e-9:
            raise ValueError('Calibrated camera TF has invalid quaternion')
        x, y, z, w = quat / length
        rotation = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        eye = np.array([t.x, t.y, t.z], dtype=np.float64)
        return eye, eye + rotation[:, 2], -rotation[:, 1]

    def process_frame(self, rgb_msg, depth_msg, info_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            if bgr.shape[:2] != depth.shape[:2]:
                raise ValueError('RGB and Depth resolutions differ')

            result = self.model.predict(
                source=bgr, imgsz=640, conf=self.confidence,
                retina_masks=True, verbose=False)[0]
            overlay = bgr.copy()
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

            if (result.boxes is None or len(result.boxes) == 0 or
                    result.masks is None):
                self.get_logger().warning('No cube detected',
                                          throttle_duration_sec=5.0)
            else:
                cls = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                candidates = np.flatnonzero(cls == 0)  # User-trained class 0 = cube
                if len(candidates) == 0:
                    self.get_logger().warning('No class-0 cube in YOLO output',
                                              throttle_duration_sec=5.0)
                else:
                    i = int(candidates[np.argmax(confidences[candidates])])
                    polygon = np.asarray(result.masks.xy[i], dtype=np.float32)
                    if polygon.ndim != 2 or len(polygon) < 3:
                        self.get_logger().warning('Empty YOLO Mask',
                                                  throttle_duration_sec=5.0)
                    else:
                        polygon = np.rint(polygon).astype(np.int32)
                        polygon[:, 0] = np.clip(polygon[:, 0], 0, bgr.shape[1] - 1)
                        polygon[:, 1] = np.clip(polygon[:, 1], 0, bgr.shape[0] - 1)
                        cv2.fillPoly(mask, [polygon], 255)

                        green = np.zeros_like(overlay)
                        green[:, :, 1] = 255
                        overlay[mask > 0] = cv2.addWeighted(
                            overlay, 0.65, green, 0.35, 0)[mask > 0]
                        x1, y1, x2, y2 = map(
                            int, result.boxes.xyxy[i].cpu().numpy())
                        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                      (0, 255, 255), 2)
                        cv2.putText(overlay, f'cube {confidences[i]:.2f}',
                                    (x1, max(16, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)
                        try:
                            eye, target, up = self.camera_geometry_for_frame(rgb_msg)
                            estimate = locate_cube(
                                mask, depth, info_msg.k,
                                eye=eye,
                                target=target,
                                up=up,
                                table_z=self.table_z,
                                cube_half=self.cube_half)
                            
                            # 1. 获取方块在相机光学坐标系下的三维坐标
                            self.publish_target(estimate, rgb_msg, overlay)

                        except ValueError as exc:
                            # Publishing invalid XYZ would be unsafe for a grasp.
                            self.get_logger().warning(
                                f'Cube found but depth localization skipped: {exc}',
                                throttle_duration_sec=3.0)
        except Exception as exc:
            self.get_logger().error(f'RGB-D inference error: {exc}')
            return

        annotated = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        annotated.header = rgb_msg.header
        mask_msg.header = rgb_msg.header
        self.annot_pub.publish(annotated)
        self.mask_pub.publish(mask_msg)

    def publish_target(self, estimate, rgb_msg, overlay):
        """将相机光学坐标系目标转换到 world，发布坐标并返回无结果。"""
        xyz_camera = estimate['camera_center']
        camera_point = PointStamped()
        camera_point.header.stamp = rgb_msg.header.stamp
        camera_point.header.frame_id = (
            rgb_msg.header.frame_id or 'camera_optical_frame'
        )
        camera_point.point.x = float(xyz_camera[0])
        camera_point.point.y = float(xyz_camera[1])
        camera_point.point.z = float(xyz_camera[2])
        self.target_camera_pub.publish(camera_point)

        try:
            world_point = self.tf_buffer.transform(camera_point, 'world')
        except TransformException as exc:
            self.get_logger().warning(
                f'TF2 transform failed: {exc}', throttle_duration_sec=2.0
            )
        else:
            self.target_pub.publish(world_point)
            x = world_point.point.x
            y = world_point.point.y
            z = world_point.point.z
            old_xyz = estimate['world_center']
            tf_xyz = np.array([x, y, z])
            difference = np.linalg.norm(tf_xyz - old_xyz)
            cv2.putText(
                overlay,
                f'TF2 XYZ: {x:.3f}, {y:.3f}, {z:.3f} m',
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )
            self.get_logger().info(
                f'TF2 world XYZ = '
                f'({x:.4f}, {y:.4f}, {z:.4f}) m; '
                f'old method difference = '
                f'{difference * 1000:.2f} mm',
                throttle_duration_sec=2.0
            )

    @staticmethod
    def publish_point(publisher, stamp, frame_id, xyz):
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.point.x, msg.point.y, msg.point.z = map(float, xyz)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
