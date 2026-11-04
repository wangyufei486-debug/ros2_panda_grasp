
"""PyBullet Panda 相机节点：发布 RGB-D 图像并执行视觉引导抓取。"""

import time
from pathlib import Path

import numpy as np
import pybullet as p
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from panda_grasp_ros.grasp_controller import PandaGraspController
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from panda_grasp_interfaces.action import ExecuteGrasp
from tf2_ros import StaticTransformBroadcaster

from panda_grasp_ros.pybullet_scene import (
    create_scene,
    TABLE_Z,
    CUBE_HALF,
)

from panda_grasp_ros.sim_rgbd_camera import (
    SimRGBDCamera,
    CAM_W,
    CAM_H,
    CAM_EYE,
)
from panda_grasp_ros.tf_utils import camera_transform, static_camera_transforms

class SimCameraNode(Node):

    # ========== 1. 初始化 ROS 2 节点 ==========

    def __init__(self):
        super().__init__('sim_camera')

        self.declare_parameter('gui', False)
        self.declare_parameter('camera_hz', 2.0)
        self.declare_parameter('extrinsic_mode', 'groundtruth')
        self.declare_parameter(
            'calibration_file',
            str(Path.home() / 'panda_grasp_ws/outputs/calibration_collect/full_18/handeye_result.npz')
        )

        # 方块初始位置，支持通过 ROS 2 参数修改
        self.declare_parameter('cube_x', 0.48)
        self.declare_parameter('cube_y', -0.12)

        self.pick_xy = [
            float(self.get_parameter('cube_x').value),
            float(self.get_parameter('cube_y').value)
        ]

        gui = bool(self.get_parameter('gui').value)
        hz = float(self.get_parameter('camera_hz').value)
        self.extrinsic_mode = str(self.get_parameter('extrinsic_mode').value)
        calibration_file = str(self.get_parameter('calibration_file').value)

        if hz <= 0:
            raise ValueError('camera_hz must be positive')

        # 启动仿真前验证标定文件；已是 panda_link0 外参，不再加 0.05 m。
        self.T_base_camera = camera_transform(
            self.extrinsic_mode, CAM_EYE, calibration_file
        )

        self.bridge = CvBridge()

        # 发布 RGB、Depth、CameraInfo
        self.rgb_pub = self.create_publisher(
            Image, '/camera/color/image_raw', 1
        )
        self.depth_pub = self.create_publisher(
            Image, '/camera/depth/image_raw', 1
        )
        self.info_pub = self.create_publisher(
            CameraInfo, '/camera/color/camera_info', 1
        )

        # 连接 PyBullet
        cid = p.connect(p.GUI if gui else p.DIRECT)

        if cid < 0:
            raise RuntimeError('Unable to connect to PyBullet')

        self.client_id = cid

        # 创建机器人、桌面和方块
        self.robot, self.cube = create_scene(
            self.pick_xy,
            gui
        )

        # ========== 2. 初始化抓取控制器 ==========

        # 抓取控制器和相机共用当前 PyBullet 世界
        self.grasp_controller = PandaGraspController(
            self.robot,
            self.cube,
            self.get_logger()
        )

        # 保存最新的视觉目标坐标
        self.latest_target = None
        self.target_received_at = 0.0

        # 每次启动仿真只允许尝试一次抓取
        self.grasp_attempted = False

        # 订阅 YOLOv8 + RGB-D 定位结果
        self.target_sub = self.create_subscription(
            PointStamped,
            '/grasp/target_point',
            self.on_target,
            10
        )

        # 创建抓取触发服务
        self.grasp_service = self.create_service(
            Trigger,
            '/grasp/execute',
            self.on_grasp
        )

          # 保存 Action 阶段反馈回调
        self._action_feedback_cb = None

        # 创建 Action Server
        self.grasp_action = ActionServer(
            self,
            ExecuteGrasp,
            '/grasp/task',
            execute_callback=self.execute_grasp_action,
            goal_callback=self.on_grasp_goal,
            cancel_callback=self.on_grasp_cancel
        )

        # ========== 3. 初始化虚拟相机 ==========

        # 创建 RGB-D 相机对象
        self.camera = SimRGBDCamera()

        # 创建静态 TF 发布器
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # 发布相机和 Panda 基座的坐标关系
        self.publish_static_tf()

        # 定时发布相机图像
        self.timer = self.create_timer(
            1.0 / hz,
            self.publish_frames
        )

        self.get_logger().info(
            f'PyBullet camera started '
            f'({CAM_W}x{CAM_H}, {hz:g} Hz, gui={gui})'
        )

    # ========== 4. 接收 YOLOv8 三维定位坐标 ==========

    def on_target(self, msg):

        # 当前仅接受世界坐标系下的目标
        if msg.header.frame_id != 'world':
            self.get_logger().warning(
                'Expected world-frame target'
            )
            return

        self.latest_target = (
            msg.point.x,
            msg.point.y,
            msg.point.z
        )

        self.target_received_at = time.monotonic()

    
    def on_grasp_goal(self, goal_request):
        """收到 Action Goal：检查是否可以开始抓取。"""
        if not goal_request.start:
            return GoalResponse.REJECT

        if self.grasp_attempted:
            return GoalResponse.REJECT

        if (
            self.latest_target is None
            or time.monotonic() - self.target_received_at > 10.0
        ):
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def on_grasp_cancel(self, goal_handle):
        """第一版暂不支持中途取消运动。"""
        return CancelResponse.REJECT

    def execute_grasp_action(self, goal_handle):
        """执行 Action，并将抓取过程转为 Feedback。"""

        result = ExecuteGrasp.Result()

        def publish_stage(stage, progress):
            feedback = ExecuteGrasp.Feedback()
            feedback.stage = stage
            feedback.progress = float(progress)

            goal_handle.publish_feedback(feedback)

            self.get_logger().info(
                f"Action Feedback: {stage} ({progress:.0%})"
            )

        self._action_feedback_cb = publish_stage

        try:
            # 复用原有抓取 Service 的检查与执行逻辑
            srv_response = self.on_grasp(
                Trigger.Request(),
                Trigger.Response()
            )

            result.success = bool(srv_response.success)
            result.message = str(srv_response.message)

            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()

        except Exception as exc:
            result.success = False
            result.message = str(exc)

            self.get_logger().error(result.message)
            goal_handle.abort()

        finally:
            self._action_feedback_cb = None

        return result


    # ========== 5. ROS 2 抓取服务 ==========

    def on_grasp(self, request, response):

        # 防止同一次仿真重复抓取
        if self.grasp_attempted:
            response.success = False
            response.message = (
                'Already attempted; restart sim_camera to reset scene'
            )
            return response

        # 检查最近是否收到视觉坐标
        if (
            self.latest_target is None
            or time.monotonic() - self.target_received_at > 10.0
        ):
            response.success = False
            response.message = (
                'No recent YOLO RGB-D target; check detector'
            )
            return response

        x, y, z = self.latest_target

        # 检查是否为有效且合理的工作空间坐标
        if (
            not np.isfinite([x, y, z]).all()
            or not (
                0.27 <= x <= 0.69
                and -0.22 <= y <= 0.22
                and abs(z - (TABLE_Z + CUBE_HALF)) <= 0.035
            )
        ):
            response.success = False
            response.message = (
                'Target outside expected cube workspace'
            )
            return response
        
        self.grasp_controller.log_visual_target_error((x, y, z))

        # 标记本次抓取已开始
        self.grasp_attempted = True

        try:
            # 使用视觉计算的坐标执行物理抓取
            response.success, response.message = (
                self.grasp_controller.execute(
                    (x, y, z),
                    feedback_cb=self._action_feedback_cb
                )
            )

        except Exception as exc:
            response.success = False
            response.message = f'Grasp stopped: {exc}'

            self.get_logger().error(response.message)

        return response

    
    def publish_static_tf(self):
        """发布 world、Panda 基座和相机之间的静态 TF。"""
        stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(
            static_camera_transforms(stamp, CAM_EYE, self.T_base_camera)
        )
        self.get_logger().info(
            f'Static TF published ({self.extrinsic_mode}): '
            'world -> panda_link0 -> camera_optical_frame'
        )


    # ========== 6. 发布 RGB-D 图像 ==========

    def publish_frames(self):
        """获取 RGB-D 数据并发布 ROS 2 Topic。"""

        # 1. 更新物理仿真
        for _ in range(12):
            p.stepSimulation()

        # 2. 从相机模块获取 RGB 和 Depth
        rgb_bgr, depth_m = self.camera.capture()
        stamp = self.get_clock().now().to_msg()
        rgb_msg, depth_msg, info = self.camera.to_messages(
            self.bridge, rgb_bgr, depth_m, stamp
        )
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info)


    # ========== 7. 关闭节点 ==========

    def destroy_node(self):

        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

        return super().destroy_node()


# ========== 8. ROS 2 主函数 ==========

def main(args=None):

    rclpy.init(args=args)

    node = SimCameraNode()

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
