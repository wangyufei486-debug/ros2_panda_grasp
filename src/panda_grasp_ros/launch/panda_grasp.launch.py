
"""一键启动 Panda 仿真与 YOLOv8 RGB-D 感知节点。"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # ========== 1. 声明启动参数 ==========

    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='是否显示 PyBullet GUI'
    )

    cube_x_arg = DeclareLaunchArgument(
        'cube_x',
        default_value='0.48',
        description='方块初始 X 坐标'
    )

    cube_y_arg = DeclareLaunchArgument(
        'cube_y',
        default_value='-0.12',
        description='方块初始 Y 坐标'
    )

    camera_hz_arg = DeclareLaunchArgument(
        'camera_hz',
        default_value='2.0',
        description='相机图像发布频率'
    )

    extrinsic_mode_arg = DeclareLaunchArgument(
        'extrinsic_mode',
        default_value='groundtruth',
        description='相机外参模式：groundtruth 或 calibrated'
    )

    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value=str(Path.home() / 'panda_grasp_ws/outputs/calibration_collect/full_18/handeye_result.npz'),
        description='calibrated 模式使用的 handeye_result.npz 路径'
    )

    # ========== 2. PyBullet 仿真节点 ==========

    sim_node = Node(
        package='panda_grasp_ros',
        executable='sim_camera',
        name='sim_camera',
        output='screen',
        parameters=[{
            'gui': ParameterValue(
                LaunchConfiguration('gui'),
                value_type=bool
            ),
            'cube_x': ParameterValue(
                LaunchConfiguration('cube_x'),
                value_type=float
            ),
            'cube_y': ParameterValue(
                LaunchConfiguration('cube_y'),
                value_type=float
            ),
            'camera_hz': ParameterValue(
                LaunchConfiguration('camera_hz'),
                value_type=float
            ),
            'extrinsic_mode': LaunchConfiguration('extrinsic_mode'),
            'calibration_file': LaunchConfiguration('calibration_file')
        }]
    )

    # ========== 3. YOLOv8 视觉感知节点 ==========

    yolo_node = Node(
        package='panda_grasp_perception',
        executable='yolo_detector',
        output='screen',
        parameters=[{'extrinsic_mode': LaunchConfiguration('extrinsic_mode')}]
    )

    # ========== 4. 统一启动 ==========

    return LaunchDescription([
        gui_arg,
        cube_x_arg,
        cube_y_arg,
        camera_hz_arg,
        extrinsic_mode_arg,
        calibration_file_arg,
        sim_node,
        yolo_node
    ])
