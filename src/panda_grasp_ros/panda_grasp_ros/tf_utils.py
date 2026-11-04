"""仿真场景中的静态 TF 消息构造。"""

import json
from zipfile import BadZipFile
from pathlib import Path

import numpy as np
from geometry_msgs.msg import TransformStamped


def rotation_matrix_to_quaternion(rotation):
    """右手系旋转矩阵转 ROS 使用的 (x, y, z, w) 四元数。"""
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = [(rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale, scale / 4.0]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = 2.0 * np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k])
        q = [0.0] * 4
        q[i] = scale / 4.0
        q[j] = (rotation[j, i] + rotation[i, j]) / scale
        q[k] = (rotation[k, i] + rotation[i, k]) / scale
        q[3] = (rotation[k, j] - rotation[j, k]) / scale
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def load_calibrated_camera_transform(filename):
    """读取已在 panda_link0 框架下求得的 T_base_camera，不再修正 0.05 m。"""
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'手眼标定文件不存在: {path}')
    try:
        with np.load(path, allow_pickle=False) as result:
            matrix = np.asarray(result['T_base_camera'], dtype=np.float64)
            inertial = np.asarray(result['T_base_base_inertial'], dtype=np.float64)
            stored_rotation = np.asarray(result['R_base_camera'], dtype=np.float64)
            stored_translation = np.asarray(result['t_base_camera_m'], dtype=np.float64)
    except (OSError, ValueError, KeyError, BadZipFile, EOFError) as exc:
        raise ValueError(f'无法读取 T_base_camera 和坐标元数据: {path}: {exc}') from exc
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-4)):
        raise ValueError(f'T_base_camera 不是有效的刚体变换: {path}')
    if (inertial.shape != (4, 4)
            or not np.allclose(inertial, np.array([
                [1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0.05], [0, 0, 0, 1]
            ]), atol=1e-7)
            or stored_rotation.shape != (3, 3)
            or stored_translation.shape != (3,)
            or not np.allclose(stored_rotation, matrix[:3, :3], atol=1e-7)
            or not np.allclose(stored_translation, matrix[:3, 3], atol=1e-7)):
        raise ValueError(f'标定文件的基座惯性修正或矩阵字段不一致: {path}')

    # 第三阶段报告明确标注了 base=panda_link0、camera=optical；缺失时拒绝猜测。
    report_path = path.with_name('handeye_report.json')
    if not report_path.is_file():
        raise FileNotFoundError(f'缺少手眼结果坐标系报告: {report_path}')
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        frames = report['frames']
        report_matrix = np.asarray(report['T_base_camera'], dtype=np.float64)
        report_offset = np.asarray(report['base_inertial_offset_m'], dtype=np.float64)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'手眼结果坐标系报告无效: {report_path}: {exc}') from exc
    if (not isinstance(frames, dict)
            or frames.get('base') != 'panda_link0 URDF Link'
            or frames.get('camera') != 'camera_optical_frame'
            or report_matrix.shape != (4, 4)
            or not np.allclose(report_matrix, matrix, atol=1e-7)
            or report_offset.shape != (3,)
            or not np.allclose(report_offset, [0, 0, 0.05], atol=1e-7)):
        raise ValueError(f'手眼结果坐标系与 panda_link0 → camera_optical_frame 不匹配: {report_path}')
    return matrix


def camera_transform(mode, camera_eye, calibration_file=None):
    """一次性选择相机外参；标定模式失败时绝不回退到真值。"""
    if mode == 'groundtruth':
        matrix = np.diag([1.0, -1.0, -1.0, 1.0])
        matrix[:3, 3] = camera_eye
        return matrix
    if mode == 'calibrated':
        if not calibration_file:
            raise ValueError('calibrated 模式必须提供 calibration_file')
        return load_calibrated_camera_transform(calibration_file)
    raise ValueError(f'未知相机外参模式: {mode}; 可选 groundtruth 或 calibrated')


def static_camera_transforms(stamp, camera_eye, camera_matrix=None):
    """构造 world → panda_link0 → camera_optical_frame，返回两段静态变换。"""
    base_tf = TransformStamped()
    base_tf.header.stamp = stamp
    base_tf.header.frame_id = 'world'
    base_tf.child_frame_id = 'panda_link0'
    base_tf.transform.translation.x = 0.0
    base_tf.transform.translation.y = 0.0
    base_tf.transform.translation.z = 0.0
    base_tf.transform.rotation.x = 0.0
    base_tf.transform.rotation.y = 0.0
    base_tf.transform.rotation.z = 0.0
    base_tf.transform.rotation.w = 1.0

    camera_tf = TransformStamped()
    camera_tf.header.stamp = stamp
    camera_tf.header.frame_id = 'panda_link0'
    camera_tf.child_frame_id = 'camera_optical_frame'
    matrix = camera_matrix if camera_matrix is not None else camera_transform(
        'groundtruth', camera_eye
    )
    q = rotation_matrix_to_quaternion(matrix[:3, :3])
    camera_tf.transform.translation.x = float(matrix[0, 3])
    camera_tf.transform.translation.y = float(matrix[1, 3])
    camera_tf.transform.translation.z = float(matrix[2, 3])
    camera_tf.transform.rotation.x = q[0]
    camera_tf.transform.rotation.y = q[1]
    camera_tf.transform.rotation.z = q[2]
    camera_tf.transform.rotation.w = q[3]
    return [base_tf, camera_tf]
