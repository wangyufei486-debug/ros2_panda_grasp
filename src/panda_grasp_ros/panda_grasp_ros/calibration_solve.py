"""Eye-to-Hand 第三阶段：从已采集数据求解并验证 T_base_camera。"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pybullet_data


METHODS = {
    'park': cv2.CALIB_HAND_EYE_PARK,
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
}
MAX_PNP_RMS_PX = 1.0
MAX_CONSISTENCY_TRANSLATION_MM = 5.0
MAX_CONSISTENCY_ROTATION_DEG = 1.0


def rigid_transform(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation).reshape(3)
    return matrix


def rotation_error_deg(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def check_transform(matrix, label):
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{label} 不是有限的 4×4 矩阵')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f'{label} 最后一行不是 [0, 0, 0, 1]')
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
        raise ValueError(f'{label} 旋转矩阵无效')


def base_inertial_offset_from_urdf():
    """读取 panda_link0 Link → 惯性坐标系的固定变换，不启动 PyBullet。"""
    urdf_path = Path(pybullet_data.getDataPath()) / 'franka_panda' / 'panda.urdf'
    root = ET.parse(urdf_path).getroot()
    origin = root.find("./link[@name='panda_link0']/inertial/origin")
    if origin is None:
        raise ValueError(f'URDF 缺少 panda_link0 惯性原点: {urdf_path}')
    xyz = np.fromstring(origin.attrib.get('xyz', ''), sep=' ')
    rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('panda_link0 惯性原点数据无效')
    if not np.allclose(rpy, 0.0, atol=1e-10):
        raise ValueError('panda_link0 惯性原点有旋转；需扩展此处的 URDF 变换解析')
    return rigid_transform(np.eye(3), xyz), urdf_path


def object_points(cols, rows, square_size):
    """SB 返回的行优先顺序：第 0 个内角点为 target 原点。"""
    return np.asarray(
        [(col * square_size, row * square_size, 0.0)
         for row in range(rows) for col in range(cols)],
        dtype=np.float64,
    )


def checked_corners(sample, dataset_dir, cols, rows, image_shape, reference_axes):
    path = dataset_dir / sample['image']
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != image_shape:
        raise ValueError(f'样本 {sample["sample_id"]} 图像缺失或尺寸不匹配: {path}')
    corners = np.asarray(sample['corners_px'], dtype=np.float64)
    if (corners.shape != (cols * rows, 2)
            or sample['corner_count'] != cols * rows
            or not np.isfinite(corners).all()):
        raise ValueError(f'样本 {sample["sample_id"]} 角点数量或坐标无效')

    # 重新检测保存的图像，核实角点数组确实与该图片逐点对应。
    found, detected = cv2.findChessboardCornersSB(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (cols, rows)
    )
    if not found or detected is None or len(detected) != cols * rows:
        raise ValueError(f'样本 {sample["sample_id"]} 的保存图像无法重新检测完整棋盘')
    max_difference = np.max(np.linalg.norm(detected.reshape(-1, 2) - corners, axis=1))
    if max_difference > 0.5:
        raise ValueError(f'样本 {sample["sample_id"]} 保存的角点与图像不对应')

    grid = corners.reshape(rows, cols, 2)
    col_axis = np.diff(grid, axis=1).mean(axis=(0, 1))
    row_axis = np.diff(grid, axis=0).mean(axis=(0, 1))
    if abs(np.linalg.det(np.column_stack((col_axis, row_axis)))) < 1.0:
        raise ValueError(f'样本 {sample["sample_id"]} 的角点网格退化')
    axes = (col_axis / np.linalg.norm(col_axis), row_axis / np.linalg.norm(row_axis))
    if reference_axes is not None and any(
        np.dot(axis, reference) < 0.0
        for axis, reference in zip(axes, reference_axes)
    ):
        raise ValueError(f'样本 {sample["sample_id"]} 的角点起点或排序发生翻转')
    return corners, axes


def camera_truth_for_validation():
    """只在求解完成后读取固定相机的场景参数，构造 optical → panda_link0。"""
    from panda_grasp_ros.sim_rgbd_camera import CAM_EYE, CAM_TARGET, CAM_UP

    eye = np.asarray(CAM_EYE, dtype=np.float64)
    forward = np.asarray(CAM_TARGET, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, CAM_UP)
    right /= np.linalg.norm(right)
    down = -np.cross(right, forward)
    # OpenCV optical: x 向右、y 向下、z 朝向场景。场景中 panda_link0 与 world 重合。
    return rigid_transform(np.column_stack((right, down, forward)), eye)


def solve(dataset_path, result_path, method_name):
    dataset_path = dataset_path.resolve()
    dataset_dir = dataset_path.parent
    data = json.loads(dataset_path.read_text(encoding='utf-8'))
    samples = data['samples']
    if len(samples) < 3:
        raise ValueError('有效样本少于 3 组，无法进行手眼标定')
    if data['camera']['frame_id'] != 'camera_optical_frame':
        raise ValueError('相机坐标系不是 camera_optical_frame')
    cols, rows = data['chessboard']['inner_corners_cols_rows']
    square_size = float(data['chessboard']['square_size_m'])
    if (cols, rows) != (8, 5) or square_size <= 0:
        raise ValueError('棋盘格应为 8×5 内角点且方格尺寸应为正数')
    if data['chessboard']['squares_cols_rows'] != [9, 6]:
        raise ValueError('棋盘格方格数与 8×5 内角点不对应')
    object_xyz = object_points(cols, rows, square_size)

    K = np.asarray(data['camera']['K'], dtype=np.float64)
    D = np.asarray(data['camera']['D'], dtype=np.float64).reshape(-1, 1)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or not np.isfinite(D).all()):
        raise ValueError('相机内参 K 或畸变 D 无效')
    image_shape = (data['camera']['height'], data['camera']['width'])

    mount_path = dataset_dir / data['chessboard']['mount_file']
    with np.load(mount_path) as mount:
        if (not np.isclose(float(mount['square_size']), square_size)
                or not np.array_equal(mount['pattern_size'], [cols, rows])):
            raise ValueError('board_mount.npz 与 dataset.json 的棋盘尺寸不一致')
    T_hand_board = np.asarray(data['chessboard']['T_hand_board'], dtype=np.float64)
    check_transform(T_hand_board, 'T_hand_board')

    # 第二阶段使用 getBasePositionAndOrientation(robot)；它是基座惯性框架，
    # panda_link0 URDF 的惯性原点高 0.05 m。先修正为真正的 Link 基座框架。
    T_base_base_inertial, urdf_path = base_inertial_offset_from_urdf()
    T_base_hand, T_hand_base, T_camera_target = [], [], []
    reprojection_rms, reprojection_max = [], []
    sample_ids, image_paths = [], []
    reference_axes = None

    for sample in samples:
        corners, axes = checked_corners(
            sample, dataset_dir, cols, rows, image_shape, reference_axes
        )
        if reference_axes is None:
            reference_axes = axes
        raw_pose = np.asarray(sample['T_base_hand'], dtype=np.float64)
        check_transform(raw_pose, f'样本 {sample["sample_id"]} 原始末端位姿')
        base_hand = T_base_base_inertial @ raw_pose
        T_base_hand.append(base_hand)
        T_hand_base.append(np.linalg.inv(base_hand))

        success, rvec, tvec = cv2.solvePnP(
            object_xyz, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            raise ValueError(f'样本 {sample["sample_id"]} solvePnP 失败')
        rotation, _ = cv2.Rodrigues(rvec)
        camera_target = rigid_transform(rotation, tvec)
        if tvec[2, 0] <= 0:
            raise ValueError(f'样本 {sample["sample_id"]} 的标定板位于相机背后')
        check_transform(camera_target, f'样本 {sample["sample_id"]} T_camera_target')
        T_camera_target.append(camera_target)

        projected, _ = cv2.projectPoints(object_xyz, rvec, tvec, K, D)
        pixel_error = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
        rms = float(np.sqrt(np.mean(pixel_error ** 2)))
        if rms > MAX_PNP_RMS_PX:
            raise ValueError(
                f'样本 {sample["sample_id"]} PnP 重投影 RMS {rms:.3f} px '
                f'超过 {MAX_PNP_RMS_PX:.1f} px'
            )
        reprojection_rms.append(rms)
        reprojection_max.append(float(pixel_error.max()))
        sample_ids.append(sample['sample_id'])
        image_paths.append(sample['image'])

    # Eye-to-Hand：把机器人运动反向成 hand ← base，
    # OpenCV 的返回值 cam2gripper 此时就是 base ← camera。
    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in T_hand_base],
        [pose[:3, 3] for pose in T_hand_base],
        [pose[:3, :3] for pose in T_camera_target],
        [pose[:3, 3] for pose in T_camera_target],
        method=METHODS[method_name],
    )
    T_base_camera = rigid_transform(rotation, translation)
    check_transform(T_base_camera, 'T_base_camera')

    # 固定在手部的 target 在每个样本中应得到同一个 T_hand_target。
    T_hand_target = np.asarray([
        hand_base @ T_base_camera @ camera_target
        for hand_base, camera_target in zip(T_hand_base, T_camera_target)
    ])
    mean_translation = T_hand_target[:, :3, 3].mean(axis=0)
    translation_residual_mm = np.linalg.norm(
        T_hand_target[:, :3, 3] - mean_translation, axis=1
    ) * 1000.0
    reference_rotation = T_hand_target[0, :3, :3]
    rotation_residual_deg = np.asarray([
        rotation_error_deg(reference_rotation.T @ pose[:3, :3])
        for pose in T_hand_target
    ])
    if (translation_residual_mm.max() > MAX_CONSISTENCY_TRANSLATION_MM
            or rotation_residual_deg.max() > MAX_CONSISTENCY_ROTATION_DEG):
        raise ValueError(
            '手眼结果不合理：T_hand_target 跨样本变化过大，'
            f'最大平移 {translation_residual_mm.max():.2f} mm、'
            f'最大旋转 {rotation_residual_deg.max():.2f}°；'
            '请检查角点排序、姿态多样性或求解方法'
        )

    # 真值仅在 solvePnP、calibrateHandEye 和内部一致性检查完成后读取。
    T_base_camera_true = camera_truth_for_validation()
    translation_error_mm = float(np.linalg.norm(
        T_base_camera[:3, 3] - T_base_camera_true[:3, 3]
    ) * 1000.0)
    rotation_truth_error_deg = rotation_error_deg(
        T_base_camera_true[:3, :3].T @ T_base_camera[:3, :3]
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_path,
        T_base_camera=T_base_camera,
        R_base_camera=T_base_camera[:3, :3],
        t_base_camera_m=T_base_camera[:3, 3],
        T_base_base_inertial=T_base_base_inertial,
        T_base_hand=np.asarray(T_base_hand),
        T_hand_base=np.asarray(T_hand_base),
        T_camera_target=np.asarray(T_camera_target),
        T_hand_target=T_hand_target,
        T_hand_board=T_hand_board,
        sample_ids=np.asarray(sample_ids),
        image_paths=np.asarray(image_paths),
        object_points_target_m=object_xyz,
        K=K,
        D=D,
        pnp_reprojection_rms_px=np.asarray(reprojection_rms),
        pnp_reprojection_max_px=np.asarray(reprojection_max),
        hand_target_translation_residual_mm=translation_residual_mm,
        hand_target_rotation_residual_deg=rotation_residual_deg,
        validation_translation_error_mm=translation_error_mm,
        validation_rotation_error_deg=rotation_truth_error_deg,
        method=method_name,
    )
    report = {
        'dataset': str(dataset_path),
        'result': str(result_path.resolve()),
        'method': method_name,
        'sample_count': len(samples),
        'frames': {
            'base': 'panda_link0 URDF Link',
            'hand': 'panda_hand URDF Link',
            'camera': 'camera_optical_frame',
            'target': 'SB 检测顺序中的第一个内角点，x 沿列递增、y 沿行递增',
        },
        'base_inertial_offset_m': T_base_base_inertial[:3, 3].tolist(),
        'base_inertial_source': str(urdf_path),
        'T_base_camera': T_base_camera.tolist(),
        'pnp_rms_px_mean': float(np.mean(reprojection_rms)),
        'pnp_rms_px_max': float(np.max(reprojection_rms)),
        'pnp_max_point_error_px': float(np.max(reprojection_max)),
        'per_sample': [
            {
                'sample_id': sample_id,
                'image': image_path,
                'pnp_reprojection_rms_px': rms,
                'pnp_reprojection_max_px': max_error,
                'hand_target_translation_residual_mm': float(translation_residual_mm[i]),
                'hand_target_rotation_residual_deg': float(rotation_residual_deg[i]),
            }
            for i, (sample_id, image_path, rms, max_error) in enumerate(zip(
                sample_ids, image_paths, reprojection_rms, reprojection_max
            ))
        ],
        'hand_target_translation_residual_mm_max': float(translation_residual_mm.max()),
        'hand_target_rotation_residual_deg_max': float(rotation_residual_deg.max()),
        'truth_translation_error_mm': translation_error_mm,
        'truth_rotation_error_deg': rotation_truth_error_deg,
        'truth_source': 'sim_rgbd_camera.CAM_EYE/CAM_TARGET/CAM_UP；仅用于求解后验证',
    }
    report_path = result_path.with_name('handeye_report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    np.set_printoptions(precision=6, suppress=True)
    print(f'方法: {method_name.upper()}，样本: {len(samples)}')
    print(f'基座惯性框架 → panda_link0 修正: {T_base_base_inertial[:3, 3]} m')
    print('R_base_camera =\n', T_base_camera[:3, :3])
    print('t_base_camera_m =', T_base_camera[:3, 3])
    print('T_base_camera =\n', T_base_camera)
    print(f'PnP 重投影 RMS: 均值 {np.mean(reprojection_rms):.3f} px，'
          f'最大 {np.max(reprojection_rms):.3f} px')
    print(f'T_hand_target 最大变化: 平移 {translation_residual_mm.max():.3f} mm，'
          f'旋转 {rotation_residual_deg.max():.3f}°')
    print(f'相对仿真真值: 平移误差 {translation_error_mm:.3f} mm，'
          f'旋转误差 {rotation_truth_error_deg:.3f}°')
    print('结果:', result_path)
    print('报告:', report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset', type=Path,
        default=Path.cwd() / 'outputs/calibration_collect/full_18/dataset.json',
    )
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--method', choices=METHODS, default='park')
    args = parser.parse_args()
    result_path = args.output or args.dataset.parent / 'handeye_result.npz'
    try:
        solve(args.dataset, result_path, args.method)
    except (ValueError, FileNotFoundError, cv2.error) as exc:
        parser.exit(1, f'标定失败: {exc}\n')


if __name__ == '__main__':
    main()
