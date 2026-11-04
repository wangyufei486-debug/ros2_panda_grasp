"""Eye-to-Hand 第二阶段：采集多姿态棋盘格图像及对应的机器人位姿。"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

from panda_grasp_ros.calibration_demo import (
    COLS,
    OUTPUT_DIR as DEMO_OUTPUT_DIR,
    PATTERN_SIZE,
    ROWS,
    SQUARE_SIZE,
    attach_to_hand,
    create_chessboard,
)
from panda_grasp_ros.pybullet_scene import create_scene
from panda_grasp_ros.sim_rgbd_camera import FRAME_ID, SimRGBDCamera


HOME = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.6, 0.7])
ARM_FORCES = [87, 87, 87, 87, 20, 20, 20]
SETTLE_STEPS = 300
JOINT_TOLERANCE_RAD = 0.02
JOINT_VELOCITY_TOLERANCE = 0.03
CORNER_MARGIN_PX = 25
BOARD_MARGIN_PX = 8

# 按序尝试：先是 5 个单关节小扰动，再加入其他关节及组合扰动。
# 数值是相对于 pybullet_scene.create_scene() 中 HOME 的偏移，单位 rad。
POSE_OFFSETS = [
    (0, 0, 0, 0, 0, 0, 0),
    (-.05, 0, 0, 0, 0, 0, 0),
    (+.08, 0, 0, 0, 0, 0, 0),
    (0, -.08, 0, 0, 0, 0, 0),
    (0, +.08, 0, 0, 0, 0, 0),
    (0, 0, -.12, 0, 0, 0, 0),
    (0, 0, +.12, 0, 0, 0, 0),
    (0, 0, 0, -.12, 0, 0, 0),
    (0, 0, 0, +.07, 0, 0, 0),
    (0, 0, 0, 0, -.15, 0, 0),
    (0, 0, 0, 0, +.15, 0, 0),
    (0, 0, 0, 0, 0, -.15, 0),
    (0, 0, 0, 0, 0, +.08, 0),
    (0, 0, 0, 0, 0, 0, -.12),
    (0, 0, 0, 0, 0, 0, +.12),
    (-.06, -.06, +.08, 0, 0, 0, 0),
    (+.06, +.06, -.08, 0, 0, 0, 0),
    (-.06, +.05, 0, -.08, +.10, 0, 0),
    (+.06, -.05, 0, +.08, -.10, 0, 0),
    (-.04, 0, -.08, 0, 0, +.10, -.12),
    (+.04, 0, +.08, 0, 0, -.10, +.12),
    (0, -.06, 0, +.08, -.10, +.08, 0),
    (0, +.06, 0, -.08, +.10, -.08, 0),
    (-.05, 0, +.10, -.06, 0, 0, +.15),
    (+.05, 0, -.10, +.06, 0, 0, -.15),
]


def transform(position, quaternion):
    """位置和 xyzw 四元数组成 T_parent_child，矩阵左乘子坐标。"""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    matrix[:3, 3] = position
    return matrix


def rotation_angle_deg(rotation):
    """旋转矩阵对应的最小转角。"""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def projected_board_outline(corners):
    """由 8×5 内角点估计含一格白边的整块板面外轮廓。"""
    grid = np.asarray(
        [(c, r) for r in range(PATTERN_SIZE[1]) for c in range(PATTERN_SIZE[0])],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(grid, corners.reshape(-1, 2))
    if homography is None or not np.isfinite(homography).all():
        return None
    # 内角点第一列/行距板外缘各两格：一格印刷白边 + 一格棋盘方格。
    outer = np.asarray([[
        [-2, -2],
        [PATTERN_SIZE[0] + 1, -2],
        [PATTERN_SIZE[0] + 1, PATTERN_SIZE[1] + 1],
        [-2, PATTERN_SIZE[1] + 1],
    ]], dtype=np.float32)
    outline = cv2.perspectiveTransform(outer, homography)[0]
    return outline if np.isfinite(outline).all() else None


def pose_statistics(samples):
    """用基座坐标系中的平移跨度和相对旋转向量检查数据多样性。"""
    if not samples:
        return {'sufficient_diversity': False, 'reason': '无有效样本'}

    matrices = np.asarray([sample['T_base_hand'] for sample in samples])
    positions = matrices[:, :3, 3]
    position_span = np.ptp(positions, axis=0)
    reference_rotation = matrices[0, :3, :3]
    rotvec_degrees = np.asarray([
        cv2.Rodrigues(matrix[:3, :3] @ reference_rotation.T)[0].ravel()
        * (180.0 / np.pi)
        for matrix in matrices
    ])
    rotation_span = np.ptp(rotvec_degrees, axis=0)
    max_rotation = max(
        rotation_angle_deg(matrix[:3, :3] @ reference_rotation.T)
        for matrix in matrices
    )
    enough = (
        len(samples) >= 15
        and np.max(position_span) >= 0.06
        and max_rotation >= 10.0
        and np.count_nonzero(rotation_span >= 5.0) >= 2
    )
    return {
        'position_min_base_m': positions.min(axis=0).tolist(),
        'position_max_base_m': positions.max(axis=0).tolist(),
        'position_span_base_m': position_span.tolist(),
        'rotation_vector_span_base_deg': rotation_span.tolist(),
        'max_rotation_from_first_deg': max_rotation,
        'sufficient_diversity': bool(enough),
        'criteria': '至少15组；一个平移轴跨度≥0.06m；最大相对旋转≥10°；至少两个旋转轴跨度≥5°',
    }


def collect(output_dir, target_samples, max_attempts):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'corners').mkdir(exist_ok=True)
    (output_dir / 'failed').mkdir(exist_ok=True)
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError('PyBullet DIRECT 连接失败')
    try:
        robot, _ = create_scene([0.48, -0.12], gui=False)
        board = create_chessboard()
        hand_link = attach_to_hand(robot, board)
        camera = SimRGBDCamera()

        # 原演示程序以 URDF Link 坐标系保存的安装关系。
        mount_source = DEMO_OUTPUT_DIR / 'board_mount.npz'
        mount_copy = output_dir / 'board_mount.npz'
        shutil.copy2(mount_source, mount_copy)
        with np.load(mount_copy) as mount:
            T_hand_board = transform(
                mount['hand_to_board_position'],
                mount['hand_to_board_quaternion'],
            )

        arm = [next(
            j for j in range(p.getNumJoints(robot))
            if p.getJointInfo(robot, j)[1].decode() == f'panda_joint{i}'
        ) for i in range(1, 8)]
        joint_limits = [(p.getJointInfo(robot, j)[8], p.getJointInfo(robot, j)[9])
                        for j in arm]
        base_pos, base_quat = p.getBasePositionAndOrientation(robot)
        T_world_base_inv = np.linalg.inv(transform(base_pos, base_quat))
        samples, failures = [], []

        for attempt, offsets in enumerate(POSE_OFFSETS[:max_attempts], start=1):
            if len(samples) >= target_samples:
                break
            target = HOME + np.asarray(offsets)
            failure = {'attempt': attempt, 'joint_target_rad': target.tolist()}
            if any(not (lower <= value <= upper)
                   for value, (lower, upper) in zip(target, joint_limits)):
                failure['reason'] = '目标关节角超出 URDF 限位'
                failures.append(failure)
                continue

            p.setJointMotorControlArray(
                robot, arm, p.POSITION_CONTROL,
                targetPositions=target.tolist(), forces=ARM_FORCES,
            )
            for _ in range(SETTLE_STEPS):
                p.stepSimulation()

            states = [p.getJointState(robot, j) for j in arm]
            actual = np.asarray([state[0] for state in states])
            velocity = np.asarray([state[1] for state in states])
            failure['joint_actual_rad'] = actual.tolist()
            if (np.max(np.abs(actual - target)) > JOINT_TOLERANCE_RAD
                    or np.max(np.abs(velocity)) > JOINT_VELOCITY_TOLERANCE):
                failure['reason'] = '关节未达到稳定目标姿态'
                failures.append(failure)
                continue

            # [4:6] 是 panda_hand 的 URDF Link 位姿；[0:2] 是质心，不能混用。
            link_state = p.getLinkState(robot, hand_link, computeForwardKinematics=True)
            T_base_hand = T_world_base_inv @ transform(link_state[4], link_state[5])
            board_pos, board_quat = p.getBasePositionAndOrientation(board)
            T_base_board = T_world_base_inv @ transform(board_pos, board_quat)
            mount_error = np.linalg.inv(T_base_hand) @ T_base_board
            mount_position_error = np.linalg.norm(
                mount_error[:3, 3] - T_hand_board[:3, 3]
            )
            mount_rotation_error = rotation_angle_deg(
                mount_error[:3, :3] @ T_hand_board[:3, :3].T
            )
            if (not np.isfinite(T_base_hand).all()
                    or mount_position_error > 0.002
                    or mount_rotation_error > 1.0):
                failure['reason'] = '手部位姿无效或标定板固定关系漂移'
                failures.append(failure)
                continue

            image, _ = camera.capture()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE)
            outline = None
            if not found or corners is None or len(corners) != 40:
                failure['reason'] = '棋盘格内角点未完整检测到 40 个'
            elif not np.isfinite(corners).all():
                failure['reason'] = '角点含非有限坐标'
            elif (np.min(corners[:, 0, 0]) < CORNER_MARGIN_PX
                  or np.max(corners[:, 0, 0]) > image.shape[1] - CORNER_MARGIN_PX
                  or np.min(corners[:, 0, 1]) < CORNER_MARGIN_PX
                  or np.max(corners[:, 0, 1]) > image.shape[0] - CORNER_MARGIN_PX):
                failure['reason'] = '棋盘格角点过于接近图像边界'
            else:
                outline = projected_board_outline(corners)
                if outline is None:
                    failure['reason'] = '无法估计完整标定板外轮廓'
                elif (np.min(outline[:, 0]) < BOARD_MARGIN_PX
                      or np.max(outline[:, 0]) > image.shape[1] - BOARD_MARGIN_PX
                      or np.min(outline[:, 1]) < BOARD_MARGIN_PX
                      or np.max(outline[:, 1]) > image.shape[0] - BOARD_MARGIN_PX):
                    failure['reason'] = '标定板外轮廓越出安全视野'
            if 'reason' in failure:
                failed_path = output_dir / 'failed' / f'attempt_{attempt:02d}.png'
                cv2.imwrite(str(failed_path), image)
                failure['diagnostic_image'] = str(failed_path.relative_to(output_dir))
                failures.append(failure)
                print(f'尝试 {attempt:02d}: 跳过，{failure["reason"]}', flush=True)
                continue

            sample_id = len(samples) + 1
            image_path = output_dir / 'images' / f'chessboard_{sample_id:03d}.png'
            corners_path = output_dir / 'corners' / f'corners_{sample_id:03d}.png'
            marked = image.copy()
            cv2.drawChessboardCorners(marked, PATTERN_SIZE, corners, True)
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f'无法保存 {image_path}')
            if not cv2.imwrite(str(corners_path), marked):
                raise RuntimeError(f'无法保存 {corners_path}')
            samples.append({
                'sample_id': sample_id,
                'attempt': attempt,
                'image': str(image_path.relative_to(output_dir)),
                'corners_image': str(corners_path.relative_to(output_dir)),
                'corners_px': corners.reshape(-1, 2).tolist(),
                'board_outline_px': outline.tolist(),
                'corner_count': len(corners),
                'joint_target_rad': target.tolist(),
                'joint_actual_rad': actual.tolist(),
                'T_base_hand': T_base_hand.tolist(),
            })
            print(f'尝试 {attempt:02d}: 收录样本 {sample_id:02d}，40 个角点', flush=True)

        summary = pose_statistics(samples)
        dataset = {
            'description': 'Panda Eye-to-Hand 多姿态采集；不含相机真值外参或标定求解结果',
            'coordinate_convention': 'T_A_B 将 B 坐标表示的点左乘变换到 A 坐标；base 是机器人基座，hand 是 panda_hand URDF Link',
            'camera': {
                'frame_id': FRAME_ID,
                'width': camera.info.width,
                'height': camera.info.height,
                'K': np.asarray(camera.info.k).reshape(3, 3).tolist(),
                'D': list(camera.info.d),
                'distortion_model': camera.info.distortion_model,
            },
            'chessboard': {
                'squares_cols_rows': [COLS, ROWS],
                'inner_corners_cols_rows': list(PATTERN_SIZE),
                'square_size_m': SQUARE_SIZE,
                'T_hand_board': T_hand_board.tolist(),
                'mount_file': str(mount_copy.relative_to(output_dir)),
            },
            'samples': samples,
            'failures': failures,
            'summary': {
                'attempted': len(samples) + len(failures),
                'valid': len(samples),
                'failed': len(failures),
                'target': target_samples,
                **summary,
            },
        }
        dataset_path = output_dir / 'dataset.json'
        dataset_path.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f'数据集: {dataset_path}', flush=True)
        print(f'有效 {len(samples)}/{target_samples}，失败 {len(failures)}', flush=True)
        print(f'位置跨度 (m): {np.round(summary.get("position_span_base_m", []), 4)}', flush=True)
        print(f'旋转向量跨度 (deg): {np.round(summary.get("rotation_vector_span_base_deg", []), 2)}', flush=True)
        print(f'最大相对旋转 (deg): {summary.get("max_rotation_from_first_deg", 0):.2f}', flush=True)
        print(f'姿态多样性达标: {summary["sufficient_diversity"]}', flush=True)
        return dataset
    finally:
        if p.isConnected(client):
            p.disconnect(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-samples', type=int, default=18)
    parser.add_argument('--max-attempts', type=int, default=len(POSE_OFFSETS))
    parser.add_argument('--output-dir', type=Path,
                        default=Path.cwd() / 'outputs' / 'calibration_collect')
    args = parser.parse_args()
    if not 1 <= args.target_samples <= 20:
        parser.error('--target-samples 必须在 1～20 之间')
    if not 1 <= args.max_attempts <= len(POSE_OFFSETS):
        parser.error(f'--max-attempts 必须在 1～{len(POSE_OFFSETS)} 之间')
    result = collect(args.output_dir, args.target_samples, args.max_attempts)
    if len(result['samples']) < args.target_samples:
        raise SystemExit('有效样本未达到目标数量，请查看 dataset.json 中的失败原因')


if __name__ == '__main__':
    main()
