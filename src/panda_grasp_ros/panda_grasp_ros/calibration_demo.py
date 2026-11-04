"""Panda Eye-to-Hand 手眼标定：虚拟棋盘格安装、拍摄和角点检测。"""
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

from panda_grasp_ros.pybullet_scene import create_scene
from panda_grasp_ros.sim_rgbd_camera import SimRGBDCamera

# 9列×6行方格 => 8×5个内角点
COLS, ROWS = 9, 6
SQUARE_SIZE = 0.02  # 米
PATTERN_SIZE = (COLS - 1, ROWS - 1)
# 初始姿态下，相对 panda_hand 质心的世界坐标偏移；随后转为固定的手部局部安装外参。
BOARD_OFFSET = (0.157, -0.05, 0.124)
FIRST_JOINT1_ANGLE = -0.03
SECOND_JOINT1_ANGLE = 0.10
OUTPUT_DIR = Path.cwd() / 'outputs' / 'calibration'


def create_chessboard():
    """用带 UV 的单块网格板显示 PNG 棋盘格。"""
    # 白色边框各占一个方格，所以物理尺寸比9×6方格大两格
    border = 1
    board_w = (COLS + 2 * border) * SQUARE_SIZE
    board_h = (ROWS + 2 * border) * SQUARE_SIZE

    # 图片只用于模拟真实印刷标定板，方格边长在三维世界中仍为0.02m
    pixels_per_square = 80
    tex_h = (ROWS + 2 * border) * pixels_per_square
    tex_w = (COLS + 2 * border) * pixels_per_square
    texture = np.full((tex_h, tex_w, 3), 255, dtype=np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                x1 = (c + border) * pixels_per_square
                y1 = (r + border) * pixels_per_square
                texture[y1:y1 + pixels_per_square,
                        x1:x1 + pixels_per_square] = (0, 0, 0)

    texture_path = OUTPUT_DIR / 'checkerboard_texture.png'
    if not cv2.imwrite(str(texture_path), texture):
        raise RuntimeError('无法保存棋盘格纹理图片')
    print('[STEP 3a] 已生成黑白棋盘格纹理：', texture_path, flush=True)

    # Tiny Renderer 不会把当前 PNG 正确映射到盒体顶面，显式给朝上面定义 UV。
    mesh_path = OUTPUT_DIR / 'checkerboard_board.obj'
    half_w, half_h = board_w / 2, board_h / 2
    mesh_path.write_text(
        f'v {-half_w} {-half_h} 0\n'
        f'v {half_w} {-half_h} 0\n'
        f'v {half_w} {half_h} 0\n'
        f'v {-half_w} {half_h} 0\n'
        'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n'
        'f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n',
        encoding='ascii',
    )
    visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(mesh_path),
        rgbaColor=[1, 1, 1, 1],
    )
    print('[STEP 3b] 已创建单块标定板视觉形状：', visual_id, flush=True)
    if visual_id < 0:
        raise RuntimeError('棋盘格几何体创建失败')

    # 薄盒只提供有限转动惯量，避免固定约束让腕部反向代偿；板不参与碰撞。
    collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[half_w, half_h, 0.002]
    )
    board_id = p.createMultiBody(
        baseMass=0.02,
        baseVisualShapeIndex=visual_id,
        baseCollisionShapeIndex=collision_id,
        basePosition=[0, 0, 1],
    )
    print('[STEP 3c] 已创建标定板刚体：', board_id, flush=True)
    if board_id < 0:
        raise RuntimeError('标定板刚体创建失败')
    p.setCollisionFilterGroupMask(board_id, -1, 0, 0)

    texture_id = p.loadTexture(str(texture_path))
    print('[STEP 3d] 已加载纹理：', texture_id, flush=True)
    if texture_id < 0:
        raise RuntimeError('棋盘格纹理加载失败')
    p.changeVisualShape(board_id, -1, textureUniqueId=texture_id)
    print('[STEP 3e] 已将棋盘格纹理贴到标定板上', flush=True)
    return board_id


def attach_to_hand(robot, board):
    """以固定约束将标定板安装到panda_hand，并记录安装外参。"""
    hand_link = next(
        j for j in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, j)[12].decode() == 'panda_hand'
    )
    state = p.getLinkState(robot, hand_link, computeForwardKinematics=True)

    # 固定相机位于上方；初始时令棋盘格正面朝世界+Z方向
    com_pos, com_orn = state[0], state[1]
    board_pos = [com_pos[i] + BOARD_OFFSET[i] for i in range(3)]
    board_orn = [0, 0, 0, 1]
    p.resetBasePositionAndOrientation(board, board_pos, board_orn)

    # createConstraint的parentFrame定义在机器人Link质心局部坐标系
    inv_pos, inv_orn = p.invertTransform(com_pos, com_orn)
    rel_pos, rel_orn = p.multiplyTransforms(
        inv_pos, inv_orn, board_pos, board_orn
    )
    constraint = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=hand_link,
        childBodyUniqueId=board,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=rel_orn,
        childFrameOrientation=[0, 0, 0, 1],
    )
    p.changeConstraint(constraint, maxForce=200)

    # 后续求解手眼标定时应统一使用URDF hand link，而非COM坐标系
    link_pos, link_orn = state[4], state[5]
    inv_link_pos, inv_link_orn = p.invertTransform(link_pos, link_orn)
    board_in_hand_pos, board_in_hand_orn = p.multiplyTransforms(
        inv_link_pos, inv_link_orn, board_pos, board_orn
    )
    np.savez(
        OUTPUT_DIR / 'board_mount.npz',
        hand_to_board_position=board_in_hand_pos,
        hand_to_board_quaternion=board_in_hand_orn,
        square_size=SQUARE_SIZE,
        pattern_size=PATTERN_SIZE,
    )
    return hand_link


def capture_and_detect(camera, filename):
    rgb_bgr, _ = camera.capture()
    image_path = OUTPUT_DIR / filename
    cv2.imwrite(str(image_path), rgb_bgr)
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE)
    marked = rgb_bgr.copy()
    if found:
        cv2.drawChessboardCorners(marked, PATTERN_SIZE, corners, found)
    marked_path = OUTPUT_DIR / filename.replace('chessboard', 'corners')
    cv2.imwrite(str(marked_path), marked)
    print(f'{filename}: 找到棋盘格 = {found}, 角点数 = {len(corners) if found else 0}')
    print('  原图:', image_path)
    print('  角点图:', marked_path)
    return found


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('[STEP 1] 连接 PyBullet DIRECT', flush=True)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError('PyBullet DIRECT 连接失败')
    try:
        print('[STEP 2] 创建 Panda 和桌面', flush=True)
        robot, _ = create_scene([0.48, -0.12], gui=False)
        print('[STEP 2] 完成', flush=True)
        print('[STEP 3] 创建棋盘格视觉模型', flush=True)
        board = create_chessboard()
        print('[STEP 3] 完成', flush=True)
        print('[STEP 4] 固定棋盘格到末端', flush=True)
        hand_link = attach_to_hand(robot, board)
        print('[STEP 4] 完成', flush=True)
        print('[STEP 5] 创建相机', flush=True)
        camera = SimRGBDCamera()
        print('[STEP 5] 完成', flush=True)

        joint1 = next(
            j for j in range(p.getNumJoints(robot))
            if p.getJointInfo(robot, j)[1].decode() == 'panda_joint1'
        )
        p.setJointMotorControl2(
            robot, joint1, p.POSITION_CONTROL,
            targetPosition=FIRST_JOINT1_ANGLE, force=87
        )
        for _ in range(150):
            p.stepSimulation()
        first_pose, first_orn = p.getBasePositionAndOrientation(board)
        print('[STEP 6] 首次拍摄并检测角点', flush=True)
        capture_and_detect(camera, 'chessboard_001.png')
        print('[STEP 6] 完成', flush=True)

        # 小幅移动第1关节，测试棋盘格是否跟随机器人运动
        p.setJointMotorControl2(
            robot, joint1, p.POSITION_CONTROL,
            targetPosition=SECOND_JOINT1_ANGLE, force=87
        )
        for _ in range(350):
            p.stepSimulation()
        second_pose, second_orn = p.getBasePositionAndOrientation(board)
        print('[STEP 7] 第二次拍摄并检测角点', flush=True)
        capture_and_detect(camera, 'chessboard_002.png')
        print('[STEP 7] 完成', flush=True)

        moved = np.linalg.norm(np.array(second_pose) - np.array(first_pose))
        print(f'棋盘格中心移动距离: {moved:.4f} m')
        print(f'panda_hand Link 索引: {hand_link}')
        print('安装外参: ', OUTPUT_DIR / 'board_mount.npz')
        print('[STEP 8] 实验结束，断开连接', flush=True)
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == '__main__':
    main()
