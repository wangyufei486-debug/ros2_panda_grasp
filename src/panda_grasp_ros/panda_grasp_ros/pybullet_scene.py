
"""PyBullet 仿真场景创建模块。"""

import pybullet as p
import pybullet_data


DT = 1.0 / 240.0
TABLE_Z = 0.30
CUBE_HALF = 0.025
PLACE_XY = [0.48, 0.12]


def create_scene(pick_xy, gui=False):
    """创建 Panda、桌面、方块和放置区域。

    参数:
        pick_xy: 方块初始位置 [x, y]
        gui: 是否显示 PyBullet GUI

    返回:
        robot: Panda 机器人 ID
        cube: 方块 ID
    """

    # 1. 初始化物理仿真参数
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(numSolverIterations=150)

    # 2. 加载地面和 Panda 机器人
    p.loadURDF('plane.urdf')

    robot = p.loadURDF(
        'franka_panda/panda.urdf',
        [0, 0, 0],
        useFixedBase=True
    )

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.35,
            cameraYaw=40,
            cameraPitch=-30,
            cameraTargetPosition=[0.42, 0, 0.35]
        )

    # 3. 创建桌面
    table_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.23, 0.25, TABLE_Z / 2]
    )

    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.23, 0.25, TABLE_Z / 2],
        rgbaColor=[0.7, 0.73, 0.77, 1]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[0.48, 0, TABLE_Z / 2]
    )

    # 4. 创建红色方块
    cube_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[CUBE_HALF] * 3
    )

    cube_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[CUBE_HALF] * 3,
        rgbaColor=[0.93, 0.08, 0.08, 1]
    )

    cube = p.createMultiBody(
        baseMass=0.03,
        baseCollisionShapeIndex=cube_col,
        baseVisualShapeIndex=cube_vis,
        basePosition=[
            pick_xy[0],
            pick_xy[1],
            TABLE_Z + CUBE_HALF
        ]
    )

    p.changeDynamics(
        cube, -1,
        lateralFriction=2.0,
        spinningFriction=0.005
    )

    # 5. 创建绿色放置区域
    goal_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.055, 0.055, 0.002],
        rgbaColor=[0.1, 0.8, 0.2, 0.65]
    )

    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_vis,
        basePosition=[
            *PLACE_XY,
            TABLE_Z + 0.002
        ]
    )

    # 6. 获取 Panda 关节编号
    by_name = {
        p.getJointInfo(robot, j)[1].decode('utf-8'): j
        for j in range(p.getNumJoints(robot))
    }

    arm = [
        by_name[f'panda_joint{i}']
        for i in range(1, 8)
    ]

    home = [
        0., -0.4, 0., -1.8,
        0., 1.6, 0.7
    ]

    # 7. 初始化机械臂关节
    for j, q in zip(arm, home):
        p.resetJointState(robot, j, q)

    p.setJointMotorControlArray(
        robot,
        arm,
        p.POSITION_CONTROL,
        targetPositions=home,
        forces=[87, 87, 87, 87, 20, 20, 20]
    )

    # 8. 初始化夹爪
    for name in [
        'panda_finger_joint1',
        'panda_finger_joint2'
    ]:
        j = by_name[name]

        p.resetJointState(robot, j, 0.04)

        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=80
        )

    # 9. 让初始状态稳定
    for _ in range(180):
        p.stepSimulation()

    return robot, cube
