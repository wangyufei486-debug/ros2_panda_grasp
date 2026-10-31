
"""Panda 机械臂物理抓取控制：与相机共用同一个 PyBullet 仿真。"""

import math
import time

import numpy as np
import pybullet as p

DT = 1.0 / 240.0
SLEEP_FACTOR = 0.25

TABLE_Z = 0.30
CUBE_HALF = 0.025
PLACE_XY = [0.48, 0.12]
TCP_LOCAL = [0.0, 0.0, 0.1034]

DOWN_QUAT = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])


class PandaGraspController:
    def __init__(self, robot, cube, logger):
        self.robot = robot
        self.cube = cube
        self.logger = logger

        by_name = {}
        self.movable = []
        self.hand = None

        for j in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, j)
            by_name[info[1].decode()] = j

            if info[12].decode() == "panda_hand":
                self.hand = j

            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.movable.append(j)

        if self.hand is None:
            raise RuntimeError("panda_hand link not found")

        self.arm = [by_name[f"panda_joint{i}"] for i in range(1, 8)]
        self.fingers = [
            by_name["panda_finger_joint1"],
            by_name["panda_finger_joint2"]
        ]

        infos = [p.getJointInfo(robot, j) for j in self.movable]
        self.lows = [float(i[8]) for i in infos]
        self.highs = [float(i[9]) for i in infos]
        self.ranges = [
            hi - lo for lo, hi in zip(self.lows, self.highs)
        ]

        for j in self.fingers:
            p.changeDynamics(
                robot, j,
                lateralFriction=3.0,
                spinningFriction=0.01
            )

    def log_visual_target_error(self, xyz):
        """只用方块仿真真值记录视觉误差，不参与抓取控制。"""
        ground_truth, _ = p.getBasePositionAndOrientation(self.cube)
        visual_xyz = np.array(xyz)
        ground_truth_xyz = np.array(ground_truth)
        visual_error = np.linalg.norm(visual_xyz - ground_truth_xyz)
        self.logger.info(f'视觉定位误差：{visual_error * 1000:.2f} mm')
        self.logger.info(f'视觉估计坐标：{np.round(visual_xyz, 4)}')
        self.logger.info(f'仿真真实坐标：{np.round(ground_truth_xyz, 4)}')

    def step(self, count):
        """执行物理仿真步进。"""
        for _ in range(count):
            p.stepSimulation()
            time.sleep(DT * SLEEP_FACTOR)

    def grip(self, width):
        """控制两个手指的开合。"""
        for j in self.fingers:
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL,
                targetPosition=width,
                force=80,
                positionGain=0.15,
                velocityGain=1.0
            )

    def current_tcp(self):
        """读取末端 TCP 的实际位置。"""
        pose = p.getLinkState(
            self.robot, self.hand,
            computeForwardKinematics=True
        )
        pos, _ = p.multiplyTransforms(
            pose[4], pose[5],
            TCP_LOCAL, [0, 0, 0, 1]
        )
        return np.asarray(pos, dtype=float)

    def move_tcp(self, xyz, stage, seconds=2.0, tolerance=0.04):
        """IK 逆运动学 + 五次多项式关节轨迹。"""
        xyz = np.asarray(xyz, dtype=float)

        offset, _ = p.multiplyTransforms(
            [0, 0, 0], DOWN_QUAT,
            TCP_LOCAL, [0, 0, 0, 1]
        )
        hand_goal = (xyz - np.asarray(offset)).tolist()

        current = [
            p.getJointState(self.robot, j)[0]
            for j in self.movable
        ]

        solution = p.calculateInverseKinematics(
            self.robot, self.hand, hand_goal,
            targetOrientation=DOWN_QUAT,
            lowerLimits=self.lows,
            upperLimits=self.highs,
            jointRanges=self.ranges,
            restPoses=current,
            maxNumIterations=250,
            residualThreshold=1e-4
        )

        if len(solution) < 7:
            raise RuntimeError(f"{stage}: IK 求解失败")

        q_start = np.array([
            p.getJointState(self.robot, j)[0]
            for j in self.arm
        ])
        q_goal = np.array(solution[:7], dtype=float)

        for k, j in enumerate(self.arm):
            info = p.getJointInfo(self.robot, j)
            q_goal[k] = np.clip(q_goal[k], info[8], info[9])

        steps = max(1, int(seconds / DT))

        for n in range(steps):
            t = (n + 1) / steps
            blend = 10*t**3 - 15*t**4 + 6*t**5
            q_ref = q_start + blend * (q_goal - q_start)

            p.setJointMotorControlArray(
                self.robot, self.arm, p.POSITION_CONTROL,
                targetPositions=q_ref.tolist(),
                forces=[87, 87, 87, 87, 20, 20, 20],
                positionGains=[0.15] * 7,
                velocityGains=[1.0] * 7
            )

            p.stepSimulation()
            time.sleep(DT * SLEEP_FACTOR)

        self.step(70)

        error = float(np.linalg.norm(self.current_tcp() - xyz))
        self.logger.info(f"{stage}: TCP 误差 = {error:.4f} m")

        if error > tolerance:
            raise RuntimeError(
                f"{stage}: TCP 误差 {error:.3f} m 超出允许范围"
            )

    def execute(self, center_xyz, feedback_cb=None):

        def report(stage, progress):
            if feedback_cb is not None:
                feedback_cb(stage, float(progress))
        """使用视觉目标坐标执行一次完整物理抓取。"""
        x, y, z = map(float, center_xyz)

        approach = [x, y, z + 0.19]
        grasp = [x, y, z]
        lift = [x, y, z + 0.19]
        travel = [*PLACE_XY, z + 0.19]
        place = [*PLACE_XY, z + 0.012]

        self.logger.info(
            f"开始视觉抓取：({x:.4f}, {y:.4f}, {z:.4f})"
        )

        report("准备抓取", 0.05)

        # 1. 张开夹爪，移动到方块上方
        self.grip(0.04)
        self.step(180)
        self.move_tcp(approach, "预抓取", seconds=2.1)
        report("预抓取完成", 0.20)

        # 2. 下降至方块中心
        self.move_tcp(
            grasp, "下降",
            seconds=1.7, tolerance=0.025
        )
        report("下降完成", 0.35)

        # 3. 闭合夹爪，验证物理接触
        self.grip(0.0)
        self.step(360)

        contacts = [
            p.getContactPoints(
                bodyA=self.robot,
                bodyB=self.cube,
                linkIndexA=j
            )
            for j in self.fingers
        ]

        self.logger.info(
            f"左右手指接触点数量：{[len(c) for c in contacts]}"
        )

        if not all(contacts):
            raise RuntimeError("双指物理接触未建立，抓取失败")
        report("双指接触已确认", 0.50)

        # 4. 抬升，并验证物块是否随夹爪移动
        self.move_tcp(
            lift, "抬升",
            seconds=2.1, tolerance=0.045
        )

        object_xyz, _ = p.getBasePositionAndOrientation(self.cube)

        if object_xyz[2] <= TABLE_Z + CUBE_HALF + 0.07:
            raise RuntimeError("物块没有随夹爪抬升")
        report("抬升完成", 0.65)

        # 5. 搬运到放置区
        self.move_tcp(travel, "搬运", seconds=2.2)
        report("搬运完成", 0.80)
        self.move_tcp(place, "放置", seconds=1.8)
        report("已到达放置位置", 0.90)

        # 6. 释放方块，机械臂撤离
        self.grip(0.04)
        self.step(240)
        self.move_tcp(travel, "撤离", seconds=1.7)
        self.step(240)

        # 7. 使用仿真真值评估最终放置效果
        final, _ = p.getBasePositionAndOrientation(self.cube)

        xy_error = float(
            np.linalg.norm(np.asarray(final[:2]) - PLACE_XY)
        )

        success = (
            xy_error < 0.065
            and abs(final[2] - TABLE_Z - CUBE_HALF) < 0.035
        )

        message = (
            f"放置 XY 误差：{xy_error:.4f} m，成功：{success}"
        )
        self.logger.info(message)

        report(
            "任务成功" if success else "放置验证失败",
            1.0
        )

        return success, message
