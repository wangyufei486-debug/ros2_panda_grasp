# Panda 机械臂视觉引导抓取

基于 **Ubuntu 24.04、ROS 2 Jazzy、PyBullet、YOLOv8 实例分割和 RGB-D** 的 Panda 机械臂仿真抓取项目。仿真相机发布彩色图、深度图和内参；感知节点从真实渲染的像素与深度估计方块中心，通过 TF2 得到世界坐标；机械臂使用该视觉坐标执行抓取、搬运和放置。项目还提供 ROS 2 Action、Service，以及用于验证跨语言通信的 C++ 节点。

> **运行范围**：当前代码控制的是 PyBullet 中的 Panda，不连接真实机械臂。方块的 PyBullet 真值仅用于抓取时记录定位误差，不作为视觉节点的目标输入。默认训练权重和数据集不随源码提交。

## 目录

- [功能与数据流](#功能与数据流)
- [环境要求](#环境要求)
- [准备权重与构建](#准备权重与构建)
- [启动完整系统](#启动完整系统)
- [执行一次抓取](#执行一次抓取)
- [独立运行节点与 C++ 验证](#独立运行节点与-c-验证)
- [工作区结构](#工作区结构)
- [ROS 2 接口](#ros-2-接口)
- [坐标系与 TF2](#坐标系与-tf2)
- [参数与模型路径](#参数与模型路径)
- [验证与预期结果](#验证与预期结果)
- [常见问题](#常见问题)
- [开发与发布说明](#开发与发布说明)

## 功能与数据流

1. `sim_camera` 创建 Panda、桌面、红色方块和虚拟 RGB-D 相机。定时器推进物理仿真，采集彩色图与深度图，使用同一个时间戳发布三条相机消息。
2. `yolo_cube_detector` 按完全相同的时间戳配对 RGB、Depth、CameraInfo。YOLOv8 分割模型识别类别 `0` 的方块，并在候选目标中选择置信度最高的实例。
3. `rgbd_geometry.py` 用实例掩膜、有效深度和相机内参计算三维点；过滤桌面区域，估计方块顶面，再换算方块中心。无有效深度时不发布目标坐标。
4. 感知节点先发布相机光学坐标系下的目标，再经 TF2 转换到 `world`，发布供抓取使用的世界坐标。
5. `sim_camera` 接收新近目标。收到 Service 请求或 Action Goal 后，复用 `PandaGraspController` 执行靠近、下降、闭合夹爪、抬升、搬运、放置和撤离。

```text
PyBullet Panda + RGB-D 相机             panda_grasp_ros/sim_camera
       │ RGB、Depth、CameraInfo                         │ /tf_static
       ▼                                               ▼
YOLOv8 分割 → RGB-D 三维定位 → 相机坐标 → TF2 → world 坐标
                 panda_grasp_perception/yolo_cube_detector
                                      │
                         /grasp/target_point
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
       Panda 抓取 Service / Action              C++ 世界坐标接收节点
```

默认 Launch 启动两个 Python ROS 2 节点；两个 C++ 节点是可选的独立订阅程序，**不会被默认 Launch 自动启动**。

## 环境要求

| 项目 | 要求或来源 |
|---|---|
| 操作系统 | Ubuntu 24.04；本项目在该环境中验证。 |
| ROS 2 | Jazzy，包含 `rclpy`、消息包、TF2、`cv_bridge` 和 Launch。包级依赖见各 `package.xml`。 |
| 构建工具 | `colcon`、`ament_python`、`ament_cmake`、Python 虚拟环境。 |
| Python 视觉/仿真依赖 | `src/panda_grasp_ros/requirements-vision.txt` 中的 NumPy、OpenCV、Ultralytics、PyBullet；YOLO 推理还需要可用的 PyTorch。 |
| 模型权重 | 与当前方块实例分割任务兼容的 YOLOv8 权重；代码选择类别 ID `0`。 |
| 图形界面 | 可选。默认 `gui=false`，使用 PyBullet DIRECT 和 Tiny Renderer。 |

若尚未安装 ROS 2 Jazzy，先按 ROS 2 官方安装说明配置系统。ROS 依赖可在工作区中按 `package.xml` 使用 `rosdep` 安装；Python 视觉依赖仍需在虚拟环境中安装。**不要把系统 Python 与虚拟环境的构建脚本混用**：已遇到过系统 `colcon` 生成 `/usr/bin/python3` 启动脚本，运行时找不到虚拟环境内的 PyBullet 和 Ultralytics。

## 准备权重与构建

下列步骤假定仓库位于 `~/panda_grasp_ws`。这是当前 YOLO 节点默认权重路径所依赖的工作区位置。

```bash
source /opt/ros/jazzy/setup.bash
cd ~/panda_grasp_ws

# 首次使用时创建虚拟环境；已有 .venv 时跳过这一行。
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 安装 Python 视觉与仿真依赖。
python -m pip install -r src/panda_grasp_ros/requirements-vision.txt

# 首次使用时放入自己的 YOLOv8 分割权重。
mkdir -p models
# 预期文件：~/panda_grasp_ws/models/cube_best.pt

# 从虚拟环境的 Python 启动 colcon，使 console_scripts 使用同一解释器。
python -m colcon build --symlink-install
source install/setup.bash
```

需要安装 ROS 包依赖时，可在已配置 `rosdep` 的系统上，从工作区根目录运行：

```bash
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
```

安装 PyTorch 时应选用与你的 CPU/GPU 环境匹配的发行包。已有虚拟环境的机器无需重复创建环境或改动现有依赖版本。构建结束后可以检查导入和可执行入口：

```bash
python -c 'import rclpy, cv_bridge, torch, ultralytics, pybullet, cv2, numpy; print("imports OK")'
ros2 pkg executables panda_grasp_ros
ros2 pkg executables panda_grasp_perception
```

预期前者包含 `sim_camera`，后者包含 `yolo_detector` 和保留的 `target_demo`。`panda_grasp_ros` 不应再有旧的 `yolo_detector` 入口。

## 启动完整系统

每个新终端都先载入 ROS、虚拟环境与本工作区：

```bash
source /opt/ros/jazzy/setup.bash
cd ~/panda_grasp_ws
source .venv/bin/activate
source install/setup.bash
```

之后使用原有的一键启动命令：

```bash
ros2 launch panda_grasp_ros panda_grasp.launch.py
```

默认 Launch 同时启动：

| 功能包 / executable | ROS 节点名 | 职责 |
|---|---|---|
| `panda_grasp_ros / sim_camera` | `/sim_camera` | 创建仿真、发布 RGB-D 与静态 TF、接收世界坐标、提供抓取 Service 和 Action。 |
| `panda_grasp_perception / yolo_detector` | `/yolo_cube_detector` | 模型加载、目标分割、RGB-D 定位、TF2 转换、发布结果。 |

Launch 参数只传给仿真节点，名称与默认值如下：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `gui` | `false` | 是否打开 PyBullet GUI；无显示服务时保持 `false`。 |
| `cube_x` | `0.48` | 方块初始世界坐标 X，单位米。 |
| `cube_y` | `-0.12` | 方块初始世界坐标 Y，单位米。 |
| `camera_hz` | `2.0` | RGB-D 发布频率，单位 Hz；必须大于 0。 |

示例：

```bash
ros2 launch panda_grasp_ros panda_grasp.launch.py gui:=false cube_x:=0.48 cube_y:=-0.12 camera_hz:=2.0
```

两个节点由 Launch 一起启动，并没有额外的启动延时门控。启动初期 TF Listener 还未收到 `/tf_static` 时，YOLO 可能短暂出现 `TF2 transform failed` 告警；随后应看到持续的 `TF2 world XYZ` 日志。

## 执行一次抓取

先确认感知节点在发布目标：

```bash
ros2 topic echo /grasp/target_point --once
```

默认场景的目标应为 `frame_id: world`，坐标大致接近 `x=0.48, y=-0.12, z=0.325` 米。该数值仅是默认场景的核对参考，并未硬编码为视觉输出。然后在另一个已载入环境的终端选择以下任一入口。

**Service：**

```bash
ros2 service call /grasp/execute std_srvs/srv/Trigger '{}'
```

成功时，响应的 `success` 为 `true`，`message` 给出最终放置 XY 误差。

**Action：**

```bash
ros2 action send_goal /grasp/task panda_grasp_interfaces/action/ExecuteGrasp '{start: true}' --feedback
```

`ExecuteGrasp.action` 的 Goal 为 `bool start`，Result 为 `bool success` 和 `string message`，Feedback 为 `string stage` 与 `float32 progress`。Action 复用同一抓取执行路径，并在准备、预抓取、下降、接触、抬升、搬运、放置及任务结束时发布进度。当前实现拒绝中途取消。

服务和 Action 都要求最近 **10 秒内**收到 `world` 坐标目标，且每次仿真只允许一次抓取尝试。Service 与 Action 不是两个连续步骤；测试第二次抓取前需要重启 `sim_camera`。当前工作空间校验范围为 `0.27 ≤ x ≤ 0.69` 米、`-0.22 ≤ y ≤ 0.22` 米，目标高度需接近桌面高度加方块半边长。

## 独立运行节点与 C++ 验证

若需要分别启动两个 Python 节点，例如使用非默认权重路径，可在两个已载入环境的终端运行：

```bash
# 终端 A
ros2 run panda_grasp_ros sim_camera
```

```bash
# 终端 B；weights 必须是本机已有的分割模型文件。
ros2 run panda_grasp_perception yolo_detector --ros-args -p weights:=/absolute/path/to/cube_best.pt
```

默认 Launch **没有**将 `weights` 暴露为 Launch 参数。想保留一键启动命令时，请把权重放在默认路径；若使用其他路径，可按上面的方式分别启动节点。

可在额外终端运行两个 C++ 程序，验证 Python/C++ 跨节点通信：

```bash
ros2 run panda_grasp_cpp target_receiver_node
```

```bash
ros2 run panda_grasp_cpp tf_target_node
```

`target_receiver_node` 订阅 `/grasp/target_point` 并输出世界坐标；`tf_target_node` 订阅 `/grasp/target_point_camera`，使用自己的 TF Buffer/Listener 转换后输出世界坐标。两个程序都不参与抓取控制，也不发布静态 TF。

如果安装了 `rqt_image_view` 且有图形显示服务，可运行 `ros2 run rqt_image_view rqt_image_view`，查看 `/camera/color/image_raw`、`/yolo/annotated_image` 或 `/yolo/cube_mask`。

## 工作区结构

以下列出源码和主要配置文件。模型权重由使用者放入 `models/`；`build/`、`install/`、`log/`、`.venv/` 为本地产物，已在 `.gitignore` 中排除。

```text
panda_grasp_ws/
├── README.md
├── .gitignore
├── models/
│   └── cube_best.pt                     # 自备模型权重，不提交到 Git
└── src/
    ├── panda_grasp_ros/
    │   ├── launch/
    │   │   └── panda_grasp.launch.py     # 原有一键启动入口
    │   ├── panda_grasp_ros/
    │   │   ├── __init__.py
    │   │   ├── sim_camera_node.py        # ROS 主节点与抓取接口
    │   │   ├── pybullet_scene.py        # 场景、机器人、方块
    │   │   ├── sim_rgbd_camera.py       # 相机、深度及消息封装
    │   │   ├── grasp_controller.py      # IK、轨迹、夹爪、放置
    │   │   └── tf_utils.py              # 静态 TF 消息构造
    │   ├── resource/panda_grasp_ros
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   └── requirements-vision.txt
    ├── panda_grasp_perception/
    │   ├── panda_grasp_perception/
    │   │   ├── __init__.py
    │   │   ├── yolo_node.py             # YOLO 节点、消息配对、TF2
    │   │   ├── rgbd_geometry.py        # 不依赖 ROS 节点的三维定位算法
    │   │   └── target_demo.py           # 保留的示例入口
    │   ├── resource/panda_grasp_perception
    │   ├── test/                        # 包模板自带的基础检查
    │   ├── package.xml
    │   ├── setup.py
    │   └── setup.cfg
    ├── panda_grasp_interfaces/
    │   ├── action/ExecuteGrasp.action
    │   ├── LICENSE
    │   ├── CMakeLists.txt
    │   └── package.xml
    └── panda_grasp_cpp/
        ├── src/
        │   ├── target_receiver_node.cpp
        │   └── tf_target_node.cpp
        ├── CMakeLists.txt
        └── package.xml
```

`panda_grasp_interfaces` 目前定义一个自定义 Action；抓取 Service 使用 ROS 标准接口 `std_srvs/Trigger`。`target_demo.py` 保留供示例使用，但默认系统不启动它。

## ROS 2 接口

### Topic

| Topic | 消息类型 | 发布者 → 订阅者 | 当前 QoS 队列深度 |
|---|---|---|---:|
| `/camera/color/image_raw` | `sensor_msgs/Image`，`bgr8` | `sim_camera` → `yolo_cube_detector` | 发布 1、订阅 5 |
| `/camera/depth/image_raw` | `sensor_msgs/Image`，`32FC1` 米 | `sim_camera` → `yolo_cube_detector` | 发布 1、订阅 5 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | `sim_camera` → `yolo_cube_detector` | 发布 1、订阅 5 |
| `/yolo/annotated_image` | `sensor_msgs/Image`，`bgr8` | `yolo_cube_detector` → 可视化工具 | 发布 1 |
| `/yolo/cube_mask` | `sensor_msgs/Image`，`mono8` | `yolo_cube_detector` → 可视化工具 | 发布 1 |
| `/grasp/target_point_camera` | `geometry_msgs/PointStamped` | `yolo_cube_detector` → `tf_target_node` | 发布 5、C++ 订阅 10 |
| `/grasp/target_point` | `geometry_msgs/PointStamped` | `yolo_cube_detector` → `sim_camera`、`target_receiver_node` | 发布 5、订阅 10 |
| `/tf_static` | `tf2_msgs/TFMessage` | `sim_camera` 的静态 TF Broadcaster → TF Listeners | TF2 管理 |

### Service 与 Action

| 名称 | 类型 | 提供者 | 用途 |
|---|---|---|---|
| `/grasp/execute` | `std_srvs/srv/Trigger` | `sim_camera` | 发起一次抓取并返回成功状态及放置误差。 |
| `/grasp/task` | `panda_grasp_interfaces/action/ExecuteGrasp` | `sim_camera` | 发起同一抓取流程，接收阶段反馈与结果。 |

Topic 与接口名称、消息格式由源码固定；当前没有额外的参数 YAML 文件。

## 坐标系与 TF2

```text
world ──恒等静态变换──► panda_link0 ──静态相机位姿──► camera_optical_frame
```

- `world → panda_link0`：仿真场景中 Panda 基座与世界原点重合，位移 `(0, 0, 0)`，单位四元数 `(0, 0, 0, 1)`。
- `panda_link0 → camera_optical_frame`：平移来自 `sim_rgbd_camera.py` 的 `CAM_EYE = [0.48, 0.0, 1.12]` 米；当前顶置相机的旋转四元数为 `(1, 0, 0, 0)`，即绕 X 轴 180°。
- `sim_camera_node.py` **只创建一个** `StaticTransformBroadcaster` 并发布上述关系。`tf_utils.py` 只构造 `TransformStamped` 消息，不创建广播器或新 ROS 节点。
- `yolo_node.py` 创建一个 TF Buffer 和 Listener，先发布相机坐标，再转换并发布 `world` 坐标。C++ 的 `tf_target_node` 有自己的 Buffer/Listener，只用于独立验证。

相机光学坐标约定为 X 向右、Y 向下、Z 向前；深度单位是米。YOLO 节点中的相机姿态参数用于 RGB-D 三维定位，需要与仿真相机的参数保持一致。TF2 转换失败时仍可发布相机坐标，世界目标则等待后续有效变换。

## 参数与模型路径

YOLO 节点声明的参数如下。默认 Launch 不覆盖这些值；如需改动，请分别启动节点并通过 `--ros-args -p` 传入。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `weights` | `~/panda_grasp_ws/models/cube_best.pt` | YOLOv8 分割权重；启动时检查文件存在。 |
| `confidence` | `0.20` | 推理置信度阈值。 |
| `camera_eye` | `[0.48, 0.0, 1.12]` | 相机在仿真世界中的观察位置，米。 |
| `camera_target` | `[0.48, 0.0, 0.30]` | 相机观察目标点，米。 |
| `camera_up` | `[0.0, 1.0, 0.0]` | 相机上方向。 |
| `table_z` | `0.30` | 桌面世界坐标高度，米。 |
| `cube_half` | `0.025` | 方块半边长，米。 |

仿真相机的默认分辨率为 `640 × 480`，视场角 `60°`，近/远裁剪面为 `0.05/2.0` 米。模型推理使用 `imgsz=640` 和 `retina_masks=True`；这些是当前源码设置，不属于 Launch 参数。项目不包含训练脚本或数据集，推理结果取决于提供的权重。

## 验证与预期结果

从构建、启动到功能检查，可依次使用以下命令：

```bash
# 在工作区根目录，已 source ROS 和 .venv。
python -m colcon build --symlink-install
source install/setup.bash
ros2 pkg executables panda_grasp_ros
ros2 pkg executables panda_grasp_perception

# 在另一终端运行原 Launch 后检查数据。
ros2 topic echo /grasp/target_point_camera --once
ros2 topic echo /grasp/target_point --once
ros2 service list
ros2 action list
```

核对点：

- Launch 日志应显示 `sim_camera` 和 `yolo_detector` 两个进程启动、静态 TF 发布、YOLO 权重加载。
- `/grasp/target_point_camera` 的 `frame_id` 应为 `camera_optical_frame`；`/grasp/target_point` 的 `frame_id` 应为 `world`。
- 默认方块位置的世界坐标大致为 `(0.48, -0.12, 0.325)` 米。感知日志会显示 TF2 世界坐标和与 RGB-D 几何计算结果的差值。
- 运行 C++ 节点后，两个节点应分别输出目标世界坐标和经 C++ TF2 转换后的世界坐标。
- Service 或 Action 抓取成功时，方块应被放置在绿色目标区域。Action 应给出阶段反馈并以 `SUCCEEDED` 结束。

此前在当前无 GUI 环境中，四个功能包构建通过，YOLO 推理和 TF2 连续工作，C++ 订阅正常；Service 与 Action 各完成一次仿真抓取，最终放置 XY 误差均为 `0.0108 m`。拆分 `tf_utils.py` 后再次确认静态变换消息与拆分前逐字段相同，并用原 Launch 验证 TF2 世界坐标持续发布。PyBullet GUI 和 RQT 图像显示尚未在该环境中验证。

## 常见问题

| 现象 | 先检查什么 |
|---|---|
| `Missing trained YOLO weights` | 确认 `~/panda_grasp_ws/models/cube_best.pt` 存在，且是分割模型；或独立启动 YOLO 节点时显式传入 `weights`。 |
| `ModuleNotFoundError: pybullet` 或 `ultralytics` | 确认已激活 `.venv`，依赖已安装，并使用 `python -m colcon build --symlink-install` 重新生成节点脚本。 |
| 启动初期 `TF2 transform failed` | 等待静态 TF 到达；若一直失败，检查 `sim_camera` 是否启动，以及 `/tf_static` 和坐标系名称。 |
| 一直没有 `/grasp/target_point` | 先看相机三条 Topic 和 YOLO 日志；检查权重的类别 `0`、分割掩膜、有效深度、相机姿态参数及 TF2。 |
| 抓取提示 `No recent YOLO RGB-D target` | 等待目标发布后在 10 秒内发起请求；确认目标消息使用 `world` 坐标系。 |
| 抓取提示 `Target outside expected cube workspace` | 检查视觉输出、方块初始位置和相机参数。当前抓取工作空间有限。 |
| 抓取提示 `Already attempted` | 当前仿真只允许一次尝试，重启 `sim_camera` 或完整 Launch 后再试。 |
| 看不到 C++ 节点日志 | 它们不在默认 Launch 中，需要额外用 `ros2 run panda_grasp_cpp ...` 启动。 |
| `gui:=true` 无法启动 | 在没有图形显示服务的环境中使用默认 `gui:=false`。 |

## 开发与发布说明

- 感知算法位于 `panda_grasp_perception`，仿真与抓取位于 `panda_grasp_ros`。两个功能包没有循环 Python 导入。
- `panda_grasp_interfaces` 目前只有 `ExecuteGrasp.action`，没有自定义 Message 或 Service；抓取服务使用标准 `std_srvs/Trigger`。
- `.gitignore` 排除了 `.venv/`、`build/`、`install/`、`log/`、模型权重与数据集。克隆仓库后需要自行准备权重并重新构建。
- `target_demo.py` 是保留的示例入口，不参与完整系统。修改感知逻辑时，请同时验证相机坐标、世界坐标和 C++ 订阅；修改抓取逻辑时，请分别回归 Service 与 Action。
- 各包的许可证字段见对应 `package.xml`。目前 `panda_grasp_cpp/package.xml` 的许可证仍为 `TODO`；公开发布前应由项目维护者确认授权信息并补全根目录许可证文件。
