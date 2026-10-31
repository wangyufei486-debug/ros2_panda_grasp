# Panda ROS 2 YOLOv8 / RGB-D grasp

Current project documentation and directory structure: [`../../README.md`](../../README.md).

A self-contained first step for Ubuntu 24.04 + ROS 2 Jazzy, based on the previous working `007c_yolo_rgbd_grasp.py` scene and its self-trained YOLOv8-seg weights. The camera publishes **real rendered RGB-D images**, not mock coordinates. YOLO subscribes to `/camera/color/image_raw`, publishes `/yolo/annotated_image` and `/yolo/cube_mask`.

## Prepare Ubuntu

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-colcon-common-extensions ros-jazzy-cv-bridge ros-jazzy-rqt-image-view
mkdir -p ~/panda_grasp_ws/src ~/panda_grasp_ws/models
# Extract archive contents into ~/panda_grasp_ws/src/ so `panda_grasp_ros/package.xml` exists.
# Copy your trained cube best.pt to ~/panda_grasp_ws/models/cube_best.pt
source /opt/ros/jazzy/setup.bash
cd ~/panda_grasp_ws
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# VMware generally has no usable NVIDIA CUDA device; choose CPU PyTorch wheels.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r src/panda_grasp_ros/requirements-vision.txt
python -c 'import rclpy, cv_bridge, torch, ultralytics, pybullet, cv2, numpy; print("Imports OK")'
python -m colcon build --symlink-install
source install/setup.bash
```

## Terminal 1: rendered PyBullet camera

```bash
source /opt/ros/jazzy/setup.bash
cd ~/panda_grasp_ws
source .venv/bin/activate
source install/setup.bash
ros2 run panda_grasp_ros sim_camera
# Optional GUI if VMware supports it:
# ros2 run panda_grasp_ros sim_camera --ros-args -p gui:=true
```

## Terminal 2: trained YOLOv8 cube segmentation

```bash
source /opt/ros/jazzy/setup.bash
cd ~/panda_grasp_ws
source .venv/bin/activate
source install/setup.bash
ros2 run panda_grasp_perception yolo_detector
```

## Terminal 3: annotated images

```bash
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view
```

Select `/yolo/annotated_image` to view segmentation, or `/camera/color/image_raw` to view the original camera image. Other topics:

- `/camera/depth/image_raw` (`32FC1`, meters)
- `/camera/color/camera_info` (camera intrinsics)
- `/yolo/cube_mask` (`mono8`)

The full system, including the visual grasp controller, starts with the original command:

```bash
ros2 launch panda_grasp_ros panda_grasp.launch.py
```

## Stage 2: real RGB-D XYZ localization (v2)

The `yolo_detector` node now synchronizes three messages with identical stamps:
`/camera/color/image_raw`, `/camera/depth/image_raw` (`32FC1` meters), and
`/camera/color/camera_info`. It converts the YOLOv8 instance mask and valid
depths into a 3-D point cloud, rejects tabletop spillover, estimates the cube
top center, and subtracts `cube_half` from the top-center world Z coordinate.

Outputs:

- `/grasp/target_point` (`geometry_msgs/PointStamped`): cube **center** in
  **PyBullet world frame**, which coincides with Panda URDF base for this scene.
- `/grasp/target_point_camera` (`geometry_msgs/PointStamped`): same point in
  `camera_optical_frame` (`x` right, `y` down, `z` forward).
- `/yolo/annotated_image`: visual overlay also displays XYZ(world) in meters.
- `/yolo/cube_mask`: original instance mask.

The YOLO node publishes camera coordinates and uses TF2 to publish world
coordinates. The simulation node accepts a recent world target for a grasp.
The camera pose parameters must match the simulated camera. Invalid depth means
no target is published.

Build as before:

```bash
cd ~/panda_grasp_ws
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install
source install/setup.bash
```

Restart sim_camera and yolo_detector, then inspect:

```bash
ros2 topic echo /grasp/target_point --once
ros2 topic echo /grasp/target_point_camera --once
```

Expected rough output for the unmodified red-cube scene is
`x ≈ 0.48 m`, `y ≈ -0.12 m`, `z ≈ 0.325 m` in the world frame.
These are reference values for verification only, NEVER fed into inference.
