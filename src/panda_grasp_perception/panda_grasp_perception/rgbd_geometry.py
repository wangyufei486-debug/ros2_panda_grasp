"""RGB-D cube localization, independent from ROS 2 and the PyBullet simulator.

The camera coordinates follow ROS optical convention: x right, y down, z forward.
The scene is the fixed overhead camera from the original 007c project. The
simulator's world origin and the Panda URDF's base origin coincide.
"""
import numpy as np


def optical_axes_in_world(eye, target, up):
    """Return columns (camera x, camera y, camera z) expressed in world frame."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    length = np.linalg.norm(forward)
    if length < 1e-9:
        raise ValueError('Camera eye and target must differ')
    forward /= length
    right = np.cross(forward, up)
    length = np.linalg.norm(right)
    if length < 1e-9:
        raise ValueError('Camera forward and up must not be parallel')
    right /= length
    true_up = np.cross(right, forward)
    return np.stack((right, -true_up, forward), axis=1)


def locate_cube(mask, depth_m, k, eye, target, up,
                table_z=0.30, cube_half=0.025,
                depth_near=0.05, depth_far=2.0):
    """Find cube center from its YOLO segmentation Mask and measured depth.

    Returns a dict of 3-D points, or raises ValueError if data is unusable.
    Unlike a bounding-box-center sample, this robustly uses the cube top face.
    Depth units are meters; K is the sensor_msgs/CameraInfo 3x3 matrix.
    """
    mask = np.asarray(mask)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if mask.ndim != 2 or depth_m.shape != mask.shape:
        raise ValueError('Mask and depth must have the same HxW dimensions')
    K = np.asarray(k, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError('Invalid CameraInfo intrinsics')

    good = (mask > 0) & np.isfinite(depth_m) & (
        depth_m > depth_near) & (depth_m < depth_far - 1e-4)
    rows, cols = np.where(good)
    if len(cols) < 35:
        raise ValueError('Fewer than 35 valid depth pixels in YOLO Mask')

    z_cam = depth_m[rows, cols]
    x_cam = (cols.astype(float) + 0.5 - cx) * z_cam / fx
    y_cam = (rows.astype(float) + 0.5 - cy) * z_cam / fy
    points_cam = np.stack((x_cam, y_cam, z_cam), axis=1)

    basis = optical_axes_in_world(eye, target, up)
    eye = np.asarray(eye, dtype=np.float64)
    points_world = points_cam @ basis.T + eye

    # Scene-specific table filter; prevents mask spillover from dragging the
    # estimated top surface down to the tabletop.
    above_table = points_world[:, 2] > table_z + 0.010
    points_world = points_world[above_table]
    if len(points_world) < 25:
        raise ValueError('Fewer than 25 cube pixels above table')

    top_z = np.percentile(points_world[:, 2], 80)
    top_points = points_world[points_world[:, 2] >= top_z - 0.006]
    if len(top_points) < 20:
        raise ValueError('Fewer than 20 cube top-surface points')

    top_world = np.median(top_points, axis=0)
    center_world = top_world.copy()
    center_world[2] -= cube_half
    center_camera = basis.T @ (center_world - eye)

    return {
        'world_center': center_world,
        'camera_center': center_camera,
        'top_world': top_world,
        'valid_depth_pixels': int(len(cols)),
        'above_table_pixels': int(len(points_world)),
        'top_surface_pixels': int(len(top_points)),
    }
