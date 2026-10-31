"""仿真场景中的静态 TF 消息构造。"""

from geometry_msgs.msg import TransformStamped


def static_camera_transforms(stamp, camera_eye):
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
    camera_tf.transform.translation.x = float(camera_eye[0])
    camera_tf.transform.translation.y = float(camera_eye[1])
    camera_tf.transform.translation.z = float(camera_eye[2])
    camera_tf.transform.rotation.x = 1.0
    camera_tf.transform.rotation.y = 0.0
    camera_tf.transform.rotation.z = 0.0
    camera_tf.transform.rotation.w = 0.0
    return [base_tf, camera_tf]
