import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'panda_grasp_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='unknown@example.com',
    description='YOLOv8 RGB-D Panda grasp starter for ROS 2 Jazzy',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'sim_camera = panda_grasp_ros.sim_camera_node:main',
        ],
    },
)
