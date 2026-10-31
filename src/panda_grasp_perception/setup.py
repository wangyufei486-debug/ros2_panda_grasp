from setuptools import find_packages, setup

package_name = 'panda_grasp_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wyf',
    maintainer_email='wyf@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'target_demo = panda_grasp_perception.target_demo:main',
            'yolo_detector = panda_grasp_perception.yolo_node:main',
        ],
    },
)
