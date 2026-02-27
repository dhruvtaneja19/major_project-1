from setuptools import find_packages, setup

package_name = 'warehouse_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/warehouse_gazebo']),
    ('share/warehouse_gazebo', ['package.xml']),
    ('share/warehouse_gazebo/launch', ['launch/sim.launch.py']),
    ('share/warehouse_gazebo/worlds', ['worlds/warehouse.world']),
    ('share/warehouse_gazebo/worlds', ['worlds/smart_warehouse.world']),
    ('share/warehouse_gazebo/worlds', ['worlds/mapf_arena.world']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dhruv',
    maintainer_email='dhruv@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
