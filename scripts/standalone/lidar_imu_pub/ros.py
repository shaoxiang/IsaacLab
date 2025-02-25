import rclpy
from rclpy.node import Node
import threading
import torch

from rosgraph_msgs.msg import Clock
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import Imu as ROS_IMU
from lidar import get_head_lidar_pointcloud

class SensorPubNode(Node):
    def __init__(self):
        super().__init__("go2_pub_node")

        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        # self.imu_pub = self.create_publisher(ROS_IMU, "/imu", 10)
        self.head_lidar_pub = self.create_publisher(PointCloud2, "/point_cloud2", 10)
        self.clock_msg = Clock()

    def publish(self, obs: dict, sim_time_sec: float):
        self._pub_clock(sim_time_sec)
        self._pub_head_lidar()

    def _pub_clock(self, sim_time_sec: float):
        msg = Clock()
        msg.clock = self.get_clock().now().to_msg()
        msg.clock.sec = int(sim_time_sec)
        msg.clock.nanosec = int((sim_time_sec - int(sim_time_sec)) * 1e9)
        self.clock_msg = msg
        self.clock_pub.publish(msg)

    def _pub_imu(self, obs: dict):
        msg = ROS_IMU()
        msg.imu_state.quaternion[0] = obs["obs"]["imu_body_orientation"][0, 1].item()
        msg.imu_state.quaternion[1] = obs["obs"]["imu_body_orientation"][0, 2].item()
        msg.imu_state.quaternion[2] = obs["obs"]["imu_body_orientation"][0, 3].item()
        msg.imu_state.quaternion[3] = obs["obs"]["imu_body_orientation"][0, 0].item()
        msg.imu_state.gyroscope[0] = obs["obs"]["imu_body_ang_vel"][0, 0].item()
        msg.imu_state.gyroscope[1] = obs["obs"]["imu_body_ang_vel"][0, 1].item()
        msg.imu_state.gyroscope[2] = obs["obs"]["imu_body_ang_vel"][0, 2].item()
        msg.imu_state.accelerometer[0] = obs["obs"]["imu_body_lin_acc"][0, 0].item()
        msg.imu_state.accelerometer[1] = obs["obs"]["imu_body_lin_acc"][0, 1].item()
        msg.imu_state.accelerometer[2] = obs["obs"]["imu_body_lin_acc"][0, 2].item()
        self.low_state_pub.publish(msg)

    def _pub_head_lidar(self):
        pcl = get_head_lidar_pointcloud().tolist()

        header = Header()
        header.frame_id = "lidar_frame"
        header.stamp.sec = self.clock_msg.clock.sec
        header.stamp.nanosec = self.clock_msg.clock.nanosec

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=16, datatype=PointField.FLOAT32, count=1
            ),
            PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1),
            PointField(name="time", offset=24, datatype=PointField.FLOAT32, count=1),
        ]

        pcl_msg = point_cloud2.create_cloud(header, fields, pcl)
        self.head_lidar_pub.publish(pcl_msg)

    def _clock_to_sec(self, clock_msg: Clock) -> float:
        return clock_msg.clock.sec + clock_msg.clock.nanosec / 1e9
