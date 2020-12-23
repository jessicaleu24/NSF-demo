import socket
import numpy as np
import struct
import time

import simulation.trajectory_cubic as traj
from simulation.lrmate_kine_base import lrmate_fk, lrmate_ik
from scipy.spatial.transform import Rotation as R

"""
This file allow Ubuntu/Mac/Windows control the robot using position/velocity.
Set your Ubuntu PC IP to 192.168.1.200, gateview to 255.255.255.1, netmask 24. (refer to https://wiki.teltonika-networks.com/view/Setting_up_a_Static_IP_address_on_a_Ubuntu_16.04_PC)
"""

class robot_controller():
    def __init__(self):
        self.UDP_IP_IN = "192.168.1.200"     # Ubuntu IP, should be the same as Matlab shows
        self.UDP_PORT_IN = 57831             # Ubuntu receive port, should be the same as Matlab shows
        self.UDP_IP_OUT = "192.168.1.100"    # Target PC IP, should be the same as Matlab shows
        self.UDP_PORT_OUT_1 = 3826           # Robot 1 receive Port
        self.UDP_PORT_OUT_2 = 3827           # Robot 2 receive Port

        self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        self.unpacker = struct.Struct("6d 6d 6d 6d 6d 6d")

        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.robot_1_joints, self.robot_1_vels, self.robot_1_ATI_force = None, None, None
        self.robot_2_joints, self.robot_2_vels, self.robot_2_ATI_force = None, None, None
        
        # traj tracking parameters
        self.freq = 125
        self.kp = np.diag([0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.kd = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    def receive(self):
        data, _ = self.s_in.recvfrom(1024)
        unpacked_data = np.array(self.unpacker.unpack(data))
        self.robot_1_joints, self.robot_1_vels, self.robot_1_ATI_force = np.deg2rad(unpacked_data[0:6]), np.deg2rad(unpacked_data[6:12]), unpacked_data[12:18]
        self.robot_2_joints, self.robot_2_vels, self.robot_2_ATI_force = np.deg2rad(unpacked_data[18:24]), np.deg2rad(unpacked_data[24:30]), unpacked_data[30:36]

    def send(self, target_joints_robot_1=np.zeros(6), target_joints_robot_2=np.zeros(6)):
        """send target joints pos/vel to the robot, vel in rad

        Args:
            target_joints_robot_1 (target joints for robot 1, optional): if vel, then in rad. Defaults to np.zeros(6).
            target_joints_robot_2 (target joints for robot 2, optional): if vel, then in rad. Defaults to np.zeros(6).
        """
        joint_1 = target_joints_robot_1.astype('d').tobytes()
        self.s_out.sendto(joint_1, (self.UDP_IP_OUT, self.UDP_PORT_OUT_1))
        joint_2 = target_joints_robot_2.astype('d').tobytes()
        self.s_out.sendto(joint_2, (self.UDP_IP_OUT, self.UDP_PORT_OUT_2))

    def step(self, T=0.08):
        st = time.time()
        while time.time() - st <= T:
            self.receive()

    def plan_traj(self, tgt_joints, waypoints=None):
        """Plan trajectory for the robot

        Args:
            tgt_joints (np.ndarray): target joints of the robot in rad
            waypoints (np.ndarray, optional): middle joints of the traj. Defaults to None.
            T (int, optional): execution time. Defaults to 3.

        Returns:
            traj's joints angle, velocity, acc in rad
        """
        self.receive()
        curr_joints = self.robot_1_joints
        T = int(max(abs(tgt_joints - curr_joints) / 0.2))
        T = max(5, T)
        start = np.hstack((curr_joints, 0)).reshape(1, 7)
        end = np.hstack((tgt_joints, T)).reshape(1, 7)
        if waypoints is None:
            middle = np.linspace(start[0], end[0], num=8, endpoint=False)[1:, 0:6]
            middle = np.hstack((middle, np.zeros((middle.shape[0], 1))))
        else:
            middle = waypoints.reshape(-1, 7)
        waypoints, pos, vel, acc = traj.trajectory_xyz(start, end, middle, self.freq)
        return pos, vel, acc
    
    def track_traj(self, ref_poses, ref_vels, ref_accs):
        """track the planned trajectory
           PD is not needed as shown in experiments

        Args:
            ref_poses (np.ndarray): ref joints of the robot in rad
            ref_vels (np.ndarray): ref joints vel of the robot in rad
            ref_accs (np.ndarray): ref joints acc of the robot in rad
        """
        for ref_pos, ref_vel, _ in zip(ref_poses, ref_vels, ref_accs):
            st = time.time()
            self.receive()
            curr_pos, curr_vel = self.robot_1_joints, self.robot_1_vels
            err_pos = ref_pos - curr_pos
            err_vel = ref_vel - curr_vel
            p_term = self.kp @ err_pos.reshape(6, 1)
            d_term = self.kd @ err_vel.reshape(6, 1)
            next_vel = ref_vel + p_term.reshape(6,) + d_term.reshape(6,)
            self.send(target_joints_robot_1=next_vel)
            used_time = time.time() - st
            if used_time < 1 / self.freq:
                time.sleep(1 / self.freq - used_time)
        self.send()
        real_robot.step(0.2)

    def init_robot_pose(self):
        tgt_joints = np.deg2rad(np.array([0.0, 0.0, 0.0, 0.0, -90.0, 0.0]))
        ref_poses, ref_vels, ref_accs = self.plan_traj(tgt_joints)
        self.track_traj(ref_poses, ref_vels, ref_accs)
    
    
    
if __name__ == "__main__":
    real_robot = robot_controller()
    real_robot.step(1)
    real_robot.init_robot_pose()
    
    init_joints = np.deg2rad(np.array([0, 0, 0, 0, -90, 0]))
    approach_joints = np.array([-17.18, 1.76, -37.74, -29.91, -48.21, -84.01])
    tgt_joints = np.array([-12.15, 13.54, -41.43, -40.62, -33.92, -75.65])
    
    init_joints = np.deg2rad(init_joints)
    approach_joints = np.deg2rad(approach_joints)
    tgt_joints = np.deg2rad(tgt_joints)
    
    ref_poses, ref_vels, ref_accs = real_robot.plan_traj(approach_joints)
    real_robot.track_traj(ref_poses, ref_vels, ref_accs)
    
    ref_poses, ref_vels, ref_accs = real_robot.plan_traj(tgt_joints)
    real_robot.track_traj(ref_poses, ref_vels, ref_accs)
    
    ref_poses, ref_vels, ref_accs = real_robot.plan_traj(approach_joints)
    real_robot.track_traj(ref_poses, ref_vels, ref_accs)
    
    # ref_poses, ref_vels, ref_accs = real_robot.plan_traj(init_joints)
    # real_robot.track_traj(ref_poses, ref_vels, ref_accs)
    
    print("Done")

