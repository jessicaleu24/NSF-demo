import pybullet as p
import time
import pybullet_data
import numpy as np

try:
    from simulation.lrmate_kine_base import lrmate_fk, lrmate_ik
except:
    from lrmate_kine_base import lrmate_fk, lrmate_ik

from scipy.spatial.transform import Rotation as R
import os, copy

import open3d as o3d
import pyrender
import trimesh
import matplotlib.pyplot as plt


class BulletSim():
    def __init__(self, root, obj_model_path, sim_freq=240.0, vis=True, max_force=3):
        self.root = root
        self.obj_model_path = obj_model_path
        self.objID = []
        self.sim_freq = sim_freq
        self.max_force = max_force
        
        if vis:
            self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")

        robotStartPos = [0,0,0]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.filename = root + "/simulation/robot_model/lrmate_parallel.urdf"
        self.robotID = p.loadURDF(self.filename, robotStartPos, robotStartOrientation, useFixedBase=1, flags=p.URDF_MAINTAIN_LINK_ORDER)

        self.robotJointsNum = p.getNumJoints(self.robotID)
        self.jointIndices = range(self.robotJointsNum)
        self.arm_joints = [0, 1, 2, 3, 4, 5]
        self.current_arm_joints_position = np.zeros(6)
        self.current_arm_joints_velocity = np.zeros(6)
        self.init_joints = np.array([0, 0, 0, 0, -np.pi/2, 0])
        
        self.extrinsic = np.eye(4)
        self.eyePos=[0.501, 0.0, 0.5]
        self.tgtPos=[0.5, 0.0, 0.0]
        self.get_camerao_pose()
        self.image_h, self.image_w = 424, 512
         
        # setup camera and renderer
        self.render_scene = pyrender.Scene()
        self.flags = pyrender.constants.RenderFlags.DEPTH_ONLY
        self.renderer = pyrender.OffscreenRenderer(viewport_height=self.image_h, viewport_width=self.image_w)
        self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
        self.intrinsic = self.camera_intrinsic.intrinsic_matrix
        self.fx, self.fy, self.cx, self.cy = (self.intrinsic[0, 0], self.intrinsic[1, 1], self.intrinsic[0, 2], self.intrinsic[1, 2])
        self.camera = pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, znear=0.025)
        self.ca = self.render_scene.add(self.camera, pose=self.extrinsic)
    
    def get_camerao_pose(self):
        z = np.array(self.tgtPos) - np.array(self.eyePos)
        z = z / np.linalg.norm(z)
        up = np.array([0,0,1])
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        self.extrinsic[0:3, 0], self.extrinsic[0:3, 1], self.extrinsic[0:3, 2], self.extrinsic[0:3, 3] = x, y, z, np.array(self.eyePos)
        self.rot = np.eye(4)
        self.rot[0:3, 0:3] = R.from_euler("x", 180, degrees=True).as_matrix()
        self.extrinsic = self.extrinsic @ self.rot
        
    def destroy_sim(self):
        p.disconnect()
    
    def load_obj(self, name, pos=None, rot=None, scale=1):
        path = self.obj_model_path + name + "/obj_urdf/obj_urdf.urdf"
        if pos is None:
            pos = abs(0.2 * np.random.randn(3))
            pos += np.array([0.35, 0, 0.1])
            pos[1] = 0.0
            rot = p.getQuaternionFromEuler(np.random.randn(3))
        ID = p.loadURDF(path, pos, rot, useFixedBase=0, globalScaling=scale)
        self.objID.append((name, ID, scale))
        p.changeDynamics(ID, -1, mass=0.01, lateralFriction=1, rollingFriction=0.0)
        return ID
        
    def load_obj_render(self):
        for (name, ID, scale) in self.objID:
            path = self.obj_model_path + name + "/full_mesh.stl"
            mesh = trimesh.load_mesh(path)
            mesh.vertices *= scale
            render_mesh = pyrender.Mesh.from_trimesh(mesh)
            pose = p.getBasePositionAndOrientation(ID)
            pos, rot_quant = np.array(pose[0]), np.array(pose[1])
            pose = np.eye(4)
            pose[0:3, 3], pose[0:3, 0:3] = pos, R.from_quat(rot_quant).as_matrix()
            self.render_scene.add(render_mesh, pose=pose)

    def set_joints_vel(self, vel):
        """
        Given 6 joint velocities for robot, return a full list of joint velocities for the simulator

        Parameters
        ----------
        vel : [numpy array: (6,)]
            Contains the 6 joint velocities of the Fanuc robots
            joints1, joints2, joints3, joints4, joints5, joints6
        """
        assert len(vel) == 6
        # self.targetVelocities = vel
        p.setJointMotorControlArray(bodyUniqueId=self.robotID, jointIndices=self.jointIndices[0:6], controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=vel)
        
    def set_joints(self, joints):
        assert len(joints) == 6
        # self.targetJoints = joints
        p.setJointMotorControlArray(bodyUniqueId=self.robotID, jointIndices=self.jointIndices[0:6], controlMode=p.POSITION_CONTROL,
                                        targetPositions=joints)
        
    def set_gripper(self, pos=0.0):
        """
        set gripper pose, 0.08m gap when fully open
        
        Parameters
        ----------
        pos : float, optional
            set the gripper open position, by default 0.0 (fully open), 0.04 (fully close)
        """
        p.setJointMotorControl2(bodyUniqueId=self.robotID, jointIndex=8, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=self.max_force)
        p.setJointMotorControl2(bodyUniqueId=self.robotID, jointIndex=9, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=self.max_force)
    
    def go2pose_cart(self, pose_cart):
        assert len(pose_cart) == 6
        joints = lrmate_ik(pose_cart)
        self.set_joints(joints)
        
    def step(self):
        p.stepSimulation()
        time.sleep(1.0 / self.sim_freq)
        temp = p.getJointStates(self.robotID, jointIndices=self.arm_joints)
        for i in range(len(self.arm_joints)):
            self.current_arm_joints_position[i] = temp[i][0]
            self.current_arm_joints_velocity[i] = temp[i][1]

    def capture_depth_img_pybullet(self):
        viewMatrix = p.computeViewMatrix(cameraEyePosition=self.eyePos, cameraTargetPosition=self.tgtPos, cameraUpVector=[0,0,1])
        projectionMatrix = p.computeProjectionMatrixFOV(fov=60, aspect=self.image_w/self.image_h, nearVal=0.01, farVal=100)
        _, _, rgb, depth, mask = p.getCameraImage(self.image_w, self.image_h, viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return self.segment_depth(depth, mask)
    
    def capture_depth_img(self):
        depth_pybullet = self.capture_depth_img_pybullet()
        self.load_obj_render()
        depth = self.renderer.render(self.render_scene, self.flags)
        return depth, depth_pybullet
        
    def segment_depth(self, depth, mask):
        imgs = []
        for _, ID, _ in self.objID:
            depth_ = copy.deepcopy(depth)
            depth_[mask != ID] = 0
            imgs.append(depth_)
        return imgs
    
if __name__ == "__main__":
    # root = os.getcwd()
    root = "/home/zxh/Documents/ContrastiveGrasp_Train/ContrastiveGrasp_Train/"
    obj_model_path = "/home/zxh/Documents/GraspNet/data/obj_models/"
    
    sim_freq = 24000.0
    vis = True
    
    env = BulletSim(root, obj_model_path, sim_freq=sim_freq, vis=vis)
    env.set_joints(env.init_joints)
    
    env.load_obj("109d55a137c042f5760315ac3bf2c13e")
    
    for _ in range(240*5):
        env.step()
        
    depth_imgs = env.capture_depth_img()
        
    while True:
        env.step()