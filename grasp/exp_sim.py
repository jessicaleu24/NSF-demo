import numpy as np
import matplotlib.pyplot as plt

import pybullet as p
from simulation.BulletSim import BulletSim
from PIL import Image

from detectron2.engine import default_argument_parser, default_setup, launch
import logging
import os, time
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch
from detectron2.modeling import build_model
from detectron2.structures import Instances
from data.cgrcnn_dataset import *
from data.data_utils import plot, plot_many, create_model_input

from scipy.spatial.transform import Rotation as R
from utils import setup, reproject_depth

import open3d as o3d


def plot_grasp(depth, grasp):
    inst = Instances(depth.shape)
    rbox = np.array([grasp])
    gt_boxes = torch.tensor(rbox, dtype=torch.float32)
    inst.gt_boxes = RotatedBoxes(gt_boxes)
    img = np.tile(depth, (3, 1, 1))
    img = torch.from_numpy(img)
    inputs_list = [{"image": img, "instances": inst}]
    plot_many(inputs_list)

def plan_grasps(depth, extrinsic):
    """Given a pre-processed depth image and the camera's extrinsic, plan grasps

    Args:
        depth (np.array 424x512): depth image from the camera
        extrinsic (np.array 4x4): camera's pose
    """
    # First, reproject the image for model's performance
    depth, extrinsic = reproject_depth(depth, extrinsic)
    # Second, load the model's weight
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    cfg = setup(args)

    model = build_model(cfg)
    pretrained = torch.load(cfg.MODEL.WEIGHTS)
    model.load_state_dict(pretrained["model"])
    model.eval()
    # Process the image for model's usage
    zero_pad = np.zeros((44, 512), dtype=np.float32)
    depth = np.vstack((zero_pad, depth, zero_pad))
    tf_depth = Image.fromarray(depth)
    tf_depth = tf_depth.resize((224, 224))
    depth = np.asarray(tf_depth)
    # Plan grasps
    with torch.no_grad():
        inputs = [create_model_input(depth)]
        proposals, rpn_proposals = model(inputs)
        assert len(proposals) == len(rpn_proposals) == 1
    # plot proposals
    n_top = 5
    for ii in range(len(proposals)):
        inst = Instances((224, 224))
        indices = torch.topk(rpn_proposals[ii].objectness_logits, n_top).indices
        inst.gt_boxes = rpn_proposals[ii].proposal_boxes[indices]
        # inst.gt_tilts = proposals[ii]["instances"].pred_tilts[indices]
        inputs_list = [{"image": inputs[ii]["image"], "instances": inst}]
        plot_many(inputs_list, n_plot=n_top)
        
    grasps = inst.gt_boxes.tensor.cpu().numpy()
    return grasps, depth, extrinsic


if __name__ == "__main__":
    root = os.getcwd() + "/grasp/"
    obj_model_path = "./grasp/simulation/obj_models/"
    img_path = root + "/imgs/demo.npy"
    objID = None
    obj_name = "109d55a137c042f5760315ac3bf2c13e" # bottle, scale=1
    scale = 1
    sim_freq = 240.0
    vis = True
    
    ####################################### initiate the camera pose #####################################
    # j_b_in_r = np.array([0, 20.3793, -35.049, 0, -34.156, 0])
    # T_b_in_r, T_b_in_c = np.eye(4), np.eye(4)
    # T_b_in_r[0:3, 3] = (0.4709, 0, 0.105)
    # T_b_in_c[0:3, 0:3] = np.array([-0.9975, -0.0677, -0.0213, 0.0511, -0.8934, 0.4464, -0.0493, 0.4441, 0.8946]).reshape(3,3) 
    # T_b_in_c[0:3, 3] = (0.1165697, 0.0739131, 1.0686)
    # T_c_in_b = np.linalg.inv(T_b_in_c)
    # rotY180 = np.eye(4)
    # rotY180[0:3, 0:3] = R.from_euler("y", -180, degrees=True).as_matrix()
    # T_c_in_r = T_b_in_r @ rotY180 @ T_c_in_b
    
    # frame_r = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    # frame_b = copy.deepcopy(frame_r).transform(T_b_in_r)
    # frame_c = copy.deepcopy(frame_r).transform(T_c_in_r)
    # o3d.visualization.draw_geometries([frame_r, frame_b, frame_c])
    
    
    ############################# start the simulation and render depth imgs #############################
    env = BulletSim(root, obj_model_path, sim_freq=sim_freq, vis=vis)
    env.set_joints(env.init_joints)
    objID = env.load_obj(obj_name, scale=scale)
    
    for _ in range(240*3):
        env.step()
    
    depth, depth_pybullet = env.capture_depth_img()
    depth_rp, extrinsic = reproject_depth(depth, env.extrinsic)
    
    plt.subplot(121)
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.subplot(212)
    plt.imshow(depth_rp, cmap=plt.cm.gray_r)
    plt.show()
    np.save(root + "/output/test_1.npy", {"intrinsic": env.intrinsic, "extrinsic": extrinsic, 
                                          "depth_512": depth_rp, "obj_location": p.getBasePositionAndOrientation(2),
                                          "depth_512_raw": depth, "extrinsic_raw": env.extrinsic})
    
    
    # ############################# use CGPN to plan grasps #############################
    contents = np.load(root + "/output/test_1.npy", allow_pickle=True).item()
    extrinsic, intrinsic = contents["extrinsic_raw"], contents["intrinsic"]
    depth = contents["depth_512_raw"]
    grasps, depth, extrinsic = plan_grasps(depth, extrinsic)
    
    ############################# execute grasps #############################
    obj_location = contents["obj_location"]
    
    if objID is None:
        objID = env.load_obj(obj_name, pos=obj_location[0], rot=obj_location[1])
    
    rot = np.eye(4)
    rot[0:3, 0:3] = R.from_euler("x", -180, degrees=True).as_matrix()
    extrinsic = extrinsic @ rot
    
    for grasp in grasps:
        p.resetBasePositionAndOrientation(objID, obj_location[0], obj_location[1])
        
        x_center, y_center, width, height, angle = np.rint(grasp).astype(np.int)
        
        z_cam = depth[y_center, x_center]
        if z_cam <= 1e-3:
            continue
        
        # project grasp back to 512*424 image
        x_center *= 512 / 224
        y_center = y_center * 512 / 224 - 44
        p_cam = np.linalg.inv(intrinsic) @ (np.array([x_center, y_center, 1]).reshape(3, 1) * z_cam)
        p_cam_homo = np.vstack((p_cam, np.array([1])))

        p_world_homo = extrinsic @ p_cam_homo
        p_world = p_world_homo[0:3].reshape(3,) # + np.array([0.0, 0.0, 0.3])
        
        plot_grasp(depth, grasp)
        
        execute = "y"
        # execute = "n"
        if execute == "y":
            robot_target_pose = np.hstack((p_world, np.array([np.pi, 0.0, np.deg2rad(-angle)])))
            env.go2pose_cart(robot_target_pose)
            for _ in range(240):
                env.step()
            
            p_grasp = p_world_homo[0:3].reshape(3,) - np.array([0.0, 0.0, 0.22])
            p_grasp[2] = max(0.22, p_grasp[2])
            robot_target_pose = np.hstack((p_grasp, np.array([np.pi, 0.0, np.deg2rad(-angle)])))
            env.go2pose_cart(robot_target_pose)
            for _ in range(240):
                env.step()
                
            env.set_gripper(0.04)
            for _ in range(240):
                env.step()
                
            robot_target_pose = np.hstack((p_world, np.array([np.pi, 0.0, np.deg2rad(-angle)])))
            env.go2pose_cart(robot_target_pose)
            for _ in range(240):
                env.step()
                
            env.set_gripper(0.0)
            for _ in range(240):
                env.step()