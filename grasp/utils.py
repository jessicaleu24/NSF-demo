import os, copy
import math
import h5py
import time
import numpy as np
import trimesh
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyrender
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from pyrender import PerspectiveCamera, DirectionalLight, SpotLight, PointLight, \
    MetallicRoughnessMaterial, Primitive, \
    Mesh, Node, Scene, Viewer, \
    OffscreenRenderer
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from skimage.filters import gaussian

try:
    import cv2
except:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

def find_contacts(obj_mesh,
                  centers,
                  axises,
                  max_width=0.047,
                  find_contacts=False):
    """Return grasp contact position, modified from vis_dexnet_dataset.py

    Parameters:
    ------------------
    obj_mesh :    trimesh.mesh 
        object mesh
    centers  :    numpy.array (m, 3) 
        center point of gripper
    axises   :    numpy.array (m, 3)
        grasp axis
    max_width:    float
        length of gripper width

    Returns:
    ------------------
    p1s     : numpy.array (m, 3)
        one end point of grasp axis
    p2s     : numpy.array (m, 3)
        one end point of grasp axis
    contacts: python dictionary
        grasp contact points 
    """

    p1s = centers - max_width * axises / 2
    p2s = centers + max_width * axises / 2
    contacts = None
    if find_contacts is False:
        return p1s, p2s, contacts

    # first shoot rays along axises from p1s to hit the obj_mesh
    locations, index_ray, index_tri = obj_mesh.ray.intersects_location(
        ray_origins=p1s, ray_directions=axises)
    # second sort contacts for each p1
    contacts = {}
    for i in range(locations.shape[0]):
        idx = index_ray[i]
        if idx in contacts:
            contacts[idx] = np.vstack((contacts[idx], locations[i]))
        else:
            contacts[idx] = locations[i]
    # third discard points that are too far, and find min/max contacts
    for i in range(centers.shape[0]):
        p1 = p1s[i]
        if len(contacts[i].shape) == 1:
            contacts[i] = np.vstack((contacts[i], contacts[i]))
        else:
            dist = np.linalg.norm(contacts[i] - p1, axis=1)
            inside_points_idx = np.where(dist < max_width)[0]
            contacts[i] = contacts[i][inside_points_idx]
            dist = np.linalg.norm(contacts[i] - p1, axis=1)
            c1 = contacts[i][np.argmin(dist)]
            c2 = contacts[i][np.argmax(dist)]
            contacts[i] = np.vstack((c1, c2))

    return p1s, p2s, contacts


def get_camera_pose(camera_position, target_position):
    """ use Look At function to get 4 x 4 transformation matrix to pose the camera face to object 

    Parameters:
    ------------------
    camera_position : numpy.array (3,)
        position of camera
    target_position : numpy.array (3,)
        position of object centroid

    Return:
    ------------------
    camera_pose     : numpy.array (4,4) 
        Transformation matrix to make camera pose to object
    Reference:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    """
    camera_pose = np.eye(4)
    forward = camera_position - target_position
    forward = forward / np.linalg.norm(forward)
    left = np.cross(np.array([0, 1, 0]), forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)
    camera_pose[:3, 0] = left
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = forward
    camera_pose[:3, 3] = camera_position

    return camera_pose


def sample_camera_position(radius, mode="uniform"):
    """sample camera position by roll/pitch/yaw in a sphere

    Parameters:
    ------------------
    radius: int 
        radius of the sphere
    Returns:
    ------------------
    sampled_position: numpy.array (1,3)
        sampled camera position
    """

    if mode == "axis":
        # Initialize the position
        inital_position = np.array([[radius, 0, 0]])
        # Sample a rotation matrix
        angles = np.random.uniform(low=0.0, high=2 * math.pi, size=(3, ))
        M = trimesh.transformations.euler_matrix(angles[0],
                                                 angles[1],
                                                 angles[2],
                                                 axes='sxyz')
        # Transformation: (1,3) = (1,3) x (3, 3)
        sampled_position = inital_position @ M[0:3, 0:3].T
        return sampled_position
    elif mode == "uniform":
        u = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 2 * math.pi)
        sampled_position = radius * random_uniform_vector(u, theta)
        return sampled_position


def sample_mesh(obj_mesh, n_points=2048, rad=0.001):
    obj_faces = np.array(obj_mesh.faces)
    obj_face_normals = np.array(obj_mesh.face_normals)
    points, face_index = trimesh.sample.sample_surface_even(
        obj_mesh, n_points, 0.001)
    normals = None
    for i in face_index:
        normal = obj_face_normals[i]
        if normals is None:
            normals = normal
        else:
            normals = np.vstack((normals, normal))
    return points, normals


def random_uniform_vector(u, theta):
    '''
    Parameters:
    ------------------
    u           :  float
        uniform random float number into -1 to 1
    theta       :  float
        uniform random float number into 0 to 2pi
    Returns:
    ------------------

    Sphere Point Picking 
    https://mathworld.wolfram.com/SpherePointPicking.html
    '''
    # print(u)
    x = np.sqrt(1 - u**2) * np.cos(theta)
    y = np.sqrt(1 - u**2) * np.sin(theta)
    z = u
    position = np.array([[x, y, z]])

    return position


def depth2pc_w(depth, camera_pose, rotM, voxel_size=0.001):
    depth = o3d.geometry.Image(depth)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
    # width: 512, height: 424, fx: 365.456 mm, fy: 365.456 mm, cx: 254.878, cy: 205.395
    try:
        pcd = o3d.geometry.create_point_cloud_from_depth_image(
            depth, camera_intrinsic)  # open3d for ubuntu 16
    except:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth, camera_intrinsic)  # open3d for ubuntu 18
    # downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # pc_o = np.array(downsample_pcd.points)
    pc_o = np.array(pcd.points)
    camera_pose = camera_pose @ rotM
    pc_w = camera_pose[0:3, 0:3] @ pc_o.T + \
        camera_pose[0:3, 3].reshape((3, 1))
    
    return pc_w.T


def mesh_normalization(mesh):
    """Normalize mesh 

    Parameters:
    ------------------
    mesh: trimesh.mesh or trimesh.scene
        mesh to be normalized 
    """
    # center mesh
    mesh_mean = np.mean(mesh_or_scene.vertices, axis=0)
    value = np.max(np.linalg.norm(mesh_or_scene.vertices - mesh_mean, axis=1))

    # normalize bounding box
    matrix = np.identity(4)
    matrix[0, 0] = 1.0 / value
    matrix[1, 1] = 1.0 / value
    matrix[2, 2] = 1.0 / value
    matrix[:3, 3] = -mesh_mean
    mesh.apply_transform(matrix)

def euclidean_distance_matrix(x):
    """compute euclidean distance matrix
    Parameters:
    ------------------
    x             :   numpy.array (n, 3)
        point cloud

    Returns:
    ------------------
    distance_mat  :   numpy.array (n, n)
        distance matrix
    """
    r = np.sum(x*x, 1)
    r = r.reshape(-1, 1)
    distance_mat = r - 2*np.dot(x, x.T) + r.T
    return distance_mat

def update_farthest_distance(far_mat, dist_mat, s):
    """update distance matrix and select the farthest point from set S after a new point is selected
    Parameters:
    ------------------
    far_mat  :  python list length = dist_mat.shape[0]
        the minimum distance from solution set to all the points 
    dist_mat :  numpy.array (n, n)
        distance matrix 
    s        :  the last added index
        solution index

    Returns:
    ------------------
    far_mat  :  python list length = dist_mat.shape[0]
        updated by add a new point, the minimum distance from solution set to all the points 
    new_s    :  int
        new point index  
    """
    for i in range(far_mat.shape[0]):
        far_mat[i] = dist_mat[i,s] if far_mat[i] > dist_mat[i,s] else far_mat[i]
    new_s = np.argmax(far_mat)
    return far_mat, new_s

def init_farthest_distance(far_mat, dist_mat, s):
    """initialize matrix to keep track of distance from set 
    Parameters:
    ------------------
    far_mat  :  python list length = dist_mat.shape[0]
        the minimum distance from solution set to all the points
    dist_mat :  numpy.array (n, n)
        distance matrix  
    s        :  the last added index
        solution index

    Returns:
    ------------------
    far_mat  :  python list length = dist_mat.shape[0]
        the minimum distance from solution set to all the points
    """
    for i in range(far_mat.shape[0]):
        far_mat[i] = dist_mat[i,s]
    return far_mat
    
def farthest_sampling(set_P, num_samples = 2048):
    """farthest point sampling 
    Parameters:
    ------------------
    set_P        : numpy.array  （m , 3)
        origin point cloud
    num_samples  : int     
        number of samples 

    Returns:
    ------------------
    set_S        : numpy.array   (m , 3)
        sampled point cloud
    """
    num_P = set_P.shape[0]
    distance_mat = euclidean_distance_matrix(set_P)
    set_S = []
    s = np.random.randint(num_P)
    far_mat = init_farthest_distance(np.zeros((num_P)), distance_mat, s)
    for i in range(num_samples):
        set_S.append(set_P[s])
        far_mat, s = update_farthest_distance(far_mat, distance_mat, s)
        
    set_S = np.array(set_S)
    return set_S

def generate_namelist():
    """generate a list of (obj_name, NumofGrasps) and save as name_lists.py
    """
    dataset_path = os.path.abspath(os.path.join(cwd, os.pardir)) +  "/data/"
    f = h5py.File(dataset_path + "dexnet_2_database.hdf5", 'r')
    dataset = f['datasets']
    dataset_names = ['3dnet', 'kit']
    obj_list = np.load(os.getcwd() + "/name_list.npy",
                       allow_pickle=True).item()
    
    for dataset_name in dataset_names:
        obj_names = obj_list[dataset_name]
        group = dataset[dataset_name]["objects"]
        obj_grasp_list = []
        for obj_name in obj_names:
            obj_dataset = group[obj_name]
            grasps = obj_dataset["grasps"]["yumi_metal_spline"]
            obj_grasp_list.append((obj_name, len(grasps)))
        obj_list[dataset_name] = obj_grasp_list
    np.save(os.getcwd() + "/name_list.npy", obj_list)


def plot_rbbx(depth, gt_rbbx, proposal_rbbx=None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(depth, cmap=plt.cm.gray_r)
    
    for x, y, z, h, w, angle, tilt, metric_type, metric in gt_rbbx:
        k = np.sqrt(h ** 2 + w ** 2) / 2
        theta = np.deg2rad(np.rad2deg(np.arctan(h / w)) + angle)
        anc = (x - k * np.cos(theta), y - k * np.sin(theta))
        
        box = patches.Rectangle(anc, height=h, width=w, color="red", fill=False, angle=angle)
        ax.add_patch(box)
        
    if proposal_rbbx is not None:
        for x, y, z, h, w, angle, tilt, metric_type, metric in proposal_rbbx:
            k = np.sqrt(h ** 2 + w ** 2) / 2
            theta = np.deg2rad(np.rad2deg(np.arctan(h / w)) + angle)
            anc = (x - k * np.cos(theta), y - k * np.sin(theta))
            
            box = patches.Rectangle(anc, height=h, width=w, color="blue", fill=False, angle=angle)
            ax.add_patch(box)
            
    plt.show()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    # Backbone
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    # RRPN
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-60, -30, 0, 30, 60, 90]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]] # TODO [[32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS: [[0.5, 2.0]] # TODO [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 0, 1) # TODO (1, 1, 1, 1, 1)
    # ROIAligh + Downstream
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 3
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.NORM = ""    # "" (no norm), "GN", "SyncBN"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 0, 1) # TODO (10, 10, 5, 5, 1)
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = (1, 1, 1, 1) 
    
    cfg.MODEL.ROI_HEADS.NAME = "GraspRROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # Dataset
    cfg.DATASETS.TRAIN = ("grasp_train",)
    cfg.DATASETS.TEST = ("grasp_test",)
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    # Parameters for the model (default, may not use)
    cfg.MODEL.PIXEL_MEAN = [0.01, 0.01, 0.01]
    cfg.MODEL.PIXEL_STD = [0.0125, 0.0125, 0.0125]
    cfg.VIS_PERIOD = 0
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.DEVICE = "cuda"
    # batch size
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 5e-4
    # data loade workers
    cfg.DATALOADER.NUM_WORKERS = 0
    # eval period
    cfg.TEST.EVAL_PERIOD = 100000000
    # checkpoint period
    cfg.SOLVER.CHECKPOINT_PERIOD = 512

    # load weights
    cfg.OUTPUT_DIR = "./grasp/output"
    PATH = "model.pth" 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, PATH)

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def inpaint_depth_img(depth):
    img = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    is_obj = (gaussian(img, 1.0, preserve_range=True) > 0).astype(np.uint8)
    mask = np.logical_and(is_obj, img==0).astype(np.uint8)
    
    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale
    return img
        
def reproject_depth(depth, extrinsic):
    assert depth.shape == (424, 512), "Only support raw Kinect depth image"
    depth = o3d.geometry.Image(depth)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
    try:
        pcd = o3d.geometry.create_point_cloud_from_depth_image(depth, camera_intrinsic)  # open3d for ubuntu 16
    except:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera_intrinsic)  # open3d for ubuntu 18
    pc_o = np.array(pcd.points)
    rotM = np.eye(4)
    rotM[0:3, 0:3] = R.from_euler("x", -180, degrees=True).as_matrix()
    camera_pose = extrinsic @ rotM
    pc_w = camera_pose[0:3, 0:3] @ pc_o.T + camera_pose[0:3, 3].reshape((3, 1))
    
    xyz = pc_w.T
    # x_low, x_high = (-0.1, 0.4)
    # y_low, y_high = (-0.6, 0)
    # z_low, z_high = (0, 0.3)
    # x_ind = np.logical_and(xyz[:, 0] > x_low, xyz[:, 0] < x_high)
    # y_ind = np.logical_and(xyz[:, 1] > y_low, xyz[:, 1] < y_high)
    # z_ind = np.logical_and(xyz[:, 2] > z_low, xyz[:, 2] < z_high)
    # ind = np.logical_and(x_ind, y_ind)
    # ind = np.logical_and(ind, z_ind)
    # xyz = xyz[ind]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    bound_rad = max(np.linalg.norm(xyz - np.mean(xyz, axis=0), axis=1))
    new_camera_pose = np.copy(camera_pose)
    new_camera_pose[0:3, 3] = np.mean(xyz, axis=0) - camera_pose[0:3, 2] * 2 * bound_rad
    
    pc = pyrender.Mesh.from_points(pc_w.T)
    render_scene = pyrender.Scene()
    image_h, image_w = 424, 512
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    renderer = pyrender.OffscreenRenderer(viewport_height=image_h, viewport_width=image_w)
    intrinsic_matrix = camera_intrinsic.intrinsic_matrix
    fx, fy, cx, cy = (intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.025)
    render_scene.add(camera, pose=new_camera_pose @ rotM)
    render_scene.add(pc)
    depth = renderer.render(render_scene, flags)
    
    depth_inpaint = inpaint_depth_img(depth)
    
    # plt.subplot(121)
    # plt.imshow(depth, cmap="gray")
    # plt.subplot(122)
    # plt.imshow(depth_inpaint, cmap="gray")
    # plt.show()
    
    # # For Debug Vis
    # frame_w = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0,0,0])
    # frame_c = copy.deepcopy(frame_w).transform(camera_pose)
    # frame_cn = copy.deepcopy(frame_w).transform(new_camera_pose)
    # o3d.visualization.draw_geometries([pcd, frame_w, frame_c, frame_cn])
    return depth_inpaint, new_camera_pose
    