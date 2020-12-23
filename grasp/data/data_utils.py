import os
import sys
import copy
import random

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy import signal
from skimage import feature

from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.data.build import build_batch_data_loader
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_single_instance(depth, rbbxs):
    inst = Instances(depth.shape)
    
    rbox = rbbxs[:, [0, 1, 4, 3, 5]]
    gt_boxes = torch.tensor(rbox, dtype=torch.float32)
    inst.gt_boxes = RotatedBoxes(gt_boxes)
    inst.gt_boxes.clip(depth.shape)
    
    inst.gt_classes = torch.ones(rbbxs.shape[0], dtype=torch.int64)
    
    gt_tilts = rbbxs[:, 6].astype(np.float32)
    inst.gt_tilts = torch.from_numpy(np.deg2rad(gt_tilts))
    
    gt_z = rbbxs[:, 2].astype(np.float32) * 10
    inst.gt_z = torch.from_numpy(gt_z)
    
    gt_metric = rbbxs[:, 8].astype(np.float32)
    inst.gt_metric = torch.from_numpy(gt_metric)
    
    return inst
      
              
def get_instance(depth_pathes, rbbxses):
    # NOTE 
    # each rbbx = (x, y, z, h, w, angle, tilt, metric_type, metric)
    # each RotatedBoxes = (x_center, y_center, width, height, angle)
    batched_inputs = []
    for depth_path, rbbxs in zip(depth_pathes, rbbxses):
        inputs = get_single_instance(depth_path, rbbxs)
        batched_inputs.append(inputs)
    return batched_inputs


def create_model_input(img, inst=None):
    if len(img.shape) == 2:
        img = np.tile(img, (3, 1, 1))
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if inst is not None:
        return {"image": img, "instances": inst}
    else:
        return {"image": img}
        
        
def plot(inputs, random_select=False):
    depth = inputs["image"].cpu().numpy()[0,]
    rbbxs = inputs["instances"].gt_boxes.tensor.cpu().numpy()
    
    if random_select:
        ind_ = list(range(rbbxs.shape[0]))
        random.shuffle(ind_)
        rbbxs = rbbxs[ind_[0: 20]]
        rbbxs[:, 3] = rbbxs[:, 3] * 0 + 25


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(depth.astype(np.float32), cmap=plt.cm.gray_r)
    if rbbxs is not None:
        for x, y, w, h, angle in rbbxs:
            k = np.sqrt(h ** 2 + w ** 2) / 2
            theta = np.deg2rad(np.rad2deg(np.arctan(h / w)) + angle)
            anc = (x - k * np.cos(theta), y - k * np.sin(theta))
            
            box = patches.Rectangle(anc, height=h, width=w, color="red", fill=False, angle=angle)
            ax.add_patch(box)
    plt.show()
            
def plot_many(inputs_list, n_plot=200, text=False):
    n_input = len(inputs_list)
    fig = plt.figure()
    
    for i in range(n_input):
        inputs = inputs_list[i]
        depth = inputs["image"].cpu().numpy()[0,]
        rbbxs = inputs["instances"].gt_boxes.tensor.cpu().numpy()
        rbbxs[:, 3] = rbbxs[:, 3] * 0 + 15
        
        if text:
            tilts = inputs["instances"].gt_tilts
        
        ax = fig.add_subplot(1, n_input, i + 1)
        ax.imshow(depth.astype(np.float32), cmap=plt.cm.gray_r)
        ax.set_axis_off()
        if rbbxs is not None:
            for j, rbbx in enumerate(rbbxs, 0):
                if j == n_plot:
                    break
                
                x, y, w, h, angle = rbbx
                
                
                k = np.sqrt(h ** 2 + w ** 2) / 2
                theta = np.deg2rad(np.rad2deg(np.arctan(h / w)) + angle)
                anc = (x - k * np.cos(theta), y - k * np.sin(theta))
                
                box = patches.Rectangle(anc, height=h, width=w, color="red", fill=False, angle=angle)
                ax.add_patch(box)
                if text:
                    tilt = tilts[j].item()
                    ax.text(x, y, "tilt="+str(round(tilt,3)), fontsize=11)
    
    plt.show()
    
    
def generate_perlin_noise_2d(shape, res, tileable=(False, False)):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
    
    
class Rotate(object):
    def __init__(self, image_size=(512, 512), rotate_range=(0, 2*np.pi)):
        self.rotate_range = rotate_range
        self.h, self.w = image_size
        
        self.t_p, self.t_i = np.eye(3), np.eye(3)
        self.t_p[0:2, 2] = 0.5 * np.array([self.w, self.h])
        self.t_i[0:2, 2] = -0.5 * np.array([self.w, self.h])

    def _get_rot_matrix(self, angle):
        ca, sa = np.cos(angle), np.sin(angle)
        rot = np.array([[ca,  sa,  0.0],
                        [-sa, ca,  0.0],
                        [0.0, 0.0, 1.0]])
        return rot

    def __call__(self, depth_, inst_):
        depth, inst = copy.deepcopy(depth_), copy.deepcopy(inst_)
        
        rbbxs = inst.gt_boxes.tensor.cpu().numpy()
        rbbxs_new = None
        
        angle = np.random.uniform(self.rotate_range[0], self.rotate_range[1])
        rot = self._get_rot_matrix(angle)
        
        depth = rotate(depth, angle=np.rad2deg(angle), reshape=False) # angle is counter-clockwise rotated
        depth[depth < 1e-5] = 0
        
        for i, rbbx in enumerate(rbbxs, 0):
            pos_new = self.t_p @ rot @ self.t_i @ np.array([rbbx[0], rbbx[1], 1.0]).reshape(3, 1)
            x_new, y_new = pos_new[0][0], pos_new[1][0]
            
            theta_new = -angle + np.deg2rad(rbbx[4]) # in the RRPN equation, alpha is clock-wise rotated;
            while theta_new < -np.pi / 2:
                theta_new += np.pi
                inst.gt_tilts[i] = -inst.gt_tilts[i]
            while theta_new > np.pi / 2:
                theta_new -= np.pi
                inst.gt_tilts[i] = -inst.gt_tilts[i]
            assert theta_new > -np.pi / 2 and theta_new < np.pi / 2
            
            rbbx_new = np.array([x_new, y_new, rbbx[2], rbbx[3], np.rad2deg(theta_new)])
            if rbbxs_new is None:
                rbbxs_new = rbbx_new
            else:
                rbbxs_new = np.vstack((rbbxs_new, rbbx_new))
        rbbxs_new = rbbxs_new.reshape(-1, 5)
        
        gt_boxes = torch.tensor(rbbxs_new, dtype=torch.float32)
        inst.gt_boxes = RotatedBoxes(gt_boxes)
        
        return depth, inst
    
    
class TranslateZ(object):
    def __init__(self, delta_range=(-0.05, 0.05)):
        self.delta_range = delta_range

    def __call__(self, depth_, inst_):
        depth, inst = copy.deepcopy(depth_), copy.deepcopy(inst_)
        
        delta_z = np.random.uniform(self.delta_range[0], self.delta_range[1])
        depth += delta_z
        depth[depth < 1e-5] = 0
        
        inst.gt_z += delta_z
        
        return depth, inst
    
        
class Flip(object):
    def __init__(self, prob=0.3, image_size=(512, 512)):
        self.prob = prob
        self.h, self.w = image_size

    def __call__(self, depth_, inst_):
        depth, inst = copy.deepcopy(depth_), copy.deepcopy(inst_)
        
        if np.random.uniform(0, 1) < self.prob:
            depth = np.fliplr(depth)
            
            rbbxs = inst.gt_boxes.tensor.cpu().numpy()
            rbbxs[:, 0] = self.w - rbbxs[:, 0]
            rbbxs[:, 4] = -rbbxs[:, 4]
            rbbxs = rbbxs.reshape(-1, 5)
            gt_boxes = torch.tensor(rbbxs, dtype=torch.float32)
            inst.gt_boxes = RotatedBoxes(gt_boxes)
            inst.gt_tilts = -inst.gt_tilts
            
        return depth, inst
    

class AreaDropout(object):
    def __init__(self, prob=0.3, image_size=(512, 512), max_point=3, max_rad=40, grad_thred=0.1):
        self.prob = prob
        self.h, self.w = image_size
        self.max_point = max_point
        self.max_rad = max_rad
        self.grad_thred = grad_thred
        self.laplacian_kernel = np.array([[-1, -1, -1], 
                                          [-1,  8, -1], 
                                          [-1, -1, -1]])
        
    def _drop(self, depth, xs, ys):
        x, y = np.arange(0, depth.shape[1]), np.arange(0, depth.shape[0])
        bg_depth = depth[0,0]
        for cx, cy in zip(xs, ys):
            r = np.random.randint(30, self.max_rad)
            mask = (x[np.newaxis, :] - cx)**2 + (y[:, np.newaxis] - cy)**2 < r**2
            depth[mask] = bg_depth
        return depth
    
    def __call__(self, depth_):
        depth = copy.deepcopy(depth_)
        
        if np.random.uniform(0, 1) < self.prob:
            grad = np.absolute(signal.convolve2d(depth, self.laplacian_kernel, boundary='symm', mode='same'))
            grad_thred = np.max(grad) * 0.8
            grad_mask = (grad > grad_thred).astype(np.int)
            x_ind, y_ind = np.where(grad_mask > 0)
            n_drop_points = np.random.randint(1, self.max_point)
            drop_index = np.random.choice(x_ind.shape[0], n_drop_points, replace=False)
            drop_points_x, drop_points_y = x_ind[drop_index], y_ind[drop_index]
            depth = self._drop(depth, drop_points_x, drop_points_y)
            
        return depth
    

class DepthSim2Real(object):
    """
    refer to this paper
    https://arxiv.org/pdf/2003.01835.pdf
    """
    def __init__(self, image_size=(512, 512)):
        self.h, self.w = image_size
        self.laplacian_kernel = np.array([[-1, -1, -1], 
                                          [-1,  8, -1], 
                                          [-1, -1, -1]])
    
    def __call__(self, depth_):
        depth = copy.deepcopy(depth_)
        
        # Laplacian Gradient Mask (https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm)
        # may need a Gaussian smooth in real (https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm)
        grad = np.absolute(signal.convolve2d(depth, self.laplacian_kernel, boundary='symm', mode='same'))
        grad_thred = np.mean(grad)
        grad_mask = grad > grad_thred
        depth[grad_mask] = 0.0
        temp = copy.deepcopy(depth)
        
        # Canny Edge Mask
        edges = feature.canny(depth).astype(np.int)
        # Generate Perlin Noise (noise in -1 to 1)
        perlin_noise = generate_perlin_noise_2d((self.h, self.w), (8, 8))
        # Add pixel-wise perlin noise
        depth_range = np.max(depth) - np.min(depth)
        perlin_noise *= depth_range / (np.max(perlin_noise) - np.min(perlin_noise))
        perlin_noise += depth_range / 2
        depth += np.multiply(edges, perlin_noise)
         
        # kk = 3
        # plt.subplot(2,kk,1)
        # plt.imshow(depth_, cmap=plt.cm.gray_r)
        # plt.title("raw synthetic depth")
        # plt.subplot(2,kk,2)
        # plt.imshow(grad_mask, cmap=plt.cm.gray_r)
        # plt.title("laplacian grad mask")
        # plt.subplot(2,kk,3)
        # plt.imshow(temp, cmap=plt.cm.gray_r)
        # plt.title("depth impaint along grad mask")
        # plt.subplot(2,kk,6)
        # plt.imshow(edges, cmap=plt.cm.gray_r)
        # plt.title("canny edge mask")
        # plt.subplot(2,kk,5)
        # plt.imshow(perlin_noise, cmap=plt.cm.gray_r)
        # plt.title("generated perlin noise")
        # plt.subplot(2,kk,4)
        # plt.imshow(depth, cmap=plt.cm.gray_r)
        # plt.title("noised depth")
        # plt.show()
        return depth
