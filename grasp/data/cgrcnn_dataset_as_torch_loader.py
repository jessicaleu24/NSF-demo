import os
import sys

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.data.build import build_batch_data_loader
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

try:
    from data.data_utils import *
except:
    from data_utils import *
"""
Only good grasp boxes are labeled in the depth image, all bad grasps are ignored.
Only use top 20% grasps
"""
    
class cgrcnn_dataset_torch(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", image_size=(224, 224), vis=False):
        self.images_path = root + "images/{}/".format(mode)
        self.rbbxs_path = root + "pruned_rbbxs/"
        self.sample_name = os.listdir(self.images_path)
        self.vis = vis
        self.image_size = image_size
        
        self.rotate = Rotate(image_size=image_size, rotate_range=(0, 2*np.pi))
        self.trans_z = TranslateZ(delta_range=(-0.01, 0.01))
        self.flip = Flip(prob=0.5, image_size=image_size)
        self.drop = AreaDropout(prob=1, image_size=image_size, max_point=2, max_rad=50, grad_thred=0.1)
        self.sim2real = DepthSim2Real(image_size=image_size)
        self.sim2_real_prob = 0.5

        self.mode = mode
        
        # image from Kinect V2 is 424 x 512, pad to a square
        self.zero_pad = np.zeros((44, 512), dtype=np.float32) 
        
    def __getitem__(self, index):
        sample = self.sample_name[index]
        depth_path = self.images_path + sample
        rbbx_path = self.rbbxs_path + sample
        rbbxs = np.load(rbbx_path, allow_pickle=True)
        depth = np.load(depth_path).astype(np.float32)
        
        depth = np.vstack((self.zero_pad, depth, self.zero_pad))
        
        tf_depth = Image.fromarray(depth)
        tf_depth = tf_depth.resize(self.image_size)
        depth = np.asarray(tf_depth)
        # transform the bounding box to square and resize to 224
        rbbxs[:, 1] += 44
        
        rbbxs[:, [0, 1, 3, 4]] *= 224 / 512
        
        inst = get_single_instance(depth, rbbxs)
        
        if self.mode == "train":
            # argument samples
            transformed_depth, transformed_inst = self.rotate(depth, inst)
            transformed_depth, transformed_inst = self.flip(transformed_depth, transformed_inst)
            transformed_depth, transformed_inst = self.trans_z(transformed_depth, transformed_inst)

            transformed_depth_1 = self.drop(transformed_depth)
            if np.random.uniform(0, 1) < self.sim2_real_prob:
                transformed_depth_1 = self.sim2real(transformed_depth_1)
            transformed_inputs_1 = create_model_input(transformed_depth_1, transformed_inst)
            
            transformed_depth_2 = self.drop(transformed_depth)
            if np.random.uniform(0, 1) < self.sim2_real_prob:
                transformed_depth_2 = self.sim2real(transformed_depth_2)
            transformed_inputs_2 = create_model_input(transformed_depth_2, transformed_inst)
            
            if self.vis:
                inputs = create_model_input(depth, inst)
                plot_many([inputs, transformed_inputs_1, transformed_inputs_2], n_plot=3, text=True)
        else:
            transformed_inputs_1 = create_model_input(depth, inst)
            transformed_inputs_2 = copy.deepcopy(transformed_inputs_1)
            
        return (transformed_inputs_1, transformed_inputs_2)
    
    def __len__(self):
        return len(self.sample_name)
        
        
if __name__ == "__main__":
    batch_size = 4
    np.random.seed(12138)
    
    cwd = os.getcwd()
    root = cwd + "/data/training_data/"
    # root = "/media/fanuc/35B4BBB7636A09EB/Xinghao/ContrastiveGrasp/data/training_data/"

    train_dataset = cgrcnn_dataset_torch(root, mode="train")
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False, num_samples=None, generator=None)
    trainloader = build_batch_data_loader(dataset=train_dataset, sampler=train_sampler, total_batch_size=batch_size, aspect_ratio_grouping=False, num_workers=8)
    img = []

    for i, inputs in enumerate(trainloader, 0):
        img += [input_[0]["image"][0] for input_ in inputs]
        if i == 250:
            count = len(img)
            img_sum = torch.stack(img, dim=0).sum(dim=0) / count
            mean, std = torch.mean(img_sum), torch.std(img_sum)
    print("OK")