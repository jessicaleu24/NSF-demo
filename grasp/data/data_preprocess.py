import os
import numpy as np
import torch
from tqdm import tqdm
import random
from shutil import copyfile, move

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from detectron2.structures import RotatedBoxes
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

def pairwise_iou(boxes1, boxes2):
    # (x_center, y_center, width, height, angle)
    boxes1, boxes2 = torch.from_numpy(boxes1).to(dtype=torch.float32), torch.from_numpy(boxes2).to(dtype=torch.float32)
    iou = pairwise_iou_rotated(boxes1, boxes2)
    return iou.numpy()

def plot(depth, rotated_bbx):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(depth) #, cmap=plt.cm.gray_r)
    
    for x, y, z, h, w, angle, tilt, metric_type, metric in rotated_bbx:
        k = np.sqrt(h ** 2 + w ** 2) / 2
        theta = np.deg2rad(np.rad2deg(np.arctan(h / w)) + angle)
        anc = (x - k * np.cos(theta), y - k * np.sin(theta))
        
        box = patches.Rectangle(anc, height=h, width=w, color="red", fill=False, angle=angle)
        ax.add_patch(box)
    plt.show()
        
def prune_bbx(rbbx, tilt_range=(-30, 30), iou_thre=0.4):
    # boxes with tilt in [-30, 30] are considered
    # box which has overlap > 0.4 and tilt angle < 15 degree should be pruned
    tilt = rbbx[:, 6]
    ind = np.where(np.logical_and(tilt >= tilt_range[0], tilt <= tilt_range[1]))[0]

    rbox = rbbx[ind][:, [0, 1, 4, 3, 5]]
    iou = pairwise_iou(rbox, rbox)
    
    while np.where(iou > iou_thre)[0].shape[0] != 0:
        x, y = np.where(iou > iou_thre)
        if np.all(x == y):
            break
        for i, j in zip(x, y):
            if i == j:
                continue
            else:
                ind = np.delete(ind, i)
                rbox = rbbx[ind][:, [0, 1, 4, 3, 5]]
                iou = pairwise_iou(rbox, rbox)
                break
    return rbbx[ind]

def split_dataset(root, seed=12138):
    print("Start splitting dataset ... ")
    random.seed(seed)
    
    img_path = root + "images/"
    image_filenames = os.listdir(img_path)
    
    random.shuffle(image_filenames)
    num_train = int(len(image_filenames) * 0.9)
    num_test = len(image_filenames) - num_train
    
    train_dataset = image_filenames[0: num_train]
    test_dataset = image_filenames[num_train:]
    dataset = {"train": train_dataset, "test": test_dataset}
    
    for cat in ["train", "test"]:
        folder_dir = img_path + cat + "/"
        if not os.path.exists(folder_dir):
            os.mkdir(folder_dir)
        for filename in dataset[cat]:
            src = img_path + filename
            dst = folder_dir + filename
            # copyfile(src, dst)
            # os.remove(src)
            move(src, dst)

def main_prune_boxes(root, iou_thre=0.4):
    print("Start prunning rotated bounding boxes ... ")
    bx_path = root + "rbbxs/"
    # img_path = root + "images/"
    pruned_path = root + "pruned_rbbxs/"
    filenames = os.listdir(bx_path)
    
    if not os.path.exists(pruned_path):
        os.mkdir(pruned_path)
    
    pbar = tqdm(total=len(filenames))
    zero_count = 0
    
    for filename in filenames:
        full_path = bx_path + filename
        rbbx = np.load(full_path, allow_pickle=True) # (x, y, z, h, w, angle, tilt, metric_type, metric)
        pruned_rbbx = prune_bbx(rbbx, iou_thre=iou_thre)
        # depth = np.load(img_path + filename).astype(np.float32)
        # plot(depth, pruned_rbbx)
        if pruned_rbbx.shape[0] == 0:
            pbar.update(1)
            zero_count += 1
            continue
        np.save(pruned_path + filename, pruned_rbbx)
        pbar.update(1)
    
    pbar.close()
    print(zero_count)
    
def remove_zero_box_images(root):
    print("Start removing images with zero rotated bounding box ... ")
    pruned_path = root + "pruned_rbbxs/"
    bbxs_names = os.listdir(pruned_path)
    img_path = root + "images/"
    zero_count = 0
    
    for cat in ["train", "test"]:
        folder_dir = img_path + cat + "/"
        filenames = os.listdir(folder_dir)
        for filename in filenames:
            if filename not in bbxs_names:
                os.remove(folder_dir + filename)
                zero_count += 1
    print(zero_count)
    
def check_dataset(root):
    print("Start checking the dataset ... ")
    pruned_path = root + "pruned_rbbxs/"
    img_path = root + "images/"
    
    for cat in ["train", "test"]:
        folder_dir = img_path + cat + "/"
        filenames = os.listdir(folder_dir)
        for filename in filenames:
            bbx = np.load(pruned_path + filename)
            assert bbx.shape[0] > 0
            assert np.all(bbx[:, 6] > -30) and np.all(bbx[:, 6] < 30)
    print("All data checked!")
    
if __name__ == "__main__":
    cwd = os.getcwd()
    file_path = cwd + "/data/"
    root = file_path + "training_data/"
    
    main_prune_boxes(root, iou_thre=100) # iou_thre > 1 means no prune on pariwise iou
    split_dataset(root)
    remove_zero_box_images(root)
    check_dataset(root)
