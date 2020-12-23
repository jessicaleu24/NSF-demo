import numpy as np

import os, sys, random, copy
import torch
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.data.build import build_detection_train_loader, build_batch_data_loader

from data.cgrcnn_dataset_as_torch_loader import cgrcnn_dataset_torch


def get_grasp_dicts(root, mode="train"):
    img_path = root + "images/{}/".format(mode)
    bbx_path = root + "pruned_rbbxs/"
    
    image_filenames = os.listdir(img_path)
    dataset_dicts = []
    
    for idx, filename in enumerate(image_filenames):
        record = {}
        record["file_name"] = img_path + filename
        height, width = np.load(record["file_name"]).astype(np.float32).shape
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        rbbxs = np.load(bbx_path + filename, allow_pickle=True)
        grasps = []
        for rbbx in rbbxs:
            rbox = rbbx[[0, 1, 4, 3, 5]]
            grasp = {
                "bbox": rbox.tolist(),
                "bbox_mode": BoxMode.XYWHA_ABS,
                "category_id": 1,
                "metric": rbbx[8],
                "tilt": rbbx[6],
                "z": rbbx[2],
            }
            grasps.append(grasp)
            
        record["annotations"] = grasps
        dataset_dicts.append(record)
    return dataset_dicts

def cgrcnn_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    depth = np.load(dataset_dict["file_name"]).astype(np.float32)
    inst = Instances(depth.shape)
    depth = torch.from_numpy(np.tile(depth, (3, 1, 1)))
    
    grasps = dataset_dict["annotations"]
    gt_boxes, gt_tilts, gt_z, gt_metric = None, None, None, None
    for grasp in grasps:
        box, z, tilt, metric = np.array(grasp["bbox"]), np.array(grasp["z"]), np.array(grasp["tilt"]), np.array(grasp["metric"])
        if gt_boxes is None:
            gt_boxes, gt_tilts, gt_z, gt_metric = box, tilt, z, metric
        else:
            gt_boxes = np.vstack((gt_boxes, box))
            gt_tilts = np.hstack((gt_tilts, tilt))
            gt_z = np.hstack((gt_z, z))
            gt_metric = np.hstack((gt_metric, metric))
            
    inst.gt_boxes = RotatedBoxes(torch.from_numpy(gt_boxes.astype(np.float32).reshape(-1, 5)))
    # inst.gt_tilts = torch.from_numpy(gt_tilts.astype(np.float32))
    # inst.gt_z = torch.from_numpy(gt_z.astype(np.float32))
    # inst.gt_metric = torch.from_numpy(gt_metric.astype(np.float32))
    inst.gt_classes = torch.ones(gt_boxes.shape[0], dtype=torch.int64)
    
    return {"image": depth, "instances": inst}
    
    
def build_as_detection_loader(cfg, root):
    # d = "train"
    # dataset_dicts = get_grasp_dicts(root, mode=d)
    # inputs = cgrcnn_mapper(dataset_dicts[0])

    for d in ["train", "test"]:
        DatasetCatalog.register("grasp_" + d, lambda d=d: get_grasp_dicts(root, mode=d))
        MetadataCatalog.get("grasp_" + d).set(thing_classes=["grasps"])
    grasp_metadata = MetadataCatalog.get("grasp_train")
    
    trainloader = build_detection_train_loader(cfg, mapper=cgrcnn_mapper)
    return trainloader

def build_as_torch_loader(root, mode="train", batch_size=16, num_workers=0):
    if mode == "train":
        train_dataset = cgrcnn_dataset_torch(root, mode=mode)
        train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False, num_samples=None, generator=None)
        trainloader = build_batch_data_loader(dataset=train_dataset, sampler=train_sampler, total_batch_size=batch_size, aspect_ratio_grouping=False, num_workers=num_workers)
        return trainloader
    elif mode == "test":
        test_dataset = cgrcnn_dataset_torch(root, mode=mode)
        test_sampler = torch.utils.data.RandomSampler(test_dataset, replacement=False, num_samples=None, generator=None)
        testloader = build_batch_data_loader(dataset=test_dataset, sampler=test_sampler, total_batch_size=batch_size, aspect_ratio_grouping=False, num_workers=num_workers)
        return testloader
    