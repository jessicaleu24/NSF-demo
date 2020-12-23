import numpy as np
import time
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import os

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# import cv2
from cv_bridge import CvBridge

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Use this line to open the kinect
roslaunch kinect2_bridge kinect2_bridge.launch
One could use rviz to see the image

Refer this website to calibrate: 
https://github.com/code-iai/iai_kinect2/tree/master/kinect2_calibration#calibrating-the-kinect-one

NOTE
Use this to build cv to virtualenv python3: https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3
I directly installed to base python3.5, but should be usable for any python3 virtualenv
"""

class kinect():
    def __init__(self, extract_bg=False):
        rospy.init_node('kinect', anonymous=False)
        self.sub_depth = rospy.Subscriber("/kinect2/sd/image_depth", Image, self.depth_image_callback)
        # self.sub_pc = rospy.Subscriber("/kinect2/sd/points", PointCloud2, self.pc_callback) 
        
        self.background = np.zeros((224, 224))
        self.depth_image = None
        self.cropped_depth_image = None
        self.cropped_no_bg_depth_image = None
        self.height, self.width = None, None
        
        time.sleep(2)
        if extract_bg:
            self.extract_bg()
        else:
            self.background = np.load("./imgs/background.npy")
        
    def extract_bg(self):
        for _ in range(20):
            self.background += self.cropped_depth_image
            time.sleep(0.05)
        self.background /= 20
        np.save("./imgs/background.npy", self.background)
        
    def depth_image_callback(self, data):
        self.height, self.width = data.height, data.width
        cv_image = CvBridge().imgmsg_to_cv2(data, data.encoding)
        self.depth_image = np.array(cv_image, dtype=np.float) / 1000
        self.cropped_depth_image = self.depth_image[100: 100 + 224, 144: 144 + 224]

        self.cropped_no_bg_depth_image = self.cropped_depth_image - self.background
        
    def pc_callback(self, data):
        points_list = []
        for point in pc2.read_points(data, skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        pc = np.array(points_list)
    
    def save_img(self, img_dir, name="depth_img"):
        np.save(img_dir + name + ".npy", np.float32(self.cropped_no_bg_depth_image))

def depth2pc(depth):
    depth = o3d.geometry.Image(depth)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
    # width: 512, height: 424, fx: 365.456 mm, fy: 365.456 mm, cx: 254.878, cy: 205.395
    pcd = o3d.geometry.create_point_cloud_from_depth_image(depth, camera_intrinsic)
    pc = np.array(pcd.points)
    return pc

def pc2depth(pc, camera_pose):
    pass
        
if __name__ == "__main__":
    root = os.getcwd()
    img_dir = root + "/imgs/"
    name = "failure_depth_img"
    
    names = ['screw_driver', 'helmet', 'controller', 
             'tape_2', 'small_robot', 'clutter', 'bottle', 'scissor', 
             'toy', 'tape', 'large_robot', 'sprayer', 'plug']
    
    k = kinect(extract_bg=False)
    
    plt.ion()
    for i in range(30):
        plt.imshow(k.cropped_no_bg_depth_image, cmap=plt.cm.gray_r) # "gray"
        plt.draw()
        plt.pause(0.05)
        plt.clf()
    k.save_img(img_dir=img_dir, name=name)
    
    # for name in names:
    fig = plt.figure()
    d = np.load(img_dir + name + ".npy")
    d = -d
    d = d.clip(0, 1)
    plt.imshow(d, cmap=plt.cm.gray_r)
    plt.axis('off')
    # plt.show()
    plt.savefig("./imgs/{}.png".format(name))
    
"""
img, bg = cropped_imgs["clut_10"], cropped_imgs["background"]
diff = img - bg
segimg = np.zeros(img.shape)
# mask = np.logical_and((np.abs(diff) < 2), np.abs(bg) > 0)
mask = np.abs(diff) < 5
segimg[mask] = 1
plt.imshow(segimg, cmap="gray")
plt.show()
"""