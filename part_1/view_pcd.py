import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import numpy as np
from copy import deepcopy
import helper
import seaborn as sns

# path="./results/Q1/pcds/between_boat_and_car.pcd"
path="./results/Q4/global_frame_pointcloud.pcd"
master_pcd = o3d.io.read_point_cloud(path)
o3d.visualization.draw([master_pcd], show_ui=True)
# o3d.io.write_point_cloud("scene.pcd", master_pcd)
# o3d.io.write_point_cloud("scene_table.pcd", pcd_dict["table"])
