import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

path='saved_pcd_data/pcd_1.pcd'
master_pcd = o3d.io.read_point_cloud(path)

path='saved_pcd_data/pcd_2.pcd'
master_pcd_2 = o3d.io.read_point_cloud(path)
o3d.visualization.draw([master_pcd, master_pcd_2], show_ui=True)
# mesh=o3d.io.read_triangle_mesh(path)
