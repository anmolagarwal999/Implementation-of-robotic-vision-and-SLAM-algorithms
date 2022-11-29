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


# object_names=["boat", "car", "laptop", "plane","table", "trashcan" ]
object_names={"table":1, "boat":30,"laptop":40,"trashcan":20,"plane":50,"car":20 }

NUM_COLORS=8
COLORS_ARR=sns.color_palette('husl', n_colors=NUM_COLORS)
bb=[]

pcd_dict=dict()

for idx, curr_obj in enumerate(object_names):
    print("curr obj is ", curr_obj)
    PCL_PATH=os.path.join("./data/Q1", f"{curr_obj}.obj")
    mesh = helper.load_mesh(PCL_PATH)
    print(f'Center of mesh is: {mesh.get_center()}')
    print("bounds are : ", mesh.get_min_bound(), mesh.get_max_bound())

    #print(mesh)
    curr_pcl=helper.fetch_pcd_from_mesh(mesh)
    pcd_dict[curr_obj] = curr_pcl

    displacement=pcd_dict[curr_obj].get_center()

    if curr_obj=="table":
        displacement[2]=0


    pcd_dict[curr_obj]=deepcopy(pcd_dict[curr_obj]).translate(tuple([-x for x in displacement]))
    pcd_dict[curr_obj].scale(object_names[curr_obj], center=pcd_dict[curr_obj].get_center())

    pcd_dict[curr_obj].paint_uniform_color(COLORS_ARR[idx])


    print("############")


'''TABLE'''
curr_pcd=pcd_dict['table']
curr_pcd.rotate(curr_pcd.get_rotation_matrix_from_axis_angle([0, 0,-np.pi/2]), center=(0,0,0))

curr_pcd.points=o3d.utility.Vector3dVector(np.asarray(curr_pcd.points) * np.array([2.4, 2, 1.]) )

height_of_table = curr_pcd.get_max_bound()[2]






'''BOAT'''
curr_pcd=pcd_dict['boat']
curr_pcd.rotate(curr_pcd.get_rotation_matrix_from_axis_angle([np.pi/2, 0,0]), center=(0,0,0))

# make boat on table
x_min, y_min, z_min = curr_pcd.get_min_bound()
diff= height_of_table  - z_min
curr_pcd.translate((-40, 0, diff))




'''PLANE'''
curr_pcd=pcd_dict['plane']
curr_pcd.rotate(curr_pcd.get_rotation_matrix_from_axis_angle([np.pi/2, 0,0]), center=(0,0,0))
x_min, y_min, z_min = curr_pcd.get_min_bound()
diff= height_of_table  - z_min
curr_pcd.translate((40, -20, diff))



'''CAR'''
curr_pcd=pcd_dict['car']
curr_pcd.rotate(curr_pcd.get_rotation_matrix_from_axis_angle([np.pi/2, 0,0]), center=(0,0,0))
x_min, y_min, z_min = curr_pcd.get_min_bound()
diff= height_of_table  - z_min
curr_pcd.translate((-30, 10, diff))



'''Laptop'''
curr_pcd=pcd_dict['laptop']

curr_pcd.rotate(curr_pcd.get_rotation_matrix_from_axis_angle([0, 0,np.pi]), center=(0,0,0))
curr_pcd.rotate(curr_pcd.get_rotation_matrix_from_axis_angle([-np.pi/2, 0,0]), center=(0,0,0))

x_min, y_min, z_min = curr_pcd.get_min_bound()
diff= height_of_table  - z_min

curr_pcd.translate((0, 0, diff))



'''Trashcan'''
curr_pcd=pcd_dict['trashcan']
curr_pcd.rotate(curr_pcd.get_rotation_matrix_from_axis_angle([np.pi/2, 0,0]), center=(0,0,0))
x_min, y_min, z_min = curr_pcd.get_min_bound()
diff= height_of_table  - z_min
curr_pcd.translate((40, 20, diff))






#######################################################

def populate_bounding_boxes():
    for idx, curr_obj in enumerate(object_names):
        

        axis_aligned_bounding_box = pcd_dict[curr_obj].get_axis_aligned_bounding_box()
        axis_aligned_bounding_box.color = (1, 0, 0)
        bb.append(axis_aligned_bounding_box)
        print("############")

populate_bounding_boxes()


o3d.visualization.draw([{"name": x, "geometry":pcd_dict[x]} for x in pcd_dict]+bb, show_ui=True)


master_pcd=o3d.geometry.PointCloud()
master_pcd.points = o3d.utility.Vector3dVector(
                                        np.concatenate(
                                            [np.asarray(x.points) for x in pcd_dict.values()]
                                        )
                                            )           


o3d.visualization.draw([master_pcd], show_ui=True)
o3d.io.write_point_cloud("scene.pcd", master_pcd)
# o3d.io.write_point_cloud("scene_table.pcd", pcd_dict["table"])
