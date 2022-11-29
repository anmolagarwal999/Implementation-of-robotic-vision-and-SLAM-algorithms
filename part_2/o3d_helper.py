import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import seaborn as sns
from copy import deepcopy
import struct

def load_mesh(file_path):
    mesh=o3d.io.read_triangle_mesh(file_path)
    return mesh

def fetch_pcd_from_mesh(mesh_obj):
    pcd = mesh_obj.sample_points_uniformly(len(np.asarray(mesh_obj.vertices)))
    return pcd


def get_armadillo_mesh():
    armadillo_path = "./Armadillo.ply"
    mesh = o3d.io.read_triangle_mesh(armadillo_path)
    mesh.compute_vertex_normals()
    return mesh

def load_bin_file(file_path):
    with open(file_path, "rb") as binary_file:
    # Read the whole file at once
        data = binary_file.read()
        # print(data)
    return data


def convert_kitti_bin_to_pcd(bin_file_path):
    size_float = 4
    list_pcd = []
    with open(bin_file_path, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

def fetch_pose_matrix(line_str):
    tokens=line_str.split(" ");
    assert(len(tokens)==12)
    tokens=[float(x) for x in tokens]
    arr=np.array(tokens)
    arr=arr.reshape(3, 4)
    return arr

def apply_rot_on_pcl(pcd_obj, rot_mat):

    if rot_mat.shape==(3,3):

        # print("shape is ", rot_mat.shape)
        new_mesh=copy.deepcopy(pcd_obj)
        new_mesh.rotate(rot_mat)
    else:
        assert(rot_mat.shape==(3,4))
        # print("rot mat is ", rot_mat)
        r_mat=rot_mat[:3, :3]
        t_mat=rot_mat[:3, -1]

        # r_mat=r_mat.T
        # t_mat=-r_mat@t_mat
        # print("r mat is ", r_mat)
        # print("t mat is ", t_mat)
        new_mesh=copy.deepcopy(pcd_obj)
        new_mesh.rotate(r_mat)
        new_mesh=new_mesh.translate(tuple(t_mat))
    return new_mesh


def rotate_as_per_camera(camera_pose_matrix, pose_obj):

    rot_mat=camera_pose_matrix

    print("SHAPE IS ", rot_mat.shape)
    print("org pose is ", pose_obj)

    new_pose=deepcopy(pose_obj)

    r_mat=rot_mat[:3, :3]
    t_mat=rot_mat[:3, -1]
    
    # Inverse of a transformation matrix
    R_INV=np.zeros((4, 4))

    R_INV[-1, -1]=1
    R_INV[:3, :3] = r_mat.T
    R_INV[:3, -1] =  - r_mat.T@t_mat

    print("R INV IS ", R_INV)

    pts_old=np.asarray(pose_obj.points)

    pts= np.hstack((pts_old, np.ones((pts_old.shape[0], 1), dtype=pts_old.dtype)))

    new_pts=R_INV@pts.T
    new_pts=new_pts.T

    # print("NEW PTS IS ", new_pts)

    final_pts = np.array([new_pts[i]/new_pts[i][-1] for i in range(0, len(new_pts))])
    final_pts=final_pts[: ,:3]

    new_pose.points=o3d.utility.Vector3dVector(final_pts)
    return new_pose






if __name__ == "__main__":
    ROOT_PATH="./data/lidar_data"

    EXP_NAME="LiDAR"
    EXP_NAME="velodyne_5"

    odom_file="odometry.txt"
    odom_file="odometry_5.txt"
    all_files=sorted(os.listdir(os.path.join(ROOT_PATH, EXP_NAME)))

    STEP_VAL=50

    # list all files
    all_files=all_files[::STEP_VAL]

    
    print(*all_files, sep="\n")

    # load all files
    pcds=[convert_kitti_bin_to_pcd( os.path.join(ROOT_PATH,EXP_NAME,x)) for x in all_files]

    # for downsampling (uncomment if not needed)
    # pcds = [x.voxel_down_sample(voxel_size=0.1) for x in pcds]

    #####################
    # READ POSE MATRICES
    odom_path=os.path.join(ROOT_PATH, odom_file)
    with open(odom_path,'r') as fd:
        all_lines=fd.readlines()
    poses=[fetch_pose_matrix(x) for x in all_lines][::STEP_VAL][:len(all_files)]

 
    ################
    # to convert from lidar frame to camera frame
    ROT_MAT=np.array([ 
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    print("len pcds is ", len(pcds))

    # contains points in respective timestamp camera frames
    new_pcds=[apply_rot_on_pcl(x, ROT_MAT.T) for x in pcds]

    # contains points in the global frame
    new_pcds_2=[apply_rot_on_pcl(x, poses[idx]) for idx, x in enumerate(new_pcds)]
    print("len new_pcds is ", len(new_pcds))

    ##################
    NUM_COLORS=300
    COLORS_ARR=sns.color_palette('hls', n_colors=NUM_COLORS)

    #####
    # COLOR FOR EASE OF VIEWING
    pcds=list(map(lambda x: x[1].paint_uniform_color(COLORS_ARR[x[0]]) , enumerate(pcds)))
    new_pcds=list(map(lambda x: x[1].paint_uniform_color(COLORS_ARR[-x[0]]) , enumerate(new_pcds)))
    new_pcds_2=list(map(lambda x: x[1].paint_uniform_color(COLORS_ARR[(x[0]+NUM_COLORS+12)%NUM_COLORS]) , enumerate(new_pcds_2)))


    # COMMENT IF YOU WANT TO VIEW ALL POINT CLOUDS
    pcds=[]
    new_pcds=[]
    # new_pcds_2=[]

    #########

    ######
    # DRAW
    # o3d.visualization.draw([{"name": all_files[idx], "geometry":pcds[idx]} for idx in range(len(pcds))]+
    #                         [{"name": all_files[idx]+"ROT", "geometry":new_pcds[idx]} for idx in range(len(new_pcds))]+
    #                         [{"name": all_files[idx]+"BOTH", "geometry":new_pcds_2[idx]} for idx in range(len(new_pcds_2))]
                            
    #                         , show_ui=True)

    camera_idx=3
    new_pcds_2=list(map(lambda x: x[1].paint_uniform_color((1, 0 , 0)) , enumerate(new_pcds_2)))
    new_pcds_3=[rotate_as_per_camera(poses[camera_idx], x) for x in new_pcds_2]
    new_pcds_3=list(map(lambda x: x[1].paint_uniform_color((0, 1,0)) , enumerate(new_pcds_3)))

    o3d.visualization.draw(
                            [{"name": all_files[idx]+"BOTH", "geometry":new_pcds_2[idx]} for idx in range(len(new_pcds_2))]+
                            [{"name": all_files[idx]+"ORG", "geometry":new_pcds_3[idx]} for idx in range(len(new_pcds_3))]
                            
                            , show_ui=True)









    
