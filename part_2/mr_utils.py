import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math
import helpers.pcd_from_rgbd
from sklearn.neighbors import KDTree
import seaborn as sns
# make PCLs and visualize both in open3d and matlab

# make a function:
# 1) make sure that sizes are same and compatible
# 2) Align the goves PCLs and show Frobenius NORM
# 3) Show final alignment
# 4) Give math proof

def read_data_files(file_path:str) -> np.ndarray:

    with open(file_path,'r') as fd:
        lines=[x.strip() for x in fd.readlines()]
        lines=lines[2:]
        pts=[[float(y) for y in x.split(" ")] for x in lines]
    pts=np.array(pts)
    print("Shape of pts is ", pts.shape)
    return pts

def plot_3d_points(pts:np.ndarray, wanted_color='red', axis_sent=None):
    if axis_sent is None:
        ax = plt.axes(projection='3d')
    else:
        ax=axis_sent
    xdata = pts[:,0]
    ydata = pts[:,1]
    zdata = pts[:,2]
    # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter3D(xdata, ydata, zdata, c=wanted_color)
    # plt.show()
    return ax


def find_transformation_matrix(pts_1:np.ndarray, pts_2:np.ndarray):

    """Find rotation and translation matrix for aligning 2 point clouds given that the correspondes are known

    Finds (R, t) which minimize sum of squares of ( R@pts_1 + t )-pts_2

    Args:
        pts_1: point cloud 1
        pts_2: point cloud 2

    Returns:
        #TODO
    """

    #TODO: Handle case where the RANK of the matrix is NOT sufficient
    pts_1 = deepcopy(pts_1)
    pts_2 = deepcopy(pts_2)

    n1, dim1 = pts_1.shape
    n2, dim2 = pts_2.shape

    assert(n1==n2 and dim1==dim2 and dim1==3)


    # here, we are not considering any weights to align the points. Hence, information matrix = identity

    # Compute the weighted centroids of both point sets
    com_1 = np.mean(pts_1, axis=0)
    com_2 = np.mean(pts_2, axis=0)

    # print("com_1 is ", com_1, "\n shape is ", com_1.shape)
    # print("com_2 is ", com_2, "\n shape is ", com_1.shape)

    # Compute the centered vectors
    pts_1 -= com_1
    pts_2 -= com_2



    # print("here")

    # print("first row of pts_0 is ", pts_1[0,:])



    # Compute the d X d covariance matrix
    pts_1 = pts_1.T # shape = d x n1
    pts_2 = pts_2.T # shape = d x n2

    # (d x n1) X (n2 x d) = d x d
    S = pts_1@pts_2.T

    # Shape of U = d x d
    # Shape of D = d x d
    # Shape of V_t = d x d
    U,D, V_t = np.linalg.svd(S)

    # print("Shape of d is ", D.shape)
    # print("d is ", D)
    # print("np.diag(d) is ", np.diag(D))
    V=V_t.T
    S_back = U@np.diag(D)@V.T

    # print(np.std(S), np.std(S_back), np.std(S-S_back),np.mean(S-S_back))
    # print('Is S close to S_back?', np.isclose(S, S_back).all())

    centre_mat = np.identity(dim1)

    det_val = np.linalg.det(V@U.T)
    # print("det val is ", det_val)
    assert(math.isclose(1, det_val) or math.isclose(-1, det_val))

    #TODO: Handle the case TO NORMALIZE THE DETERMINANT
    centre_mat[-1,-1] = det_val

    R = V@centre_mat@U.T

    trans = com_2 - R@com_1

    # print("simple com diff is ", com_2-com_1)

    # print("T is ", trans)

    return (R,trans)


def convert_depth_data_to_pcd_data(rgb_img_path, depth_img_path, pcd_save_path):
    """
    Retrieves point cloud using rgb and depth images and saves it as a point cloud file

    Args:
        self-explanatory

    Returns:
        None
    """

if False:
    rgb_img_paths = ["data/1/2/rgbd1/533.jpg","data/1/2/rgbd2/729.jpg"]
    depth_img_paths = ["data/1/2/rgbd1/533.png","data/1/2/rgbd2/729.png" ]
    pcd_save_paths=["./saved_pcd_data/pcd_1.pcd","./saved_pcd_data/pcd_2.pcd" ]


    # for a N x N pixel image, we get a point for each of the pixels.

    for rgb_path, depth_path, pcd_path in zip(rgb_img_paths, depth_img_paths, pcd_save_paths):
        pcd, srcPxs = helpers.pcd_from_rgbd.getPointCloud()









class icp_pcl:

    def __init__(self,pcl_path_1, pcl_path_2, num_itrs=100, initial_rot = None, initial_trans = None, downsample=False):

        # save paths
        self.pcl_path_1=pcl_path_1
        self.pcl_path_2=pcl_path_2

        
        # save actual pcls
        self.pcd1 = o3d.io.read_point_cloud(self.pcl_path_1)
        self.pcd2 = o3d.io.read_point_cloud(self.pcl_path_2)

        if downsample:
            VOX_SIZE = 250
            self.pcd1 = self.pcd1.voxel_down_sample(voxel_size=VOX_SIZE)
            self.pcd2 = self.pcd2.voxel_down_sample(voxel_size=VOX_SIZE)

        print(f"pcd 1 has {len(self.pcd1.points)} points")
        print(f"pcd 2 has {len(self.pcd2.points)} points")

        #####################################
        ### COlor PCLS:
        NUM_COLORS=10
        COLORS_ARR=sns.color_palette('hls', n_colors=NUM_COLORS)
        self.pcd1.paint_uniform_color(COLORS_ARR[0])
        self.pcd2.paint_uniform_color(COLORS_ARR[1])

        ##################################

        self.num_itrs = num_itrs

        self.N_dims = np.asarray(self.pcd1.points).shape[1]

        


        # correspondence not known, hence, find a way to find nearest enughbour
        # o3d.visualization.draw_geometries([self.pcd1, self.pcd2])


        # find coordinate frames
        self.cf1 = self.fetch_cf(self.pcd1)
        self.cf2 = self.fetch_cf(self.pcd2)
        #TODO: remove this
        # self.cf2.rotate(self.cf2.get_rotation_matrix_from_axis_angle([0, 0,-np.pi/2]))
        

        print("Before initial rotation and translation")
        o3d.visualization.draw_geometries([self.pcd1])
        o3d.visualization.draw_geometries([self.pcd2])
        # o3d.visualization.draw_geometries([self.cf1, self.cf2])
        # o3d.visualization.draw_geometries([self.pcd1, self.pcd2, self.cf1, self.cf2])
        # o3d.visualization.draw_geometries([self.pcd1, self.pcd2, self.cf1, self.cf2])



        print("After initial rotation and translation")

        if initial_rot is None:
            initial_rot = np.eye(self.N_dims)
        if initial_trans is None:
            initial_trans = np.zeros(self.N_dims)

        pts3 = self.apply_rot_and_trans(np.asarray(self.pcd1.points), initial_rot, initial_trans)
        self.pcd1.points = o3d.utility.Vector3dVector(pts3)
        pts3 = self.apply_rot_and_trans(np.asarray(self.cf1.vertices), initial_rot, initial_trans)
        self.cf1.vertices = o3d.utility.Vector3dVector(pts3)

        o3d.visualization.draw_geometries([self.pcd1, self.pcd2, self.cf1, self.cf2])



        self.kd_tree = KDTree(np.asarray(self.pcd2.points), leaf_size=2) 
        self.kd_tree_2 = KDTree(np.asarray(self.pcd1.points), leaf_size=2) 

        self.target_mat = np.zeros((len(self.pcd1.points), 3))
        self.target_mat_2 = np.zeros((len(self.pcd2.points), 3))

        self.frob_history =[]
        self.chamfer_history =[]

        # self.final_rot = np.eye(self.N_dims)
        # self.final_trans= np.zeros(self.N_dims)
        self.final_transf = np.eye(self.N_dims+1)

        self.cal_frob_norm = False
        if len(self.pcd1.points) == len(self.pcd2.points):
            self.cal_frob_norm = True




        


    def update_correspondences(self):

        pts1 = np.asarray(self.pcd1.points)
        pts2 = np.asarray(self.pcd2.points)
        
        for idx, curr_point in enumerate(pts1):
            nearest_pt = None
            min_dist = None
      
            least_dist, best_kd_ind = self.kd_tree.query(curr_point.reshape(1,-1), k=1)
            best_kd_ind = best_kd_ind[0][0]

            nearest_pt = pts2[best_kd_ind]

            self.target_mat[idx,:] = nearest_pt

        for idx, curr_point in enumerate(pts2):
            nearest_pt = None
            min_dist = None
      
            least_dist, best_kd_ind = self.kd_tree_2.query(curr_point.reshape(1,-1), k=1)
            best_kd_ind = best_kd_ind[0][0]

            nearest_pt = pts1[best_kd_ind]

            self.target_mat_2[idx,:] = nearest_pt

    def apply_rot_and_trans(self, np1, R, t):
        pts3= R@np.asarray(np1).T
        pts3=pts3.T
        pts3+=t
        return pts3

    def start_icp(self):

        TOLERANCE = 1e-4
        frob_norm = None

        STEP_VAL = 30
        for curr_itr_num in range(1, self.num_itrs+1):
            print("Current itr number is ", curr_itr_num)


            # we need to find correspondence first
            self.update_correspondences()

            R,t= find_transformation_matrix(np.asarray(self.pcd1.points), self.target_mat)

            curr_transf = np.eye(self.N_dims+1)
            curr_transf[:self.N_dims, :self.N_dims] = R
            curr_transf[:self.N_dims, self.N_dims] = t.T
            self.final_transf = curr_transf@self.final_transf
            # self.final_trans = R@self.final_trans.T
            # self.final_trans = self.final_trans.T
            # self.final_trans +=t
            # self.final_rot = R@self.final_rot

            pts3 = self.apply_rot_and_trans(np.asarray(self.pcd1.points), R, t)
            self.pcd1.points = o3d.utility.Vector3dVector(pts3)

            pts3 = self.apply_rot_and_trans(np.asarray(self.cf1.vertices), R, t)
            self.cf1.vertices = o3d.utility.Vector3dVector(pts3)
            # sleep(3)

            
            
            
            term_1 = np.linalg.norm(np.asarray(self.pcd1.points) - self.target_mat, 'fro')
            term_2 = np.linalg.norm(np.asarray(self.pcd2.points) - self.target_mat_2, 'fro')
            chamfer_dist = term_1 + term_2

            prev_frob_norm = frob_norm

            if self.cal_frob_norm:
                frob_norm = np.linalg.norm(np.asarray(self.pcd1.points) - np.asarray(self.pcd2.points), 'fro')
            else:
                frob_norm = term_1

            print("Frob norm is ", frob_norm)
            print("Chamfer is ", chamfer_dist)
            print("term 1 is ", term_1)
            print("term 2 is ", term_2)
            self.frob_history.append(frob_norm)
            self.chamfer_history.append(chamfer_dist)

            if prev_frob_norm!=None:
                if prev_frob_norm<frob_norm:
                    print("Frobenius norm has increased, breaking")
                    break
           
                if(math.fabs(prev_frob_norm-frob_norm)<TOLERANCE):
                    print("Tolerance value not crossed, breaking loop")
                    break
            print("###########")
            if curr_itr_num%STEP_VAL==0:
                o3d.visualization.draw_geometries([self.pcd1, self.pcd2, self.cf1, self.cf2])


    def fetch_cf(self, pcl):
        self.cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.cf.scale(0.2, center=self.cf.get_center())

        self.cf.translate(tuple(pcl.get_center() - self.cf.get_center()))
        return self.cf
    

  
    
 






























# if __name__=="__main__":
# # if True:
#     p1="./data/1/1/points1.txt"
#     p2="./data/1/1/points2.txt"

#     np1=read_data_files(p1)
#     np2=read_data_files(p2)
#     ax1=plot_3d_points(np1)
#     ax2=plot_3d_points(np2,wanted_color='blue', axis_sent=ax1)
#     # plt.show()

#     R,t=find_transformation_matrix(np1, np2)

#     pts3= R@np1.T
#     pts3=pts3.T
#     print("Shape of pts 3 is ", pts3.shape)
#     print("Shape of T is ", t.shape)
#     pts3+=t
#     ax3=plot_3d_points(pts3,wanted_color='green', axis_sent=ax2)
#     print("done")
#     plt.show()

#     #TODO: Add FROBENIUS NORM




    
