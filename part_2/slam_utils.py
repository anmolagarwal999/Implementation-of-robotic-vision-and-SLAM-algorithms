import math
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import copy
import matplotlib.patches as mpatches
import jax
import jax.numpy as jnp
# plt.rcParams["figure.figsize"] = (15,15)

def parse_files(file_name):

    with open(file_name, 'r') as fd:
        all_lines=fd.readlines()
    
    vertices = []
    edges= []

    for curr_line in all_lines:
        if "VERTEX_SE2" in curr_line:
            # print("yep")
            (ver, ind, x, y, theta) = curr_line.split()
            obj = dict()
            obj['vertex_id'] = int(ind)
            obj['x_c'] = float(x)
            obj['y_c'] = float(y)
            obj['theta_c'] = float(theta.rstrip('\n'))
            vertices.append(obj)
        
        if "EDGE_SE2" in curr_line:
            (edge, ind1, ind2, dx, dy, dtheta, x_conf, _, _, y_conf, _, theta_conf) = curr_line.split()
            obj = dict()
            obj['v1'] = int(ind1)
            obj['v2'] = int(ind2)
            obj['dx'] = float(dx)
            obj['dy'] = float(dy)
            obj['dtheta'] = float(dtheta)
            obj['x_conf'] = float(x_conf)
            obj['y_conf'] = float(y_conf)
            obj['theta_conf'] = float(theta_conf)
          
            edges.append(obj)
    return vertices, edges



def draw_trajectory(X, Y, THETA, color='red', label="", title=""):
    ax = plt.subplot(111) # equivalent to 1, 1, 1

    
    ax.plot(X, Y, 'ro') # r is red, o is circle
    plt.plot(X, Y, 'c-') # c is cyan, - is dash

    plot_symbol = color[0]+"->"
    # print("plot symbol is ", plot_symbol)
    for i in range(len(THETA)):
        x2 = 0.25*math.cos(THETA[i]) + X[i]
        y2 = 0.25*math.sin(THETA[i]) + Y[i]
        plt.plot([X[i], x2], [Y[i], y2], plot_symbol) # 'g' is green

    img_label=  mpatches.Patch(color=color, label=label)
    plt.legend(handles=[img_label])
    if title!="":
        plt.title(title)
    plt.show()


def draw_trajectories(details_arr,  colors_arr=[], labels_arr=[], title=""):
    ax = plt.subplot(111) # equivalent to 1, 1, 1
    handles_arr=[]
    for curr_detail, color, label in zip(details_arr, colors_arr, labels_arr):
        # if label=="noisy":
        #     continue
        X, Y, THETA = curr_detail 
        ax.plot(X, Y, 'ro') # r is red, o is circle
        plt.plot(X, Y, 'c-') # c is cyan, - is dash

        plot_symbol = color[0]+"->"
        # print("plot symbol is ", plot_symbol)
        for i in range(len(THETA)):
            x2 = 0.25*math.cos(THETA[i]) + X[i]
            y2 = 0.25*math.sin(THETA[i]) + Y[i]
            plt.plot([X[i], x2], [Y[i], y2], plot_symbol) # 'g' is green

        img_label=  mpatches.Patch(color=color, label=label)
        handles_arr.append(img_label)
        # break
    plt.legend(handles=handles_arr)
    # if len(details_arr)>2:

    #     for i in range(len(THETA)):
    #         plt.plot([details_arr[0][0][i], details_arr[2][0][i]], [details_arr[0][1][i], details_arr[2][1][i]], "y->")

    if title!="":
        plt.title(title, fontsize = 30)
    plt.show()



def find_residuals(flattened_curr_est, edges_info, anchor_c):
    assert(len(anchor_c) ==3 )

    f = lambda node_id, dim_id: 3 * node_id+dim_id
    # def f(node_id, dim_id):
    #     ans = 3 * node_id+dim_id
    #     print("ans is ", ans)
    #     return ans

    num_constraints = len(edges_info)+1
    num_elements_in_residual = 3 * num_constraints

    residuals = jnp.zeros(num_elements_in_residual)

    # do for all constraints
    for edge_id, curr_edge in enumerate(edges_info):
        n1 = curr_edge['v1']
        n2 = curr_edge['v2']

        dx,dy, dtheta = curr_edge['dx'], curr_edge['dy'], curr_edge['dtheta']

        r_dx_id = edge_id * 3
        r_dy_id = edge_id * 3 + 1
        r_dtheta_id = edge_id * 3 + 2
        # print("n1 and n2 are ", n1, n2)
        # print(r_dx_id, r_dy_id, r_dtheta_id)

        residuals = residuals.at[r_dx_id].set((flattened_curr_est[f(n1, 0)] + dx * jnp.cos(flattened_curr_est[f(n1, 2)]) - dy * jnp.sin(flattened_curr_est[f(n1, 2)]) ) - flattened_curr_est[f(n2, 0)])
        # residuals.at[r_dx_id].set(1)
        # print("1) ", residuals[r_dx_id])
        # print("FLat n1 x ", flattened_curr_est[f(n1, 0)])
        # print("FLat n2 x ", flattened_curr_est[f(n2, 0)])

        residuals = residuals.at[r_dy_id].set((flattened_curr_est[f(n1, 1)] + dy * jnp.cos(flattened_curr_est[f(n1, 2)]) + dx * jnp.sin(flattened_curr_est[f(n1, 2)]) ) - flattened_curr_est[f(n2, 1)])

        residuals = residuals.at[r_dtheta_id].set(flattened_curr_est[f(n1, 2)] + dtheta - flattened_curr_est[f(n2, 2)])
    
    # then also do for FIX edge
    residuals = residuals.at[-3].set(flattened_curr_est[0] - anchor_c[0])
    residuals = residuals.at[-2].set(flattened_curr_est[1] - anchor_c[1])
    residuals = residuals.at[-1].set(flattened_curr_est[2] - anchor_c[2])

    return residuals

def analytical_jacobian(curr_flattened_est, edges_info):

    f = lambda node_id, dim_id: 3 * node_id+dim_id

    num_vertices = len(curr_flattened_est)//3
    num_variables = len(curr_flattened_est)

    num_constraints = len(edges_info) + 1
    num_elements_in_residual = 3 * num_constraints

    J_anyl = np.zeros((num_elements_in_residual,num_variables ))

    # do for all constraints
    for edge_id, curr_edge in enumerate(edges_info):
        # print("keys are " , curr_edge.keys())
        n1 = curr_edge['v1']
        n2 = curr_edge['v2']

        dx,dy, dtheta = curr_edge['dx'], curr_edge['dy'], curr_edge['dtheta']

        theta1=curr_flattened_est[f(n1, 2)]

        #####################################
        # FOR X component of residual
        residual_id = edge_id * 3
        ### Node 1:
        # x
        J_anyl[residual_id, f(n1, 0)] = 1
        # theta
        J_anyl[residual_id, f(n1, 2)] = (-dx*np.sin(theta1) -dy*np.cos(theta1))


        ### Node 2:
        # x
        J_anyl[residual_id, f(n2, 0)] = -1
        #########################################


        #####################################

        # FOR Y component of residual
        residual_id = edge_id * 3 + 1
        ### node 1:
        # y
        J_anyl[residual_id, f(n1,1)] = 1
        # theta
        J_anyl[residual_id, f(n1, 2)] = (-dy*np.sin(theta1) + dx*np.cos(theta1))


        ### node 2:
        # y
        J_anyl[residual_id, f(n2,1)] = -1
        #####################################


        #####################################

        # FOR THETA component of residual
        residual_id = edge_id * 3 + 2
        ### node 1:
        # theta
        J_anyl[residual_id, f(n1, 2)] = 1

        ### node 1:
        # theta
        J_anyl[residual_id, f(n2, 2)] = -1
        #####################################

    #TODO: for fix edge
    # specially for FIX EDGE
    J_anyl[-3, 0] = 1
    J_anyl[-2, 1] = 1
    J_anyl[-1, 2] = 1

    return J_anyl



def fetch_flattened_coordinates(vertices_info):
    arr=[]
    for idx, curr_v in enumerate(vertices_info):
        assert(curr_v['vertex_id']==idx)
        x, y, theta = curr_v['x_c'], curr_v['y_c'], curr_v['theta_c']
        arr.extend([x, y, theta])
    assert(len(arr)%3==0)
    return jnp.array(arr)

def fetch_information_matrix(edges, odom_wt, loop_wt, anchor_wt):

    diag_arr = []
    found_loop = False
    for curr_edge in edges:
        idx1 = min(curr_edge["v1"], curr_edge["v2"])
        idx2 = max(curr_edge["v1"], curr_edge["v2"])

        if idx1+1==idx2:
            diag_arr.extend([odom_wt] * 3)
        else:
            diag_arr.extend([loop_wt] * 3)

    # Anchor FIX 0 edge
    diag_arr.extend([anchor_wt] * 3)
    
    return jnp.diag(jnp.array(diag_arr))

def frobNorm(P1, P2, str1="mat1", str2="mat2"):
    jnp.set_printoptions(suppress=True)
    val = jnp.linalg.norm(P1 - P2, 'fro')
    print(f"Frobenius norm between {str1} and {str2} is: {val}")

def get_tot_error(residual_arr, information_mat):
    #residual = 420 x 1
    err = residual_arr.T@information_mat@residual_arr
    return 0.5 * err

def restore_vertices_from_flattened_arr(flattened_arr):
    assert(len(flattened_arr)%3==0)
    vertices_X = np.asarray(flattened_arr[0::3]).tolist()
    vertices_Y = np.asarray(flattened_arr[1::3]).tolist()
    vertices_THETA = np.asarray(flattened_arr[2::3]).tolist()
    return vertices_X, vertices_Y, vertices_THETA

def read_gt(filepath):
    with open(filepath, 'r') as fd:
        all_lines=fd.readlines()
    
    vertices = [] 

    for curr_line in all_lines:
        if "VERTEX_SE2" in curr_line:
            # print("yep")
            (ver, ind, x, y, theta) = curr_line.split()
            obj = dict()
            obj['vertex_id'] = int(ind)
            obj['x_c'] = float(x)
            obj['y_c'] = float(y)
            obj['theta_c'] = float(theta.rstrip('\n'))
            vertices.append(obj)
        
    vertices = sorted(vertices, key=lambda x:x['vertex_id'])
    vertices_X = [x['x_c'] for x in vertices]
    vertices_Y = [x['y_c'] for x in vertices]
    vertices_THETA = [x['theta_c'] for x in vertices]

    return vertices_X, vertices_Y, vertices_THETA


def perform_optimization(flattened_curr_est, edges_info, inf_matrix, anchor_c, gt_details, noisy_details, num_iters=4, method="gn"):
    assert(method in ['gn','gd','lm'])
    
    
    initial_res =  find_residuals(flattened_curr_est, edges_info, anchor_c)
    initial_abs_err = get_tot_error(initial_res, inf_matrix)

    best_pose_yet = copy.deepcopy(flattened_curr_est)
    best_err_yet = initial_abs_err
 
    abs_err_arr = [initial_abs_err]
    best_err_arr = [initial_abs_err]
    print("Method is ", method)

    print("Before beginning, initial abs error is ", initial_abs_err)
    print("-----------------------------------")
    for curr_itr_number in range(1, num_iters+1 ):
        print("CURR ITR is ", curr_itr_number)

        curr_J = analytical_jacobian(flattened_curr_est, edges_info)
        curr_residual = find_residuals(flattened_curr_est, edges_info, anchor_c)


        # for gaussian newton
        if method=="gn":
            print("Method is ", method)
            d_flattened_arr = - np.linalg.inv(curr_J.T @ inf_matrix @ curr_J) @ curr_J.T @ inf_matrix @ curr_residual




        # # for LM
        # pass
        if method=="lm":
            print("Method is ", method)

            lambda_val = 1
            d_flattened_arr = - np.linalg.inv(curr_J.T @ inf_matrix @ curr_J + lambda_val * np.eye(curr_J.shape[1])) @ curr_J.T @ inf_matrix.T @ curr_residual

        # # for gradient descent
        if method=="gd":
            print("Method is ", method)

            lr = 0.0001
            d_flattened_arr =  - curr_J.T @ inf_matrix @ curr_residual


        


        ### UPDATE params
        # new_poses = poses + step_lm(poses, edges, weights, anchor, lamda)
        ALPHA = 1
        flattened_curr_est = [a + ALPHA * b for a, b in zip(flattened_curr_est, d_flattened_arr)]

        # calculate error
        curr_res =  find_residuals(flattened_curr_est, edges_info, anchor_c)
        curr_abs_err = get_tot_error(curr_res, inf_matrix)
        print("Error obtained in this itr is ", curr_abs_err)

        # update best error
        if curr_abs_err < best_err_yet:
            best_err_yet = curr_abs_err
            best_pose_yet = copy.deepcopy(flattened_curr_est)
            print("This is the best pose so far")

        # make updates for plotting error vs iterations
        abs_err_arr.append(curr_abs_err)
        best_err_arr.append(best_err_yet)

        # estimate best trajectory
        current_details = restore_vertices_from_flattened_arr(flattened_curr_est)
        draw_trajectories([gt_details, noisy_details, current_details], ['blue','yellow','green'], ["ground truth", "noisy","current estimate"],f"Comparison for ITR {curr_itr_number}")
        print("####################################")
    abs_err_arr = np.array(abs_err_arr)
    best_err_arr = np.array(best_err_arr)

    return best_pose_yet, best_err_yet, flattened_curr_est, abs_err_arr, best_err_arr



    
def overwrite_vertices(vertices_X, vertices_Y, vertices_THETA, vertices):

    vertices = copy.deepcopy(vertices)
    for idx, curr_ver in enumerate(vertices):
        v_id = curr_ver['vertex_id']
        assert(v_id==idx)
        vertices[idx]['x_c'] = vertices_X[idx]
        vertices[idx]['y_c'] = vertices_Y[idx]
        vertices[idx]['theta_c'] = vertices_THETA[idx]
    return vertices
    