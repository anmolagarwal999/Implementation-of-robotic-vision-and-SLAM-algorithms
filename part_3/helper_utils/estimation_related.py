import numpy as np
import math
import scipy.linalg
import cv2
from typing import Tuple, List
from copy import deepcopy


def homogenize_2d_coordinates(pts_list):
    new_list = deepcopy(pts_list)
    for curr_elem in new_list:
        curr_elem.append(1)
    new_list = np.array(new_list)
    return new_list


def fetch_dis_from_origin(arr):
    sum_of_d = 0
    for i in arr:
        # print(i)
        dis_sq = i[0]**2 + i[1]**2
        sum_of_d += math.sqrt(dis_sq)
    return sum_of_d


def normalize_2d_coordinates(pts_list: List[List]):

    print(
        f"Trying to normalize 2d coordinated for an array with {len(pts_list)} points")

    pts_df = np.array(deepcopy(pts_list))

    # the mean
    mean_df = pts_df.mean(axis=0)
    print("the mean is: ", mean_df)

    translated_df = pts_df - mean_df
    print("points after translatibg via the mean is , ", translated_df)

    print("[DEBUG] the mean of translated df is ", translated_df.mean(axis=0))

    sum_of_d = fetch_dis_from_origin(translated_df)
    print("Tot dis from origin is: ", sum_of_d)

    scaling_fac = ((sum_of_d)) / (pts_df.shape[0] * math.sqrt(2))
    scaling_fac = 1 / scaling_fac

    print("Scaling factor is: ", scaling_fac)
    print("mean dis from origin if operator would be applied is : ",
          fetch_dis_from_origin(translated_df * scaling_fac) / len(pts_list))

    # constructing the normalizing matrix
    NORM_MAT = np.zeros((3, 3))
    NORM_MAT[2, 2] = 1
    NORM_MAT[0, 0] = scaling_fac
    NORM_MAT[1, 1] = scaling_fac
    NORM_MAT[0, 2] = -mean_df[0] * scaling_fac
    NORM_MAT[1, 2] = -mean_df[1] * scaling_fac

    print("Now, [verifying the normalizing matrix]")
    three_n_df = np.array(homogenize_2d_coordinates(pts_list))

    # 3 x N = (3 x 3) X (3 x N)
    changed_pts = NORM_MAT @ three_n_df.T
    changed_pts = changed_pts.T
    print("Shape of changd pts is ", changed_pts.shape)
    print("MEAN of changed pts is ", changed_pts.mean(axis=0))
    print(
        "distance is ",
        fetch_dis_from_origin(changed_pts) /
        changed_pts.shape[0])
    print("######\n")
    return changed_pts, NORM_MAT
    # return fundamental matrix and normalizing matrix


def fetch_best_solution(A):
    # assume that AX=0 in this case
    # assume size of X = 9
    print(
        f"Given A matrix, trying to find best P_vector value where P has {A.shape[1]} dimensions")
    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    S = S.tolist()

    # making sure that the last index is the one with min singular value
    min_value, min_index = min([(j, i) for i, j in enumerate(S)])
    assert (min_value >= 0)
    assert (min_index == len(S) - 1)

    V = np.transpose(Vh)

    # fetch the last column as the solution
    return V[:, -1], U, S, V


def prep_A_matrix(norm_px_1, norm_px_2):

    assert (norm_px_1.shape[0] == norm_px_2.shape[0])
    num_points = norm_px_1.shape[0]
    A_mat = np.zeros()

    pass


def compute_fundamental_matrix(X1_list: List[List], X2_list: List[List]):

    # first, assert that both matrices have the same number of points
    assert (len(X1_list) == len(X2_list))

    # fetch the normalized coordinates and the normalization matrix for both
    # the images
    norm_px_1, norm_matrix_1 = normalize_2d_coordinates(X1_list)
    norm_px_2, norm_matrix_2 = normalize_2d_coordinates(X2_list)

    # prepare the funademantal matrix to minimze algebraic error

    # perform correction on the fundamental matrix

    # multiply corrected fundamental matrix with other matrices to reverse the
    # effect of normalization

    pass


def essential_from_fundamental(
        fundamental_matrix,
        calibration_matrix ):

    K = calibration_matrix
    F = fundamental_matrix

    E_initial = np.dot(K.T, np.dot(F, K))

    # we want to enforce the rank 2 condition on E as well, hence, we keep
    # only the first 2 singular values to be non-zeero
    U, S, V = np.linalg.svd(E_initial)

    return (np.dot(U, np.dot(np.diag([1, 1, 0]), V)))


def FetchCameraPose(F, K):

    # retrieval algorithm being followed from here: https://cmsc426.github.io/sfm/ and Zisserman's book 
    
    E = essential_from_fundamental(F, K)
    
    # SVD of E
    U, S, V_T = np.linalg.svd(E)

    W = [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]]

    R_candidates = []
    C_candidates = []

    R_candidates.append(np.dot(U, np.dot(np.array(W), V_T)))
    C_candidates.append(U[:, 2])
    
    
    R_candidates.append(np.dot(U, np.dot(np.array(W), V_T)))
    C_candidates.append(-U[:, 2])
    
    
    R_candidates.append(np.dot(U, np.dot(np.array(W).T, V_T)))
    C_candidates.append(U[:, 2])
    
    
    R_candidates.append(np.dot(U, np.dot(np.array(W).T, V_T)))
    C_candidates.append(-U[:, 2])


    for i in range(4):
        if (np.linalg.det(R_candidates[i]) < 0):
            R_candidates[i] = -R_candidates[i]
            C_candidates[i] = -C_candidates[i]

    return R_candidates, C_candidates


def fetch_best_3d_point(A):
    # assume that AX=0 in this case
    # assume size of X = 4
    # print(
    # f"Given A matrix, trying to find best P_vector value where P has
    # {A.shape[1]} dimensions")
    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    S = S.tolist()

    # making sure that the last index is the one with min singular value
    min_value, min_index = min([(j, i) for i, j in enumerate(S)])
    assert (min_value >= 0)
    assert (min_index == len(S) - 1)

    V = np.transpose(Vh)

    # fetch the last column as the solution
    return V[:, -1], U, S, V


def fetch_coeffs_for_triangulate(px, P_mat):
    assert (P_mat.shape == (3, 4))
    p1_t = np.array([P_mat[0, :]])
    p2_t = np.array([P_mat[1, :]])
    p3_t = np.array([P_mat[2, :]])

    x, y, _ = px

    r1 = y * p3_t - p2_t

    r2 = p1_t - x * p3_t

    coeffs = np.vstack([r1, r2])

    assert (coeffs.shape == (2, 4))

    return coeffs


def fetch_triangulate_point(px_1, px_2, P_mat_1, P_mat_2):

    # both pixel coordiantes are homogenized
    assert (px_1.shape[0] == 3)
    assert (px_2.shape[0] == 3)

    coeffs_1 = fetch_coeffs_for_triangulate(px_1, P_mat_1)
    coeffs_2 = fetch_coeffs_for_triangulate(px_2, P_mat_2)

    A_mat = np.vstack([coeffs_1, coeffs_2])

    x_world_best, U, S, V = fetch_best_3d_point(A_mat)

    return x_world_best

