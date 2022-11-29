import numpy as np
import math
import scipy.linalg


def part():
    print("############################")


def verify_consistency(P_org, K_mat, R_mat, t_vec):

    M_int = np.hstack((K_mat, np.zeros((K_mat.shape[0], 1))))
    with np.printoptions(suppress=True):
        print('The M_int matrix is \n', M_int)

    M_ext = np.vstack([R_mat, np.array([0, 0, 0])])

    t_vec_l = t_vec.tolist()
    t_vec_l.append(1)

    M_ext = np.hstack([M_ext, np.array([t_vec_l]).transpose()])
    with np.printoptions(suppress=True):
        print('The M_ext matrix is \n', M_ext)

    P_est = M_int @ M_ext
    with np.printoptions(suppress=True):
        print('The P-reverse engineered is\n', P_est)
    print("\nElementwise-Ratio between P original and P reverse engineered: ")

    with np.printoptions(suppress=True):
        print('', P_org / P_est)


def fetch_specific_columns(M, cols_list):
    arr = []
    for curr_col in cols_list:
        a = M[:, curr_col]
        arr.append(a)
    return np.vstack(arr).transpose()


def apply_DLT_algo(X_world: np.ndarray, X_img: np.ndarray,
                   debug_print: bool = False):
    '''Input: world coordinates, image coordinates mapping,
    Output: P_matrix, K_matrix (normalized), R_matrix, translation vector, camera coordinates in WORLD FRAME vector
    '''

    # decide which points to use for applying the dlt algorithm. DLT only
    # needs a minimum of 6. But adding more points can always contribute to
    # minimizing the mean squared error
    arr_to_use = [0, 3, 6, 9, 11, 12, 13]

    # Constructing the matrixes with only those points who are to be used in
    # DLT ie the indices in `arr_to_use`
    X_world_use, X_img_use = fetch_specific_columns(X_world, arr_to_use), \
        fetch_specific_columns(X_img, arr_to_use)

    if debug_print:
        print("Shape of X_world: ", X_world_use.shape)
        print("Shape of X_img: ", X_img_use.shape)

    # fetch the A matrix to solve: Ax = 0
    A_mat = fetch_A_from_mappings(X_world_use, X_img_use)
    print("Shape of A_mat is ", A_mat.shape)

    P_mat = fetch_best_solution(A_mat)

    # reshaping the P_matrix
    P_mat = P_mat.reshape(3, 4)
    with np.printoptions(suppress=True):
        print('The P matrix is \n', P_mat)

    print("## Now, finding the projected image point using the camera matrix")

    # finding the image points which would be produced using the approximated
    # camera matrix
    # ASKED FUNCTION: get_reprojected_points(P, X_world)
    X_proj = P_mat @ X_world

    ###################################################
    # Finding algebraic error
    # Display the squared error
    # For A x P_soln = b, algebraic error would be norm(b).

    b = A_mat @ P_mat.reshape(12)

    print("Shape of A_mat is ", A_mat.shape)

    part()

    error_algebraic = sum(b**2)
    print("### Algebraic error is ", sum(b**2))

    #################################################
    # DISPLAY GEOMETRIC ERROR
    geom_error = fetch_homogenized_diff_mat(
        X_proj, X_img, False) / X_img.shape[1]
    print("\n### Mean Geometric error is ", geom_error)

    ####################################################

    print("\nRatio between image coordinates is:: ")
    with np.printoptions(suppress=True):
        print('', X_proj / X_img)

    ##########################################

    print("Applying RQ decomposition on P_matrix  to retrieve intrinsic and extrinisic parameters")
    # R_mat = rotation matrix, K_mat= camera intrinsic matrix
    K_mat, R_mat = scipy.linalg.rq(P_mat[:3, :3])

    with np.printoptions(suppress=True):
        print('\nThe K matrix (without normalization) is \n', K_mat)

    print("\nFinding the translation vector")
    t_vec = np.linalg.inv(K_mat) @ P_mat[:, -1]

    with np.printoptions(suppress=True):
        print('\nTranslation vector is \n', t_vec)

    # normalizing the K matrix
    K_mat /= K_mat[-1, -1]
    part()

    ###############################################################
    with np.printoptions(suppress=True):
        print('\nThe K matrix (AFTER normalization) is \n', K_mat)
    part()
    with np.printoptions(suppress=True):
        print('\nRotation matrix is \n', R_mat)

    print("\nVerifying rotation matrix is ORTHOGONAL")
    with np.printoptions(suppress=True):
        print('\nThe R matrix orthogonality validation  (R@R.t comes out to be):\n ',
              R_mat @ R_mat.transpose())

    C_vec = -np.linalg.inv(K_mat @ R_mat) @ t_vec
    part()
    with np.printoptions(suppress=True):
        print('\nCamera coordinates seem to be: ', C_vec)

    return P_mat, K_mat, R_mat, t_vec, C_vec, error_algebraic, geom_error, X_proj


def fetch_best_solution(A):
    # assume that AX=0 in this case
    # assume size of X = 12
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
    return V[:, -1]


def fetch_2_cols(x_img, x_world):
    NUM_CONSTANTS = 12

    # normalize both the points
    x_img = np.array(x_img) / x_img[-1]
    x_world = np.array(x_world) / x_world[-1]

    # print("x img is ", x_img)
    ####################
    a1 = np.zeros(12)
    a1[:4] = x_world
    # print("0th elem is ", type(x_img[0]))
    # print("x_worls is ", x_world)
    # print("x_world elem type is ", type(x_world[0]))
    # print("sample is ", x_img[0] * x_world)
    a1[8:12] = -x_world * x_img[0]
    # 3
    a2 = np.zeros(12)
    a2[4:8] = x_world
    a2[8:12] = -x_world * x_img[1]
    #####################
    return np.vstack([a1, a2])


def fetch_A_from_mappings(X_world, X_img):
    print(
        f"Deriving the A_matrix using the world-image coordinate mappings with {X_world.shape[1]} points")
    # print("Number of points is ", X_world.shape[1])
    assert (X_world.shape[1] == X_img.shape[1])
    points_stack = [fetch_2_cols(X_img[:, i], X_world[:, i])
                    for i in range(X_world.shape[1])]
    A_mat = np.vstack(points_stack)
    # print("Shape of A_mat is ", A_mat.shape)
    return A_mat


def fetch_homogenized_diff_mat(img_1, img_2, debug_stat=False):
    # each column in both matrices represent a point
    # print("img type is ", type(img_1))
    # print("Shape is ", img_1.shape)
    NUM_DIMS = img_1.ndim
    if NUM_DIMS == 1:
        return fetch_homogenized_diff_mat(
            np.vstack([img_1]).transpose(), np.vstack([img_2]).transpose(), debug_stat)
    else:
        num_cols = img_1.shape[1]
        # print("img 1 is ", img_1)
        # print("img 2 is ", img_2)
        # print("num cols is ", num_cols)
        # print("last elem is ",img_1[-1,0])
        error = 0
        for i in range(num_cols):
            # print("elem is ",np.array(img_1[-1,i]))
            # print("elem is ",img_1[:,i])
            img_1[:, i] /= np.array(img_1[-1, i], dtype='float')
            img_2[:, i] /= np.array(img_2[-1, i], dtype='float')
            error += math.sqrt((img_1[0, i] - img_2[0, i])
                               ** 2 + (img_1[1, i] - img_2[1, i])**2)
        # diff=img_1-img_2

        if debug_stat:
            with np.printoptions(suppress=True):
                print("Normalized img_1 is ", img_1)
            # print("########################")
            with np.printoptions(suppress=True):
                print("Normalized img_2 is ", img_2)
            # print("########################")
            # with np.printoptions(suppress=True): print("diff is \n", diff)
        # print("########################")
        # print("TOt error is ", np.sum(diff**2))

        return error
