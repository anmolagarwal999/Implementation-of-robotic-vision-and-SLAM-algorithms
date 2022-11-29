import cv2
import numpy as np
import colorsys
from copy import deepcopy

def draw_dots(cv_img_obj, centre_x, centre_y, circle_radius, color_tuple, circle_boundary_thinkness):

    # origin is at leftmost corner of image
    assert (len(color_tuple) == 3)
    assert (max(color_tuple) <= 255)

    # img_height, img_width, img_channels = cv_img_obj.shape
    img_height, img_width = cv_img_obj.shape[0], cv_img_obj.shape[1]
    # print("centre x is ", centre_x)
    # print("img height is ", img_height)
    assert (centre_x <= img_width)
    assert (centre_y <= img_height)
    # cv_img_obj = cv.circle(cv_img_obj, centerOfCircle, radius, color, thickness)
    cv_img_obj = cv2.circle(cv_img_obj, (centre_x, centre_y), radius=circle_radius,
                            color=color_tuple, thickness=circle_boundary_thinkness)
    return cv_img_obj


def plot_pixel_points(curr_img, coord_details, color_wanted, circle_rad=35):
    # each elem of coord_details should be a pair of <x,y>
    img = curr_img.copy()
    for curr_elem in coord_details:
        img = draw_dots(
            img, curr_elem[0], curr_elem[1], circle_rad, color_wanted, -1)
    return img


def get_unique_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors = [[int(y*255) for y in x] for x in colors]
    return colors


def plot_diff_pixel_points(curr_img, coord_details, colors_wanted, circle_rad=10):
    # each elem of coord_details should be a pair of <x,y>
    img = curr_img.copy()
    for curr_elem, color_wanted in zip(coord_details, colors_wanted):
        img = draw_dots(
            img, curr_elem[0], curr_elem[1], circle_rad, color_wanted, -1)
    return img


def plot_epipolar_lines(img1,img2, points1, points2, F):
    '''Aim: To plot epipolar lines in image 2 when fundamental matrix has been derived as: 
    (coordinates in image 2) X FUND MATRIX X (coordinates in image 1) =  0 
    '''
    
    img1 = deepcopy(img1)
    img2 = deepcopy(img2)
    
    # find number of points
    x_max = img1.shape[1]

    line_width = 3
    point_width = 8
    lines_in_img_2 = []
    for i in range(points1.shape[0]):
        # making homogenous coordinates
        point_arr = np.array([points1[i][0], points1[i][1], 1])        
        line = F @ point_arr


        a = line[0]; b = line[1]; c = line[2]
        lines_in_img_2.append([a, b, c])


        x1, y1 = [0, int(-c / b) ] 
        x2, y2 = [x_max, int(-(c + a * x_max) / b) ]
        color = [np.random.randint(1, 255), np.random.randint(1, 255), np.random.randint(1, 255)]
        # annotate the original lines in image 1
        img1 = cv2.circle(img1, (int(points1[i][0]), int(points1[i][1])), point_width, color, -1)
        
        # annotate epipolar lines in image 2
        img2 = cv2.line(img2, (x1, y1), (x2, y2), color, line_width) 
        img2 = cv2.circle(img2, (int(points2[i][0]), int(points2[i][1])), point_width, color, -1)
        
        
    return img1, img2, lines_in_img_2