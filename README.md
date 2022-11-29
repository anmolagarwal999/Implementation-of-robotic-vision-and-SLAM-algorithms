### Robotic-Vision

This repository contains my implementations and explorations of several algorithms and pipelines used in Computer Vision and Robotics, as a part of the "Mobile Robotics" (Monsoon 2022) course.


#### Part-1
Consists of:
* a basic overview of euclidean transformations (includes an artificially constructed table scene via Open3D)
* Ways of representing rotations and the inter-conversions between rotation matrix, quaternions and euler-angle based representations
* Approximating points of infinity in an image
* An explanation and exercises regarding the Gimbal Lock problem
* Registering the different point clouds in different frames from the Kitti dataset to construct the ground truth

#### Part-2
Consists of: 
* Implementation of the Procrustes Alignment
* Implementation of point-to-point Iterative Closest Point (ICP) [without known correspondences as well] and analyzes why some other ICP variants might work better
* Looks at various non-linear optimizations - the various algorithms (Gaussian Newton, Levenberg-Marquardt method and Simple Gradient Descent) and their characteristics. 
* Implementation of 1D and 2D euclidean pose-graph optimisation for SLAM from scratch, with the solution verified via the jax auto-differentiation library. It then compares the results with those obtained using the g2o solver. The analysis of the extent of errors and on which parts of the trajectory, the error is occurring is done using `evo`. It also compares the solution accuracy and convergence across factors like "number of iterations", "weights given to anchor, odometry and loop closure edges"  etc.

#### Part-3
* Implements camera calibration using the direct linear transform algorithm
* Investigates the characteristics of camera calibration matrices and uses eopiloar geometry to spot corresponding points in other images by reducing their search within the epipolar line.
