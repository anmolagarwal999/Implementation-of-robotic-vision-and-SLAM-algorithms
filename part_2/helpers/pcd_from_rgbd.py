
from sys import argv, exit
import numpy as np
import open3d as o3d
from PIL import Image


def readDepth(depthFile):
	depth = Image.open(depthFile)
	if depth.mode != "I":
		raise Exception("Depth image is not in intensity format")

	return np.asarray(depth)

def getPointCloud(rgbFile, depthFile, scalingFactor = -1, centerX = 400, centerY = 400, focalX=-400, focalY=400):

	thresh = 10000

	depth = readDepth(depthFile)
	rgb = Image.open(rgbFile)


	points = []
	colors = []
	srcPxs = []
	
    

	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):


			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - centerX) * Z / focalX
			Y = (v - centerY) * Z / focalY

			srcPxs.append((u, v))
			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))

	srcPxs = np.asarray(srcPxs).T
	points = np.asarray(points)
	colors = np.asarray(colors)

	pcd = o3d.geometry.PointCloud()
	print(points)
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)
	o3d.visualization.draw_geometries([pcd])
	return pcd, srcPxs

if __name__=="__main__":
	# argv[1] : rgb img
	# argv[2] : depth img
	# argv[3] : path to save point cloud
    pcd, srcPxs = getPointCloud(argv[1], argv[2])
    o3d.io.write_point_cloud(argv[3], pcd)
