import numpy as np
from visualization.KittiUtils import *

'''
LIDAR 
        z    x
        |   / 
        |  /
        | /
y--------/

BEV 
--------- x (used with y lidar)
| 
|
|
y (used with x lidar)
'''
# np.set_printoptions(threshold=np.inf)

# =========================  Config ===============================
boundary = {
    "minX": -30,
    "maxX": 30,
    "minY": -25,
    "maxY": 25,
    "minZ": -3,
    "maxZ": 1
}

BEV_WIDTH  = 608
BEV_HEIGHT = 608


descretization_x = BEV_HEIGHT / (boundary["maxX"] - boundary['minX'])
descretization_y = BEV_WIDTH / (boundary["maxY"] - boundary["minY"])
descretization_z = 1 / float(np.abs(boundary['maxZ'] - boundary['minZ']))
max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))

# =========================== BEV RGB Map ==================================

def pointcloud_to_bev(pointcloud):
    pointcloud = clip_pointcloud(pointcloud)

    # sort by z ... to get the maximum z when using unique 
    # (as unique function gets the first unique elemnt so we attatch it with max value)
    z_indices = np.argsort(pointcloud[:,2])
    pointcloud = pointcloud[z_indices]

    n_points = pointcloud.shape[0]

    MAP_HEIGHT = BEV_HEIGHT + 1
    MAP_WIDTH  = BEV_WIDTH  + 1

    height_map    = np.zeros((MAP_HEIGHT, MAP_WIDTH)) # max z
    intensity_map = np.zeros((MAP_HEIGHT, MAP_WIDTH)) # intensity (contains reflectivity or 1 if not supported)
    density_map   = np.zeros((MAP_HEIGHT, MAP_WIDTH)) # density of the mapped 3D points to a the pixel

    # shape = (n_points, 1)
    x_bev = np.int_((BEV_HEIGHT/2)  - pointcloud[:, 0] * descretization_x )
    y_bev = np.int_((BEV_WIDTH/2) - pointcloud[:, 1] * descretization_y)
    z_bev = pointcloud[:, 2] 
    
    # shape = (n_points, 2)
    xy_bev = np.stack((x_bev, y_bev), axis=1)
    
    # xy_bev_unique.shape (n_unique_elements, 2)
    # indices.shape (n_unique_elements,) (needed for maximum Z)
    # counts.shape  (n_unique_elements,) .. counts is count of repeate times of each unique element (needed for density)
    xy_bev_unique, indices, counts = np.unique(xy_bev, axis=0, return_index=True, return_counts=True)

    # 1 or reflectivity if supported
    # intensity_map[x_bev_unique, y_bev_unique] = pointcloud[x_indices, 3]
    intensity_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = 1

    # points are sorted by z, so unique indices (first found indices) is the max z
    height_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = z_bev[indices]

    # density of points in each pixel
    density_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = np.minimum(1, np.log(counts + 1)/np.log(64) )

    # stack the BEV channels along 3rd axis
    BEV = np.dstack((intensity_map, height_map, density_map))
    return BEV

def corner_to_bev_coord(corner):
    x_bev = np.int_((BEV_HEIGHT/2)  - corner[0] * descretization_x )
    y_bev = np.int_((BEV_WIDTH/2) - corner[1] * descretization_y)
    return np.array([y_bev, x_bev])

# ===================== Clipping =====================
def clip_pointcloud(pointcloud):

    mask = np.where((pointcloud[:, 0] >= boundary["minX"]) & (pointcloud[:,0] <= boundary["maxX"]) &
                    (pointcloud[:, 1] >= boundary["minY"]) & (pointcloud[:,1] <= boundary["maxY"]) &
                    (pointcloud[:, 2] >= boundary["minZ"]) & (pointcloud[:,2] <= boundary["maxZ"])
    )

    return pointcloud[mask]

def is_bbox_in_boundary(point):
    in_x = (point[0] >= boundary["minX"]) & (point[0] <= boundary["maxX"])  
    in_y = (point[1] >= boundary["minY"]) & (point[1] <= boundary["maxY"])
    in_z = (point[2] >= boundary["minZ"]) & (point[2] <= boundary["maxZ"])
    return in_x & in_y & in_z    

def clip_3d_boxes(objects, calib):
    cliped_objects = []
    for obj in objects:
        bbox= obj.bbox_3d

        # check if point in Camera Coord. and convert it to LIDAR Coord.
        point = np.array([bbox.x, bbox.y, bbox.z]).reshape(1,3)
        if bbox.coordinates == Coordinates.CAM_3D_RECT:
            point = calib.rectified_camera_to_velodyne(point)[0,0:3] # reshape from homoginous

        point = point.reshape(3,1)

        # if box center in the cliped region
        if is_bbox_in_boundary(point):
            cliped_objects.append(obj)
    # print("before ", len(objects), "  .. after ", len(cliped_objects))
    return cliped_objects

