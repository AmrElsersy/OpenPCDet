# from mayavi import mlab
# mlab.options.offscreen = True

import numpy as np
from math import sin, cos, radians
# from .KittiDataset import KittiDataset
from visualization.KittiUtils import *
import visualization.BEVutils as BEVutils
import os, sys
import cv2, PIL


from PIL import Image
from PIL import ImageDraw

sys.path.insert(0, '../')
# from Models.SFA.data_process.kitti_bev_utils import makeBEVMap

class KittiVisualizer:
    def __init__(self):
        self.__scene_2D_mode = False
        self.scene_2D_width = 750
        self.ground_truth_color = (0,1,0) # green
        self.thickness = 1
        self.user_press = None
        self.confidence_score_thresh = 0.25 
        # for bev only
        self.semantic_colors = {
            0: (255,0,0),
            1: (0,0,255),
            2: (0,255,0),
            3: (255,0,255)
        }

        self.colors = [
            (255, 10, 0),
            (255,0 , 255),
            (255,255,0),
            (0, 0, 255)
        ]

    def visualize_scene_3D(self, pointcloud, objects, labels=None, calib=None):
        """
            Visualize the Scene including Point Cloud & 3D Boxes 

            Args:
                pointcloud: numpy array (points_n, 3)
                objects: list of KittiObject represents model output
                labels: list of KittiObjects represents dataset labels
                calib: Kitti Calibration Object (must be specified if you pass boxes with cam_rect_coord)
        """
        # self.figure = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1), size=(1280, 720))

        # Point Cloud
        self.visuallize_pointcloud(pointcloud)

        # 3D Boxes of model output
        for obj in objects:
            bbox_3d = obj.bbox_3d
            color = self.__get_box_color(obj.label)
            self.visualize_3d_bbox(bbox=bbox_3d, color=color, calib=calib)

            self.__draw_text_3D(*bbox_3d.pos, text=str( round(obj.score,2) ), color=color)

        # 3D Boxes of dataset labels 
        if labels is not None:
            for obj in labels:
                self.visualize_3d_bbox(obj.bbox_3d, (1,1,0), calib)

        self.__show_3D()

    def visualize_scene_2D(self, pointcloud, image, objects, labels=None, calib=None, visualize=True):
        # read BEV & image
        self.__scene_2D_mode = True
        _image = self.visualize_scene_image(image, objects, calib)
        _bev   = self.visualize_scene_bev(pointcloud, objects, labels, calib)
        self.__scene_2D_mode = False

        # all will have the same width, just map the height to the same ratio to have the same image
        scene_width = self.scene_2D_width        
        image_h, image_w = _image.shape[:2]
        bev_h, bev_w = _bev.shape[:2]

        # print(_image.shape, _bev.shape)

        new_image_height = int(image_h * scene_width / image_w)
        new_bev_height = int(bev_h * scene_width / bev_w)

        _bev   = cv2.resize(_bev,   (scene_width, new_bev_height) )
        _image = cv2.resize(_image, (scene_width, new_image_height) )

        image_and_bev = np.zeros((new_image_height + new_bev_height, scene_width, 3), dtype=np.uint8)
        # print(_image.shape, _bev.shape, image_and_bev.shape)
        image_and_bev[:new_image_height, :, :] = _image
        image_and_bev[new_image_height:, :, :] = _bev

        if visualize:
            cv2.imshow("scene 2D", image_and_bev)
            self.__show_2D()
        else:
            return image_and_bev


    def visualize_stereo_scene(self, imgL, disp, pointcloud):
        self.__scene_2D_mode = True
        bev   = self.visualize_scene_bev(pointcloud, [])
        self.__scene_2D_mode = False

        scene_width = self.scene_2D_width        
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        bev = bev[:,:,0]
        
        img_h, img_w = imgL.shape[:2]
        bev_h, bev_w = bev.shape[:2]
        disp_h, disp_w = disp.shape[:2]

        new_img_h  = int(img_h  * scene_width / img_w)
        new_disp_h = int(disp_h * scene_width / disp_w)
        new_bev_h  = int(bev_h * 2/3) 

        bev   = cv2.resize(bev,  (scene_width, new_bev_h) )
        image = cv2.resize(imgL, (scene_width, new_img_h) )
        disp  = cv2.resize(disp, (scene_width, new_disp_h))

        scene_height_img = new_img_h + new_disp_h
        scene = np.zeros((new_bev_h + new_disp_h + new_img_h, scene_width), dtype=np.uint8)

        print(bev.shape, image.shape, disp.shape, "total = ", scene.shape)
        scene[:new_disp_h, :] = disp
        scene[new_disp_h: scene_height_img, :] = image
        scene[scene_height_img:, :] = bev

        cv2.imshow("disparity_scene", scene)
        self.__show_2D()

    def bev_to_colored_bev_semantic(self, bev):
        semantic_map = bev[:,:,3]
        shape = semantic_map.shape[:2]
        color_map = np.zeros((shape[0], shape[1], 3))

        for label in self.semantic_colors:
            color = self.semantic_colors[label]
            color_map[semantic_map == id] = color[2], color[1], color[0]

        return color_map
        
    def visualize_scene_bev(self, pointcloud, objects, labels=None, calib=None):
        # BEV = makeBEVMap(pointcloud, None, pointpainting=True)
        BEV = BEVutils.pointcloud_to_bev(pointcloud)
        BEV = self.__bev_to_colored_bev(BEV)

        # clip boxes
        objects = BEVutils.clip_3d_boxes(objects, calib)
        
        # 3D Boxes of model output
        for obj in objects:
            color = self.__get_box_color(obj.label)
            color = list(color)
            color[0], color[2] = color[2], color[0] # swap to converrt from BGR to RGB
            color = tuple(color)
            self.__draw_bev_box3d(BEV, obj.bbox_3d, color, calib)

        # # 3D Boxes of dataset labels 
        if labels is not None:
            labels = BEVutils.clip_3d_boxes(labels, calib)
            for obj in labels:
                color = [c * 255 for c in self.ground_truth_color]
                self.__draw_bev_box3d(BEV, obj.bbox_3d, color, calib)

        if self.__scene_2D_mode:
            return BEV 

        cv2.imshow("BEV", BEV)
        self.__show_2D()

    def visualize_scene_image(self, image, kitti_objects, calib, scene_2D_mode=True):
        
        # Preprocessing for viz.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        self.current_image = image

        for object in kitti_objects:
            if object.score < self.confidence_score_thresh:
                continue

            corners = self.__convert_3d_bbox_to_corners(object.bbox_3d, calib)
            proj_corners = calib.project_lidar_to_image(corners)
            
            isTruncated = self.filter_truncated_boxes(proj_corners)
            if isTruncated:
                continue

            color = self.__get_box_color(object.label)
            self.__draw_box_corners(proj_corners, color, VisMode.SCENE_2D)
            
            point = (min(proj_corners[2][0],proj_corners[3][0], proj_corners[1][0]), \
                max(proj_corners[1][1], proj_corners[2][1], proj_corners[3][1]))

            bbox_volume = object.bbox_3d.height * object.bbox_3d.width * object.bbox_3d.length
            box_depth = object.bbox_3d.x
            
            score_point = (point[0], point[1]-20)
            score_per_box = int(object.score * 100)
            self.__draw_text_2D(f"Score: {score_per_box}", score_point, bbox_volume, color)

            label_point = (point[0], point[1]-20)
            # self.__draw_text_2D(f"{object.label}", (point[0], point[1]))

        self.current_image = np.array(self.current_image)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR) 

        if scene_2D_mode:
            return np.array(self.current_image) 

        cv2.imshow('Image',self.current_image)
        self.__show_2D()        

    def __show_3D(self):
        # mlab.show(stop=True)
        print("Called show")
        # pass

    def __show_2D(self):
        print("**************** Press n for next example ... Press ESC to quit *****************")
        self.user_press = cv2.waitKey(0) & 0xff      

    def visuallize_pointcloud(self, pointcloud):
        pointcloud = self.__to_numpy(pointcloud)
        # mlab.points3d(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], 
        #             colormap='gnuplot', scale_factor=1, mode="point",  figure=self.figure)
        # self.__draw_axes()

    def visualize_3d_bbox(self, bbox: BBox3D, color=(0,1,0), calib=None):
        corners = self.__convert_3d_bbox_to_corners(bbox, calib)
        self.__draw_box_corners(corners, color)

    def __draw_bev_box3d(self, bev, bbox_3d, color, calib):
        corners = self.__convert_3d_bbox_to_corners(bbox_3d, calib)
        c0 = BEVutils.corner_to_bev_coord(corners[0])
        c1 = BEVutils.corner_to_bev_coord(corners[1])
        c2 = BEVutils.corner_to_bev_coord(corners[2])
        c3 = BEVutils.corner_to_bev_coord(corners[3])
        
        cv2.line(bev, (c0[0], c0[1]), (c1[0], c1[1]), color, self.thickness)
        cv2.line(bev, (c0[0], c0[1]), (c2[0], c2[1]), color, self.thickness)
        cv2.line(bev, (c3[0], c3[1]), (c1[0], c1[1]), color, self.thickness)
        cv2.line(bev, (c3[0], c3[1]), (c2[0], c2[1]), color, self.thickness)

    def __bev_to_colored_bev(self, bev):

        bev = (bev * 255).astype(np.uint8)

        intensity_map = bev[:,:,0]
        height_map = bev[:,:,1]
        density_map = bev[:,:,2]

        minZ = BEVutils.boundary["minZ"]
        maxZ = BEVutils.boundary["maxZ"]
        # height_map = 255 - 255 * (height_map - minZ) / (maxZ - minZ) 
        # bev = np.dstack((intensity_map, height_map, density_map))

        return bev

    def __draw_axes(self):
        l = 4 # axis_length
        w = 1
        # mlab.plot3d([0, l], [0, 0], [0, 0], color=(0, 0, 1), line_width=w, figure=self.figure) # x
        # mlab.plot3d([0, 0], [0, l], [0, 0], color=(0, 1, 0), line_width=w, figure=self.figure) # y
        # mlab.plot3d([0, 0], [0, 0], [0, l], color=(1, 0, 0), line_width=w, figure=self.figure) # z

    def convert_3d_bbox_to_corners(self, bbox, calib):
        return self.__convert_3d_bbox_to_corners(bbox, calib)
        
    def __convert_3d_bbox_to_corners(self, bbox: BBox3D, calib=None):
        """
            convert BBox3D with x,y,z, width, height, depth .. to 8 corners
                    h
              3 -------- 1
          w  /|         /|
            2 -------- 0 . d
            | |        | |
            . 7 -------- 5
            |/         |/
            6 -------- 4

                        z    x
                        |   / 
                        |  /
                        | /
                y--------/
        """
        x = bbox.x
        y = bbox.y
        z = bbox.z
        w = bbox.width  # y
        h = bbox.height # z
        l = bbox.length # x
        angle = bbox.rotation

        # convert from Camera 3D coordinates to LIDAR coordinates.
        if bbox.coordinates == Coordinates.CAM_3D_RECT:
            if calib is None:
                print("WARNING: Visualization is in LIDAR coord & you pass a bbox of camera coord")

            point = np.array([x, y, z]).reshape(1,3)

            # convert x, y, z from rectified cam coordinates to velodyne coordinates
            point = calib.rectified_camera_to_velodyne(point)

            x = point[0,0]
            y = point[0,1] 
            # model output is z center but dataset annotations consider z at the bottom 
            z = point[0,2] + h/2

            # angle in annotations is inverted .. while angle in predictions is correct
            angle = -angle
        
        # convert (x,y,z) from center to top left corner (corner 0)
        x = x - w/2
        y = y - l/2
        z = z + h/2

        top_corners = np.array([
            [x, y, z],
            [x+w, y, z],
            [x, y+l, z],
            [x+w, y+l, z]
        ])

        # same coordinates but z = z_top - box_height
        bottom_corners = top_corners - np.array([0,0, h])

        # concatinate 
        corners = np.concatenate((top_corners,bottom_corners), axis=0)

        # 3x3 Rotation Matrix along z 
        cosa = cos(angle)
        sina = sin(angle)
        R = np.array([
            [cosa, -sina, 0],
            [sina, cosa, 0],
            [0,    0,    1]
        ])

        # Translate the box to origin to perform rotation
        center = np.array([x+w/2, y+l/2, z-h/2])
        centered_corners = corners - center

        # Rotate
        rotated_corners = np.dot( R, centered_corners.T ).T

        # Translate it back to its position
        corners = rotated_corners + center

        # output of sin & cos sometimes is e-17 instead of 0
        corners = np.round(corners, decimals=10)

        return corners

    def __draw_box_corners(self, corners, clr, vis_mode=VisMode.SCENE_3D):
        if corners.shape[0] != 8:
            print("Invalid box format")
            return

        c0 = corners[0]
        c1 = corners[1] 
        c2 = corners[2] 
        c3 = corners[3] 
        c4 = corners[4] 
        c5 = corners[5] 
        c6 = corners[6] 
        c7 = corners[7] 

        # top suqare
        self.__draw_line(c0, c1, clr, vis_mode)
        self.__draw_line(c0, c2, clr, vis_mode)
        self.__draw_line(c3, c1, clr, vis_mode)
        self.__draw_line(c3, c2, clr, vis_mode)
        # bottom square
        self.__draw_line(c4, c5, clr, vis_mode)
        self.__draw_line(c4, c6, clr, vis_mode)
        self.__draw_line(c7, c5, clr, vis_mode)
        self.__draw_line(c7, c6, clr, vis_mode)
        # vertical edges
        self.__draw_line(c0, c4, clr, vis_mode)
        self.__draw_line(c1, c5, clr, vis_mode)
        self.__draw_line(c2, c6, clr, vis_mode)
        self.__draw_line(c3, c7, clr, vis_mode)

    def __draw_line(self, corner1, corner2, clr, vis_mode):
        x = 0
        y = 1
        z = 2
        if vis_mode == VisMode.SCENE_3D:
            clr=tuple([x/255.0 for x in clr])
            # mlab.plot3d([corner1[x], corner2[x]], [corner1[y], corner2[y]], [corner1[z], corner2[z]],
            #         line_width=2, color=clr, figure=self.figure)

        elif vis_mode == VisMode.SCENE_2D:
            draw_ = ImageDraw.Draw(self.current_image, mode='RGB')
            # cv2.line(self.current_image, (corner1[x], corner1[y]), (corner2[x], corner2[y]), \
                # color=tuple([255 * x for x in clr]), thickness=2)
            draw_.line(xy=[(corner1[x], corner1[y]), (corner2[x], corner2[y])], \
                fill=clr, width=self.thickness)           

    def __draw_text_2D(self, text, point, bbox_volume, color=(255, 255, 255), font_scale=0.4, thickness=2):
        # cv2.putText(self.current_image, text, point, font, font_scale, color, thickness)
        draw = ImageDraw.Draw(self.current_image, mode='RGB')
        # font = ImageFont.truetype('Extras/arial.ttf', int(bbox_volume))
        draw.text(xy=(point), 
            text=text,
            fill=color
            # font=font)
        )

    def __draw_text_3D(self, x, y, z, text, color):
        color=tuple([x/255.0 for x in color])
        # mlab.text3d(x,y,z, text, scale=0.3, color=color, figure=self.figure)

    def __get_box_color(self, class_id):
        if type(class_id) == str:
            class_id = class_name_to_label(class_id)

        return self.colors[class_id]

    def __to_numpy(self, pointcloud):
        if not isinstance(pointcloud, np.ndarray):
            return pointcloud.cpu().numpy()
        return pointcloud

    def filter_truncated_boxes(self, box_corners=None):
        assert box_corners is not None

        for corner in box_corners:
            x = corner[0]
            y = corner[1]

            if x < 0 or y < 0:
                return True

        return False



# KITTI = KittiDataset('/home/pointpillars/kitti/training')
# image, pointcloud, labels, calib = KITTI[10]
# visualizer = KittiVisualizer()
# # visualizer.visualize_scene_2D(pointcloud, image, labels, [], calib)
# visualizer.visualize_scene_3D(pointcloud, labels, [], calib)
# objects = model_output_to_kitti_objects(pred)
# visualizer.visualize_scene_3D(pointcloud, objects, labels, calib)

# visualizer.visualize_scene_3D(pointcloud, objects)
# visualizer.visualize_scene_bev(pointcloud, objects, labels, calib=calib)
# visualizer.visualize_2D_image(image, objects, calib)
