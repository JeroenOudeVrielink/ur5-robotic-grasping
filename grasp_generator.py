import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
import torch.utils.data
from PIL import Image
from datetime import datetime

from network.hardware.device import get_device
from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results, save_results
from network.utils.dataset_processing.grasp import detect_grasps
import os


class GraspGenerator:
    IMG_WIDTH = 224
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius):
        self.net = torch.load(net_path, map_location='cpu')
        self.device = get_device(force_cpu=True)

        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(-img_center/self.PIX_CONVERSION,
                                                    img_center/self.PIX_CONVERSION,
                                                    0,
                                                    self.IMG_ROTATION)
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION)

    def get_transform_matrix(self, x, y, z, rot):
        return np.array([
                        [np.cos(rot),   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot),    0,  y],
                        [0,             0,              1,  z],
                        [0,             0,              0,  1]
                        ])

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = np.clip(x_p-self.depth_r, 0, self.IMG_WIDTH)
        x_max = np.clip(x_p+self.depth_r, 0, self.IMG_WIDTH)
        y_min = np.clip(y_p-self.depth_r, 0, self.IMG_WIDTH)
        y_max = np.clip(y_p+self.depth_r, 0, self.IMG_WIDTH)
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p /= self.PIX_CONVERSION
        y_p /= self.PIX_CONVERSION
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.IMG_ROTATION)
        if roll < -np.pi / 2:
            roll += np.pi

        # Covert pixel width to gripper width
        opening_length = (grasp.length / int(self.MAX_GRASP *
                          self.PIX_CONVERSION)) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        # return x, y, z, roll, opening length gripper
        return robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], roll, opening_length, obj_height

    def predict(self, rgb, depth, n_grasps=1, show_output=False):
        depth = np.expand_dims(np.array(depth), axis=2)
        img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
        x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.net.predict(xc)
            pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
            q_img, ang_img, width_img = post_process_output(pred['pos'],
                                                            pred['cos'],
                                                            pred['sin'],
                                                            pred['width'],
                                                            pixels_max_grasp)
            save_name = None
            if show_output:
                fig = plt.figure(figsize=(10, 10))
                plot_results(fig=fig,
                             rgb_img=img_data.get_rgb(rgb, False),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=3,
                             grasp_width_img=width_img)

                if not os.path.exists('network_output'):
                    os.mkdir('network_output')
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_name = 'network_output/{}'.format(time)
                fig.savefig(save_name + '.png')

            grasps = detect_grasps(
                q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
            return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False):
        predictions, save_name = self.predict(
            rgb, depth, n_grasps=n_grasps, show_output=show_output)
        grasps = []
        for grasp in predictions:
            x, y, z, roll, opening_len, obj_height = self.grasp_to_robot_frame(
                grasp, depth)
            grasps.append((x, y, z, roll, opening_len, obj_height))

        return grasps, save_name
