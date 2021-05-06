import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
from datetime import datetime

from network.hardware.device import get_device
from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results, save_results
from network.utils.dataset_processing.grasp import detect_grasps


class GraspGenerator:
    
    def __init__(self, net_path, camera, cam_rot, img_rot, pix_conv):
        self.net = torch.load(net_path)
        self.device = get_device(force_cpu=False)
        
        self.pix_conv = pix_conv
        self.img_rot = img_rot
        self.near = camera.near
        self.far = camera.far
        
        # Get rotation matrix
        img_center = camera.width / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(-img_center/pix_conv, img_center/pix_conv, 0, img_rot)
        self.cam_to_robot_base = self.get_transform_matrix(camera.x, camera.y, camera.z, cam_rot)

        # self.img_to_cam_3D = np.array([[np.cos(self.alpha), -np.sin(self.alpha), 0, -self.img_center / self.pix_conv],
        #                                 [np.sin(self.alpha), np.cos(self.alpha), 0, self.img_center / self.pix_conv], 
        #                                 [0, 0, 1, 0], 
        #                                 [0, 0, 0, 1]])
        # self.cam_to_robot_frame = np.array([[1, 0, 0, 0.1],
        #                                     [0, 1, 0, -0.60], 
        #                                     [0, 0, 1, 1.90], 
        #                                     [0, 0, 0, 1]])

    def get_transform_matrix(self, x, y, z, rot):
        return np.array([
                        [np.cos(rot),   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot),    0,  y], 
                        [0,             0,              1,  z], 
                        [0,             0,              0,  1]
                        ])

    def grasp_to_robot_frame(self, grasp, depth_img):
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]
        z_p = depth_img[x_p, y_p, 0]

        # Get depth value for the desk
        z_desk = np.amax(depth_img)

        # Convert pixels to meters
        x_p /= self.pix_conv
        y_p /= self.pix_conv
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)
        z_desk = self.far * self.near / (self.far - (self.far - self.near) * z_desk)

        # Calculate height of object and adjust z_p by half the height

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)
        
        #Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.img_rot)
        if roll < -np.pi / 2:
            roll += np.pi

        # Covert pixel width to gripper width
        opening_length = (grasp.length / 45) * 0.14

        # return x, y, z, roll, opening length gripper
        return robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], roll, opening_length

    def predict(self, rgb, depth, show_output=False):
    
        depth = np.expand_dims(np.array(depth), axis=2)
        img_data = CameraData(width=244, height=244)
        x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

        # plt.imshow(depth_img[0])
        # plt.colorbar(label='Pixel value')
        # plt.title('Depth image')
        # plt.show()

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.net.predict(xc)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

            if show_output:
                fig = plt.figure(figsize=(10, 10))
                plot_results(fig=fig,
                                rgb_img=img_data.get_rgb(rgb, False),
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                no_grasps=1,
                                grasp_width_img=width_img)
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                fig.savefig('results/{}.png'.format(time))
            
            grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)  
            return self.grasp_to_robot_frame(grasps[0], depth)