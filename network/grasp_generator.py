import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from network.hardware.device import get_device
from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results, save_results
from network.utils.dataset_processing.grasp import detect_grasps


class GraspGenerator:
    def __init__(self, net_path, alpha, ppc, cam_xyz):
        self.net = torch.load(net_path)
        self.device = get_device(force_cpu=False)
        
        self.img_to_cam_3D = np.array([[np.cos(alpha), -np.sin(alpha), 0, -121.5 / 300],
                                        [np.sin(alpha), np.cos(alpha), 0, 121.5 / 300], 
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]])
        self.cam_to_robot_frame = np.array([[1, 0, 0, 0.1],
                                            [0, 1, 0, -0.55], 
                                            [0, 0, 1, 1.90], 
                                            [0, 0, 0, 1]])


        self.pix_p_m = ppc * 100
        self.alpha = alpha
        # top of desk hight is 0.785
        # cam eye y=1.90
        # near plane is 0.2
        # far plane is 2
        self.near = 0.2
        self.far = 2.0
        self.min_z = 0.785



    def grasp_to_robot_frame(self, grasp, depth_img):
        # Convert camera image to camera's 3D space
        
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]
        z_p = depth_img[x_p, y_p, 0]

        # Get depth value for the desk
        z_desk = np.amax(depth_img)

        # Convert pixels to meters
        x_p /= self.pix_p_m
        y_p /= self.pix_p_m
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)
        z_desk = self.far * self.near / (self.far - (self.far - self.near) * z_desk)

        # Calculate height of object and adjust z_p by half the height
        height = z_desk - z_p
        print(f'z_p:{z_p} z_desk:{z_desk} height:{height}')
        z_p = z_desk + (height / 2)
        print(f'z_p after:{z_p}')
        
        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam_3D, img_xyz)
        
        #Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_frame, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.alpha)
        if roll < -np.pi / 2:
            roll += np.pi

        opening_length = grasp.length / self.pix_p_m

        # return x, y, z, roll, and opening length gripper, quality
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
                fig.savefig('img_result.pdf')

            grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
            
            return self.grasp_to_robot_frame(grasps[0], depth)