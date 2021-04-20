from environment.ur5_environment import Environment
from objects.pybullet_object_models import ycb_objects
import matplotlib.pyplot as plt
from imageio import imsave
import numpy as np
import os
import pybullet as p
from network.grasp_generator import GraspGenerator
import sys
sys.path.append('network')

# from grasp_generator import GraspGenerator


def calc_ppc(seg_mask):
    print(seg_mask.shape)
    print(seg_mask[122, :])
    print(len(seg_mask))
    unique, counts = np.unique(seg_mask[122, :], return_counts=True)
    print(dict(zip(unique, counts)))


if __name__ == '__main__':

    env = Environment(gui_mode=False, show_debug=False)
    
    obj_start_pos = [0.1, -0.65, 0.81]
    obj_start_orn = [0, 0, 0.5]

    # Setup Network
    conv_net = GraspGenerator(
        'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98', 
        alpha= -np.pi * 0.5,
        ppc=3, 
        cam_xyz= env.camera_eye_xyz,)
    
    #x=0.1, y=-0.5, z=1.45, roll=0, pitch=1.57, yaw=-1.57, grip=0.085, delayed_grip=False
    robot_start_pos = [0.4, -0.1, 0.9, 0, 1.57, -1.57, 0.085, False]
    robot_target_zone_pos = [0.50, -0.40, 0.85, 0, 1.57, -1.57, 0.085, True]
    # robot_target_zone_pos = [0.2, -0.50, 0.85, 0, 1.57, -1.57, 0.085, True]


    objects = ['YcbPottedMeatCan']
    # objects = ['objects/test_cube2.urdf']

    env.run_simulation(0.1, -0.55, 1.0, 0, 1.57, -1.57, 0.085, False, True)

    for obj_name in objects:
        # Load object
        env.load_object(os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf"), obj_start_pos, obj_start_orn)
        # env.load_object(obj_name, obj_start_pos, obj_start_orn)
        # Bring robot in start pos
        env.run_simulation(*robot_start_pos)
        # Get camera images
        rgb_img, depth_img, seg_mask = env.get_camera_image()

        # imsave(f'{obj_name}1.png', rgb_img)
        # imsave(f'{obj_name}1.tiff', depth_img.astype(np.float32))
        
        # plt.imshow(depth_img)
        # plt.colorbar(label='Pixel value')
        # plt.title('Seg mask image')
        # plt.show()


        # plt.imshow(rgb_img)
        # plt.show()

        # ==============================================
        # GR-ConvNet
        x, y, z, roll, opening_len = conv_net.predict(rgb_img, depth_img, show_output=True)
        print(f'x:{x} y:{y}, z:{z}, roll:{roll}, opening len:{opening_len}')

        # ==============================================

        env.run_simulation(x=x, y=y, z=1.0, roll=roll, pitch=1.57, yaw=-1.57, grip=0.085, delayed_grip=False)
        print('pre pos')
        env.run_simulation(x=x, y=y, z=z, roll=roll, pitch=1.57, yaw=-1.57, grip=0.03, delayed_grip=True)
        print('move done')

        robot_target_zone_pos[3] = roll
        env.run_simulation(*robot_target_zone_pos)

        # Check if object has been successfully placed in the target zone
        print(env.check_if_successful())
        # Remove object
        env.remove_current_object()