from environment.ur5_environment import Environment
from objects.pybullet_object_models import ycb_objects
import matplotlib.pyplot as plt
from imageio import imsave
import numpy as np
import os
import pybullet as p
from grasp_generator import GraspGenerator
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

    env = Environment(gui_mode=False)
    
    obj_start_pos = [0.1, -0.60, 0.786]
    obj_start_orn = [0, 0, 0]

    # Setup Network
    conv_net = GraspGenerator('network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98')
    
    #x=0.1, y=-0.5, z=1.45, roll=0, pitch=1.57, yaw=-1.57, grip=0.085, delayed_grip=False
    robot_start_pos = [-0.3, -0.3, 0.9, 0, 1.57, -1.57, 0.14, False]
    robot_target_zone_pos = [env.target_zone_pos[0], env.target_zone_pos[1], 0.95, 0, 1.57, -1.57, 0.14, True]
    # robot_target_zone_pos = [0.2, -0.50, 0.85, 0, 1.57, -1.57, 0.085, True]

    # Bring robot in neutral position
    env.run(0.1, -0.55, 1.0, 0, 1.57, -1.57, 0.14, delayed_grip=False, setup=True)

    objects = ['objects/test_cube1.urdf']

    for obj_name in objects:
        # Load object
        # env.load_object(os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf"), obj_start_pos, obj_start_orn)
        env.load_object(obj_name, obj_start_pos, obj_start_orn)
        # Bring robot in start pos
        env.run(*robot_start_pos)
        # print(0)
        # env.run(0.1, -0.55, 0.785, 0, 1.57, -1.57, 0, delayed_grip=True)
        # print(1)
        # env.run(0.1, -0.55, 0.9, 0, 1.57, -1.57, 0.14, delayed_grip=True)

        # env.run(0.1, -0.55, 0.785, 0, 1.57, -1.57, 0.01, delayed_grip=True)
        # print(2)
        # env.run(0.1, -0.55, 0.9, 0, 1.57, -1.57, 0.14, delayed_grip=True)
        # env.run(0.1, -0.55, 0.785, 0, 1.57, -1.57, 0.02, delayed_grip=True)
        # print(3)
        # env.run(0.1, -0.55, 0.9, 0, 1.57, -1.57, 0.14, delayed_grip=True)
        # env.run(0.1, -0.55, 0.785, 0, 1.57, -1.57, 0.03, delayed_grip=True)
        # print(4)
        # env.run(0.1, -0.55, 0.9, 0, 1.57, -1.57, 0.14, delayed_grip=True)
        # env.run(0.1, -0.55, 0.785, 0, 1.57, -1.57, 0.04, delayed_grip=True)
        # print(5)
        # env.run(0.1, -0.55, 0.9, 0, 1.57, -1.57, 0.14, delayed_grip=True)
        # env.run(0.1, -0.55, 0.785, 0, 1.57, -1.57, 0.05, delayed_grip=True)
        # print(6)
        # env.run(0.1, -0.55, 0.9, 0, 1.57, -1.57, 0.14, delayed_grip=True)
        # env.run(0.1, -0.55, 0.785, 0, 1.57, -1.57, 0.06, delayed_grip=True)

        # env.run_simulation(x=0.1, y=-0.55, z=0.785, roll=0, pitch=1.57, yaw=-1.57, grip=0.01, delayed_grip=True)
        # env.keep_running()

        # Get camera images
        rgb_img, depth_img, seg_mask = env.get_camera_image()
        calc_ppc(seg_mask)

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
        # x, y, z, roll, opening_len, obj_height = conv_net.predict(rgb_img, depth_img, show_output=True)
        # print(f'x:{x} y:{y}, z:{z}, roll:{roll}, opening len:{opening_len}')

        # ==============================================

        # x=0.08166666666666667
        # y=-0.6016666666666666
        # z=0.8171228060396258
        # roll=0.885853306451116
        # opening_len=0.21882232666015625

        
        # env.run(x=x, y=y, z=1.0, roll=roll, pitch=1.57, yaw=-1.57, grip=0.14, delayed_grip=False)
        # # # print('pre pos')
        # env.run(x=x, y=y, z=z, roll=roll, pitch=1.57, yaw=-1.57, grip=opening_len / 10, delayed_grip=True)
        # # print('move done')
        
        # robot_target_zone_pos[3] = roll
        # env.run(*robot_target_zone_pos, rpy_margin=0.2, show_debug=False)

        # # # Check if object has been successfully placed in the target zone
        # print(env.check_if_successful())
        # # # Remove object
        # env.remove_current_object()