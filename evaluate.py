from math import e
from environment.ur5_environment import Environment
from objects.pybullet_object_models import ycb_objects
from grasp_generator import GraspGenerator
import numpy as np
import random
import os
import sys
sys.path.append('network')

def calc_ppc(seg_mask):
    print(seg_mask.shape)
    print(seg_mask[122, :])
    print(len(seg_mask))
    unique, counts = np.unique(seg_mask[122, :], return_counts=True)
    print(dict(zip(unique, counts)))


def random_pos(x, y, x_range, y_range):
    r_x = np.random.uniform(x - x_range, x + x_range)
    r_y = np.random.uniform(y - y_range, y + y_range)
    orn = np.random.uniform(0, np.pi)
    return r_x, r_y, orn

def evaluate(n_trials):
    gui_mode = True
    show_output = True

    # Get list of object namesS
    list_dirs = os.listdir('objects/pybullet_object_models/ycb_objects')
    objects = [item for item in list_dirs if item.startswith('Ycb')]
    random.shuffle(objects)

    # Create dictironary to count succesfull grasp
    succes_rate = dict.fromkeys(objects , 0)

    # Setup Network
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    conv_net = GraspGenerator(network_path)
    
    # Create environment
    env = Environment(gui_mode)
    robot_start_pos = [-0.3, -0.3, 1.1, 0, 1.57, -1.57, 0.14]
    # target zone coordinates
    x_t, y_t = env.target_zone_pos[0], env.target_zone_pos[1]
    # Bring robot in start neutral position
    env.run(0.1, -0.55, 1.1, 0, 1.57, -1.57, 0.14, delayed_grip=False, setup=True)

    # objects = ['YcbCrackerBox','YcbChipsCan', 'YcbPowerDrill', 'YcbMustardBottle', 'YcbGelatinBox', 'YcbTomatoSoupCan']
    # objects = ['YcbStrawberry', 'YcbScissors', 'YcbMediumClamp', 'YcbTennisBall', 'YcbPear']
    objects = [ 'YcbBanana']

    x_o, y_o, z_o = 0.1, -0.60, 0.785

    for obj in objects:
        for n in range(n_trials):

            # Get random x, y, and orientation
            x, y, orn = random_pos(x_o, y_o, 0.1, 0.1)
            # Load object
            env.load_object(os.path.join(ycb_objects.getDataPath(), obj, "model.urdf"), [x, y, z_o], [0, 0, orn])
            # env.load_object(os.path.join(ycb_objects.getDataPath(), obj, "model.urdf"), [0.1, -0.6, z_o], [0, 0, 0])
            # env.load_object('objects/test_cube1.urdf', [0.1, -0.6, z_o], [0, 0, 0])
        
            # Bring robot in start pos
            env.run(*robot_start_pos)

            env.pause_till_obj_at_rest()
            # Get camera images
            rgb_img, depth_img, seg_mask = env.get_camera_image()

            # GR-ConvNet
            x, y, z, roll, opening_len, obj_height = conv_net.predict(rgb_img, depth_img, show_output)
            # x, y, z, roll, opening_len, obj_height = 0.1, -0.6, 0.785, 0.5 * np.pi, 0.0438, 0.1
            print(obj + f'x:{x}, y:{y}, z{z}, roll{roll}, opening len:{opening_len}, obj_height:{obj_height}')
            opening_len *= 0.50
            # print(f'reduced:{opening_len}')

            # Move above object
            env.run(x, y, 1.1, roll, 1.57, -1.57, 0.14, delayed_grip=False)
            # Grip object
            env.run(x, y, z, roll, 1.57, -1.57, opening_len, delayed_grip=True)

            z_t = 0.8 + obj_height
            env.run(x_t, y_t, z_t, roll, 1.57, -1.57, opening_len, rpy_margin = 0.35, delayed_grip=False)
            env.run(x_t, y_t, z_t, roll, 1.57, -1.57, 0.14, rpy_margin = 0.35, delayed_grip=True)

            # Check if object has been successfully placed in the target zone
            if env.check_if_successful():
                succes_rate[obj] += 1
            
            # Remove object
            env.remove_current_object()
    
    print(succes_rate)


if __name__ == '__main__':
    evaluate(1)