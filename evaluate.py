from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from objects.objects_wrapper import YcbObjects
import pybullet as p
import numpy as np
import sys
import random
import os
import random
sys.path.append('network')

def random_pos(x, y, x_range, y_range):
    r_x = random.uniform(x - x_range, x + x_range)
    r_y = random.uniform(y - y_range, y + y_range)
    orn = random.uniform(0, np.pi)
    return r_x, r_y, orn

def test(n):
    vis = True
    output = True

    center_x, center_y = 0.05, -0.52
    objects = YcbObjects('objects/ycb_objects', 'results', n)
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((center_x, center_y, 1.9), (center_x, center_y, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=vis, debug=True, num_objs=0, gripper_type='140')
    generator = GraspGenerator(network_path, camera, 5)

    objects.obj_names = ['TomatoSoupCan', 'ChipsCan']

    for obj_name in objects.obj_names:
        print(obj_name)
        
        for _ in range(n):
            x, y, orn = random_pos(center_x, center_y, 0.1, 0.1)
            q_orn = p.getQuaternionFromEuler([0, 0, orn])
            env.load_object(objects.get_obj_path(obj_name), [x, y, 0.785], q_orn)
            env.move_away_arm()
            
            rgb, depth, _ = camera.get_cam_img()
            pred, save_name = generator.predict(rgb, depth, output)
            x, y, z, roll, opening_len, obj_height = pred
            # print(f'x:{x} y:{y}, z:{z}, roll:{roll}, opening len:{opening_len}, obj height:{obj_height}')
            if vis:
                debugID = p.addUserDebugLine([x, y, z], [x, y, 1.2], [0, 0, 1])
            succes_grasp, succes_target = env.grasp((x, y, z), roll, opening_len, obj_height)
            # print(f'Grasped:{succes_grasp} Target:{succes_target}')
            
            env.remove_object()
            if vis:
                p.removeUserDebugItem(debugID)
            
            if succes_target:
                objects.add_succes_target(obj_name)
            if succes_grasp:
                objects.add_succes_grasp(obj_name)
                os.rename(save_name + '.png', save_name + '_SUCCESS.png')

    objects.summarize_results()

if __name__ == '__main__':
    test(1)