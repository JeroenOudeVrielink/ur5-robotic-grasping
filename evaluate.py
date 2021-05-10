from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from objects.objects_wrapper import YcbObjects
import pybullet as p
import numpy as np
import sys
import random
import os
sys.path.append('network')

def random_pos(x, y, x_range, y_range):
    r_x = np.random.uniform(x - x_range, x + x_range)
    r_y = np.random.uniform(y - y_range, y + y_range)
    orn = np.random.uniform(0, np.pi)
    return r_x, r_y, orn

def test(n):

    objects = YcbObjects('objects/ycb_objects', 'results', n)
    
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((0.05, -0.55, 1.9), (0.05, -0.55, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=False, debug=True, num_objs=0, gripper_type='140')
    generator = GraspGenerator(network_path, camera, 5)

    # objects.obj_names = ['CrackerBox']


    for obj_name in objects.obj_names:
        print(obj_name)
        
        for _ in range(n):
            x, y, orn = random_pos(0.05, -0.55, 0.2, 0.2)
            q_orn = p.getQuaternionFromEuler([0, 0, orn])

            env.load_object(objects.get_obj_path(obj_name), [0.1, -0.55, 0.785], q_orn)
            env.move_away_arm()
            
            rgb, depth, _ = camera.get_cam_img()
            x, y, z, roll, opening_len, obj_height = generator.predict(rgb, depth, False)
            # print(f'x:{x} y:{y}, z:{z}, roll:{roll}, opening len:{opening_len}, obj height:{obj_height}')
            # debugID = p.addUserDebugLine([x, y, z], [x, y, 1.1], [0, 0, 1])
            succes_grasp, succes_target = env.grasp((x, y, z), roll, opening_len, obj_height)
            # print(f'Grasped:{succes_grasp} Target:{succes_target}')
            env.remove_object()
            # p.removeBody(debugID)
            
            if succes_target:
                objects.add_succes_target(obj_name)
            if succes_grasp:
                objects.add_succes_grasp(obj_name)

    objects.summarize_results()

if __name__ == '__main__':
    test(15)