from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
import pybullet as p
import numpy as np
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

def test():

    camera = Camera((0.1, -0.55, 1.9), 0.2, 2.0, (244, 244), 40)

    env = Environment(camera, vis=True, debug=True, num_objs=0, gripper_type='140')
    q_orn = p.getQuaternionFromEuler([0 ,0, 0])
    
    # env.load_object('ycb_objects/YcbCrackerBox/model.urdf', [0.1, -0.55, 0.785], q_orn)
    # env.move_away_arm()
    # camera.shot()

    # succes_grasp, succes_target = env.grasp((0.1, -0.55, 0.925), np.pi*0.5, 0.040, 0.210)
    # print(f'Grasped:{succes_grasp} Target:{succes_target}')
    # env.remove_object()

    env.load_object('objects/ycb_objects/YcbBanana/model.urdf', [0.1, -0.55, 0.785], q_orn)
    env.move_away_arm()
    camera.shot()

    succes_grasp, succes_target = env.grasp((0.1, -0.55, 0.785), 0, 0.030, 0.035)
    print(f'Grasped:{succes_grasp} Target:{succes_target}')
    env.remove_object()

    env.load_object('objects/ycb_objects/YcbMustardBottle/model.urdf', [0.1, -0.55, 0.785], q_orn)
    env.move_away_arm()
    camera.shot()

    succes_grasp, succes_target = env.grasp((0.1, -0.55, 0.87), np.pi*0.5, 0.040, 0.175)
    print(f'Grasped:{succes_grasp} Target:{succes_target}')
    env.remove_object()


if __name__ == '__main__':
    test()