from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects
import pybullet as p
import argparse
import os
import sys
sys.path.append('network')


def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')

    parser.add_argument('--scenario', type=str, default='isolated',
                        help='Grasping scenario (isolated/pack/pile)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs the scenario is executed')
    parser.add_argument('--show-network-output', dest='output', type=bool, default=False,
                        help='Show network output (True/False)')

    args = parser.parse_args()
    return args


def isolated_obj_scenario(n, vis, output, debug):

    objects = YcbObjects('objects/ycb_objects',
                         mod_orn=['ChipsCan', 'MustardBottle',
                                  'TomatoSoupCan'],
                         mod_stiffness=['Strawberry'])
    center_x, center_y = 0.05, -0.52
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((center_x, center_y, 1.9), (center_x,
                    center_y, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=vis, debug=debug)
    generator = GraspGenerator(network_path, camera, 5)

    objects.shuffle_objects()
    for _ in range(n):
        for obj_name in objects.obj_names:
            print(obj_name)

            path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
            env.load_isolated_obj(path, mod_orn, mod_stiffness)
            env.move_away_arm()

            rgb, depth, _ = camera.get_cam_img()
            grasps, save_name = generator.predict_grasp(
                rgb, depth, n_grasps=3, show_output=output)
            for i, grasp in enumerate(grasps):
                x, y, z, roll, opening_len, obj_height = grasp
                if vis:
                    debug_id = p.addUserDebugLine(
                        [x, y, z], [x, y, 1.2], [0, 0, 1], lineWidth=3)

                succes_grasp, succes_target = env.grasp(
                    (x, y, z), roll, opening_len, obj_height)
                if vis:
                    p.removeUserDebugItem(debug_id)
                if succes_target:
                    if save_name is not None:
                        os.rename(save_name + '.png', save_name +
                                  f'_SUCCESS_grasp{i}.png')
                    break
                env.reset_all_obj()
            env.remove_all_obj()


def pile_scenario(n, vis, output, debug):

    objects = YcbObjects('objects/ycb_objects',
                         mod_orn=['ChipsCan', 'MustardBottle',
                                  'TomatoSoupCan'],
                         mod_stiffness=['Strawberry'],
                         exclude=['CrackerBox', 'Hammer'])
    center_x, center_y = 0.05, -0.52
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((center_x, center_y, 1.9), (center_x,
                    center_y, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
    generator = GraspGenerator(network_path, camera, 5)

    for i in range(n):
        print(f'Trial {i}')
        straight_fails = 0
        objects.shuffle_objects()

        env.move_away_arm()
        info = objects.get_n_first_obj_info(5)
        env.create_pile(info)

        straight_fails = 0
        while len(env.obj_ids) != 0 and straight_fails < 3:
            env.move_away_arm()
            env.reset_all_obj()
            rgb, depth, _ = camera.get_cam_img()
            grasps, save_name = generator.predict_grasp(
                rgb, depth, n_grasps=3, show_output=output)

            for i, grasp in enumerate(grasps):
                x, y, z, roll, opening_len, obj_height = grasp

                if vis:
                    debugID = p.addUserDebugLine(
                        [x, y, z], [x, y, 1.2], [0, 0, 1], lineWidth=3)

                succes_grasp, succes_target = env.grasp(
                    (x, y, z), roll, opening_len, obj_height)
                if vis:
                    p.removeUserDebugItem(debugID)
                if succes_target:
                    straight_fails = 0
                    if save_name is not None:
                        os.rename(save_name + '.png', save_name +
                                  f'_SUCCESS_grasp{i}.png')
                    break
                else:
                    straight_fails += 1

                if straight_fails == 3 or len(env.obj_ids) == 0:
                    break

                env.reset_all_obj()
        env.remove_all_obj()


def pack_scenario(n, vis, output, debug):
    vis = vis
    output = output
    debug = debug

    objects = YcbObjects('objects/ycb_objects',
                         mod_orn=['ChipsCan', 'MustardBottle',
                                  'TomatoSoupCan'],
                         mod_stiffness=['Strawberry'])
    center_x, center_y = 0.05, -0.52
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((center_x, center_y, 1.9), (center_x,
                    center_y, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
    generator = GraspGenerator(network_path, camera, 5)

    for i in range(n):
        print(f'Trial {i}')
        straight_fails = 0
        objects.shuffle_objects()
        info = objects.get_n_first_obj_info(5)
        env.create_packed(info)

        straight_fails = 0
        while len(env.obj_ids) != 0 and straight_fails < 3:
            env.move_away_arm()
            env.reset_all_obj()
            rgb, depth, _ = camera.get_cam_img()
            grasps, save_name = generator.predict_grasp(
                rgb, depth, n_grasps=3, show_output=output)

            for i, grasp in enumerate(grasps):
                x, y, z, roll, opening_len, obj_height = grasp

                if vis:
                    debugID = p.addUserDebugLine(
                        [x, y, z], [x, y, 1.2], [0, 0, 1], lineWidth=3)

                succes_grasp, succes_target = env.grasp(
                    (x, y, z), roll, opening_len, obj_height)
                if vis:
                    p.removeUserDebugItem(debugID)
                if succes_target:
                    straight_fails = 0
                    if save_name is not None:
                        os.rename(save_name + '.png', save_name +
                                  f'_SUCCESS_grasp{i}.png')
                    break
                else:
                    straight_fails += 1

                if straight_fails == 3 or len(env.obj_ids) == 0:
                    break

                env.reset_all_obj()
        env.remove_all_obj()


if __name__ == '__main__':
    args = parse_args()
    output = args.output
    runs = args.runs

    if args.scenario == 'isolated':
        isolated_obj_scenario(runs, vis=True, output=output, debug=False)
    elif args.scenario == 'pack':
        pack_scenario(runs, vis=True, output=output, debug=False)
    elif args.scenario == 'pile':
        pile_scenario(runs, vis=True, output=output, debug=False)
