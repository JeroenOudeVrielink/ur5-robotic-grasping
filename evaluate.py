from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
import pybullet as p
import os
import sys
sys.path.append('network')


def isolated_obj_scenario(n, vis, output, debug):

    objects = YcbObjects('objects/ycb_objects',
                         mod_orn=['ChipsCan', 'MustardBottle',
                                  'TomatoSoupCan'],
                         mod_stiffness=['Strawberry'])
    data = IsolatedObjData(objects.obj_names, n, 'results')

    center_x, center_y = 0.05, -0.52
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((center_x, center_y, 1.9), (center_x,
                    center_y, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=vis, debug=debug)
    generator = GraspGenerator(network_path, camera, 5)

    for obj_name in objects.obj_names:
        print(obj_name)
        for _ in range(n):

            path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
            env.load_isolated_obj(path, mod_orn, mod_stiffness)
            env.move_away_arm()

            rgb, depth, _ = camera.get_cam_img()
            grasps, save_name = generator.predict_grasp(
                rgb, depth, n_grasps=3, show_output=output)
            for i, grasp in enumerate(grasps):
                data.add_try(obj_name)
                x, y, z, roll, opening_len, obj_height = grasp
                if vis:
                    debug_id = p.addUserDebugLine(
                        [x, y, z], [x, y, 1.2], [0, 0, 1])

                succes_grasp, succes_target = env.grasp(
                    (x, y, z), roll, opening_len, obj_height)
                if vis:
                    p.removeUserDebugItem(debug_id)
                if succes_grasp:
                    data.add_succes_grasp(obj_name)
                if succes_target:
                    data.add_succes_target(obj_name)
                    if save_name is not None:
                        os.rename(save_name + '.png', save_name +
                                  f'_SUCCESS_grasp{i}.png')
                    break
                env.reset_all_obj()
            env.remove_all_obj()

    data.write_json()
    summarize(data.save_dir, n)


def pile_scenario(n, vis, output, debug):

    data = PackPileData(5, n, 'results', 'pile')
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
            print(f'N objs:{len(env.obj_ids)} straight fails:{straight_fails}')

            env.move_away_arm()
            env.reset_all_obj()
            rgb, depth, _ = camera.get_cam_img()
            grasps, save_name = generator.predict_grasp(
                rgb, depth, n_grasps=3, show_output=output)

            for i, grasp in enumerate(grasps):
                data.add_try()
                x, y, z, roll, opening_len, obj_height = grasp

                if vis:
                    debugID = p.addUserDebugLine(
                        [x, y, z], [x, y, 1.2], [0, 0, 1])

                succes_grasp, succes_target = env.grasp(
                    (x, y, z), roll, opening_len, obj_height)
                if vis:
                    p.removeUserDebugItem(debugID)
                if succes_grasp:
                    data.add_succes_grasp()
                if succes_target:
                    data.add_succes_target()
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
    data.summarize()


def pack_scenario(n, vis, output, debug):
    vis = vis
    output = output
    debug = debug

    data = PackPileData(5, n, 'results', 'pack')
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
                data.add_try()
                x, y, z, roll, opening_len, obj_height = grasp

                if vis:
                    debugID = p.addUserDebugLine(
                        [x, y, z], [x, y, 1.2], [0, 0, 1])

                succes_grasp, succes_target = env.grasp(
                    (x, y, z), roll, opening_len, obj_height)
                if vis:
                    p.removeUserDebugItem(debugID)
                if succes_grasp:
                    data.add_succes_grasp()
                if succes_target:
                    data.add_succes_target()
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
    data.summarize()


if __name__ == '__main__':
    # isolated_obj_scenario(100, vis=False, output=True, debug=False)
    # pack_scenario(100, vis=False, output=True, debug=False)
    pile_scenario(100, vis=False, output=True, debug=False)
