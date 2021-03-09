from environment.ur5_environment import Environment


if __name__ == '__main__':
    env = Environment(gui_mode=True)

    obj_start_pos = [0.1, -0.50, 0.85]
    obj_start_orn = [0, 0, 0]
    env.load_object("/environment/urdf/objects/block.urdf", obj_start_pos, obj_start_orn)

    
    #x=0.1, y=-0.5, z=1.45, roll=0, pitch=1.57, yaw=-1.57, grip=0.085, delayed_grip=False
    robot_start_pos = [0.1, -0.5, 1.45, 0, 1.57, -1.57, 0.085, False]

    env.run_simulation(*robot_start_pos)
    env.run_simulation(x=0.1, y=-0.5, z=1.05, roll=0, pitch=1.57, yaw=-1.57, grip=0.045, delayed_grip=True)
    env.run_simulation(x=-0.1, y=-0.7, z=1.30, roll=0, pitch=1.57, yaw=-1.57, grip=0.085, delayed_grip=True)
    env.run_simulation(*robot_start_pos)
    print(env.check_if_successful())
    env.remove_current_object()
    env.keep_running()