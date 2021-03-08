from environment.ur5_environment import ur5_environment


if __name__ == '__main__':
    env = ur5_environment(gui_mode=True)
    env.load_object("/environment/urdf/objects/block.urdf")

    env.run_simulation(x=0.11, y=-0.5, z=1.45, roll=0, pitch=1.57, yaw=-1.57, grip=0.085, delayed_grip=False)
    env.run_simulation(x=0.11, y=-0.5, z=1.05, roll=0, pitch=1.57, yaw=-1.57, grip=0.045, delayed_grip=True)
    env.run_simulation(x=0.11, y=-0.5, z=1.30, roll=0, pitch=1.57, yaw=-1.57, grip=0.045, delayed_grip=False)