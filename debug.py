from environment.env import Environment
from environment.utilities import Camera
from grasp_generator import GraspGenerator
from network.utils.dataset_processing.grasp import Grasp
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import sys
import pybullet_data
sys.path.append('network')




def load_object(path, pos, orn):
    # load object
    objID = p.loadURDF(path, pos, orn)
    # adjust position according to height
    aabb = p.getAABB(objID, -1)
    y_min, y_max = aabb[0][2], aabb[1][2]    
    pos[2] += (y_max - y_min) / 2 
    p.resetBasePositionAndOrientation(objID, pos, orn)
    #change dynamics
    p.changeDynamics(objID, -1, lateralFriction=1, restitution=0.01)
    # wait until object is at rest
    for _ in range(20):
        p.stepSimulation()


vis = False

physicsClient = p.connect(p.GUI if vis else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeID = p.loadURDF("plane.urdf")
tableID = p.loadURDF("environment/urdf/objects/table.urdf",
                            [0.0, -0.65, 0.76],
                            p.getQuaternionFromEuler([0, 0, 0]),
                            useFixedBase=True)
target_tableID = p.loadURDF("environment/urdf/objects/target_table.urdf",
                            [0.7, 0.0, 0.76],
                            p.getQuaternionFromEuler([0, 0, 0]),
                            useFixedBase=True)
UR5StandID = p.loadURDF("environment/urdf/objects/ur5_stand.urdf",
                                [-0.7, -0.36, 0.0],
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)

p.addUserDebugLine([0.1, -0.55, 0], [0.1, -0.55, 1.9], [0, 1, 0])
p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))




network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
camera = Camera((0.1, -0.55, 1.9), 0.2, 2.0, (244, 244), 40)
generator = GraspGenerator(network_path, camera, 0, -np.pi*0.5, 300)

q_orn = p.getQuaternionFromEuler([0 ,0, 0])
pos = [0.1, -0.55, 0.785]
load_object('objects/test_cube4.urdf', pos, q_orn)

print('Shooting...')
rgb, depth, seg = camera.get_cam_img()
# plt.imshow(seg)
# plt.colorbar(label='Pixel value')
# plt.title('Seg mask image')
# plt.show()

p_x, p_y = np.where(seg == 4.0)
p_x = p_x[0]
p_y = p_y[0]

print(depth[p_x][p_y])
print(depth[p_y][p_x])
print(f'p_x:{p_x}, p_y:{p_y}')

grasp = Grasp([p_x, p_y], 0, 0)

depth = np.expand_dims(np.array(depth), axis=2)
x, y, z, roll, opening_len = generator.grasp_to_robot_frame(grasp, depth)
print(f'x:{x} y:{y}, z:{z}, roll:{roll}, opening len:{opening_len}')


for _ in range(100):
    p.stepSimulation()