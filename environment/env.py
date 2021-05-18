from environment.utilities import setup_sisbot, Camera
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import random


class FailToReachTargetError(RuntimeError):
    pass


class Environment:
    OBJECT_INIT_HEIGHT = 1.05
    GRIPPER_MOVING_HEIGHT = 1.25
    GRIPPER_GRASPED_LIFT_HEIGHT = 1.4
    TARGET_ZONE_POS = [0.7, 0.0, 0.685]
    SIMULATION_STEP_DELAY = 1 / 400.
    FINGER_LENGTH = 0.06
    Z_TABLE_TOP = 0.785
    GRIP_REDUCTION = 0.60

    def __init__(self, camera: Camera, vis=False, debug=False, num_objs=3, gripper_type='85', finger_length=0.06) -> None:
        self.vis = vis
        self.debug = debug
        self.num_objs = num_objs
        self.camera = camera

        self.obj_init_pos = (camera.x, camera.y)
        self.obj_id = None
        self.obj_ids = []
        self.obj_pos = None
        self.obj_orn = None
        self.obj_positions = []
        self.obj_orientations = []

        if gripper_type not in ('85', '140'):
            raise NotImplementedError('Gripper %s not implemented.' % gripper_type)
        self.gripper_type = gripper_type
        self.finger_length = finger_length

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF('plane.urdf')
        self.tableID = p.loadURDF('environment/urdf/objects/table.urdf',
                                  [0.0, -0.65, 0.76],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True)
        self.target_table_id = p.loadURDF('environment/urdf/objects/target_table.urdf',
                                  [0.7, 0.0, 0.66],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True)     
        self.target_id = p.loadURDF('environment/urdf/objects/traybox.urdf',
                                  self.TARGET_ZONE_POS,
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True,
                                  globalScaling=0.7)
        self.UR5Stand_id = p.loadURDF('environment/urdf/objects/ur5_stand.urdf',
                                     [-0.7, -0.36, 0.0],
                                     p.getQuaternionFromEuler([0, 0, 0]),
                                     useFixedBase=True)
        self.robot_id = p.loadURDF('environment/urdf/ur5_robotiq_%s.urdf' % gripper_type,
                                  [0, 0, 0.0],  # StartPosition
                                  p.getQuaternionFromEuler([0, 0, 0]),  # StartOrientation
                                  useFixedBase=True,
                                  flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
            setup_sisbot(p, self.robot_id, gripper_type)
        self.eef_id = 7  # ee_link
        
        # Add force sensors
        p.enableJointForceTorqueSensor(self.robot_id, self.joints['left_inner_finger_pad_joint'].id)
        p.enableJointForceTorqueSensor(self.robot_id, self.joints['right_inner_finger_pad_joint'].id)
        
        # Change the friction of the gripper
        p.changeDynamics(self.robot_id, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=1)
        p.changeDynamics(self.robot_id, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=1)
        
        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # Task space (Cartesian space)
        if debug:
            self.xin = p.addUserDebugParameter('x', -0.4, 0.4, 0.11)
            self.yin = p.addUserDebugParameter('y', -0.8, 0, -0.49)
            self.zin = p.addUserDebugParameter('z', 0.9, 1.3, 1.1)
            self.rollId = p.addUserDebugParameter('roll', -3.14, 3.14, 0)  # -1.57 yaw
            self.pitchId = p.addUserDebugParameter('pitch', -3.14, 3.14, np.pi/2)
            self.yawId = p.addUserDebugParameter('yaw', -np.pi/2, np.pi/2, 0)  # -3.14 pitch
            self.gripper_opening_length_control = p.addUserDebugParameter('gripper_opening_length', 0, 0.1, 0.085)

        # Add debug lines for end effector and camera
        if vis:
            self.eef_debug_lineID = None
            p.addUserDebugLine([camera.x, camera.y, 0], [camera.x, camera.y, camera.z], [0, 1, 0])
            p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
            # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) 

        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.1)
        self.ee_position_limit = ((-0.8, 0.8),
                                  (-0.8, 0.8),
                                  (0.785, 1.4))
        self.reset_robot()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:    
            if self.eef_debug_lineID is not None:
                p.removeUserDebugItem(self.eef_debug_lineID)
            eef_xyz = p.getLinkState(self.robot_id, self.eef_id)[0:1]
            end = np.array(eef_xyz[0])
            end[2] -= 0.5
            self.eef_debug_lineID = p.addUserDebugLine(np.array(eef_xyz[0]), end, [1, 0, 0])
            # time.sleep(self.SIMULATION_STEP_DELAY)        

    def remove_obj(self):
        if self.obj_id == None:
            print('WARNING: No object in simulation')
            return
        p.removeBody(self.obj_id)
        self.obj_id = None

    def reset_obj(self):
        p.resetBasePositionAndOrientation(self.obj_id, self.obj_pos, self.obj_orn)

    def reset_all_obj(self):
        for i, obj_id in enumerate(self.obj_ids):
            p.resetBasePositionAndOrientation(obj_id, self.obj_positions[i], self.obj_orientations[i])
        self.wait_until_all_still()      

    @staticmethod
    def is_still(handle):
        still_eps = 1e-3
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        # print(np.abs(lin_vel).sum() + np.abs(ang_vel).sum())
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def wait_until_still(self, objID ,max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if self.is_still(objID):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' % max_wait_epochs)

    
    def wait_until_all_still(self, max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if np.all(list(self.is_still(obj_id) for obj_id in self.obj_ids)):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' % max_wait_epochs)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def reset_robot(self):
        user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                           -1.5707970583733368, 0.0009377758247187636, 0.085)
        for _ in range(60):
            for i, name in enumerate(self.controlJoints):
                if i == 6:
                    self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=user_parameters[i])
                    break
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                self.step_simulation()

    def move_away_arm(self):
        joint = self.joints['shoulder_pan_joint']
        for _ in range(200):
            p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                    targetPosition=0., force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
            self.step_simulation()

    def check_grasped(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left + contact_right if item[2] in [self.obj_id])
        if len(contact_ids) == 1:
            return True
        return False

    def check_grasped_id(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left + contact_right if item[2] in self.obj_ids)
        if len(contact_ids) > 1:
            print('Warning: Multiple items in hand!')
        return list(item_id for item_id in contact_ids if item_id in self.obj_ids)

    def check_contact(self, id_a, id_b):
        contact_a = p.getContactPoints(bodyA=id_a)
        contact_ids = set(item[2] for item in contact_a if item[2] in [id_b])
        if len(contact_ids) == 1:
            return True
        return False

    def check_target_reached(self, obj_id):
        aabb = p.getAABB(self.target_id, -1)
        x_min, x_max = aabb[0][0], aabb[1][0]
        y_min, y_max = aabb[0][1], aabb[1][1]
        pos = p.getBasePositionAndOrientation(obj_id)
        x, y = pos[0][0], pos[0][1]
        if x > x_min and x < x_max and y > y_min and y < y_max:
            return True
        return False

    def gripper_contact(self, bool_operator='and', force=100):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id, linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(self.robot_id, left_index)[2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robot_id, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
            self.step_simulation()

    def auto_close_gripper(self, step: int = 120, check_contact: bool = False) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(self.robot_id, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            # time.sleep(1 / 120)
            if check_contact and self.gripper_contact():
                return True
        return False

    def calc_z_offset(self, gripper_opening_length: float):
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        if self.gripper_type == '140':
            gripper_length = 10.3613 * np.sin(1.64534-0.24074 * (gripper_opening_angle / np.pi)) - 10.1219
        else:
            gripper_length = 1.231 - 1.1
        return gripper_length

    def load_isolated_obj(self, path, special_case=False):
        r_x = random.uniform(self.obj_init_pos[0] - 0.1, self.obj_init_pos[0] + 0.1)
        r_y = random.uniform(self.obj_init_pos[1] - 0.1, self.obj_init_pos[1] + 0.1)
        yaw = random.uniform(0, np.pi)
        
        pos = [r_x, r_y, self.Z_TABLE_TOP]
        orn = p.getQuaternionFromEuler([0, 0, yaw])        
        self.obj_id = p.loadURDF(path, pos, orn)
        # adjust position according to height
        aabb = p.getAABB(self.obj_id, -1)
        if special_case:
            minm, maxm = aabb[0][1], aabb[1][1]
            orn = p.getQuaternionFromEuler([0, np.pi*0.5, yaw])
        else:
            minm, maxm = aabb[0][2], aabb[1][2]    
        
        pos[2] += (maxm - minm) / 2 
        p.resetBasePositionAndOrientation(self.obj_id, pos, orn)   
        #change dynamics
        p.changeDynamics(self.obj_id, -1, lateralFriction=1, restitution=0.01)
        # wait until object is at rest
        for _ in range(10):
            self.step_simulation()
        self.wait_until_still(self.obj_id)
        self.obj_pos, self.obj_orn = p.getBasePositionAndOrientation(self.obj_id)

    def create_temp_box(self):
        box_size = 0.35
        box_height = 0.2
        box_z = self.Z_TABLE_TOP + (box_height/2)
        id1 = p.loadURDF('environment/urdf/objects/slab.urdf',
                                [self.obj_init_pos[0] - box_size/2, self.obj_init_pos[1], box_z],
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)
        id2 = p.loadURDF('environment/urdf/objects/slab.urdf',
                                [self.obj_init_pos[0] + box_size/2, self.obj_init_pos[1], box_z],
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)
        id3 = p.loadURDF('environment/urdf/objects/slab.urdf',
                                [self.obj_init_pos[0], self.obj_init_pos[1] + box_size/2, box_z],
                                p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                                useFixedBase=True)
        id4 = p.loadURDF('environment/urdf/objects/slab.urdf',
                                [self.obj_init_pos[0] , self.obj_init_pos[1] - box_size/2, box_z],
                                p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                                useFixedBase=True)
        return [id1, id2, id3, id4]

    def create_pile(self, objects):
        box_ids = self.create_temp_box()
        for obj_path in objects:
            yaw = random.uniform(0, np.pi)
            roll = random.uniform(0, np.pi)
            # roll = 0

            pos = [self.obj_init_pos[0], self.obj_init_pos[1], 1.0]
            orn = p.getQuaternionFromEuler([roll, 0, yaw])        
            objID = p.loadURDF(obj_path, pos, orn)
            self.obj_ids.append(objID)
            p.changeDynamics(objID, -1, lateralFriction=1, restitution=0.01)
            for _ in range(10):
                self.step_simulation()
            self.wait_until_still(objID, 150)
        
        self.wait_until_all_still()
        for handle in box_ids:
            p.removeBody(handle)
        self.wait_until_all_still()
        
        self.obj_positions = []
        self.obj_orientations = []
        for obj_id in self.obj_ids:
            obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
            self.obj_positions.append(obj_pos)
            self.obj_orientations.append(obj_orn)

    def move_ee(self, action, max_step=300, check_collision_config=None, custom_velocity=None,
                try_close_gripper=False, verbose=False):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0
        still_open_flag_ = True  # Hot fix
        for _ in range(max_step):
            # apply IK
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.eef_id, [x, y, z], orn,
                                                       maxNumIterations=100, jointDamping=jd
                                                       )
            for i, name in enumerate(self.controlJoints[:-1]):  # Filter out the gripper
                joint = self.joints[name]
                pose = joint_poses[i]
                # control robot end-effector
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            self.step_simulation()
            if try_close_gripper and still_open_flag_ and not self.gripper_contact():
                still_open_flag_ = self.close_gripper(check_contact=True)
            # Check if contact with objects
            if check_collision_config and self.gripper_contact(**check_collision_config):
                if self.debug:
                    print('Collision detected!', self.check_grasped_id())
                return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]
            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(self.robot_id, self.eef_id)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                if verbose:
                    print('Reach target with', _, 'steps')
                return True, (real_xyz, real_xyzw)

        # raise FailToReachTargetError
        if self.debug:
            print('Failed to reach the target')
        return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]

    def grasp(self, pos: tuple, roll: float, gripper_opening_length: float, obj_height: float, debug: bool = False):
        """
        Method to perform grasp
        pos [x y z]: The axis in real-world coordinate
        roll: float,   for grasp, it should be in [-pi/2, pi/2)
        """
        succes_grasp, succes_target = False, False
        obj_id = None        
        
        x, y, z = pos
        # Substracht gripper finger length from z
        z -= self.finger_length
        z = np.clip(z, *self.ee_position_limit[2])

        # Move above target
        self.reset_robot()
        self.move_gripper(0.1)
        orn = p.getQuaternionFromEuler([roll, np.pi/2, 0.0])
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # Reduce grip to get a tighter grip
        gripper_opening_length *= self.GRIP_REDUCTION

        # Grasp and lift object
        z_offset = self.calc_z_offset(gripper_opening_length)
        self.move_ee([x, y, z + z_offset, orn])
        # self.move_gripper(gripper_opening_length)
        self.auto_close_gripper(check_contact=True)
        for _ in range(10):
            self.step_simulation()
        self.move_ee([self.camera.x, self.camera.y, self.GRIPPER_MOVING_HEIGHT, orn])

        # If the object has been grasped and lifted off the table
        grasped_id = self.check_grasped_id()
        if len(grasped_id) == 1:
            succes_grasp = True
            obj_id = grasped_id[0]
        else:
            return succes_target, succes_grasp

        # Move object to target zone
        y_drop = self.TARGET_ZONE_POS[2] + z_offset + obj_height + 0.15
        y_orn = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])
        # y_orn = np.array(p.getLinkState(self.robotID, self.eefID)[1:2][0])

        self.move_away_arm()
        self.move_ee([self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1], 1.25, y_orn])
        self.move_ee([self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1], y_drop, y_orn])
        self.move_gripper(0.085)
        self.move_ee([self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1], self.GRIPPER_MOVING_HEIGHT, y_orn])

        #Wait then check if object is in target zone
        for _ in range(50):
            self.step_simulation()
        if self.check_target_reached(obj_id):
            succes_target = True
            idx = self.obj_ids.index(obj_id)
            self.obj_orientations.pop(idx)
            self.obj_positions.pop(idx)
            self.obj_ids.pop(idx)
            p.removeBody(obj_id)

        return succes_grasp, succes_target

    def close(self):
        p.disconnect(self.physicsClient)
