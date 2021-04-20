import pybullet as p
import pybullet_data
from . import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import tf


class Environment():
    """ A class for the running a ur5 robot with a gripper"""

    def __init__(self, gui_mode=False, show_debug=False):
        """ Constructor: Initializes the robot and its environment"""
        # Define camera image parameter
        self.width = 244
        self.height = 244
        self.fov = 40
        self.aspect = self.width / self.height
        self.near = 0.2
        # ORIGINAL
        # self.far = 0.65
        # self.camera_eye_xyz = [0.1, -0.5, 1.35]        
        # # TEST
        self.far = 2.
        self.camera_eye_xyz = [0.1, -0.55, 1.90]
        self.camera_target_xyz = [0.1, -0.55, 0.90]
        # Compute view and projection matrix
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        self.view_matrix = p.computeViewMatrix(self.camera_eye_xyz, self.camera_target_xyz, [0, 1, 0])
        
        # xyz of the target position
        self.target_zone_pos = [0.60, -0.50, 0.785]
        # Compute min and max x,y coordinates of the target zone
        target_size = 0.2
        margin = target_size / 2;
        self.min_x = self.target_zone_pos[0] - margin;
        self.max_x = self.target_zone_pos[0] + margin;
        self.min_y = self.target_zone_pos[1] - margin;
        self.max_y = self.target_zone_pos[1] + margin;

        # Server mode
        self.gui_mode = gui_mode
        self.show_debug = show_debug
        # ID of the to be grasped object
        self.object_ID = None
        # Standard opening length of the gripper
        self.gripper_opening_length = 0.085
        # Error to adjust for gripper length
        self.gripper_length = 0.236

        self.z_min = 0.785
    
        # Connect to engine servers
        p.connect(p.GUI if gui_mode else p.DIRECT)

        # Add search path for loadURDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define world
        p.setGravity(0,0,-10)
        p.loadURDF("plane.urdf")
        
        file_dir = os.path.dirname(os.path.realpath(__file__))

        # Load desk
        table_start_pos = [0.0, -0.84, 0.76]
        table_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF(file_dir + "/urdf/objects/table.urdf", table_start_pos, table_start_orn, useFixedBase=True)

        # Load target square
        target_zone_orn = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF(file_dir + '/urdf/objects/zone2.urdf', self.target_zone_pos, target_zone_orn, useFixedBase=True)
        
        # Load stand
        ur5_stand_start_pos = [-0.7, -0.36, 0.0]
        ur5_stand_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        tst = p.loadURDF(file_dir + "/urdf/objects/ur5_stand.urdf", ur5_stand_start_pos, ur5_stand_start_orn, useFixedBase=True)

        # setup ur5 with robotiq 140 gripper
        # Path to robot file
        sisbot_urdf_path = file_dir + "/urdf/ur5_robotiq_140.urdf"
        # Robot start pos in cartesian coords
        robot_start_pos = [0,0,0.0]
        # Robot orientation in quaternion
        robot_start_orn = p.getQuaternionFromEuler([0,0,0])     
        print("----------------------------------------")
        print("Loading robot from {}".format(sisbot_urdf_path))
        #Load robot returns robot ID
        self.robot_ID = p.loadURDF(sisbot_urdf_path, robot_start_pos, robot_start_orn, useFixedBase=True,
                            flags=p.URDF_USE_INERTIA_FROM_FILE)
        # Some utils function that gets robot info
        self.joints, self.control_robotic_c2, self.control_joints, self.mimic_parent_name = utils_ur5_robotiq140.setup_sisbot(p, self.robot_ID)
        # Index for the end effector link
        self.eef_ID = 7 

    def get_camera_image(self):
        """ A method to return the camera image"""
        # Returns Width, height (both int), RGB Pixels, depth pixels, segmentation mask buffer
        images = p.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix, 
            shadow=False, renderer=p.ER_TINY_RENDERER)
        rgb_img = images[2]
        rgb_img = rgb_img[:,:,0:3]
        depth_img = images[3]
        seg_mask = images[4]
        return rgb_img, depth_img, seg_mask

    def load_object(self, file_path, pose, orn, fixed_base=False):
        """ A method to load a to be grasped object"""
        self.object_ID = p.loadURDF(file_path, pose, p.getQuaternionFromEuler(orn), useFixedBase=fixed_base)

    def remove_current_object(self):
        """Method to remove the currently loaded object"""
        p.removeBody(self.object_ID)
        self.object_ID = None

    def check_if_successful(self):
        """ Method to check if object has been placed successfully in the target zone"""
        object_pos, _ = p.getBasePositionAndOrientation(self.object_ID)
        x, y, _ = object_pos
        if x < self.max_x and x > self.min_x and y < self.max_y and y > self.min_y:
            return True 
        return False

    def run_simulation(self, x, y, z, roll, pitch, yaw, grip, delayed_grip, start_up=False):
        """ A method to run the simulation"""
       
        if z < self.z_min:
            z = self.z_min

        z += self.gripper_length

        # Define function to caculate absolute difference between desired and real values
        ABSE = lambda a,b: abs(a-b)

        # @TODO what is this?
        # Set damping for robot arm and gripper
        jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        jd = jd*0

        # Joint space to keep certain joints at a fixed angle
        fixed_joints = dict()
        fixed_joints[0] = -1.57
        fixed_joints[1] = -1.57
        fixed_joints[2] = 1.57
        fixed_joints[3] = -1.57
        fixed_joints[4] = -1.57
        fixed_joints[5] = roll

        # If we don't have to wait for the delayed grip
        if not delayed_grip:
            self.gripper_opening_length = grip

        disred_pose = False
        control_cnt = 0;
        
        # Run while desired pose has not been reached or waiting for the delayed grip
        while(not disred_pose or delayed_grip):
            
            if self.gui_mode:
                # Returns Width, height (both int), RGB Pixels, depth pixels, segmentation mask buffer
                p.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix, 
                    shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            
            # @TODO What is this?
            # rgb_opengl = np.reshape(images[2], (self.height, self.width, 4)) * 1. / 255.
            # plt.imshow(images[2])
            # plt.title('RGB image')
            # plt.pause(0.00001)

            # Read the value of task parameter
            orn = p.getQuaternionFromEuler([roll, pitch, yaw])

            # Compute gripper opening angle
            gripper_opening_angle = 0.715 - math.asin((self.gripper_opening_length - 0.010) / 0.1143)    # angle calculation

            # apply IK(robotId, end effector ID, target x,y,z, target orientation)
            # calculateInverseKinematics returns a list of joint positions for each degree of freedom, 
            # so the length of this list is the number of degrees of freedom of the joints 
            joint_pose = p.calculateInverseKinematics(self.robot_ID, self.eef_ID, [x,y,z], orn, jointDamping=jd)

            for i, name in enumerate(self.control_joints):
                joint = self.joints[name]
                pose = joint_pose[i]
                
                # If the joint i not the robtic 140 gripper
                if i != 6:
                    pose1 = fixed_joints[i]
                if name == self.mimic_parent_name:
                    self.control_robotic_c2(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
                else:
                    if start_up and control_cnt < 100:
                        # control robot joints
                        p.setJointMotorControl2(self.robot_ID, joint.id, p.POSITION_CONTROL,
                                            targetPosition=pose1, force=joint.maxForce, 
                                            maxVelocity=joint.maxVelocity)
                    else:
                        # control robot end-effector
                        p.setJointMotorControl2(self.robot_ID, joint.id, p.POSITION_CONTROL,
                                            targetPosition=pose, force=joint.maxForce, 
                                            maxVelocity=joint.maxVelocity)
                        test = 1
            
            control_cnt = control_cnt + 1
            
            # Get real XYZ and roll pitch yaw
            rXYZ = p.getLinkState(self.robot_ID, self.eef_ID)[0]
            rxyzw = p.getLinkState(self.robot_ID, self.eef_ID)[1]
            rroll, rpitch, ryaw = tf.transformations.euler_from_quaternion(rxyzw)

            # Calulate abs error between real XYZ - rpy and desired values
            error_x, error_y, error_z = map(ABSE, [x,y,z], rXYZ)
            error_roll, error_pitch, error_yaw = map(ABSE, [roll, pitch, yaw], [rroll, rpitch, ryaw])

            if self.show_debug:
                print('Desired:')
                print("x= {:.2f}, y= {:.2f}, z= {:.2f}".format(x, y, z))
                print("roll= {:.2f}, pitch= {:.2f}, yaw= {:.2f}".format(roll, pitch, yaw))
                print('Real:')
                print("r_x_= {:.2f}, r_y= {:.2f}, r_z= {:.2f}".format(rXYZ[0], rXYZ[1], rXYZ[2]))
                print("r_roll_= {:.2f}, r_pitch= {:.2f}, r_yaw= {:.2f}".format(rroll, rpitch, ryaw))
                print('Error:')
                print("err_x= {:.2f}, err_y= {:.2f}, err_z= {:.2f}".format(error_x, error_y, error_z))
                print("err_roll= {:.2f}, err_pitch= {:.2f}, err_yaw= {:.2f}\n".format(error_roll, error_pitch, error_yaw))

                # current object coordinates
                # object_pos, object_orn = p.getBasePositionAndOrientation(self.object_ID)
                # print(object_pos, object_orn)

            # Allowed error margin
            margin = 0.01
            # If desired pose has not been reached and all the errors are smaller than the margin
            if not disred_pose and error_x < margin and error_y < margin and error_z < margin and error_roll < margin \
                and error_pitch < margin and error_yaw < margin:
                disred_pose = True
                target_delay = control_cnt + 150
                self.gripper_opening_length = grip

            # If desired pose has been reached and we are wating for a delayed grip
            if disred_pose and control_cnt == target_delay:
                delayed_grip = False

            p.stepSimulation()

    def keep_running(self):
        """Debugger function """
        
        while True:
            p.stepSimulation()