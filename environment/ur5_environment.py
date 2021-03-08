import pybullet as p
import pybullet_data
import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt


class ur5_environment():
    """ A class for the running a ur5 robot with a gripper"""
    # Define camera image parameter
    width = 128
    height = 128
    fov = 40
    aspect = width / height
    near = 0.1
    far = 0.62
    camera_eye_xyz = [0.1, -0.5, 1.425]
    camera_target_xyz = [0.1, -0.50, 0.90]


    def __init__(self, gui_mode=False):
        """ Constructor: Initializes the robot and its environment"""
        # Connect to engine servers
        if gui_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        # Server mode
        self.gui_mode = gui_mode
        # ID of the to be grasped object
        self.object_ID = 0
        # Standard opening length of the gripper
        self.gripper_opening_length = 0.085
    
        # Add search path for loadURDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define world
        p.setGravity(0,0,-10)
        p.loadURDF("plane.urdf")

        # Load desk
        table_start_pos = [0.0, -0.9, 0.8]
        table_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF("./urdf/objects/table.urdf", table_start_pos, table_start_orn, useFixedBase=True)

        # Load stand
        ur5_stand_start_pos = [-0.7, -0.36, 0.0]
        ur5_stand_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF("./urdf/objects/ur5_stand.urdf", ur5_stand_start_pos, ur5_stand_start_orn, useFixedBase=True)

        # setup ur5 with robotiq 140 gripper
        # Path to robot file
        sisbot_urdf_path = "./urdf/ur5_robotiq_140.urdf"
        # Robot start pos in cartesian coords
        robot_start_pos = [0,0,0.0]
        # Robot orientation in quaternion
        robot_start_orn = p.getQuaternionFromEuler([0,0,0])     
        print("----------------------------------------")
        print("Loading robot from {}".format(sisbot_urdf_path))
        #Load robot returns robot ID
        self.robot_ID = p.loadURDF(sisbot_urdf_path, robot_start_pos, robot_start_orn,useFixedBase = True,
                            flags=p.URDF_USE_INERTIA_FROM_FILE)
        # Some utils function that gets robot info
        self.joints, self.control_robotic_c2, self.control_joints, self.mimic_parent_name = utils_ur5_robotiq140.setup_sisbot(p, self.robot_ID)
        # Index for the end effector link
        self.eef_ID = 7 

    def get_camera_image(self):
        """ A method to return the camera image"""

        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        view_matrix = p.computeViewMatrix(self.camera_eye_xyz, self.camera_target_xyz, [0, 1, 0])
        # Get images
        # Returns Width, height (both int), RGB Pixels, depth pixels, segmentation mask buffer
        images = p.getCameraImage(self.width, self.height, view_matrix, projection_matrix,shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return images

    def load_object(self, file_path):
        """ A method to load a to be grasped object"""

        object_start_pos = [0.1, -0.49, 0.85]
        object_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.object_ID = p.loadURDF(file_path, object_start_pos, object_start_orn)

    def run_simulation(self, x, y, z, roll, pitch, yaw, grip, delayed_grip):
        """ A method to run the simulation"""
       
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
        fixed_joints[5] = 0

        if self.gui_mode:
            view_matrix = p.computeViewMatrix(self.camera_eye_xyz, self.camera_target_xyz, [0, 1, 0])
            projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

        # If we don't have to wait for the delayed grip
        if not delayed_grip:
            self.gripper_opening_length = grip

        disred_pose = False
        control_cnt = 0;
        
        # Run while desired pose has not been reached or waiting for the delayed grip
        while(not disred_pose or delayed_grip):
            
            if self.gui_mode:
                # Returns Width, height (both int), RGB Pixels, depth pixels, segmentation mask buffer
                images = p.getCameraImage(self.width, self.height, view_matrix, projection_matrix,shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            
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
            joint_pose = p.calculateInverseKinematics(self.robot_ID, self.eef_ID, [x,y,z],orn,jointDamping=jd)
            
            for i, name in enumerate(self.control_joints):
                joint = self.joints[name]
                pose = joint_pose[i]
                
                # read joint value
                if i != 6:
                    pose1 = fixed_joints[i]
                if name == self.mimic_parent_name:
                    self.control_robotic_c2(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
                else:
                    if control_cnt < 100:
                        # control robot joints
                        p.setJointMotorControl2(self.robot_ID, joint.id, p.POSITION_CONTROL,
                                            targetPosition=pose1, force=joint.maxForce, 
                                            maxVelocity=joint.maxVelocity)
                    else:
                        # control robot end-effector
                        p.setJointMotorControl2(self.robot_ID, joint.id, p.POSITION_CONTROL,
                                            targetPosition=pose, force=joint.maxForce, 
                                            maxVelocity=joint.maxVelocity)
            
            control_cnt = control_cnt + 1
            
            # Get real XYZ and roll pitch yaw
            rXYZ = p.getLinkState(self.robot_ID, self.eef_ID)[0]
            rxyzw = p.getLinkState(self.robot_ID, self.eef_ID)[1]
            rroll, rpitch, ryaw = p.getEulerFromQuaternion(rxyzw)

            # Calulate abs error between real XYZ - rpy and desired values
            error_x, error_y, error_z = map(ABSE, [x,y,z], rXYZ)
            error_roll, error_pitch, error_yaw = map(ABSE, [roll, pitch, yaw], [rroll, rpitch, ryaw])

            print("err_x= {:.2f}, err_y= {:.2f}, err_z= {:.2f}".format(error_x, error_y, error_z))
            print("err_r= {:.2f}, err_o= {:.2f}, err_y= {:.2f}".format(error_roll, error_pitch, error_yaw))
            print("x_= {:.2f}, y= {:.2f}, z= {:.2f}".format(rXYZ[0], rXYZ[1], rXYZ[2]))
            print("rroll_= {:.2f}, rpitch= {:.2f}, ryaw= {:.2f}".format(rroll, rpitch, ryaw))
            
            # current object coordinates
            object_pos, object_orn = p.getBasePositionAndOrientation(self.object_ID)
            print(object_pos, object_orn)

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


env = ur5_environment(gui_mode=True)
env.load_object("./urdf/objects/block.urdf")

env.run_simulation(x=0.11, y=-0.5, z=1.45, roll=0, pitch=1.57, yaw=-1.57, grip=0.085, delayed_grip=False)
env.run_simulation(x=0.11, y=-0.5, z=1.05, roll=0, pitch=1.57, yaw=-1.57, grip=0.045, delayed_grip=True)
env.run_simulation(x=0.11, y=-0.5, z=1.30, roll=0, pitch=1.57, yaw=-1.57, grip=0.045, delayed_grip=False)

