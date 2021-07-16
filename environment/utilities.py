import pybullet as p
from collections import namedtuple
from attrdict import AttrDict
import functools
import os
from datetime import datetime


def setup_sisbot(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info

    # explicitly deal with mimic joints
    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) is not 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName


def setup_sisbot_force(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable", "jointAxis", "parentFramePos", "parentFrameOrn"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        jointAxis = info[13]
        parentFramePos = info[14]
        parentFrameOrn = info[15]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable,
                         jointAxis, parentFramePos, parentFrameOrn)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    for j in joints:
        print(joints[j])
    # explicitly deal with mimic joints

    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # p.setJointMotorControl2(robotID, parent.id, p.TORQUE_CONTROL,
            #                         force=10, maxVelocity=parent.maxVelocity)
            return
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) is not 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    # Create all the gear constraint
    for name in children:
        child = children[name]
        c = p.createConstraint(robotID, parent.id, robotID, child.id, p.JOINT_GEAR, child.jointAxis,
                               # child.parentFramePos, (0, 0, 0), child.parentFrameOrn, (0, 0, 0))
                               (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        p.changeConstraint(c, gearRatio=-mimicChildren[name], maxForce=10000)
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName


class Camera:
    def __init__(self, cam_pos, cam_target, near, far, size, fov):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

        self.rec_id = None

    def get_cam_img(self):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        return rgb[:, :, 0:3], depth, seg

    def start_recording(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = f'{save_dir}/{now}.mp4'

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    def stop_recording(self):
        p.stopStateLogging(self.rec_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
