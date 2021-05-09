import matplotlib.pyplot as plt
from imageio import imsave
import numpy as np
import os
import pybullet as p
from grasp_generator import GraspGenerator
import sys
sys.path.append('network')

# from grasp_generator import GraspGenerator


def calc_ppc(seg_mask):
    print(seg_mask.shape)
    print(seg_mask[122, :])
    print(len(seg_mask))
    unique, counts = np.unique(seg_mask[122, :], return_counts=True)
    print(dict(zip(unique, counts)))


if __name__ == '__main__':
    print('yolo')