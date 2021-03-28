# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:22:13 2020

@author: John
"""

import numpy as np

def chassis_and_arm_props():

    # Fixed Offset from chassis frame to b to arm frame O
    Tb0 = np.array([[1,0,0,0.1662],
                     [0,1,0,0],
                     [0,0,1,0.0026],
                     [0,0,0,1]])

    # Fixed Offset between the end effector and the base of the arm frameb calculated from trajectory generation
    M0e = np.array([[1,0,0,0.033],
                     [0,1,0,0],
                     [0,0,1,0.6546],
                     [0,0,0,1]])

    # Arm Configuration screw axis
    B = np.array([[0,0,1,0,0.033,0],
                  [0,-1,0,-0.5076,0,0],
                  [0,-1,0,-0.3526,0,0],
                  [0,-1,0,-0.2176,0,0],
                  [0,0,1,0,0,0]])

    return Tb0, M0e, B

def chassis_dims():

    r = 0.0475
    l = 0.235
    w = 0.15

    return r,l,w