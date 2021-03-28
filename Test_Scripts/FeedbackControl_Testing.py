# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:01:06 2020

@author: John
"""

import sys
import os
import csv
import matplotlib as plt
import numpy as np
import math as m

from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import robot_properties as rprops
import Modern_robotics_functions_JE_edits as mr
import TrajectoryGeneration as tg
import NextState as ns

pi = m.pi
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

#%% Functions
def pinv_tol(matrix,tol=0.0000001):

    matrix[np.where(np.abs(matrix) < tol)] = 0

    matrix = np.linalg.pinv(matrix)

    return matrix

def test_joint_limits(theta_arm):

    test_joints = np.ones((theta_arm.shape),dtype=bool)

    if theta_arm[2] > -0.1:
        test_joints[2] = False

    if theta_arm[3] > -0.1:
        test_joints[3] = False

    return test_joints

def calc_complete_jacobian(position_state):

    chassis_config = position_state[0]
    theta_arm = position_state[1]

    # Get robot properties from properties file
    Tb0, M0e, B = rprops.chassis_and_arm_props()

    # Calculate Arm Jacobian
    J_arm = mr.JacobianBody(B.T,theta_arm)

    # Calculate Base Jacobian
    # Chassis Dimensions from robot properties config file
    r,l,w = rprops.chassis_dims()

    # Chassis Base Kinematic Model
    H_0 = (1/r)*np.array([[-l-w,1,-1],
                  [l+w,1,1],
                  [l+w,1,-1],
                  [-l-w,1,1]])

    F = np.linalg.pinv(H_0)

    F6 = np.zeros([6,4])

    F6[2:-1,:] = F

    # Calculate current end effector configuration
    T0e = mr.FKinBody(M0e, B.T, theta_arm)

    Teb = np.dot(np.linalg.inv(T0e),np.linalg.inv(Tb0))

    J_base = np.dot(mr.Adjoint(Teb), F6)

    # Put it together
    J_e = np.concatenate((J_base,J_arm), axis = 1 )

    return J_e

def feedback_control(position_state,Xd,Xdnext,K, Ki, delta_t):

    chassis_config = position_state[0]
    theta_arm = position_state[1]

    # Get robot properties from properties file
    Tb0, M0e, B = rprops.chassis_and_arm_props()

    # Calculate current end effector configuration
    X = mr.FKinBody(M0e, B.T, theta_arm)

    # Feedback Control Law Calculation
    Xerr = mr.se3ToVec(mr.MatrixLog6(np.dot(pinv_tol(X),Xd)))

    AdX = mr.Adjoint(np.dot(pinv_tol(X),Xd))

    Vd = mr.se3ToVec((1/delta_t)*mr.MatrixLog6(np.dot(pinv_tol(Xd), Xdnext)))

    V_t = np.dot(AdX,Vd) + K*Xerr + Ki*Xerr*delta_t

    J_e = calc_complete_jacobian(position_state)

    u_theta = np.dot(pinv_tol(J_e),V_t)

    return u_theta, Xerr

# %% Testing the functions
chassis_config = np.array((0.0,0.0,0.0))
theta_arm = np.array((0.0,0.0,0.2,-1.6,0.0))
position_state = (chassis_config,theta_arm)

# Feedback control law
Xd = np.array([[0,0,1,0.5],
                [0,1,0,0],
                [-1,0,0,0.5],
                [0,0,0,1]])

Xdnext = np.array([[0,0,1,0.6],
                 [0,1,0,0],
                 [-1,0,0,0.3],
                 [0,0,0,1]])

K = 0

Ki = 0

delta_t = 0.01

u_theta, Xerr = feedback_control(position_state,Xd,Xdnext,K, Ki, delta_t)

print("++++++++++++++++++++++++++++++")
print("Xerr")
print(Xerr)
print("++++++++++++++++++++++++++++++")
print("U_theta")
print(u_theta)

# %% Testing Math outside of functions
# robot configuration
chassis_config = np.array((0.0,0.0,0.0))
theta_arm = np.array((0.0,0.0,0.2,-1.6,0.0))

test_joints = test_joint_limits(theta_arm)

# Feedback control law
Xd = np.array([[0,0,1,0.5],
                [0,1,0,0],
                [-1,0,0,0.5],
                [0,0,0,1]])

Xdnext = np.array([[0,0,1,0.6],
                 [0,1,0,0],
                 [-1,0,0,0.3],
                 [0,0,0,1]])

X = np.array([[0.170,0,0.985,0.387],
                 [0,1,0,0],
                 [-0.985,0,0.170,0.570],
                 [0,0,0,1]])

K = 0

Ki = 0

delta_t = 0.01

invX = pinv_tol(X)

invXd = pinv_tol(Xd)

dot_invX_Xd = np.dot(invX,Xd)

dot_invXd_Xdnext = np.dot(invXd, Xdnext)

Xerr = mr.se3ToVec(mr.MatrixLog6(dot_invX_Xd))

AdX = mr.Adjoint(np.dot(invX,Xd))

Vd = mr.se3ToVec((1/delta_t)*mr.MatrixLog6(dot_invXd_Xdnext))

AdX_Vd = np.dot(AdX,Vd)

V_t = AdX_Vd + K*Xerr + Ki*Xerr*delta_t

# Calc Jacobian
# =======================================
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

# Test J arm
theta_list = theta_arm
J_arm = mr.JacobianBody(B.T,theta_list)

# Test J base
# Chassis Dimensions set here by default
r = 0.0475
l = 0.235
w = 0.15

# Chassis Base Kinematic Model
H_0 = (1/r)*np.array([[-l-w,1,-1],
              [l+w,1,1],
              [l+w,1,-1],
              [-l-w,1,1]])

F = np.linalg.pinv(H_0)

F6 = np.zeros([6,4])

F6[2:-1,:] = F

# Calculate
# End effector configuration based on arm configuration
T0e = mr.FKinBody(M0e, B.T, theta_list)

Teb = np.dot(np.linalg.inv(T0e),np.linalg.inv(Tb0))

Ad_Teb = mr.Adjoint(Teb)

J_base = np.dot(Ad_Teb, F6)

J_e = np.concatenate((J_base,J_arm), axis = 1 )

u_theta = np.dot(pinv_tol(J_e),V_t)

print("Vd")
print(Vd)
print("++++++++++++++++++++++++++++++")
print("AdX")
print(AdX_Vd)
print("++++++++++++++++++++++++++++++")
print("V_t")
print(V_t)
print("++++++++++++++++++++++++++++++")
print("Xerr")
print(Xerr)
print("++++++++++++++++++++++++++++++")
print("J_e")
print(J_e)
print("++++++++++++++++++++++++++++++")
print("U_theta")
print(u_theta)
