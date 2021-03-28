# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:01:06 2020
***************************************************************************
Modern Robotics: Capstone Project
Next State Module
Description:
This module uses the reference trajectory to caclulate the corresponding
joint and wheel velocities to move the robot to the desired configuration.

***************************************************************************
Author: John Evering
Date: September 2020
***************************************************************************
Language: Python
Required library: numpy
***************************************************************************


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

def pinv_tol(matrix,tol=0.0000001):

    matrix[np.where(np.abs(matrix) < tol)] = 0

    matrix = np.linalg.pinv(matrix)

    return matrix

def test_joint_limits(position_state, ignore = True):

    theta_arm = position_state[1]
    theta_wheels = position_state[2]

    arm_wheels = np.concatenate((theta_wheels,theta_arm),axis = None)

    test_joints = np.ones((arm_wheels.shape),dtype = bool)

    offset_arm = theta_wheels.shape[0]-1

    if ignore == False:
        if theta_arm[2] < -0.05:
            test_joints[offset_arm + 2] = False

        if theta_arm[3] < -0.05:
            test_joints[offset_arm + 2] = False

    return test_joints

def calc_complete_jacobian(position_state,test_joints):

    chassis_config = position_state[0]
    theta_arm = np.array([position_state[1]]).T # Change to Column Vector

    # Get robot properties from properties file
    Tb0, M0e, B = rprops.chassis_and_arm_props()

    # Calculate Arm Jacobian
    J_arm = mr.JacobianBody(B.T,theta_arm)

    # Calculate Base Jacobian
    # Chassis Dimensions from robot properties config file
    r,l,w,z = rprops.chassis_dims()

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

    Teb = np.dot(pinv_tol(T0e),pinv_tol(Tb0))

    J_base = np.dot(mr.Adjoint(Teb), F6)

    # Put it together
    J_e = np.concatenate((J_base,J_arm), axis = 1 )

    # Set columns to zero where joint limits are violated
    J_e[:,np.where(test_joints == False)] = 0

    return J_e

def calc_Tsb_from_q(chassis_config):
    r,l,w,z = rprops.chassis_dims()

    # Extract from initial configuration matrix
    phi = chassis_config[0]
    x = chassis_config[1]
    y = chassis_config[2]

    Tsb = np.array([[m.cos(phi),-m.sin(phi),0,x],
                    [m.sin(phi),m.cos(phi),0,y],
                    [0,0,1,z],
                    [0,0,0,1]])

    return Tsb

def calc_Tse(chassis_config,theta_arm):
    # Get robot properties from properties file
    Tb0, M0e, B = rprops.chassis_and_arm_props()

    # Calculate current end effector configuration
    T0e = mr.FKinBody(M0e, B.T, theta_arm)

    Tsb = calc_Tsb_from_q(chassis_config)

    Teb = np.dot(pinv_tol(T0e),pinv_tol(Tb0))

    Tse = np.dot(Tsb,pinv_tol(Teb))

    return Tse

def feedback_control(position_state,Xd,Xdnext,K, Ki, Ki_error, delta_t, test_joints):

    chassis_config = position_state[0]
    theta_arm = np.array([position_state[1]]).T # Change to Column Vector

    X = calc_Tse(chassis_config,theta_arm)

    # Feedback Control Law Calculation
    Xerr = mr.se3ToVec(mr.MatrixLog6(np.dot(pinv_tol(X),Xd)))

    AdX = mr.Adjoint(np.dot(pinv_tol(X),Xd))

    Vd = mr.se3ToVec((1/delta_t)*mr.MatrixLog6(np.dot(pinv_tol(Xd), Xdnext)))

    Ki_error = Ki_error + Ki*Xerr*delta_t

    V_t = np.dot(AdX,Vd) + np.dot(K,Xerr) + Ki_error

    J_e = calc_complete_jacobian(position_state, test_joints)

    u_theta = np.dot(pinv_tol(J_e),V_t)

    return u_theta, Xerr, Ki_error, J_e, V_t, Vd

def unit_test():
    # Testing the functions
    chassis_config = np.array((0.0,0.0,0.0))
    theta_arm = np.array((0.0,0.0,0.2,-1.6,0.0))
    theta_wheels = np.array((0.0,0.0,0.0,0.0))
    position_state = (chassis_config,theta_arm,theta_wheels)

    test_joints = test_joint_limits(position_state, ignore = True)

    # Feedback control law
    Xd = np.array([[0,0,1,0.5],
                    [0,1,0,0],
                    [-1,0,0,0.5],
                    [0,0,0,1]])

    Xdnext = np.array([[0,0,1,0.6],
                      [0,1,0,0],
                      [-1,0,0,0.3],
                      [0,0,0,1]])

    K = np.eye(6)
    # K = 0

    Ki = 0

    Ki_error = 0

    delta_t = 0.01

    u_theta, Xerr, Ki_error, J_e, V_t, Vd = feedback_control(position_state,Xd,Xdnext,K, Ki, Ki_error, delta_t,test_joints)

    print("++++++++++++++++++++++++++++++")
    print("Vd")
    print(Vd)
    print("++++++++++++++++++++++++++++++")
    print("Xerr")
    print(Xerr)
    print("++++++++++++++++++++++++++++++")
    print("Je")
    print(J_e)
    print("++++++++++++++++++++++++++++++")
    print("V_t")
    print(V_t)
    print("++++++++++++++++++++++++++++++")
    print("U_theta")
    print(u_theta)

#%% Run Unit Test
# unit_test()

