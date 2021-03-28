# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:01:06 2020
***************************************************************************
Modern Robotics: Capstone Project
Main Module
Description:
This module combines the trajectory generation, next state, and feedback control
modules to move the cube in scene 6 in the robot simulator.

***************************************************************************
Author: John Evering
Date: September 2020
***************************************************************************
Language: Python
Required library: numpy, tqdm
***************************************************************************

@author: John
"""

import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import math as m
import datetime as dt
from tqdm import tqdm

from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
from IPython import get_ipython

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import robot_properties as rprops
import Modern_robotics_functions_JE_edits as mr
import TrajectoryGeneration as tg
import NextState as ns
import FeedbackControl as fc

pi = m.pi
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# for writing log file
f = open("logfile.txt", 'w')
sys.stdout = f

print(dt.datetime.now())

def plot_Xerr(Xerr_list, K, Ki):
    get_ipython().run_line_magic('matplotlib', 'qt')
    labels = []

    for i in range(Xerr_list.shape[1]):
        plt.plot(Xerr_list[:,i])
        labels.append(i)

    plt.legend(labels)
    plt.title("K: " + str(K[0,0]) + " Ki: " + str(Ki))
    plt.show()

def save_xerr(Xerr_list):

    filepath = os.path.dirname(os.path.abspath(__file__))
    filename = filepath + '//' + "Xerr.csv"

    np.savetxt(filename, Xerr_list, delimiter=",")

if __name__ == '__main__':
    print("Generating Reference Trajectory")
    # Set initial robot configuration
    #==============================================================
    chassis_config = np.array((0.0,0.0,0.0))
    theta_arm = np.array((0.0,-0.3,-0.3,-0.3,0.0))
    theta_wheels = np.array((0.0,0.0,0.0,0.0))
    position_state = (chassis_config,theta_arm,theta_wheels)

    # Compare Tse to config set by robot angles
    # Get robot properties from properties file
    Tb0, M0e, B = rprops.chassis_and_arm_props()

    # Calculate current end effector configuration
    Xtest = fc.calc_Tse(chassis_config,theta_arm)

    # Generate Reference Trajectory
    #==============================================================
    # Define initial cube, final cube, and initial end effector configurations
    # Task from Modern Robotics Page
    # Tsc_init = np.array([[1,0,0,1],
    #                   [0,1,0,0],
    #                   [0,0,1,0.025],
    #                   [0,0,0,1]])

    # Tsc_final = np.array([[0,1,0,0],
    #                   [-1,0,0,-1],
    #                   [0,0,1,0.025],
    #                   [0,0,0,1]])

    # # New Task Settings
    Tsc_init = np.array([[1,0,0,1.25],
                  [0,1,0,0],
                  [0,0,1,0.025],
                  [0,0,0,1]])

    Tsc_final = np.array([[0,1,0,0],
                      [-1,0,0,-1.5],
                      [0,0,1,0.025],
                      [0,0,0,1]])

    # Tse from Modern Robotics Wiki Page
    Tse_init = np.array([[0,0,1,0],
                      [0,1,0,0],
                      [-1,0,0,0.5],
                      [0,0,0,1]])

    # Set Tse from Joint Angles
    # chassis_config = np.array((0.0,0.0,0.0))
    # theta_arm = np.array((0.0,-0.5,-0.5,-0.5,0.0))
    # Tse_init = fc.calc_Tse(chassis_config,theta_arm)

    # Define Time for each trajectory segment
    k = 1 # number of trajectory reference configurations per 0.01 seconds
    max_velocity = 0.25 #m/s
    joint_speed_limit = 15

    traj_list_complete, gripper_state_complete = tg.gen_full_trajectory(Tsc_init, Tsc_final, Tse_init, k, max_velocity, write_test_traj = False)


    # Run Feedback Control
    #==============================================================
    print("Generating CSV Animation")

    # List to store error
    Xerr_list = np.array([[0,0,0,0,0,0]])

    K = np.eye(6)*0.75
    Ki = 0.00
    # initial Integral error
    Ki_error = 0
    delta_t = 0.01/k

    for i,traj_section in tqdm(enumerate(traj_list_complete),position=0):

        ref_traj = traj_list_complete[i]
        gripper_state = gripper_state_complete[i]
        data_rows = []

        for i in tqdm(range(len(ref_traj)-2),position=0):

            data_row = ns.create_data_row(position_state,gripper_state)
            data_rows.append(data_row)

            test_joints = fc.test_joint_limits(position_state, ignore = True)

            Xd = ref_traj[i]
            Xdnext = ref_traj[i+1]

            u_theta, Xerr, Ki_error, J_e, V_t, Vd = fc.feedback_control(position_state,Xd,Xdnext,K, Ki, Ki_error, delta_t, test_joints)

            Xerr_list = np.concatenate((Xerr_list,[Xerr]), axis = 0)

            theta_dot_wheels = np.array(u_theta[0:4])
            theta_dot_arm = np.array(u_theta[4:])

            velocity_state = (theta_dot_arm,theta_dot_wheels)

            position_state = ns.next_state(position_state, velocity_state, delta_t, speed_limit = joint_speed_limit)

        # Write data to a file
        ns.write_row_list(data_rows,fn = 'result.csv')

    print("Writing Error Plot")
    print("K: " + str(K[0,0]) + " Ki: " + str(Ki))
    plot_Xerr(Xerr_list,K,Ki)
    save_xerr(Xerr_list)

    print("Done")

    f.close()