# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:01:06 2020
***************************************************************************
Modern Robotics: Capstone Project
Next State Module
Description:
This module calculates new joint, chassis configuration, and wheel angles
based on input postion and velocity. A simple Euler step calculation is used
to calculate the new position.

***************************************************************************
Author: John Evering
Date: September 2020
***************************************************************************
Language: Python
Required library: numpy
***************************************************************************


@author: John
"""

import sys
import os
import csv
import numpy as np
import math as m

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import robot_properties as rprops

from numpy.linalg import inv


def euler_step(angles,speeds,speed_limit,delta_t):
# Input: angles radians (np array), speed radians/s (np array), timestep s
# Output: new angles assuming constant acceleration through dt
    angles_new = angles.copy()
    speeds_temp = speeds.copy()
    # Absolute value covers positive and negative speeds
    speeds_temp[np.where(np.absolute(speeds_temp) > speed_limit)] = speed_limit
    angles_new = angles_new+speeds_temp*delta_t

    return angles_new

def calc_new_chassis_config(chassis_config, delta_theta_wheels):
# Input: Numpy array of current chassis_configuration, Numpy array of change in wheel angle in radians
# Output: new chassis configuration

    # Chassis Dimensions from robot properties config file
    r,l,w,z = rprops.chassis_dims()

    # Chassis Base Kinematic Model
    H_0 = (1/r)*np.array([[-l-w,1,-1],
                  [l+w,1,1],
                  [l+w,1,-1],
                  [-l-w,1,1]])

    F = np.linalg.pinv(H_0)

    # Chassis twist
    V_b = np.dot(F,delta_theta_wheels.T)

    # Twist Components
    wbz = V_b[0]
    vbx = V_b[1]
    vby = V_b[2]

    # Extract from initial configuration matrix
    q_init = chassis_config
    phi = q_init[0]
    x = q_init[1]
    y = q_init[2]

    # Change in cordinates relative to the body
    if wbz < 0.0000000001: # If wbz is basically zero
        dqb = np.array([0,vbx,vby])
    else:
        dqb = np.array([wbz,
                        (vbx*m.sin(wbz)+vby*(m.cos(wbz)-1))/wbz,
                        (vby*m.sin(wbz)+vbx*(1-m.cos(wbz)))/wbz])

    # Configuration of space to body frame
    T_sb = np.array([[1,0,0],
                    [0,m.cos(phi),-1*m.sin(phi)],
                    [0,m.sin(phi),m.cos(phi)]])

    # Change from body to space frame
    dq = np.dot(T_sb,dqb)

    # New chassis configuration q
    q_new = dq + q_init.T

    return q_new

def next_state(position_state, velocity_state, delta_t, speed_limit = 12.5):
    # Input
    # position state (( Chassis Configuration ), ( Arm Joint Angles ), ( Wheel Angles ))
    # (chassis phi, chassis x, chassis y), (J1, J2, J3, J4, J5), (W1, W2, W3, W4)
    # velocity state (( arm joint speeds ), ( wheel speeds ))
    # (J_dot1, J_dot2, J_dot3, J_dot4, J_dot5), (W1_dot, W2_dot, W3_dot, W4_dot)
    #  Speed Limit Radians and time step s
    # Output
    # new arm joint angles, new wheel angles, new chassis configuration all numpy arrays

    # unpack things
    chassis_config = position_state[0]
    theta_arm = position_state[1]
    theta_wheels = position_state[2]
    theta_dot_arm = velocity_state[0]
    theta_dot_wheels = velocity_state[1]

    # Calculate new arm position
    theta_arm_new = euler_step(theta_arm,theta_dot_arm,speed_limit,delta_t)
    delta_theta_arm = theta_arm_new - theta_arm

    # Calculate new wheel angles, new wheel angular velocities, change in wheel angles
    theta_wheels_new = euler_step(theta_wheels,theta_dot_wheels,speed_limit,delta_t)
    delta_theta_wheels = theta_wheels_new - theta_wheels

    # Calculate new chassis configuration based on change in wheel angles
    chassis_config_new = calc_new_chassis_config(chassis_config,delta_theta_wheels)

    position_state_new = (chassis_config_new,theta_arm_new,theta_wheels_new)

    return position_state_new

def create_data_row(position_state,gripper_state):
    # Input
    # position state (( Chassis Configuration ), ( Arm Joint Angles ), ( Wheel Angles ))
    # (chassis phi, chassis x, chassis y), (J1, J2, J3, J4, J5), (W1, W2, W3, W4) , gripper state

    # unpack position state to one list
    data_row = np.concatenate((position_state[0],position_state[1],position_state[2],np.array(gripper_state)),axis = None)
    data_row = data_row.tolist()

    return data_row

def write_csv_row(position_state,gripper_state,fn = 'result.csv'):
    # Used for writing one row at a time

    # unpack the data
    data_row = create_data_row(position_state,gripper_state)

    # Data Order ['chassis phi', 'chassis x', 'chassis y', 'J1', 'J2', 'J3', 'J4', 'J5','W1', 'W2', 'W3', 'W4', 'gripper state']

    # create the file and write the row
    filepath = os.path.dirname(os.path.abspath(__file__))
    filename = filepath + '//' + fn

    with open(filename, 'a', newline='') as f:
        csvwriter = csv.writer(f,)

        csvwriter.writerow(data_row)

def write_row_list(data_rows,fn = 'result.csv'):
    # Used for writing a list of data rows

    # Data Order ['chassis phi', 'chassis x', 'chassis y', 'J1', 'J2', 'J3', 'J4', 'J5','W1', 'W2', 'W3', 'W4', 'gripper state']

    # create the file and write the row
    filepath = os.path.dirname(os.path.abspath(__file__))
    filename = filepath + '//' + fn

    with open(filename, 'a', newline='') as f:
        csvwriter = csv.writer(f,)

        csvwriter.writerows(data_rows)


def next_state_test(position_state, gripper_state, velocity_state, iterations, max_time, speed_lim = 12.5):

    data_rows = []

    delta_t = max_time/iterations

    # write row for each iteration
    for i in range(iterations):
        data_row = create_data_row(position_state,gripper_state)
        data_rows.append(data_row)

        position_state_new = next_state(position_state, velocity_state, delta_t, speed_limit = speed_lim)

        # Update position for each iteration
        position_state = position_state_new

    write_row_list(data_rows,fn = 'next_state_test.csv')

def unit_test_single():
    # Testing one iteration
    # Test parameters
    chassis_config = np.array((0.0,0.0,0.0))
    theta_arm = np.array((0.0,0.0,0.0,0.0,0.0))
    theta_wheels = np.array((0.0,0.0,0.0,0.0))
    theta_dot_arm = np.array((0.5,0.5,0.5,0.5,0.5))
    theta_dot_wheels = np.array((10.0,10.0,10.0,10.0))
    # theta_dot_wheels = np.array((-10.0,10.0,-10.0,10.0))
    # theta_dot_wheels = np.array((-10.0,10.0,10.0,-10.0))

    gripper_state = 0
    position_state = (chassis_config,theta_arm,theta_wheels)

def unit_test_multi_line():
    # Testing multiple iterations
    # Test parameters
    chassis_config = np.array((0.0,0.0,0.0))
    theta_arm = np.array((0.0,0.0,0.0,0.0,0.0))
    theta_wheels = np.array((0.0,0.0,0.0,0.0))
    theta_dot_arm = np.array((0.5,0.5,0.5,0.5,0.5))
    # theta_dot_wheels = np.array((10.0,10.0,10.0,10.0))
    # theta_dot_wheels = np.array((-10.0,10.0,-10.0,10.0))
    theta_dot_wheels = np.array((-10.0,10.0,10.0,-10.0))

    gripper_state = 0
    position_state = (chassis_config,theta_arm,theta_wheels)
    velocity_state = (theta_dot_arm,theta_dot_wheels)

    max_time = 1
    iterations = 200

    next_state_test(position_state, gripper_state, velocity_state, iterations, max_time, speed_lim = 12.5)


