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

sys.path.append(r"D:\Engineering Courses\Modern Robotics Specialization\Code")

from Modern_robotics_functions_JE_edits import*

# Next State Functions
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

def next_state(position_state, velocity_state, speed_limit = 12.5, delta_t = 1.0):
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

def write_csv_row(position_state,gripper_state):

    # unpack the data
    data_row = create_data_row(position_state,gripper_state)

    # create the file and write the row
    filepath = os.path.dirname(os.path.abspath(__file__))
    filename = filepath + '//' + 'next_state_test.csv'

    header = ['chassis phi', 'chassis x', 'chassis y', 'J1', 'J2', 'J3', 'J4', 'J5','W1', 'W2', 'W3', 'W4', 'gripper state']

    # should_write_header = os.path.exists(filename)
    should_write_header = True

    with open(filename, 'a', newline='') as f:
        csvwriter = csv.writer(f,)

        if not should_write_header:
            csvwriter.writerow(header)
            csvwriter.writerow(data_row)
        else:
            csvwriter.writerow(data_row)

def next_state_test(position_state, gripper_state, velocity_state, iterations, max_time, speed_lim = 12.5):

    # write initial configation
    write_csv_row(position_state, gripper_state)

    del_t = max_time/iterations

    # write row for each iteration
    for i in range(iterations):
        position_state_new = next_state(position_state, velocity_state, speed_limit = speed_lim, delta_t = del_t)

        write_csv_row(position_state_new, gripper_state)

        # Update position for each iteration
        position_state = position_state_new

#%% Testing it all
# Test parameters
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
velocity_state = (theta_dot_arm,theta_dot_wheels)

max_time = 1
iterations = 100

next_state_test(position_state, gripper_state, velocity_state, iterations, max_time, speed_lim = 12.5)

#%% Testing CSV Write
filepath = os.path.dirname(os.path.abspath(__file__))
filename = filepath + '//' + 'next_state_test.csv'

header = ['chassis phi', 'chassis x', 'chassis y', 'J1', 'J2', 'J3', 'J4', 'J5','W1', 'W2', 'W3', 'W4', 'gripper state']

should_write_header = os.path.exists(filename)

with open(filename, 'a') as f:
    csvwriter = csv.writer(f)

    if should_write_header:
        csvwriter.writerow(header)
    else:
        csvwriter.writerows(row)

#%% Testing Data unpack
gripper_state = 0
data_row = np.concatenate((position_state[0],position_state[1],position_state[2],np.array(gripper_state)),axis = None).to_list

#%% Testing Next State all
# Test parameters
theta_arm = np.array((0.0,0.0,0.0,0.0,0.0))
theta_wheels = np.array((0.0,0.0,0.0,0.0))
theta_dot_arm = np.array((0.5,0.5,0.5,0.5,0.5))
theta_dot_wheels = np.array((10.0,10.0,10.0,10.0))
# theta_dot_wheels = np.array((-10.0,10.0,-10.0,10.0))
# theta_dot_wheels = np.array((-10.0,10.0,10.0,-10.0))
chassis_config = np.array((0.0,0.0,0.0))

position_state = (chassis_config,theta_arm,theta_wheels)
velocity_state = (theta_dot_arm,theta_dot_wheels)

position_state_new = next_state(position_state, velocity_state, speed_limit = 12.5, delta_t = 0.5)

#%% Testing next state functions indiviudal
speed_limit = 12.5
delta_t = 1.0

# Calculate new arm position
theta_arm_new = euler_step(theta_arm,theta_dot_arm,speed_limit,delta_t)
delta_theta_arm = theta_arm_new - theta_arm

# Calculate new wheel angles, new wheel angular velocities, change in wheel angles
theta_wheels_new = euler_step(theta_wheels,theta_dot_wheels,speed_limit,delta_t)
delta_theta_wheels = theta_wheels_new - theta_wheels

# Calculate new chassis configuration based on change in wheel angles
chassis_config_new = calc_new_chassis_config(chassis_config,delta_theta_wheels)


#%% Testing Chassis Odometry
# delta_theta_wheels = np.array([[-1.18,0.68,0.02,-0.52]])

# Chassis Dimensions
r = 0.0475
l = 0.235
w = 0.15

# Chassis Base Kinematic Model
H_0 = (1/r)*np.array([[-l-w,1,-1],
              [l+w,1,1],
              [l+w,1,-1],
              [-l-w,1,1]])

F = np.linalg.pinv(H_0)

# F2 = r/4* np.array([[-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],
#                     [1,1,1,1],
#                     [-1,1,-1,1]])

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