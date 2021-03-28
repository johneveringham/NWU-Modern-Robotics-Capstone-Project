# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:01:06 2020

@author: John
"""

import sys
import os
import csv
import numpy as np
import math as m

from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

sys.path.append(r"D:\Engineering Courses\Modern Robotics Specialization\Code")

from Modern_robotics_functions_JE_edits import*

#%% Functions
def create_data_row(config_matrix_T,gripper_state):
    """
    Input
    Configuration matrix of end effector Tse
    example:
    Tse_init = np.array([[0,1,0,0],
                 [-1,0,0,0],
                 [0,0,1,2],
                 [0,0,0,1]])
    Output:
    list in format
    [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state]

    """
    rotation = config_matrix_T[:-1,0:-1].flatten()
    position = config_matrix_T[:-1,-1:].flatten()

    # unpack position state to one list
    data_row = np.concatenate((rotation,position,gripper_state),axis = None)
    data_row = data_row.tolist()

    return data_row

def write_csv_row(config_matrix_T,gripper_state):

    # unpack the data
    data_row = create_data_row(config_matrix_T,gripper_state)

    # create the file and write the row
    filepath = os.path.dirname(os.path.abspath(__file__))
    filename = filepath + '//' + 'trajectory_gen_test.csv'

    with open(filename, 'a', newline='') as f:
        csvwriter = csv.writer(f,)

        csvwriter.writerow(data_row)

def write_trajectory_to_csv(traj_list,gripper_state):
    for matrix in traj_list:
        write_csv_row(matrix,gripper_state)

def gripper_open_close_trajectory(config_matrix, k):

    Tf = 1 #1 second for gripper to close

    N = int(Tf*k/0.01)

    traj_list = [config_matrix]*N

    return traj_list

def distance(p1, p2):

    d = m.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)+((p1[2]-p2[2])**2))

    return d

def trajectory_speed(frame_start,frame_end, max_velocity, k):
    p1 = frame_start[:-1,-1:].flatten()
    p2 = frame_end[:-1,-1:].flatten()

    d = distance(p1, p2)

    Tf = round(d/max_velocity,2)
    N = Tf*k/0.01

    return Tf, N

def gentrajectory(T_start,T_end,max_velocity,k):
    Tf, N = trajectory_speed(T_start,T_end,max_velocity,k)
    traj_list = ScrewTrajectory(T_start, T_end, Tf, N, 3)

    return traj_list

#%% Module Testing
# Components of the trajectory
# Move from initial configuration to standoff
# Move to grasp configuration
# Close gripper
# Move from current configuration to final standoffconfiguration
# Move to final grasp configuration
# Open gripper

# Define initial, final, standoff, and grasping configurations
Tsc_init = np.array([[1,0,0,1],
                 [0,1,0,0],
                 [0,0,1,0.0],
                 [0,0,0,1]])

Tsc_final = np.array([[0,1,0,0],
                 [-1,0,0,-1],
                 [0,0,1,0],
                 [0,0,0,1]])

Tse_init = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,1],
                 [0,0,0,1]])

pce_standoff = [-0.025,0,0.15]
r = Rot.from_euler('zyx', [0, 145, 0], degrees = True)
Rce = r.as_matrix()
Tce_standoff = RpToTrans(Rce,pce_standoff)

pce_grasp = [0.0,0,0.025]
Tce_grasp = RpToTrans(Rce,pce_grasp)

# Define Time for each trajectory segment
k = 1 # number of trajectory reference configurations per 0.01 seconds
max_velocity = 0.25 #m/s

# Change all configurations to Space Frame
Tse_init_standoff = np.dot(Tsc_init,Tce_standoff)
Tse_init_grasp = np.dot(Tsc_init,Tce_grasp)
Tse_final_standoff = np.dot(Tsc_final,Tce_standoff)
Tse_final_grasp = np.dot(Tsc_final,Tce_grasp)

#====================================================================
# Move from initial configuration to Tce_standoff
T_start = Tse_init
T_end = Tse_init_standoff
Tf, N = trajectory_speed(T_start,T_end,max_velocity,k)
traj_list = ScrewTrajectory(T_start, T_end, Tf, N, 3)
gripper_state = 0
write_trajectory_to_csv(traj_list,gripper_state)
#====================================================================
# Move to grasp configuration
T_start = T_end
T_end = Tse_init_grasp
Tf, N = trajectory_speed(T_start,T_end,max_velocity,k)
traj_list = ScrewTrajectory(T_start, T_end, Tf, N, 3)
gripper_state = 0
write_trajectory_to_csv(traj_list,gripper_state)
#====================================================================
# Close gripper
traj_list = gripper_open_close_trajectory(T_end, k)
gripper_state = 1
write_trajectory_to_csv(traj_list,gripper_state)
#====================================================================
# # Move from current configuration to final standoff configuration
T_start = T_end
T_end = Tse_final_standoff
Tf, N = trajectory_speed(T_start,T_end,max_velocity,k)
traj_list = ScrewTrajectory(T_start, T_end, Tf, N, 3)
gripper_state = 1
write_trajectory_to_csv(traj_list,gripper_state)
#====================================================================
# Move from current configuration to final standoff configuration
T_start = T_end
T_end = Tse_final_grasp
Tf, N = trajectory_speed(T_start,T_end,max_velocity,k)
traj_list = ScrewTrajectory(T_start, T_end, Tf, N, 3)
gripper_state = 1
write_trajectory_to_csv(traj_list,gripper_state)
#====================================================================
# Open gripper
traj_list = gripper_open_close_trajectory(T_end, k)
gripper_state = 0
write_trajectory_to_csv(traj_list,gripper_state)
#====================================================================

#%% Test Flatten
Test_init = np.array([[ 11,12,13,14],
     [21,22,23,24],
     [31,32,33,34],
     [41,42,43,44]])

rotation = Test_init[:-1,0:-1].flatten()
position = Test_init[:-1,-1:].flatten()
data_row = np.concatenate((rotation,position),axis = None)