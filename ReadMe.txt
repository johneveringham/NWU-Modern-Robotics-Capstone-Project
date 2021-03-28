Software:
#=====================================================
The software is composed of 5 modules

robot_properties.py - contains the static geometric parameters about the mobile manipulator
NextState.py - functions as described in the modern robotics wiki
TrajectoryGeneration.py - functions as described in the modern robotics wiki
FeedbackControl.py - functions as described in the modern robotics wiki
mobile_manipulation.py - The main module that uses all 4 of the sub modules 

I implemented sigularity avoidance through a special psuedo inverse function in feedbackcontrol.py

detailed comments are in the code. 

Results:
#=====================================================
My new task was not as different from the normal task as I wanted. I had issues with the robot going out of 
control when moving the cube further distances. I would like to investigate this further and potentially try the 
robot throwing the cube problem. 