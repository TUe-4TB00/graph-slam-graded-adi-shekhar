
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    dx = 2.0 * np.cos(np.pi / 4.0)  
    dy = 2.0 * np.sin(np.pi / 4.0)  
    dtheta = np.pi / 2.0  

    delta_pose = gtsam.Pose2(dx, dy, dtheta) 

    # TODO: Add the odometry factor between X(3) and X(4) to the graph (BetweenFactorPose2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), delta_pose, ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(4) and add it to the graph
    ideal_pose3 = gtsam.Pose2(4.0, 0.0, 0.0)
    pose4_est = ideal_pose3.compose(delta_pose)
    
    initial_estimate.insert(X(4), pose4_est) 

    return graph, initial_estimate