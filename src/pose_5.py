import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None
    best_landmark = None
    min_trace_sum = float('inf')

    for p_key, p_val in pose_options.items():
        for landmark in [1, 2]:
            graph_copy = gtsam.NonlinearFactorGraph(graph)
            initial_estimate_copy = gtsam.Values(initial_estimate)

            graph_copy, initial_estimate_copy = add_pose(graph_copy, initial_estimate_copy, p_val)
            result = optimize(graph_copy, initial_estimate_copy)

            graph_copy = add_landmark_measurement(graph_copy, result, p_val, landmark)
            result = optimize(graph_copy, initial_estimate_copy)

            marginals = gtsam.Marginals(graph_copy, result)

            cov1 = marginals.marginalCovariance(L(1))
            cov2 = marginals.marginalCovariance(L(2))
            trace_sum = np.trace(cov1) + np.trace(cov2)

            if trace_sum < min_trace_sum:
                min_trace_sum = trace_sum
                best_pose = p_key
                best_landmark = landmark

    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    final_marginals = gtsam.Marginals(graph, result)
    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    sum_of_marginals = final_marginals.marginalCovariance(L(1)).sum() + final_marginals.marginalCovariance(L(2)).sum()
    
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None      # chosen pose option
    best_landmark = None    # chosen landmark (1 or 2)
    errors = float('inf')

    ideal_poses = {
        1: gtsam.Pose2(0.0, 0.0, 0.0),
        2: gtsam.Pose2(2.0, 0.0, 0.0),
        3: gtsam.Pose2(4.0, 0.0, 0.0)
    }

    for p_key, p_val in pose_options.items():
        for lm_key in [1, 2]:
            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_initial_estimate = gtsam.Values(initial_estimate)

            temp_graph, temp_initial_estimate = add_pose(temp_graph, temp_initial_estimate, p_val)
            temp_result = optimize(temp_graph, temp_initial_estimate)

            temp_graph = add_landmark_measurement(temp_graph, temp_result, p_val, lm_key)
            temp_result = optimize(temp_graph, temp_initial_estimate)
    
    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = []
            for i in [1, 2, 3]:
                est_pose = temp_result.atPose2(X(i))
                ideal_pose = ideal_poses[i]

                error = np.linalg.norm([
                    est_pose.x() - ideal_pose.x(),
                    est_pose.y() - ideal_pose.y(),
                    est_pose.theta() - ideal_pose.theta()
                ])
                list_of_errors.append(error)
    # TODO: compute the sum of the errors and return it along with the best pose and landmark
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < errors:
                errors = sum_of_errors
                best_pose = p_key
                best_landmark = lm_key

    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)
    
    return best_pose, best_landmark, errors