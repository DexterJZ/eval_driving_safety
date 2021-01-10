import os
import matplotlib.pyplot as plt
from IPython import display
import argparse

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.trajectory import State

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.scenario import Location
from commonroad.scenario.scenario import Tag
from commonroad.common.solution import *
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2


def parse_args():
    parser = argparse.ArgumentParser(description='Plot scenario solution')

    parser.add_argument('--scenario_path', dest='scenario_path',
                        help='path to scenario file',
                        default="", type=str)

    parser.add_argument('--solution_path', dest='solution_path',
                        help='path to solution file',
                        default="", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    scenario_path = args.scenario_path
    solution_path = args.solution_path

    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    solution = CommonRoadSolutionReader().open(solution_path)

    ego_vehicle_trajectory = solution.planning_problem_solutions[0].trajectory
    initial_state = ego_vehicle_trajectory.state_at_time_step(0)

    vehicle2 = parameters_vehicle2()
    ego_vehicle_shape = Rectangle(length=vehicle2.l, width=vehicle2.w)
    ego_vehicle_prediction = TrajectoryPrediction(
        trajectory=ego_vehicle_trajectory, shape=ego_vehicle_shape)

    ego_vehicle_type = ObstacleType.CAR
    ego_vehicle = DynamicObstacle(obstacle_id=100,
                                  obstacle_type=ego_vehicle_type,
                                  obstacle_shape=ego_vehicle_shape,
                                  initial_state=initial_state,
                                  prediction=ego_vehicle_prediction)

    for i in range(0, 40):
        display.clear_output(wait=True)
        plt.figure()
        draw_object(scenario, draw_params={'time_begin': i})
        draw_object(ego_vehicle,
                    draw_params={'time_begin': i, 'facecolor': 'g'})
        draw_object(planning_problem_set)
        plt.gca().set_aspect('equal')
        plt.xlim(-10, 40)
        plt.ylim(-15, 15)
        plt.savefig('solution{}.png'.format(i))
        break
