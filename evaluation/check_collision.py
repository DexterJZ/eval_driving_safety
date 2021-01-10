import os
import matplotlib.pyplot as plt
from IPython import display
import argparse
from tqdm import tqdm

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

from commonroad_cc.visualization.draw_dispatch import draw_object
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_checker


def parse_args():
    parser = argparse.ArgumentParser(description='Check collision')

    parser.add_argument('--check_all', action='store_true', default=False,
                        help='will check collision for all the solutions in the folder')
    parser.add_argument('--scenario_path', dest='scenario_path',
                        help='path to scenario file',
                        default="", type=str)
    parser.add_argument('--solution_path', dest='solution_path',
                        help='path to solution file',
                        default="", type=str)

    args = parser.parse_args()
    return args

scenarios = []
solutions = []
filenames = []
collision_num = 0

if __name__ == "__main__":
    args = parse_args()
    scenario_path = args.scenario_path
    solution_path = args.solution_path

    # generate correspond scenario path and solution path
    if args.check_all:
        # for all scenarios and solutions in the correspond folder
        filenames = os.listdir(solution_path)
        filenames.sort()

        scenarios = [(scenario_path + sol[17:23] + '.xml') for sol in filenames]
        solutions = [(solution_path + sol) for sol in filenames]
    else:
        # for single scenario and solution check
        scenarios.append(scenario_path)
        solutions.append(solution_path)

    for scenario_p, solution_p in tqdm(zip(scenarios, solutions)):
        scenario, planning_problem_set = CommonRoadFileReader(scenario_p).open()
        solution = CommonRoadSolutionReader().open(solution_p)
        plot_name = scenario_p[-10:-4]

        ego_vehicle_trajectory = solution.planning_problem_solutions[0].trajectory
        # ego_vehicle_trajectory_first = Trajectory(initial_time_step=0, state_list=ego_vehicle_trajectory.state_list[:2])

        vehicle2 = parameters_vehicle2()
        ego_vehicle_shape = Rectangle(length=vehicle2.l, width=vehicle2.w)
        ego_vehicle_prediction = TrajectoryPrediction(
            trajectory=ego_vehicle_trajectory, shape=ego_vehicle_shape)

        cc = create_collision_checker(scenario)
        co = create_collision_object(ego_vehicle_prediction)

        if args.check_all and cc.collide(co):
            print(plot_name)
            collision_num += 1
        else:
            print('Does collision exist? ', cc.collide(co))

            plt.figure()
            draw_object(scenario.lanelet_network)
            draw_object(cc, draw_params={'collision': {'facecolor': 'blue'}})
            draw_object(co, draw_params={'collision': {'facecolor': 'green'}})
            plt.autoscale()
            plt.axis('equal')
            plt.xlim(-10, 40)
            plt.ylim(-15, 15)
            plt.savefig('plot/{}.png'.format(plot_name))

    if args.check_all:
        print("collision rate: ", collision_num / len(solutions))

        # record the number of collision case
        with open('collision.txt', 'w') as f:
            f.write(str(collision_num))
