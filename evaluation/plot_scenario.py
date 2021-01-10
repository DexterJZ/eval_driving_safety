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


def parse_args():
    parser = argparse.ArgumentParser(description='Plot CommonRoad scenario')

    parser.add_argument('--file_path', dest='file_path',
                        help='path to scenario file',
                        default="", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    file_path = args.file_path

    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    for i in range(0, 20):
        display.clear_output(wait=True)
        plt.figure()
        draw_object(scenario, draw_params={'time_begin': i})
        draw_object(planning_problem_set)
        plt.gca().set_aspect('equal')
        plt.xlim(-10, 40)
        plt.ylim(-15, 15)
        plt.savefig('scenario{}.png'.format(i))

        break
