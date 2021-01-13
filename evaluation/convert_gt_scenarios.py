import os
import matplotlib.pyplot as plt
from IPython import display
import argparse

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import random
from tqdm import tqdm
import numpy as np
import math

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from commonroad.scenario.lanelet import Lanelet, LaneletType

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.scenario import Location
from commonroad.scenario.scenario import Tag


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert object detection ground truths to CommonRoad scenarios')

    parser.add_argument('--input_folder', dest='input_folder',
                        help='folder containing ground truths',
                        default="", type=str)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='folder to store CommonRoad scenarios',
                        default="", type=str)
    parser.add_argument('--init_scenario_folder', dest='init_scenario_folder',
                        help='folder to store initial scenarios',
                        default="", type=str)

    args = parser.parse_args()
    return args


# min, max velocity of ego vehicle for different scenarios
default_vmin = 6  # 6m/s minimum velocity for street scenarios
default_vmax = 8
road_vmin = 11  # 11m/s minimum velocity for highway scenarios
road_vmax = 13

# output file configuration
author = 'Jindi Zhang'
affiliation = 'City University of Hong Kong'
source = ''
tags = {Tag.CRITICAL, Tag.INTERSTATE}


def load_label(label_path):
    """
    item[0]: type       Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
    item[1]: truncated  Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries
    item[2]: occluded   Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
    item[3]: alpha      Observation angle of object, ranging [-pi..pi]
    item[4]: bbox       2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
    item[5]: dimensions 3D object dimensions: height, width, length (in meters)
    item[6]: location   3D object location x,y,z in camera coordinates
                        (in meters)
    item[7]: rotation_y Rotation ry around Y-axis in camera coordinates
                        [-pi..pi]
    """

    label = []

    with open(label_path, 'r') as file:
        for line in file:
            item = []
            elements = line.strip().split(' ')
            item.append(elements[0])
            item.append(float(elements[1]))
            item.append(float(elements[2]))
            item.append(float(elements[3]))
            item.append([float(elements[4]),
                         float(elements[5]),
                         float(elements[6]),
                         float(elements[7])])
            item.append([float(elements[8]),
                         float(elements[9]),
                         float(elements[10])])
            item.append([float(elements[11]),
                         float(elements[12]),
                         float(elements[13])])
            item.append(float(elements[14]))
            label.append(item)

    return label


def create_static_obstacle(scenario, label, i):
    static_obstacle_id = scenario.generate_object_id()
    static_obstacle_type = ObstacleType.PARKED_VEHICLE
    static_obstacle_shape = Rectangle(width=label[i][5][1], length=label[i][5][2])
    static_obstacle_initial_state = State(position=np.array([label[i][6][2], -label[i][6][0]]),
                                          orientation=-(label[i][7] - 0.5 * np.pi), time_step=0)
    static_obstacle = StaticObstacle(static_obstacle_id, static_obstacle_type, static_obstacle_shape,
                                     static_obstacle_initial_state)

    return static_obstacle


def create_dynamic_obstacle(scenario, label, minv, maxv, i):
    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = ObstacleType.CAR
    dynamic_obstacle_shape = Rectangle(width=label[i][5][1], length=label[i][5][2])

    dynamic_obstacle_initial_state = State(position=np.array([label[i][6][2], -label[i][6][0]]),
                                           orientation=-(label[i][7] - 0.5 * np.pi), time_step=0)

    state_list = []
    v = random.uniform(minv, maxv)
    for j in range(1, 20):
        # divide velocity
        angle = -(label[i][7] - 0.5 * np.pi)
        angle = (angle + np.pi) % (2 * np.pi)
        x = dynamic_obstacle_initial_state.position[0] + ((v * math.cos(angle)) * (scenario.dt * j))
        y = dynamic_obstacle_initial_state.position[1] + ((v * math.sin(angle)) * (scenario.dt * j))
        new_position = np.array([x, y])

        new_state = State(position=new_position, velocity=v, orientation=-(label[i][7] - 0.5 * np.pi), time_step=j)
        state_list.append(new_state)

    dynamic_obstacle_trajectory = Trajectory(1, state_list)
    dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)
    dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                       dynamic_obstacle_type,
                                       dynamic_obstacle_shape,
                                       dynamic_obstacle_initial_state,
                                       dynamic_obstacle_prediction)

    return dynamic_obstacle


def convert_scenario(input_folder, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # dynamic
    dynamic_dir = input_folder + 'dynamic_label/'
    dynamic_annotation_dir = input_folder + 'dynamic_annotation/'
    dynamic_names = os.listdir(dynamic_dir)
    dynamic_names.sort()

    # static
    static_dir = input_folder + 'static_label/'
    static_names = os.listdir(static_dir)
    static_names.sort()

    # road dynamic
    road_dynamic_dir = input_folder + 'road_dynamic_label/'
    road_dynamic_annotation_dir = input_folder + 'road_dynamic_annotation/'
    road_dynamic_names = os.listdir(road_dynamic_dir)
    road_dynamic_names.sort()

    # convert static scenarios
    for file_name in tqdm(static_names):
        label_path = static_dir + file_name

        label = load_label(label_path)

        file_path = args.init_scenario_folder + \
                    'initial_scenario_{0}_{1}.xml'.format(default_vmin, default_vmax)

        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

        for i in range(len(label)):
            if label[i][0] != 'Car' and label[i][0] != 'Van' and label[i][0] != 'Truck' and label[i][0] != 'Misc':
                continue

            # treat objects in the static scenario as static obstacle
            static_obstacle = create_static_obstacle(scenario, label, i)
            scenario.add_objects(static_obstacle)

        fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)

        output_file_name = output_folder + '/' + file_name.split('.')[0] + '.xml'
        fw.write_to_file(output_file_name, OverwriteExistingFile.ALWAYS)

    # convert dynamic scenarios
    for file_name in tqdm(dynamic_names):
        label_path = dynamic_dir + file_name
        annotation_path = dynamic_annotation_dir + file_name
        annos = []

        label = load_label(label_path)

        file_path = args.init_scenario_folder + \
                    'initial_scenario_{0}_{1}.xml'.format(default_vmin, default_vmax)

        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

        # get annotations
        with open(annotation_path) as f:
            for line in f:
                annos.append(line.strip('\n'))

        for i in range(len(label)):
            if label[i][0] != 'Car' and label[i][0] != 'Van' and label[i][0] != 'Truck' and label[i][0] != 'Misc':
                continue

            # for dynamic scenarios, convert objects to static/dynamic obstacle according to the annotation
            # n: ignored(pedestrain or cycltist)
            # 0: static vehicle(e.g. car parked on the side of the road)
            # 1: dynamic vehicle with same driving direction
            # -1: dynamic vehicle with opposite driving direction
            if annos[i] == 'n':
                continue
            elif annos[i] == '0':
                static_obstacle = create_static_obstacle(scenario, label, i)
                scenario.add_objects(static_obstacle)
            elif annos[i] == '1' or annos[i] == '-1':
                dynamic_obstacle = create_dynamic_obstacle(scenario, label, default_vmin, default_vmax, i)
                scenario.add_objects(dynamic_obstacle)

        fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)

        output_file_name = output_folder + '/' + file_name.split('.')[0] + '.xml'
        fw.write_to_file(output_file_name, OverwriteExistingFile.ALWAYS)

    # convert road dynamic scenarios
    for file_name in tqdm(road_dynamic_names):

        label_path = road_dynamic_dir + file_name
        annotation_path = road_dynamic_annotation_dir + file_name
        annos = []

        label = load_label(label_path)

        file_path = args.init_scenario_folder + \
                    'initial_scenario_{0}_{1}.xml'.format(road_vmin, road_vmax)

        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

        # get annotations
        with open(annotation_path) as f:
            for line in f:
                annos.append(line.strip('\n'))

        for i in range(len(label)):
            if label[i][0] != 'Car' and label[i][0] != 'Van' and label[i][0] != 'Truck' and label[i][0] != 'Misc':
                continue

            if annos[i] == 'n':
                continue
            elif annos[i] == '0':
                static_obstacle = create_static_obstacle(scenario, label, i)
                scenario.add_objects(static_obstacle)
            elif annos[i] == '1' or annos[i] == '-1':
                dynamic_obstacle = create_dynamic_obstacle(scenario, label, road_vmin, road_vmax, i)
                scenario.add_objects(dynamic_obstacle)

        fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)

        output_file_name = output_folder + '/' + file_name.split('.')[0] + '.xml'
        fw.write_to_file(output_file_name, OverwriteExistingFile.ALWAYS)


if __name__ == "__main__":
    args = parse_args()
    convert_scenario(args.input_folder, args.output_folder)
