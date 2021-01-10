import os
import matplotlib.pyplot as plt
from IPython import display
import argparse

from tqdm import tqdm
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

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
    parser = argparse.ArgumentParser(description='Convert object detection results/ground truths to CommonRoad '
                                                 'scenarios')

    parser.add_argument('--input_folder', dest='input_folder',
                        help='folder containing object detection results/ground truths',
                        default="", type=str)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='folder to store CommonRoad scenarios',
                        default="", type=str)
    parser.add_argument('--init_scenario_folder', dest='init_scenario_folder',
                        help='folder to store initial scenarios',
                        default="", type=str)
    parser.add_argument('--dyna_obj_folder', dest='dyna_obj_folder',
                        help='folder to store the result of moving object classifier',
                        default="", type=str)

    args = parser.parse_args()
    return args


dyna_init_scenario = 'initial_scenario_11_13.xml'
static_init_scenario = 'initial_scenario_6_8.xml'


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


def convert_scenario(input_folder, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    file_names = os.listdir(input_folder)

    for file_name in tqdm(file_names):
        label_path = input_folder + file_name
        label = load_label(label_path)

        # Use the result of moving object classifier to set corresponding the dynamic constraints
        if os.path.exists(args.dyna_obj_folder + file_name):
            file_path = args.init_scenario_folder + dyna_init_scenario
        else:
            file_path = args.init_scenario_folder + static_init_scenario

        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

        for i in range(len(label)):
            if label[i][0] != 'Car' and label[i][0] != 'Van' and label[i][0] != 'Truck' and label[i][0] != 'Misc':
                continue

            # regulate orientation to [-pi, pi]
            orient = label[i][7]
            while orient < -np.pi: orient += 2 * np.pi
            while orient > np.pi: orient -= 2 * np.pi

            static_obstacle_id = scenario.generate_object_id()
            static_obstacle_type = ObstacleType.PARKED_VEHICLE
            static_obstacle_shape = Rectangle(width=label[i][5][1], length=label[i][5][2])
            static_obstacle_initial_state = State(position=np.array([label[i][6][2], -label[i][6][0]]),
                                                  orientation=-(orient - 0.5 * np.pi), time_step=0)
            static_obstacle = StaticObstacle(static_obstacle_id, static_obstacle_type, static_obstacle_shape,
                                             static_obstacle_initial_state)

            scenario.add_objects(static_obstacle)

        author = 'Jindi Zhang'
        affiliation = 'City University of Hong Kong'
        source = ''
        tags = {Tag.CRITICAL, Tag.INTERSTATE}

        fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)

        output_file_name = output_folder + '/' + file_name.split('.')[0] + '.xml'
        fw.write_to_file(output_file_name, OverwriteExistingFile.ALWAYS)


if __name__ == "__main__":
    args = parse_args()
    convert_scenario(args.input_folder, args.output_folder)
