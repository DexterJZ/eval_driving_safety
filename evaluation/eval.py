import os
import argparse
import ast
import math

from commonroad.common.solution import *


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the motion planning result')
    parser.add_argument('--success_rate', action='store_true', default=False,
                        help='evaluate the number of successfully generated trajectories')
    parser.add_argument('--collision_rate', action='store_true', default=False,
                        help='evaluate the number of collision-free solutions')
    parser.add_argument('--safe_driving_rate', action='store_true', default=False,
                        help='evaluate the safe driving performance')
    parser.add_argument('--path_len', action='store_true', default=False,
                        help='calculate the average path length')
    parser.add_argument('--travel_time', action='store_true', default=False,
                        help='calculate the average travel time')
    parser.add_argument('--eval_all', action='store_true', default=False,
                        help='evaluate all above metrics')

    parser.add_argument('--scenario_folder', dest='scenario_path',
                        help='path to scenario file',
                        default="", type=str)
    parser.add_argument('--solution_folder', dest='solution_path',
                        help='path to solution file',
                        default="", type=str)
    parser.add_argument('--gt_folder', dest='gt_path',
                        help='path to ground truth scenario file',
                        default="", type=str)

    args = parser.parse_args()
    return args


def calc_len(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


if __name__ == "__main__":
    args = parse_args()

    if args.eval_all:
        args.success_rate = args.collision_rate = args.safe_driving_rate = \
            args.path_len = args.travel_time = True

    # calc successful planning rate
    if args.success_rate:
        """
        the percentage of successfully planned trajectories in all scenarios.

        m_suc = k_trj/k_dts, where k_dts is the total number of scenarios in a dataset,
        and k_trj is the number of scenarios in that dataset where a trajectory can be successfully
        generated, no matter whether it is collision-free or not.
        """

        # get scenarios and solutions
        if args.scenario_path and args.solution_path:
            scenarios = os.listdir(args.scenario_path)
            solutions = os.listdir(args.solution_path)
        else:
            print('please input scenario_path or solution_path')
            raise AssertionError

        print("Success Rate: ", (len(solutions) / len(scenarios)))

    # calc collision rate
    if args.collision_rate:
        """
        the percentage of scenarios in all successfully planned trajectories where a collision occurs.

        m_cls = k_cls/k_trj, where k_cls is the number of scenarios with collision occurrence.
        """

        os.system('python check_collision.py --check_all --scenario_path {0} --solution_path {1}'.format(args.gt_path,
                                                                                                         args.solution_path))

    # calc safe driving rate
    if args.safe_driving_rate:
        """
        the percentage of scenarios in a dataset where a collision-free trajectory can be produced by the motion
        planning module.

        m_saf = (k_trj - k_cls) / k_dts.
        """

        # read collision number
        with open('collision.txt', 'r') as f:
            collision_num = int(f.readline().strip('\n'))

        scenarios = len(os.listdir(args.scenario_path))
        solutions = len(os.listdir(args.solution_path))

        print("Safe driving rate: ", (solutions - collision_num) / scenarios)

    # calc average path length and average travel travel time
    if args.path_len or args.travel_time:
        solutions = os.listdir(args.solution_path)
        len_sum = 0.
        time_sum = 0.

        for solution_path in solutions:
            # retrieve the state list in the solution to calc
            solution = CommonRoadSolutionReader().open(args.solution_path + solution_path)
            ego_vehicle_trajectory = solution.planning_problem_solutions[0].trajectory

            trajectory_len = 0.
            travel_time = ego_vehicle_trajectory.state_list[-1].time_step
            last_pos = None

            for i in range(len(ego_vehicle_trajectory.state_list)):
                current_pos = ego_vehicle_trajectory.state_list[i].position
                if last_pos is not None:
                    trajectory_len += calc_len(current_pos, last_pos)
                last_pos = current_pos

            len_sum += trajectory_len
            time_sum += (0.1 * travel_time)  # one unit of time_step stands for 0.1s

        if args.path_len:
            print("Average path length: ", (len_sum / len(solutions)))
        if args.travel_time:
            print("Average travel time: ", (time_sum / len(solutions)))
