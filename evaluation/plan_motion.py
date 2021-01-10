import os
import sys
sys.path.append("../../GSMP/motion_automata")
from automata.HelperFunctions import *

# select motion planning algorithm
# from automata.MotionPlanner_gbfs import MotionPlanner
from automata.MotionPlanner_Astar import MotionPlanner
# from automata.MotionPlanner_gbfs_only_time import MotionPlanner

from commonroad.common.solution_writer import CommonRoadSolutionWriter, VehicleModel, VehicleType, CostFunction
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Plan the motion')

    parser.add_argument('--input_folder', dest='input_folder',
                        help='folder containing CommonRoad scenarios',
                        default="", type=str)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='folder to store solutions',
                        default="", type=str)
    parser.add_argument('--motion_primitive_folder', dest='motion_primitive_folder',
                        help='folder to store motion primitives',
                        default="", type=str)
    parser.add_argument('--dyna_obj_folder', dest='dyna_obj_folder',
                        help='folder to store the result of moving object classifier',
                        default="", type=str)

    args = parser.parse_args()
    return args


dyna_mp_path = 'V_11.0_13.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.36_T_0.5_Model_BMW320i.xml'
static_mp_path = 'V_6.0_8.0_Vstep_0_SA_-1.066_1.066_SAstep_0.36_T_0.5_Model_BMW320i.xml'


def plan_motion(input_folder, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    file_names = os.listdir(input_folder)
    file_names.sort()

    scenario_path_prefix = input_folder

    for file_name in file_names:
        time_start = time.time()
        scenario_id = file_name.split('.')[0]
        print(scenario_id)

        scenario_path = scenario_path_prefix + '/' + scenario_id + '.xml'

        scenario, planning_problem_set = load_scenario(scenario_path)

        veh_type_id = 2

        if veh_type_id == 1:
            veh_type = VehicleType.FORD_ESCORT
        elif veh_type_id == 2:
            veh_type = VehicleType.BMW_320i
        elif veh_type_id == 3:
            veh_type = VehicleType.VW_VANAGON

        # choose the motion primitive with the result of moving object classifier(need to be substituted by your own
        # motion primitives!!!)
        if os.path.exists(args.dyna_obj_folder + file_name[:-4] + '.txt'):
            mp_path = dyna_mp_path
        else:
            mp_path = static_mp_path

        automata = generate_automata(veh_type_id, mp_file=args.motion_primitive_folder + mp_path)

        planning_problem_idx = 0
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[planning_problem_idx]

        automata, initial_motion_primitive = \
            add_initial_state_to_automata(automata, planning_problem)

        motion_planner = MotionPlanner(scenario, planning_problem, automata)

        result_path, result_dict_status = start_search(
            scenario,
            planning_problem,
            automata,
            motion_planner,
            initial_motion_primitive,
            flag_plot_intermediate_results=False,
            flag_plot_planning_problem=False)

        # check if search fails
        if result_path is None or len(result_path) <= 1:
            continue

        # access to cost
        print('cost: ', result_dict_status['cost_current'])

        list_state = list()

        for state in result_path:
            kwarg = {'position': state.position,
                     'velocity': state.velocity,
                     'steering_angle': state.steering_angle,
                     'orientation': state.orientation,
                     'time_step': state.time_step}
            list_state.append(State(**kwarg))

        trajectory = Trajectory(initial_time_step=list_state[0].time_step,
                                state_list=list_state)

        csw = CommonRoadSolutionWriter(output_dir=output_folder,
                                       scenario_id=scenario_id,
                                       step_size=0.1,
                                       vehicle_type=veh_type,
                                       vehicle_model=VehicleModel.KS,
                                       cost_function=CostFunction.SM1)

        csw.add_solution_trajectory(
            trajectory=trajectory,
            planning_problem_id=planning_problem.planning_problem_id)

        csw.write_to_file(overwrite=True)
        time_end = time.time()
        print('time cost', time_end - time_start, 's')


if __name__ == "__main__":
    args = parse_args()
    plan_motion(args.input_folder, args.output_folder)
