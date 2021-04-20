## An End-to-End Driving Safety Evaluation Framework

This folder contains the implementation of the end-to-end driving safety evaluation framework that we proposed.

### Requirement

This implementation is tested in following environment:
- Ubuntu 16.04.1
- Python 3.7
- [CommonRoad-Search commit 0d434310](https://gitlab.lrz.de/tum-cps/commonroad-search/-/tree/0d434310f393e5af6c155507e691982ab8fd27890)

### Installation

Please follow [Installation guide] to set up the repo.

Third party libraries can be installed(in a `python3` virtualenv) using:

```
pip install -r requirements.txt
```

### Data preparation

After adding our code, the overall directory will look like below:

```
.                                           (root directory)
|-- evaluation
    |-- commonroad-search
        |-- notebooks
            |-- tutorials
                |-- plan_motion.py          (motion planning file)
    |--init_scenario
        |-- ...
    |-- kitti_labeled
        |-- ...
    |-- check_collision.py
    |-- convert_gt_scenarios.py
    |-- convert_scenarios.py
    |-- eval.py
    |-- plan_motion.py
    |-- plot_scenario.py
    |-- plot_solution.py
    |-- README.md
    |-- requirements.txt
```

### Evaluation

To evaluate driving safety under adversarial attacks, please follow steps below.

#### 0. Preparation

Before you start the evaluation, please refer [driving_constraint/README.md](../driving_constraint/README.md) and [dynamic_vehicles](../dynamic_vehicles/README.md) to generate the dynamic information of driving scenarios.
#### 1. convert scenarios
```
python convert_scenarios.py --input_folder [path to your detection result folder] --output_folder [path to your scenario folder] --init_scenario_folder [path to your initial scenario]
```

#### 2. convert ground truth scenarios 

Ground truth scenarios are combine with planned trajectory to evaluate

```
python convert_gt_scenarios.py --input_folder [path to ground truth detection result folder] --output_folder [path to your ground truth scenario folder] --init_scenario_folder [path to your initial scenario]
```

#### 3. plot scenarios (optional)
```
python plot_scenario.py --input_folder [path to the scenario file] 
```

#### 4. plan motion
```
python plan_motion.py --input_folder [path to your scenario folder] --output_folder [path to your solution(trajectory) folder]
```

#### 5. plot solution(optional)
```
python plot_solution.py --scenario_path [path to your scenarios folder]
```

#### 6. evaluate
```
python eval.py --scenario_folder [path to scenarios] --solution_folder [path to planned trajectories] --gt_folder [path to corresponding ground truth scenarios]
```

### Driving Safety Performance Metrics

The `eval.py` file will output following metrics:
- Successful planning rate
- Collision rate
- Safe driving rate
- Travel time
- Trajectory length


