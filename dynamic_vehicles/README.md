# Dynamic Vehicle Classifier

This classifier is to distinguish between dynamic vehicles and static vehicles.

### Requirement
The code of this project is test in the following environment:
* Ubuntu 18.04.4 LTS
* Python 3.6.9
* PyTorch 1.0.0
* Torchvision 0.2.2

And there are other dependencies:
```
    tqdm==4.55.1
    Pillow==7.2.0
    pandas==1.1.5
    numpy==1.18.5
```

### Dataset Preparation
Make sure your directory structure looks like:
```
.
|--eval_driving_safety
   |--dynamic_vehicles
      |--data
      |  |--training_image_2
      |  |--validation_image_2
      |--model
      |  |--pretrained_model
      |--create_training_csv.py
      |--create_validation_csv.py
      |-- ...
```

In terms of training and validating the dynamic vehicle classifier, we choose the first 1200 scenarios of the dataset provided by the KITTI 3D Object Detection Benchmark and filter out the scenarios where no car appears. We manually annotate each vehicle in every scenaio to indicate whether or not it is moving based on the previous and the subsequent frames and crop it out. The naming format of each vehicle image in the dataset is "aaaaaa_bc.png", where "aaaaaa" is the scenario index, "b" is the vehicle index in that scenario, and "c" is the label suggesting its dynamic status ("s" stands for static and "d" for dynamic).

The training dataset contains 4605 scenaios, while the validation dataset set has 533 scenaios. You can download them from [Google Drive](https://).

Our pretrained model is available in [Google Drive](https://). If you want to try it out, put it into eval_driving_safety/dynamic_vehicles/model/pretrained_model/.

### Training
For training please run:
```
    python train.py
```

The checkpoints are saved under eval_driving_safety/dynamic_vehicles/model/.

### Validation
For validation please run:
```
    python validate.py
```

For validating specific model, you may change the loading path in the code.