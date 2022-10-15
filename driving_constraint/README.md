# Driving Constraint Selector

This classifier is used to classify the road type of driving scenarios and select proper real driving constraint for evaluation.

### Requirement
The code of this project is test in the following environment:
* Ubuntu 20.04.2 LTS
* Python 3.7.9
* PyTorch 1.7.0
* Torchvision 0.8.1

And there are other dependencies:
```
tqdm==4.59.0
Pillow==8.0.1
pandas==1.1.3
numpy==1.19.2
```

### Data Preparation
Make sure your directory structure looks like:

```
.
|--eval_driving_safety
    |--driving_constraint
        |--data
            |--image_2
            |--training_csv.csv
            |--validation_csv.csv
        |--Dataset.py
        |--Model.py
        |--train.py
        |--validation.py
```

To train and validate the driving constraint selector, we choose the first 1,200 scenarios of the dataset provided by [KITTI 3D Object Detection Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and filter out the scenarios where no car appears. We manually annotate the road type of each scenario to indicate whether it's in `highway` or `road`. 

We randomly divide the dataset into the training set and validation set with `split ratio = 4:1`. The training set contains 444 scenarios, while the validation set has 111 scenarios. The split files are saved in `training_csv.csv` and `validation_csv.csv`, respectively. You can create the `image_2` folder by creating the symbolic link which points to `path_to_your_kitti_dataset/training/image_2`.
```
cd data
ln -s path_to_your_kitti_dataset/training/image_2
```

Our pretrained model is available in [Google Drive](https://drive.google.com/file/d/1O8DmkrxxaQBQopWAXZcLOntCyf19vvTg/view?usp=sharing). If you want to try it out, put it into `eval_driving_safety/driving_constraint/model/`.

### Training
For training please run:

```
python train.py
```

The checkpoints are saved under `eval_driving_safety/driving_constraint/model/`.

### Validation
For validation please run:

```
python validate.py --loadmodel path_to_your_model
```

To validate specific model, you can change the `path_to_your_model` above.