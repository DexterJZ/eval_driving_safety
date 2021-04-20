## Adversarial attacks against Stereo R-CNN

This folder contains code about performing `perturbation attack` and `patch attack` against one of our target vision-based 3D object detectors, namely, `Stereo-RCNN`.

### Requirement

This implementation is tested in following environment:
- Ubuntu 16.04.12
- Python 3.6
- PyTorch 1.0.0
- Torchvision 0.2.2
- [Stereo R-CNN branch 1](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/tree/1.0)

The perturbation and patch are trained with 1 *NVIDIA Tesla T4* GPU. 

### Install

Please refer instructions in [Stereo R-CNN 0.Install](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/tree/1.0#0-install) to install the repo in two folder, `./Stereo R-CNN` and `./eval/Stereo R-CNN`, for attack and eval respectively.

Third party libraries can be installed(in a `python3` virtualenv) using:

```
pip install -r requirements.txt
```

### Dataset preparation

Please follow steps in the [Dataset Preparation](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/tree/1.0#2-dataset-preparation) to set the dataset. 
And use symbolic links to make `data/kitti/object/training/image_2` and `data/kitti/object/training/image_3` changeable. So that, they can point to image folder we generated under attacks.

After setting up the repo, please add our attack code to two directories. Remember to substitute `stereo_rpn.py`, `stereo_rcnn.py` and `roibatchLoader.py` under attack repo. The overall directory will look like below:

```
.                                           (root directory)
|-- Stereo R-CNN
    |-- lib
        |-- model
            |-- stereo_rpn.py               (substitute file)
            |-- stereo_rcnn.py              (substitute file)
            |-- ...
        |-- roi_data_layer
            |-- roibatchLoader.py           (substitute file)
            |-- ...
    |-- patch_attack.py                     (patch attack file)
    |-- pgd_attack.py                       (pgd attack file)
    |-- ...            
|-- DSGN
    |-- predict_and_save_patch.py           (evaluate patch attack)
    |-- predict_and_save_pgd.py             (evaluate pgd attack)
    |-- ...    
```

### Perturbation attack

**Launch the perturbation attack**:
```
python pgd_attack.py --iter [iteration number of attack] --alpha [perturbation update step of attack]
```
The attacked images are saved in `./stereo_rcnn_pgd_iters_[iteration number]` folder.

**Evaluate Stereo R-CNN under perturbation attack** (please ensure that you have change the symbolic links of `data/kitti/object/training/image_2` and `data/kitti/object/training/image_3` to the image folder under attack):
```
python predict_and_save_pgd.py --iter [iteration number of attack] --alpha [perturbation update step of attack]
```
The detection results are saved in `result_stereo_rcnn_pgd_[iteration number]_{alpha value}`. If you want to evaluate regarding average precision, please refer to [kitti_eval](https://github.com/prclibo/kitti_eval).

### Patch attack

**Launch the patch attack**:
```
python patch_attack.py --debug --debugnum [number of training data] --epochs [training epoch number] --ratio [trained patch size]
```
The trained patch is saved in `./stereo_rcnn_patch_ratio_[ratio size]` folder.

**Evaluate Stereo R-CNN under patch attack** (please ensure the symbolic links of `image_2` and `image_3` point to raw data):
```
python predict_and_save_patch.py -ratio [trained patch ratio] --epochs [training epoch number] --patch_dir [path to your patch folder] --atk_mode [random/specific attack]
```

