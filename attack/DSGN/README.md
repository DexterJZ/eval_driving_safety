## Adversarial attacks against DSGN

This folder contains code about performing `perturbation attack` and `patch attack` against one of our target vision-based 3D object detectors, namely, `DSGN`.

### Requirement

This implementation is tested in following environment:
- Ubuntu 18.04.1
- Python 3.7
- PyTorch 1.3.0
- Torchvision 0.4.1
- [DSGN](https://github.com/Jia-Research-Lab/DSGN) (latest version)

The perturbation and patch are trained with 1 *NVIDIA Tesla V100 (32G)* GPU. 

### Install

Please refer instructions in [DSGN Installation](https://github.com/Jia-Research-Lab/DSGN#installation) to set up the `DSGN` folder.

Third party libraries can be installed(in a `python3` virtualenv) using:

```
pip install -r requirements.txt
```

### Data preparation

Please follow steps in the [Data Preparation](https://github.com/Jia-Research-Lab/DSGN#data-preparation) to set the repo. 

For the KITTI dataset, please save it in a separate folder. In this way, we can create symbolic links to change `data/kitti/training/image_2` and `data/kitti/training/image_3` to point to image folder we generated under attacks.

After setting up the model, please add our code to `tools` folder. The overall directory will look like below:

```
.                                           (root directory)
|-- dsgn                 
|-- configs                          
|-- data
|   |-- kitti
|       |-- training
|           |-- image_2                     (symbolic link to raw data / attacked data)
|           |-- image_3                     (symbolic link to raw data / attacked data)
|-- ...
|-- tools
    |-- env_utils
        |-- ...
    |-- generate_targets.py
    |-- merge_results.py
    |-- test_net.py
    |-- train_net.py
    |-- patch_attack.py                     (patch attack file)
    |-- pgd_attack.py                       (pgd attack file)
    |-- predict_and_save_patch.py           (evaluate the model under patch attack)
    |-- predict_and_save_pgd.py             (evaluate the model under pgd attack)
```

### Perturbation attack

**Launch the perturbation attack**:
```
python3 tools/pgd_attack.py --loadmodel ./outputs/temp/DSGN_car_pretrained/finetune_53.tar -btest 1 -d 0 --iter [iteration number] --alpha [perturbation update step]
```
The attacked images all saved in `./dsgn_pgd_iters_[iteration number]` folder.

**Evaluate DSGN** regarding the average precision under perturbation attack (please ensure that you have change the symbolic links of `data/kitti/training/image_2` and `data/kitti/training/image_3` to the image folder under attack):

```
python3 tools/predict_and_save_pgd.py --loadmodel ./outputs/temp/DSGN_car_pretrained/finetune_53.tar -btest 1 -d 0 --iter [iteration number of attack] --alpha [perturbation update step of attack]
```

### Patch attack

**Launch the patch attack**:
```
python3 ./tools/patch_attack.py --loadmodel ./outputs/temp/DSGN_car_pretrained/finetune_53.tar -btest 1 -d 0 --debug --debugnum [number of training data] --epochs [training epoch number] --ratio [trained patch size]
```
The trained patch is saved in `./dsgn_patch_ratio_[ratio size]` folder. The patch attack is time-consuming, for `debugnum=50, epochs=80, ratio=0.2` setting in our testing environment, it takes about 24 hours to finish the training.

**Evaluate the DSGN model under patch attack** (please ensure the symbolic links of `image_2` and `image_3` point to raw data): 

```
python3 tools/predict_and_save_patch.py --ratio [trained patch ratio] --epochs [training epoch number] --patch_dir [path to your patch folder] --atk_mode [random/specific attack]
```

