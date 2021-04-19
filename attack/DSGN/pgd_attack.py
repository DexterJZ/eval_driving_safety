from __future__ import print_function

import argparse
import os
import random
import sys
import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import cv2
from PIL import Image
from tqdm import tqdm
import scipy.misc as ssc

from dsgn.models import *
from dsgn.utils.numpy_utils import *
from dsgn.utils.numba_utils import *
from dsgn.utils.torch_utils import *
from dsgn.models.loss3d import RPN3DLoss
from dsgn.models.inference3d import make_fcos3d_postprocessor
from env_utils import *

parser = argparse.ArgumentParser(description="PGD attack")
parser.add_argument('-cfg', '--cfg', '--config',
                    default=None, help='config path')
parser.add_argument(
    '--data_path', default='./data/kitti/training', help='select model')
parser.add_argument('--loadmodel', default=None, help='loading model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split_file', default='./data/kitti/val.txt',
                    help='split file')
parser.add_argument('--btest', '-btest', type=int, default=None)
parser.add_argument('--devices', '-d', type=str, default=None)
parser.add_argument('--tag', '-t', type=str, default='')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode')
parser.add_argument('--debugnum', default=None, type=int,
                    help='debug mode')

parser.add_argument('--iter', type=int, default=4, help='iteration number of pgd attack')
parser.add_argument('--alpha', type=float, default=(1.0 / 255))
parser.add_argument('--eps', type=float, default=0.3)
args = parser.parse_args()

if not args.devices:
    args.devices = str(np.argmin(mem_info()))

if args.devices is not None and '-' in args.devices:
    gpus = args.devices.split('-')
    gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
    gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
    args.devices = ','.join(map(lambda x: str(x), list(range(*gpus))))

if args.debugnum is None:
    args.debugnum = 100

exp = Experimenter(os.path.dirname(args.loadmodel), args.cfg)
cfg = exp.config

if args.debug:
    args.btest = len(args.devices.split(','))
    num_workers = 0
    cfg.debug = True
    args.tag += 'debug{}'.format(args.debugnum)
else:
    num_workers = 12

assert args.btest

print('Using GPU:{}'.format(args.devices))
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# ------------------- Data Loader -----------------------
from dsgn.dataloader import KITTILoader3D as ls
from dsgn.dataloader import KITTILoader_dataset3d as DA

all_left_img, all_right_img, all_left_disp, = ls.dataloader(args.data_path,
                                                            args.split_file,
                                                            depth_disp=True,
                                                            cfg=cfg,
                                                            is_train=True)

ImageFloader = DA.myImageFloder(
    all_left_img, all_right_img, all_left_disp, True, split=args.split_file, cfg=cfg)


class BatchCollator(object):
    def __init__(self, cfg):
        super(BatchCollator, self).__init__()
        self.cfg = cfg

    def __call__(self, batch):
        transpose_batch = list(zip(*batch))
        ret = dict()
        ret['imgL'] = torch.cat(transpose_batch[0], dim=0)
        ret['imgR'] = torch.cat(transpose_batch[1], dim=0)
        ret['disp_L'] = torch.stack(transpose_batch[2], dim=0)
        ret['calib'] = transpose_batch[3]
        ret['calib_R'] = transpose_batch[4]
        ret['image_indexes'] = transpose_batch[5]
        ii = 6
        if self.cfg.RPN3D_ENABLE:
            ret['targets'] = transpose_batch[ii]
            ii += 1
        if self.cfg.RPN3D_ENABLE:
            ret['ious'] = transpose_batch[ii]
            ii += 1
            ret['labels_map'] = transpose_batch[ii]
            ii += 1
        return ret


# unable shuffle and sampler here
TestImgLoader = torch.utils.data.DataLoader(
    ImageFloader,
    batch_size=args.btest, shuffle=False, num_workers=num_workers,
    collate_fn=BatchCollator(cfg))

# ------------------- Model -----------------------
model = StereoNet(cfg=cfg)

model = nn.DataParallel(model)
model.cuda()
model.eval()

if args.loadmodel is not None and args.loadmodel.endswith('tar'):
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    print('Loaded {}'.format(args.loadmodel))
else:
    print('------------------------------ Load Nothing ---------------------------------')

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

# mean and standard deviation of the dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def tensor2im(input_image, imtype=np.uint8):
    """"convert the tensor into a numpy array and denormalize it

    Parameters:
        input_image (tensor) --  tensor of input image
        imtype (type)        --  output type of numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_img(im, path, w, h):
    """ im can be tensor type data without any processing,
        and the data is stored in path

    Parameters:
        im (tensor) --  tensor of input image
        path (str)  --  patch to save the image
        size (int)  --  size of the image
    """
    im_numpy = tensor2im(im)  # convert to numpy array and denormalize
    im_array = Image.fromarray(im_numpy)
    cropped = im_array.crop((0, 0, w, h))
    cropped.save(path)


def denormalize(im):
    for i in range(len(mean)):
        im.data[0][i] = im.data[0][i] * std[i] + mean[i]

    return im


def normalize(im):
    for i in range(len(mean)):
        im.data[0][i] = (im.data[0][i] - mean[i]) / std[i]

    return im


def test(imgL, imgR, image_sizes=None, calibs_fu=None, calibs_baseline=None, calibs_Proj=None, calibs_Proj_R=None):
    model.eval()
    with torch.no_grad():
        outputs = model(imgL, imgR, calibs_fu, calibs_baseline,
                        calibs_Proj, calibs_Proj_R=calibs_Proj_R)
    pred_disp = outputs['depth_preds']

    rets = [pred_disp]

    if cfg.RPN3D_ENABLE:
        box_pred = make_fcos3d_postprocessor(cfg)(
            outputs['bbox_cls'], outputs[
                'bbox_reg'], outputs['bbox_centerness'],
            image_sizes=image_sizes, calibs_Proj=calibs_Proj)
        rets.append(box_pred)

    return rets


def main():
    alpha = args.alpha
    eps = args.eps
    iter_num = args.iter

    for batch_idx, databatch in enumerate(TestImgLoader):

        imgL = databatch['imgL']
        imgR = databatch['imgR']
        disp_L = databatch['disp_L']
        calib = databatch['calib']
        calib_R = databatch['calib_R']
        image_indexes = databatch['image_indexes']
        targets = databatch['targets']
        ious = databatch['ious']
        labels_map = databatch['labels_map']

        if cfg.debug:
            if batch_idx * len(imgL) > args.debugnum:
                break

        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        disp_L = Variable(torch.FloatTensor(disp_L))
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        ori_imgL_data = denormalize(imgL.clone().detach_().data)
        ori_imgR_data = denormalize(imgR.clone().detach_().data)

        if targets is not None:
            for i in range(len(targets)):
                targets[i].bbox = targets[i].bbox.cuda()
                targets[i].box3d = targets[i].box3d.cuda()

        calibs_fu = torch.as_tensor([c.f_u for c in calib])
        calibs_baseline = torch.abs(
            torch.as_tensor([(c.P[0, 3] - c_R.P[0, 3]) / c.P[0, 0] for c, c_R in zip(calib, calib_R)]))
        calibs_Proj = torch.as_tensor([c.P for c in calib])
        calibs_Proj_R = torch.as_tensor([c.P for c in calib_R])

        # ---------
        mask = (disp_true > cfg.min_depth) & (disp_true <= cfg.max_depth)
        mask.detach_()
        # ---------

        # find image shape without pre-processing
        img_name = "%06d" % image_indexes[0]
        orig_img_path = '{0}/image_2/{1}.png'.format(args.data_path, img_name)
        w, h = Image.open(orig_img_path).convert('RGB').size

        # save clean image(without attacked)
        save_dir = 'dsgn_pgd_iters_0'
        save_dir_l = save_dir + '/image_2'
        save_dir_r = save_dir + '/image_3'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir_l):
            os.makedirs(save_dir_l)
        if not os.path.isdir(save_dir_r):
            os.makedirs(save_dir_r)

        imgL_save = imgL.clone().detach_().cpu()
        save_img(imgL_save[0], '{0}/{1}.png'.format(save_dir_l, img_name), w, h)

        imgR_save = imgR.clone().detach_().cpu()
        save_img(imgR_save[0], '{0}/{1}.png'.format(save_dir_r, img_name), w, h)

        # store denormalized clean images for pgd attack
        clean_imgL_data = denormalize(imgL.clone().detach_().data)
        clean_imgR_data = denormalize(imgR.clone().detach_().data)

        for iteration in range(iter_num):
            loss = 0.
            losses = dict()

            # require gradient to perform pgd attack
            imgL.requires_grad = True
            imgR.requires_grad = True

            outputs = model(imgL, imgR, calibs_fu, calibs_baseline, calibs_Proj, calibs_Proj_R=calibs_Proj_R)

            if getattr(cfg, 'PlaneSweepVolume', True) and cfg.loss_disp:
                depth_preds = [torch.squeeze(o, 1) for o in outputs['depth_preds']]

                disp_loss = 0.
                weight = [0.5, 0.7, 1.0]
                for i, o in enumerate(depth_preds):
                    disp_loss += weight[3 - len(depth_preds) + i] * F.smooth_l1_loss(o[mask[0]], disp_true[mask],
                                                                                     size_average=True)
                losses.update(disp_loss=disp_loss)
                loss += disp_loss

            if cfg.RPN3D_ENABLE:
                bbox_cls, bbox_reg, bbox_centerness = outputs['bbox_cls'], outputs['bbox_reg'], outputs[
                    'bbox_centerness']
                rpn3d_loss, rpn3d_cls_loss, rpn3d_reg_loss, rpn3d_centerness_loss = RPN3DLoss(cfg)(
                    bbox_cls, bbox_reg, bbox_centerness, targets, calib, calib_R,
                    ious=ious, labels_map=labels_map)
                losses.update(rpn3d_cls_loss=rpn3d_cls_loss,
                              rpn3d_reg_loss=rpn3d_reg_loss,
                              rpn3d_centerness_loss=rpn3d_centerness_loss)
                loss += rpn3d_loss
            losses.update(loss=loss)

            model.zero_grad()
            imgL.retain_grad()
            imgR.retain_grad()
            loss.backward()

            # denormalize image (->[0,1])
            imgL = denormalize(imgL)
            imgR = denormalize(imgR)

            # pgd attack
            adv_imgL_data = imgL + alpha * imgL.grad.sign()
            adv_imgR_data = imgR + alpha * imgR.grad.sign()

            eta_left = torch.clamp(adv_imgL_data - clean_imgL_data, min=-eps, max=eps)
            eta_right = torch.clamp(adv_imgR_data - clean_imgR_data, min=-eps, max=eps)

            imgL = torch.clamp(ori_imgL_data + eta_left, min=0, max=1)
            imgR = torch.clamp(ori_imgR_data + eta_right, min=0, max=1)

            # normalize image(->[-2.1.2.6])
            imgL = normalize(imgL).detach()
            imgR = normalize(imgR).detach()

            # save attacked images
            save_dir = 'dsgn_pgd_iters_{0}'.format(iteration + 1)
            save_dir_l = save_dir + '/image_2'
            save_dir_r = save_dir + '/image_3'

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            if not os.path.isdir(save_dir_l):
                os.makedirs(save_dir_l)

            if not os.path.isdir(save_dir_r):
                os.makedirs(save_dir_r)

            imgL_save = imgL.clone().detach_().cpu()
            save_img(imgL_save[0], '{0}/{1}.png'.format(save_dir_l, img_name), w, h)

            imgR_save = imgR.clone().detach_().cpu()
            save_img(imgR_save[0], '{0}/{1}.png'.format(save_dir_r, img_name), w, h)


if __name__ == '__main__':
    main()
