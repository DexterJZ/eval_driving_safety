from __future__ import print_function

import argparse
import os
import random
import sys
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
import numpy as np
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

parser = argparse.ArgumentParser(description="Patch attack")
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

parser.add_argument('--iter', type=int, default=2, help='iteration number of pgd attack')
parser.add_argument('--eps', type=float, default=8/255)
parser.add_argument('--epochs', dest='epochs', type=int, default=80)
parser.add_argument('--ratio', dest='ratio', type=float, default=0.2)
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

# return dir of all images
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


def init_patch(patch_ratio, save_dir):
    save_dir = save_dir + '/epoch0'
    patch_dim = int(384 * patch_ratio)  # patch diameter = shortest side length * patch ratio

    if patch_dim % 2 == 0:
        patch_dim += 1

    radius = int(patch_dim / 2)

    if os.path.isdir(save_dir):
        print("existed patch!")
        patch = np.load('{0}/patch.npy'.format(save_dir))

        # resize patch trained from Stereo R-CNN
        patch = np.transpose(patch[0], (1, 2, 0))
        patch = cv2.resize(patch, (patch_dim, patch_dim), interpolation=cv2.INTER_LINEAR)
        patch = np.array([np.transpose(patch, (2, 0, 1))])
    else:
        os.makedirs(save_dir)

        patch = np.zeros((1, 3, patch_dim, patch_dim), dtype=np.float32)
        np.save('{0}/patch.npy'.format(save_dir), patch)

    return patch_dim, radius, patch


def generate_round_mask(radius):
    # random generate center point of the patch
    center_row = random.randint(int(384 * 0.4), int(384 - radius - 1))
    center_col = random.randint(int(1248 * 0.2), int(1248 * 0.8))

    center_l = [center_row, center_col]
    center_r = [center_row, int(center_col - (40 * 1.6))]

    Y, X = np.ogrid[:384, :1248]
    dist_from_center_l = np.sqrt((Y - center_l[0]) ** 2 +
                                 (X - center_l[1]) ** 2)
    mask_l = (dist_from_center_l <= radius).astype('float32')
    mask_l = np.array([[mask_l, mask_l, mask_l]])

    dist_from_center_r = np.sqrt((Y - center_r[0]) ** 2 +
                                 (X - center_r[1]) ** 2)
    mask_r = (dist_from_center_r <= radius).astype('float32')
    mask_r = np.array([[mask_r, mask_r, mask_r]])

    return center_l, center_r, mask_l, mask_r


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
    alpha = 1e3
    eps = args.eps
    iters = args.iter
    epochs = args.epochs

    patch_ratio = args.ratio

    save_dir = 'dsgn_patch_ratio_{}'.format(patch_ratio)

    patch_dim, radius, patch = init_patch(patch_ratio, save_dir)

    patch = torch.from_numpy(patch).cuda()

    for epoch in range(epochs):
        print('Epoch {0}'.format(epoch))

        loss_sum = 0.
        loss_num = 0
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

            imgL = Variable(torch.FloatTensor(imgL))
            imgR = Variable(torch.FloatTensor(imgR))
            disp_L = Variable(torch.FloatTensor(disp_L))
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

            if cfg.debug:
                if batch_idx * len(imgL) > args.debugnum:
                    break

            if list(databatch['imgL'].shape) != [1, 3, 384, 1248] or \
                    list(databatch['imgR'].shape) != [1, 3, 384, 1248]:
                continue

            # generate mask
            center_l, center_r, mask_l, mask_r = generate_round_mask(radius)
            mask_l = torch.from_numpy(mask_l).cuda()
            mask_r = torch.from_numpy(mask_r).cuda()
            padding_l = nn.ConstantPad2d((center_l[1] - radius,
                                          1247 - (center_l[1] + radius),
                                          center_l[0] - radius,
                                          383 - (center_l[0] + radius)), 0.0)
            padding_r = nn.ConstantPad2d((center_r[1] - radius,
                                          1247 - (center_r[1] + radius),
                                          center_r[0] - radius,
                                          383 - (center_r[0] + radius)), 0.0)

            # change ground truth
            if targets is not None:
                for i in range(len(targets[0].bbox.data)):
                    targets[0].bbox.data[i] = torch.zeros(size=targets[0].bbox.data[i].shape)
                    targets[0].box3d.data[i] = torch.zeros(size=targets[0].box3d.data[i].shape)

                # change bbox(x, y, x, y)
                targets[0].bbox.data[0, 0] = 569.33
                targets[0].bbox.data[0, 1] = 180.88
                targets[0].bbox.data[0, 2] = 613.91
                targets[0].bbox.data[0, 3] = 225.02

                # change box3d(h, w, l, x, y, z, theta)
                targets[0].box3d.data[0, 0] = 1.65
                targets[0].box3d.data[0, 1] = 1.67
                targets[0].box3d.data[0, 2] = 3.64
                targets[0].box3d.data[0, 3] = -0.78
                targets[0].box3d.data[0, 4] = 1.98
                targets[0].box3d.data[0, 5] = 29.11
                targets[0].box3d.data[0, 6] = -1.60

            calibs_fu = torch.as_tensor([c.f_u for c in calib])
            calibs_baseline = torch.abs(
                torch.as_tensor([(c.P[0, 3] - c_R.P[0, 3]) / c.P[0, 0] for c, c_R in zip(calib, calib_R)]))
            calibs_Proj = torch.as_tensor([c.P for c in calib])
            calibs_Proj_R = torch.as_tensor([c.P for c in calib_R])

            # ---------
            mask = (disp_true > cfg.min_depth) & (disp_true <= cfg.max_depth)
            mask.detach_()
            # ---------

            for iteration in range(iters):
                # pad the patch with 0s so that it has the same dimension as images
                patch_l = padding_l(patch)
                patch_r = padding_r(patch)

                # add patch to images
                imgL.data = torch.mul((1 - mask_l), imgL.data) + \
                            torch.mul(mask_l, patch_l)
                imgR.data = torch.mul((1 - mask_r), imgR.data) + \
                            torch.mul(mask_r, patch_r)

                loss = 0.
                losses = dict()

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

                loss_sum += loss.clone().cpu().data

                imgL_grad = imgL.grad.clone()
                imgR_grad = imgR.grad.clone()

                # extract the gradients at the positions of the patch
                imgL_grad = imgL_grad[
                            :, :, (center_l[0] - radius):(center_l[0] + radius + 1),
                            (center_l[1] - radius):(center_l[1] + radius + 1)]
                imgR_grad = imgR_grad[
                            :, :, (center_r[0] - radius):(center_r[0] + radius + 1),
                            (center_r[1] - radius):(center_r[1] + radius + 1)]

                patch -= torch.clamp(
                    0.5 * alpha * (imgL_grad + imgR_grad),
                    min=-eps, max=eps
                ).detach()

            loss_num += 1

        avg_loss = loss_sum / loss_num
        print("Average loss for epoch{0}: {1}".format(str(epoch + 1), avg_loss))

    # save trained patch
    trained_patch_dir = '{0}/epoch{1}'.format(save_dir, str(epochs))

    if not os.path.isdir(trained_patch_dir):
        os.makedirs(trained_patch_dir)

    np.save('{0}/patch.npy'.format(trained_patch_dir), patch.clone().cpu().numpy())


if __name__ == '__main__':
    main()
