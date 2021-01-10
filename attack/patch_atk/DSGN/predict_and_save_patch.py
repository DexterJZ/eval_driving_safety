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
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.misc as ssc

sys.path.append('/public/yanglou3/DSGN/')
from dsgn.models import *
from dsgn.utils.numpy_utils import *
from dsgn.utils.numba_utils import *
from dsgn.utils.torch_utils import *
from dsgn.models.inference3d import make_fcos3d_postprocessor

sys.path.append('/public/yanglou3/DSGN/tools/')
from env_utils import *

parser = argparse.ArgumentParser(description='Patch attack predict and save')
parser.add_argument('-cfg', '--cfg', '--config',
                    default=None, help='config path')
parser.add_argument(
    '--data_path', default='./data/kitti/training', help='select model')
parser.add_argument('--loadmodel', default='./outputs/temp/DSGN_car_pretrained/finetune_53.tar', help='loading model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split_file', default='./data/kitti/val.txt',
                    help='split file')
parser.add_argument('--save_path', type=str, default='./outputs/result', metavar='S',
                    help='path to save the predict')
parser.add_argument('--save_lidar', action='store_true',
                    help='if true, save the numpy file, not the png file')
parser.add_argument('--save_depth_map', action='store_true',
                    help='if true, save the numpy file, not the png file')
parser.add_argument('--btest', '-btest', type=int, default=1)
parser.add_argument('--devices', '-d', type=str, default=0)
parser.add_argument('--tag', '-t', type=str, default='')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode')
parser.add_argument('--debugnum', default=None, type=int,
                    help='debug mode')
parser.add_argument('--train', '-train', action='store_true', default=False,
                    help='test on train set')

parser.add_argument('--ratio', dest='ratio', type=float, default=0.2)
parser.add_argument('--epochs', dest='epochs', type=int, default=80)
parser.add_argument('--patch_dir', dest='patch_dir', type=str, help='path to folder that save all trained patches')
parser.add_argument('--atk_mode', dest='atk_mode', type=str, default='random',
                    help='four patch attack modes(random, sp_left, sp_straight, sp_right)')
parser.add_argument('--save_feat_map', action='store_true', help='will save feature maps')
parser.add_argument('--save_feat_path', type=str, default='', help='path to save feature maps')
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

if args.train:
    args.split_file = './data/kitti/train.txt'
    args.tag += '_train'

if args.ratio or args.epochs:
    args.tag += '_ratio{0}_epochs{1}'.format(args.ratio, args.epochs)

assert args.btest

print('Using GPU:{}'.format(args.devices))
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from dsgn.dataloader import KITTILoader3D as ls
from dsgn.dataloader import KITTILoader_dataset3d as DA

all_left_img, all_right_img, all_left_disp, = ls.dataloader(args.data_path,
                                                            args.split_file,
                                                            depth_disp=cfg.eval_depth,
                                                            cfg=cfg,
                                                            is_train=False)


class BatchCollator(object):

    def __call__(self, batch):
        transpose_batch = list(zip(*batch))
        l = torch.cat(transpose_batch[0], dim=0)
        r = torch.cat(transpose_batch[1], dim=0)
        disp = torch.stack(transpose_batch[2], dim=0)
        calib = transpose_batch[3]
        calib_R = transpose_batch[4]
        image_sizes = transpose_batch[5]
        image_indexes = transpose_batch[6]
        outputs = [l, r, disp, calib, calib_R, image_sizes, image_indexes]
        return outputs


ImageFloader = DA.myImageFloder(
    all_left_img, all_right_img, all_left_disp, False, split=args.split_file, cfg=cfg)

TestImgLoader = torch.utils.data.DataLoader(
    ImageFloader,
    batch_size=args.btest, shuffle=False, num_workers=num_workers, drop_last=False,
    collate_fn=BatchCollator())

# initialize hook to save feature maps
module_name_extractors = []
feat_out_extractors = []
feat_extractors = []


def hook_feat(module, fea_in, fea_out):
    print("feature hook working")
    module_name_extractors.append(module.__class__)
    feat_out_extractors.append(fea_out)

    return None


model = StereoNet(cfg=cfg)

if args.save_feat_map:
    child_idx = 0
    net_children = model.children()
    for child in net_children:
        if isinstance(child, submodule.feature_extraction):
            feat_extractors.append(child)

    # focus on the feature extractor part in the network architecture
    extractor_children = feat_extractors[0].children()
    for child in extractor_children:
        child.register_forward_hook(hook=hook_feat)

model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None and args.loadmodel.endswith('tar'):
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    print('Loaded {}'.format(args.loadmodel))
else:
    print('------------------------------ Load Nothing ---------------------------------')

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))


def test(imgL, imgR, image_sizes=None, calibs_fu=None, calibs_baseline=None, calibs_Proj=None, calibs_Proj_R=None):
    model.eval()
    with torch.no_grad():
        outputs = model(imgL, imgR, calibs_fu, calibs_baseline,
                        calibs_Proj, calibs_Proj_R=calibs_Proj_R)

        if args.feat:
            # test feature hook
            print("*" * 5 + "hook record extractor features" + "*" * 5)
            print(module_name_extractors)
            print("*" * 5 + "hook record extractor features" + "*" * 5)

    pred_disp = outputs['depth_preds']

    rets = [pred_disp]

    if cfg.RPN3D_ENABLE:
        box_pred = make_fcos3d_postprocessor(cfg)(
            outputs['bbox_cls'], outputs[
                'bbox_reg'], outputs['bbox_centerness'],
            image_sizes=image_sizes, calibs_Proj=calibs_Proj)
        rets.append(box_pred)

    return rets


def error_estimating(pred_disp, ground_truth, maxdisp=192):
    all_err = 0.
    for i in range(len(pred_disp)):
        gt = ground_truth[i]
        disp = pred_disp[i]

        mask = (gt > 0) & (gt < maxdisp)

        errmap = torch.abs(disp - gt)
        err3 = ((errmap[mask] > 3.) & (errmap[mask] /
                                       gt[mask] > 0.05)).sum().float() / mask.sum()
        all_err += err3
    return float(all_err), len(pred_disp)


def depth_error_estimating(pred_disp, ground_truth, maxdisp=70,
                           depth_disp=False, calib_batch=None, calib_R_batch=None):
    all_err = 0.
    all_err_med = 0.
    for i in range(len(pred_disp)):
        gt = ground_truth[i]
        disp = pred_disp[i]
        calib = calib_batch[i]
        calib_R = calib_R_batch[i]

        if not depth_disp:
            baseline = (calib.P[0, 3] - calib_R.P[0, 3]) / \
                       calib.P[0, 0]  # ~ 0.54
            disp = (calib.f_u * baseline) / disp

        if not depth_disp:
            mask = (gt > 0) & (gt < 60)
        else:
            # ignore points outside max-depth
            mask = (gt > 0) & (gt < cfg.max_depth)

        if mask.sum() > 0:
            errmap = torch.abs(disp - gt)
            err3 = errmap[mask].mean()
            err_median = errmap[mask].median()
        else:
            err3 = 0.
            err_median = 0.
        all_err += err3
        all_err_med += err_median
    return float(all_err), len(pred_disp), float(all_err_med)


def kitti_output(box_pred_left, image_indexes, output_path, box_pred_right=None):
    for i, (prediction, image_index) in enumerate(zip(box_pred_left, image_indexes)):
        with open(os.path.join(output_path, '{:06d}.txt'.format(image_index)), 'w') as f:
            for i, (cls, bbox, score) in enumerate(zip(
                    prediction.get_field(
                        'labels').cpu(), prediction.bbox.cpu(),
                    prediction.get_field('scores').cpu())):
                if prediction.has_field('box_corner3d'):
                    assert cls != 0
                    box_corner3d = prediction.get_field(
                        'box_corner3d').cpu()[i].reshape(8, 3)
                    box_center3d = box_corner3d.mean(dim=0)
                    x, y, z = box_center3d
                    box_corner3d = box_corner3d - box_center3d.view(1, 3)
                    h, w, l, ry = get_dimensions(box_corner3d.transpose(0, 1))

                    if getattr(cfg, 'learn_viewpoint', False):
                        ry = ry - np.arctan2(z, x) + np.pi / 2
                else:
                    h, w, l = 0., 0., 0.
                    box_center3d = [0., 0., 0.]
                    ry = 0.

                cls_type = 'Pedestrian' if cls == 1 else 'Car' if cls == 2 else 'Cyclist'
                f.write(
                    '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.8f}\n'.format(
                        cls_type,
                        -np.arctan2(box_center3d[0], box_center3d[2]) + ry,
                        bbox[0], bbox[1],
                        bbox[2], bbox[3],
                        h, w, l,
                        box_center3d[0], box_center3d[1] + h / 2., box_center3d[2],
                        ry,
                        score))
        print('Wrote {}'.format(image_index))


def kitti_eval(output_path, weight_path):
    for i, cls in enumerate(cfg.valid_classes):
        results = cmd('cd ./dsgn/eval/kitti-object-eval-python && bash eval.sh {} {} {} {}'.format(
            os.path.join(output_path, 'kitti_output' + args.tag),
            0 if cls == 2 else (1 if cls == 1 else 2),
            '>' if i == 0 else '>>',
            os.path.join(output_path,
                         'result_kitti_{}{}.txt'.format(
                             weight_path.split('/')[-1].split('.')[0] if weight_path is not None else 'default',
                             args.tag)),
        ))
    os.system('cat {}'.format(os.path.join(output_path,
                                           'result_kitti_{}{}.txt'.format(weight_path.split('/')[-1].split('.')[
                                                                              0] if weight_path is not None else 'default',
                                                                          args.tag))))


def project_disp_to_depth_map(calib, disp, max_high, baseline=0.54, depth_disp=False):
    disp[disp < 0] = 0
    mask = disp > 0
    if not depth_disp:
        depth = calib.f_u * baseline / (disp + 1. - mask)
    else:
        depth = disp
    return depth


def project_disp_to_depth(calib, disp, max_high, baseline=0.54, depth_disp=False):
    disp[disp < 0] = 0
    mask = disp > 0
    if not depth_disp:
        depth = calib.f_u * baseline / (disp + 1. - mask)
    else:
        depth = disp
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def init_patch(patch_ratio, save_dir):

    patch_dim = int(384 * patch_ratio)  # patch diameter = shortest side length * patch ratio

    if patch_dim % 2 == 0:
        patch_dim += 1

    radius = int(patch_dim / 2)

    if os.path.isdir(save_dir):
        patch = np.load('{0}/patch.npy'.format(save_dir))

        # resize patch trained from Stereo R-CNN
        patch = np.transpose(patch[0], (1, 2, 0))
        patch = cv2.resize(patch, (patch_dim, patch_dim), interpolation=cv2.INTER_LINEAR)
        patch = np.array([np.transpose(patch, (2, 0, 1))])
    else:
        raise Exception('Patch directory NOT found.')

    return patch_dim, radius, patch


def generate_round_mask(radius):

    # generate center point of the patch
    center_row = random.randint(int(384*0.4), int(384-radius-1))

    if args.atk_mode == 'random':
        center_col = random.randint(int(1248 * 0.2), int(1248 * 0.8))  # random attack
    elif args.atk_mode == 'sp_left':
        center_col = random.randint(int(1248 * 0.2), int(1248 * 0.4))  # specific attack - left
    elif args.atk_mode == 'sp_straight':
        center_col = random.randint(int(1248 * 0.4), int(1248 * 0.6))  # specific attack - straight
    elif args.atk_mode == 'sp_right':
        center_col = random.randint(int(1248 * 0.6), int(1248 * 0.8))  # specific attack - right
    else:
        raise Exception('Patch attack mode NOT found.')

    center_l = [center_row, center_col]
    center_r = [center_row, int(center_col-(40*1.6))]

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


def main():

    patch_ratio = args.ratio
    patch_epoch = args.epochs
    patch_dir = '{0}/dsgn_patch_ratio_{1}/epoch{2}'.format(args.patch_dir, patch_ratio, patch_epoch)

    patch_dim, radius, patch = init_patch(patch_ratio, patch_dir)
    patch = torch.from_numpy(patch).cuda()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    output_path = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), '..', os.path.dirname(args.loadmodel))
    if not os.path.exists(output_path + '/kitti_output' + args.tag):
        os.makedirs(output_path + '/kitti_output' + args.tag)
    else:
        os.system('rm -rf {}/*'.format(output_path + '/kitti_output' + args.tag))

    all_err = 0.
    all_err_med = 0.

    stat = []

    for batch_idx, databatch in enumerate(TestImgLoader):

        imgL, imgR, gt_disp, calib_batch, calib_R_batch, image_sizes, image_indexes = databatch

        if cfg.debug:
            if batch_idx * len(imgL) > args.debugnum:
                break

        if list(imgL.shape) != [1, 3, 384, 1248] or \
                list(imgR.shape) != [1, 3, 384, 1248]:
            continue

        # initialize mask
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

        patch_l = padding_l(patch)
        patch_r = padding_r(patch)

        imgL, imgR, gt_disp = imgL.cuda(), imgR.cuda(), gt_disp.cuda()

        calibs_fu = torch.as_tensor([c.f_u for c in calib_batch])
        calibs_baseline = torch.as_tensor(
            [(c.P[0, 3] - c_R.P[0, 3]) / c.P[0, 0] for c, c_R in zip(calib_batch, calib_R_batch)])
        calibs_Proj = torch.as_tensor([c.P for c in calib_batch])
        calibs_Proj_R = torch.as_tensor([c.P for c in calib_R_batch])

        # add patch to images
        imgL.data = torch.mul((1 - mask_l), imgL.data) + \
                    torch.mul(mask_l, patch_l)
        imgR.data = torch.mul((1 - mask_r), imgR.data) + \
                    torch.mul(mask_r, patch_r)

        start_time = time.time()
        cfg.time = time.time()
        output = test(imgL, imgR, image_sizes=image_sizes, calibs_fu=calibs_fu,
                      calibs_baseline=calibs_baseline, calibs_Proj=calibs_Proj, calibs_Proj_R=calibs_Proj_R)

        # save feature maps
        if args.save_feat_map:
            idx = 0
            feat_out_dir = '{0}/{1}'.format(args.save_feat_path, image_indexes)

            if not os.path.isdir(feat_out_dir):
                os.makedirs(feat_out_dir)

            for feat_out in feat_out_extractors:
                if isinstance(feat_out, torch.Tensor):
                    np.save('{}/feat_out{2}.npy'.format(feat_out_dir, idx),
                            feat_out.cpu().numpy())
                    idx += 1
                else:
                    for feat in feat_out:
                        np.save('{}/feat_out{2}.npy'.format(feat_out_dir, idx),
                                feat.cpu().numpy())
                        idx += 1

        if cfg.RPN3D_ENABLE:
            pred_disp, box_pred = output
            kitti_output(box_pred[0], image_indexes, output_path + '/kitti_output' + args.tag)
        else:
            pred_disp, = output
        print('time = %.2f' % (time.time() - start_time))

        if getattr(cfg, 'PlaneSweepVolume', True) and getattr(cfg, 'loss_disp', True):
            if len(pred_disp) > 0:
                if cfg.eval_depth:
                    err, batch, err_med = depth_error_estimating(pred_disp, gt_disp,
                                                                 depth_disp=True, calib_batch=calib_batch,
                                                                 calib_R_batch=calib_R_batch)
                    print('Mean depth error(m): {} Median(m): {} (batch {})'.format(
                        err / batch, err_med / batch, batch))
                    all_err += err
                    all_err_med += err_med
                else:
                    err, batch = error_estimating(pred_disp, gt_disp)
                    print('>3px error: {} (batch {})'.format(err / batch, batch))
                    all_err += err

        if args.save_depth_map:
            for i in range(len(image_indexes)):
                depth_map = project_disp_to_depth_map(calib_batch[i], pred_disp[i].cpu().numpy()[:image_sizes[i][0],
                                                                      :image_sizes[i][1]], max_high=1.,
                                                      baseline=(calib_batch[i].P[0, 3] - calib_R_batch[i].P[0, 3]) /
                                                               calib_batch[i].P[0, 0],
                                                      depth_disp=True)
                if not os.path.exists('{}/depth_maps/'.format(args.save_path)):
                    os.makedirs('{}/depth_maps/'.format(args.save_path))
                np.save('{}/depth_maps/{:06d}.npy'.format(args.save_path,
                                                          image_indexes[i]), depth_map)

        if args.save_lidar:
            for i in range(len(image_indexes)):
                lidar = project_disp_to_depth(calib_batch[i],
                                              pred_disp[i].cpu().numpy()[:image_sizes[i][0], :image_sizes[i][1]],
                                              max_high=1.,
                                              baseline=(calib_batch[i].P[0, 3] - calib_R_batch[i].P[0, 3]) /
                                                       calib_batch[i].P[0, 0],
                                              depth_disp=True)
                lidar = np.concatenate(
                    [lidar, np.ones((lidar.shape[0], 1))], 1)
                lidar = lidar.astype(np.float32)
                lidar.tofile(
                    '{}/{:06d}.bin'.format(args.save_path, image_indexes[i]))

    stat = np.asarray(stat)

    if cfg.RPN3D_ENABLE:
        kitti_eval(output_path, args.loadmodel)

    print(args.loadmodel)
    all_err /= len(ImageFloader)
    all_err_med /= len(ImageFloader)

    print("Mean Error", all_err)
    os.system("echo Mean Error: {} >> {}/result_{}.txt".format(all_err,
                                                               os.path.dirname(args.loadmodel),
                                                               args.loadmodel.split('/')[-1].split('.')[0]))
    if cfg.eval_depth:
        print('Median Error', all_err_med)
        os.system("echo Median Error: {} >> {}/result_{}.txt".format(all_err_med,
                                                                     os.path.dirname(args.loadmodel),
                                                                     args.loadmodel.split('/')[-1].split('.')[0]))


if __name__ == '__main__':
    main()
