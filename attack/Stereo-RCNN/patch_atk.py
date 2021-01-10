from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import argparse
import shutil
import time

import cv2
import math as m
import random
import datetime
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, \
        border_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.stereo_rcnn.resnet import resnet
from model.utils import kitti_utils
from model.utils import vis_3d_utils as vis_utils
from model.utils import box_estimator as box_estimator
from model.dense_align import dense_align


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Attack the Stereo R-CNN network')

    parser.add_argument('--iter', type=int, default=2,
                        help='iteration number of pgd attack')
    parser.add_argument('--eps', dest='eps', type=float, default=0.1)
    parser.add_argument('--epochs', dest='epochs', type=int, default=40)
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    parser.add_argument('--debugnum', default=None, type=int,
                        help='debug mode')

    args = parser.parse_args()
    return args


def init_patch(patch_ratio, save_dir):
    save_dir = save_dir + '/epoch0'
    patch_dim = int(600 * patch_ratio)  # 600 is the shortest side length after preprocessing

    if patch_dim % 2 == 0:
        patch_dim += 1

    radius = int(patch_dim / 2)

    if os.path.isdir(save_dir):
        print("existed patch!")
        patch = np.load('{0}/patch.npy'.format(save_dir))
    else:
        os.makedirs(save_dir)

        patch = np.zeros((1, 3, patch_dim, patch_dim), dtype=np.float32)
        np.save('{0}/patch.npy'.format(save_dir), patch)

    return patch_dim, radius, patch


def generate_round_mask(radius):
    # random generate center_point of the patch
    center_row = random.randint(int(600*0.4), int(600-radius-1))
    center_col = random.randint(int(1987 * 0.2), int(1987 * 0.8))
    center_l = [center_row, center_col]
    center_r = [center_row, int(center_col-(40*1.6))]

    Y, X = np.ogrid[:600, :1987]
    dist_from_center_l = np.sqrt((Y - center_l[0])**2 +
                                 (X - center_l[1])**2)
    mask_l = (dist_from_center_l <= radius).astype('float32')
    mask_l = np.array([[mask_l, mask_l, mask_l]])

    dist_from_center_r = np.sqrt((Y - center_r[0])**2 +
                                 (X - center_r[1])**2)
    mask_r = (dist_from_center_r <= radius).astype('float32')
    mask_r = np.array([[mask_r, mask_r, mask_r]])

    return center_l, center_r, mask_l, mask_r


if __name__ == '__main__':
    args = parse_args()
    alpha = 1e3
    eps = args.eps
    iters = args.iter
    epochs = args.epochs

    # the length ratio between patch diameter and the short side of image
    patch_ratio = args.ratio

    save_dir = 'stereo_rcnn_patch_ratio_{0}'.format(patch_ratio)

    patch_dim, radius, patch = init_patch(patch_ratio, save_dir)

    patch = torch.from_numpy(patch).cuda()

    np.random.seed(cfg.RNG_SEED)
    cfg.TRAIN.USE_FLIPPED = False

    # training parameter in combined_roidb is set to True by default to filter
    # image pairs without bounding box
    imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_val')
    train_size = len(roidb)

    # training parameter is set to True to prepare ground truth
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=8)

    # initialize the tensor holder here.
    im_left_data = Variable(torch.FloatTensor(1).cuda())
    im_right_data = Variable(torch.FloatTensor(1).cuda())
    im_info = Variable(torch.FloatTensor(1).cuda())
    num_boxes = Variable(torch.LongTensor(1).cuda())
    gt_boxes_left = Variable(torch.FloatTensor(1).cuda())
    gt_boxes_right = Variable(torch.FloatTensor(1).cuda())
    gt_boxes_merge = Variable(torch.FloatTensor(1).cuda())
    gt_dim_orien = Variable(torch.FloatTensor(1).cuda())
    gt_kpts = Variable(torch.FloatTensor(1).cuda())
    uncert = Variable(torch.rand(6).cuda())

    # initialize the network here.
    stereoRCNN = resnet(imdb.classes, 101, pretrained=False)
    stereoRCNN.create_architecture()

    model_pth = './models_stereo/stereo_rcnn_12_6477.pth'
    checkpoint = torch.load(model_pth)
    stereoRCNN.load_state_dict(checkpoint['model'])
    uncert.data = checkpoint['uncert']

    stereoRCNN.cuda()

    for epoch in range(epochs):
        print('Epoch {0}'.format(epoch))

        stereoRCNN.eval()

        data_iter = iter(dataloader)

        loss_sum = 0.
        loss_num = 0
        for i in tqdm(range(train_size)):

            if args.debug and i >= args.debugnum:
                break

            data = next(data_iter)

            if list(data[0].shape) != [1, 3, 600, 1987] or \
               list(data[1].shape) != [1, 3, 600, 1987]:
                continue

            # generate mask
            center_l, center_r, mask_l, mask_r = generate_round_mask(radius)
            mask_l = torch.from_numpy(mask_l).cuda()
            mask_r = torch.from_numpy(mask_r).cuda()
            padding_l = nn.ConstantPad2d((center_l[1] - radius,
                                          1986 - (center_l[1] + radius),
                                          center_l[0] - radius,
                                          599 - (center_l[0] + radius)), 0.0)
            padding_r = nn.ConstantPad2d((center_r[1] - radius,
                                          1986 - (center_r[1] + radius),
                                          center_r[0] - radius,
                                          599 - (center_r[0] + radius)), 0.0)

            # change ground truth
            data[8].data = torch.tensor(1)

            data[3] = torch.zeros(size=data[3].shape)
            data[4] = torch.zeros(size=data[4].shape)
            data[5] = torch.zeros(size=data[5].shape)

            data[3][0, 0, 0] = center_l[1] - radius
            data[3][0, 0, 1] = center_l[0] - radius
            data[3][0, 0, 2] = center_l[1] + radius
            data[3][0, 0, 3] = center_l[0] + radius

            data[4][0, 0, 0] = center_r[1] - radius
            data[4][0, 0, 1] = center_r[0] - radius
            data[4][0, 0, 2] = center_r[1] + radius
            data[4][0, 0, 3] = center_r[0] + radius

            data[5][0, 0, 0] = center_l[1] - radius
            data[5][0, 0, 1] = center_l[0] - radius
            data[5][0, 0, 2] = center_l[1] + radius
            data[5][0, 0, 3] = center_l[0] + radius

            im_left_data.data.resize_(data[0].size()).copy_(data[0])
            im_right_data.data.resize_(data[1].size()).copy_(data[1])
            im_info.data.resize_(data[2].size()).copy_(data[2])
            gt_boxes_left.data.resize_(data[3].size()).copy_(data[3])
            gt_boxes_right.data.resize_(data[4].size()).copy_(data[4])
            gt_boxes_merge.data.resize_(data[5].size()).copy_(data[5])
            gt_dim_orien.data.resize_(data[6].size()).copy_(data[6])
            gt_kpts.data.resize_(data[7].size()).copy_(data[7])
            num_boxes.data.resize_(data[8].size()).copy_(data[8])

            for iteration in range(iters):
                # pad the patch with 0s so that it has the same dimension as images
                patch_l = padding_l(patch)
                patch_r = padding_r(patch)

                # add patch to images
                im_left_data.data = \
                    torch.mul((1-mask_l), im_left_data.data) + \
                    torch.mul(mask_l, patch_l)
                im_right_data.data = \
                    torch.mul((1-mask_r), im_right_data.data) + \
                    torch.mul(mask_r, patch_r)

                im_left_data.requires_grad = True
                im_right_data.requires_grad = True

                rois_left, rois_right, cls_prob, bbox_pred, dim_orien_pred, \
                    kpts_prob, left_border_prob, right_border_prob, \
                    rpn_loss_cls, rpn_loss_box_left_right, RCNN_loss_cls, \
                    RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, \
                    rois_label = stereoRCNN(
                        im_left_data, im_right_data, im_info, gt_boxes_left,
                        gt_boxes_right, gt_boxes_merge, gt_dim_orien, gt_kpts,
                        num_boxes)

                loss = rpn_loss_cls.mean() * torch.exp(-uncert[0]) + \
                    uncert[0] + rpn_loss_box_left_right.mean() * \
                    torch.exp(-uncert[1]) + uncert[1] + \
                    RCNN_loss_cls.mean() * torch.exp(-uncert[2]) + \
                    uncert[2] + RCNN_loss_bbox.mean() * \
                    torch.exp(-uncert[3]) + uncert[3] + \
                    RCNN_loss_dim_orien.mean() * torch.exp(-uncert[4]) + \
                    uncert[4] + RCNN_loss_kpts.mean() * \
                    torch.exp(-uncert[5]) + uncert[5]

                stereoRCNN.zero_grad()
                loss.backward()
                loss_sum += loss.clone().cpu().data
                im_left_data_grad = im_left_data.grad.clone()
                im_right_data_grad = im_right_data.grad.clone()

                # extract the gradients at the positions of the patch
                im_left_data_grad = im_left_data_grad[
                    :, :, (center_l[0]-radius):(center_l[0]+radius+1),
                    (center_l[1]-radius):(center_l[1]+radius+1)]
                im_right_data_grad = im_right_data_grad[
                    :, :, (center_r[0]-radius):(center_r[0]+radius+1),
                    (center_r[1]-radius):(center_r[1]+radius+1)]

                patch -= torch.clamp(
                    0.5 * alpha * (im_left_data_grad + im_right_data_grad),
                    min=-eps, max=eps)

                data_holder_0 = torch.clamp(patch[0][0], min=(0 - 102.9801),
                                            max=(255 - 102.9801))
                data_holder_1 = torch.clamp(patch[0][1], min=(0 - 115.9465),
                                            max=(255 - 115.9465))
                data_holder_2 = torch.clamp(patch[0][2], min=(0 - 122.7717),
                                            max=(255 - 122.7717))

                patch = torch.stack(
                    [data_holder_0, data_holder_1, data_holder_2],
                    0).unsqueeze(0).detach()

            loss_num += 1

        print("Average loss for epoch{0}: {1}".format(str(epoch+1), (loss_sum/loss_num)))

    # save trained patch
    trained_patch_dir = '{0}/epoch{1}'.format(save_dir, str(epochs))

    if not os.path.isdir(trained_patch_dir):
        os.makedirs(trained_patch_dir)

    np.save('{0}/patch.npy'.format(trained_patch_dir), patch.clone().cpu().numpy())

