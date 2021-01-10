from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import math as m

from tqdm import tqdm
import numpy as np
import argparse
import shutil
import time
import cv2
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
from torchvision.utils import save_image

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Perturbation attack against the Stereo R-CNN network')

    parser.add_argument('--iter', type=int, default=4, help='iteration number of pgd attack')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--eps', default=0.3, type=float)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    parser.add_argument('--debugnum', default=None, type=int,
                        help='debug mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    alpha = args.alpha
    eps = 255 * args.eps
    iter_num = args.iter
    print("Start iteration: ", iter_num)

    np.random.seed(cfg.RNG_SEED)
    cfg.TRAIN.USE_FLIPPED = False

    # training parameter in combined_roidb is set to True by default to filter
    # image pairs without bounding box
    imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_val')
    attack_size = len(roidb)

    # training parameter is set to True to prepare ground truth
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

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


    # initilize the network here.
    stereoRCNN = resnet(imdb.classes, 101, pretrained=False)
    stereoRCNN.create_architecture()

    model_pth = './models_stereo/stereo_rcnn_12_6477.pth'
    checkpoint = torch.load(model_pth)
    stereoRCNN.load_state_dict(checkpoint['model'])
    uncert.data = checkpoint['uncert']

    stereoRCNN.cuda()

    stereoRCNN.eval()

    data_iter = iter(dataloader)

    for i in tqdm(range(attack_size)):

        if args.debug and i >= args.debugnum:
            break

        data = next(data_iter)
        im_left_data.data.resize_(data[0].size()).copy_(data[0])
        im_right_data.data.resize_(data[1].size()).copy_(data[1])
        im_info.data.resize_(data[2].size()).copy_(data[2])
        gt_boxes_left.data.resize_(data[3].size()).copy_(data[3])
        gt_boxes_right.data.resize_(data[4].size()).copy_(data[4])
        gt_boxes_merge.data.resize_(data[5].size()).copy_(data[5])
        gt_dim_orien.data.resize_(data[6].size()).copy_(data[6])
        gt_kpts.data.resize_(data[7].size()).copy_(data[7])
        num_boxes.data.resize_(data[8].size()).copy_(data[8])
        img_name = roidb[i]['img_left'].split('/')[-1].strip()

        clean_im_left_data = im_left_data.data
        clean_im_right_data = im_right_data.data

        # save clean images(unattacked or iteration=0)
        save_dir = 'stereo_rcnn_pgd_iters_0'
        save_dir_l = save_dir + '/image_2'
        save_dir_r = save_dir + '/image_3'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if not os.path.isdir(save_dir_l):
            os.makedirs(save_dir_l)

        if not os.path.isdir(save_dir_r):
            os.makedirs(save_dir_r)

        img_left = im_left_data.clone().detach_().cpu()
        img_left = img_left.squeeze(0).permute(1, 2, 0) \
            .contiguous().data.numpy()
        img_left += cfg.PIXEL_MEANS
        cv2.imwrite('{0}{1}'.format(save_dir_l, img_name), img_left)

        img_right = im_right_data.clone().detach_().cpu()
        img_right = img_right.squeeze(0).permute(1, 2, 0) \
            .contiguous().data.numpy()
        img_right += cfg.PIXEL_MEANS
        cv2.imwrite('{0}{1}'.format(save_dir_r, img_name), img_right)

        for iteration in range(iter_num):

            im_left_data.requires_grad = True
            im_right_data.requires_grad = True

            rois_left, rois_right, cls_prob, bbox_pred, dim_orien_pred, \
                kpts_prob, left_border_prob, right_border_prob, rpn_loss_cls, \
                rpn_loss_box_left_right, RCNN_loss_cls, RCNN_loss_bbox, \
                RCNN_loss_dim_orien, RCNN_loss_kpts, \
                rois_label = stereoRCNN(im_left_data, im_right_data, im_info,
                                        gt_boxes_left, gt_boxes_right,
                                        gt_boxes_merge, gt_dim_orien, gt_kpts,
                                        num_boxes)

            loss = rpn_loss_cls.mean() * torch.exp(-uncert[0]) + uncert[0] + \
                rpn_loss_box_left_right.mean() * torch.exp(-uncert[1]) + \
                uncert[1] + RCNN_loss_cls.mean() * torch.exp(-uncert[2]) + \
                uncert[2] + RCNN_loss_bbox.mean() * torch.exp(-uncert[3]) + \
                uncert[3] + RCNN_loss_dim_orien.mean() * \
                torch.exp(-uncert[4]) + uncert[4] + RCNN_loss_kpts.mean() * \
                torch.exp(-uncert[5]) + uncert[5]

            stereoRCNN.zero_grad()
            loss.backward()

            # pgd attack
            adv_im_left_data = im_left_data + alpha * im_left_data.grad.sign()
            adv_im_right_data = im_right_data + alpha * \
                im_right_data.grad.sign()

            eta_left = torch.clamp(adv_im_left_data - clean_im_left_data,
                                   min=-eps, max=eps)
            eta_right = torch.clamp(adv_im_right_data - clean_im_right_data,
                                    min=-eps, max=eps)

            data_holder_l = clean_im_left_data + eta_left
            data_holder_r = clean_im_right_data + eta_right

            data_holder_l_0 = torch.clamp(data_holder_l[0][0],
                                          min=(0 - 102.9801),
                                          max=(255 - 102.9801))
            data_holder_l_1 = torch.clamp(data_holder_l[0][1],
                                          min=(0 - 115.9465),
                                          max=(255 - 115.9465))
            data_holder_l_2 = torch.clamp(data_holder_l[0][2],
                                          min=(0 - 122.7717),
                                          max=(255 - 122.7717))

            data_holder_r_0 = torch.clamp(data_holder_r[0][0],
                                          min=(0 - 102.9801),
                                          max=(255 - 102.9801))
            data_holder_r_1 = torch.clamp(data_holder_r[0][1],
                                          min=(0 - 115.9465),
                                          max=(255 - 115.9465))
            data_holder_r_2 = torch.clamp(data_holder_r[0][2],
                                          min=(0 - 122.7717),
                                          max=(255 - 122.7717))

            im_left_data = torch.stack([data_holder_l_0,
                                        data_holder_l_1,
                                        data_holder_l_2],
                                       0).unsqueeze(0).detach()

            im_right_data = torch.stack([data_holder_r_0,
                                         data_holder_r_1,
                                         data_holder_r_2],
                                        0).unsqueeze(0).detach()

            # save attacked images
            save_dir = '/stereo_rcnn_pgd_iters_{0}'.format(iteration+1)
            save_dir_l = save_dir + '/image_2'
            save_dir_r = save_dir + '/image_3'

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            if not os.path.isdir(save_dir_l):
                os.makedirs(save_dir_l)

            if not os.path.isdir(save_dir_r):
                os.makedirs(save_dir_r)

            img_left = im_left_data.clone().detach_().cpu()
            img_left = img_left.squeeze(0).permute(1, 2, 0) \
                .contiguous().data.numpy()
            img_left += cfg.PIXEL_MEANS
            cv2.imwrite('{0}{1}'.format(save_dir_l, img_name), img_left)

            img_right = im_right_data.clone().detach_().cpu()
            img_right = img_right.squeeze(0).permute(1, 2, 0) \
                .contiguous().data.numpy()
            img_right += cfg.PIXEL_MEANS
            cv2.imwrite('{0}{1}'.format(save_dir_r, img_name), img_right)