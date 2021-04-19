from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import random
import argparse
import shutil
import time
import math as m

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    parser.add_argument('--debugnum', default=None, type=int,
                        help='debug mode')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.1)
    parser.add_argument('--epochs', dest='epochs', type=int, default=40)
    parser.add_argument('--patch_dir', dest='patch_dir', type=str, help='path to folder that save all trained patches')
    parser.add_argument('--atk_mode', dest='atk_mode', type=str, default='random',
                        help='four patch attack modes(random, sp_left, sp_straight, sp_right)')
    parser.add_argument('--save_feat_map', action='store_true', help='will save feature maps')
    parser.add_argument('--save_feat_path', type=str, default='', help='path to save feature maps')

    args = parser.parse_args()
    return args


def init_patch(patch_ratio, patch_dir):

    patch_dim = int(600 * patch_ratio)  # 600 is the shortest side length after preprocessing

    if patch_dim % 2 == 0:
        patch_dim += 1

    radius = int(patch_dim / 2)

    if os.path.isdir(patch_dir):
        patch = np.load('{0}/patch.npy'.format(patch_dir))

        # resize patch trained from Stereo R-CNN
        patch = np.transpose(patch[0], (1, 2, 0))
        patch = cv2.resize(patch, (patch_dim, patch_dim), interpolation=cv2.INTER_LINEAR)
        patch = np.array([np.transpose(patch, (2, 0, 1))])
    else:
        raise Exception('Patch directory NOT found.')

    return patch_dim, radius, patch


def generate_round_mask(radius):

    # generate center point of the patch
    center_row = random.randint(int(600*0.4), int(600-radius-1))

    if args.atk_mode == 'random':
        center_col = random.randint(int(1987 * 0.2), int(1987 * 0.8))  # random attack
    elif args.atk_mode == 'sp_left':
        center_col = random.randint(int(1987 * 0.2), int(1987 * 0.4))  # specific attack - left
    elif args.atk_mode == 'sp_straight':
        center_col = random.randint(int(1987 * 0.4), int(1987 * 0.6))  # specific attack - straight
    elif args.atk_mode == 'sp_right':
        center_col = random.randint(int(1987 * 0.6), int(1987 * 0.8))  # specific attack - right
    else:
        raise Exception('Patch attack mode NOT found.')

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


# initialize hook to save feature maps
module_name = []
features_in_hook = []
features_out_hook = []


def hook(module, fea_in, fea_out):
    print("hook working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)

    return None


if __name__ == '__main__':

    args = parse_args()
    is_plot = True
    patch_ratio = args.ratio
    patch_epoch = args.epochs

    result_dir = 'result_stereo_rcnn_ratio_{0}/epoch{1}'.format(patch_ratio, patch_epoch)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    patch_dir = '{0}/stereo_rcnn_patch_ratio_{1}/epoch{2}'.format(args.patch_dir, patch_ratio, patch_epoch)

    patch_dim, radius, patch = init_patch(patch_ratio, patch_dir)
    patch = torch.from_numpy(patch).cuda()

    np.random.seed(cfg.RNG_SEED)
    cfg.TRAIN.USE_FLIPPED = False

    # training parameter in combined_roidb is set to True by default to filter
    # image pairs without bounding box
    imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_val')
    image_size = len(roidb)

    # initialize the network here.
    stereoRCNN = resnet(imdb.classes, 101, pretrained=False)
    stereoRCNN.create_architecture()

    model_pth = './models_stereo/stereo_rcnn_12_6477.pth'
    checkpoint = torch.load(model_pth)
    stereoRCNN.load_state_dict(checkpoint['model'])

    if args.save_feat_map:
        net_children = stereoRCNN.children()
        for child in net_children:
            child.register_forward_hook(hook=hook)

    with torch.no_grad():

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

        stereoRCNN.cuda()

        eval_thresh = 0.05
        vis_thresh = 0.7

        # training parameter is set to False for not preparing ground truth
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                                 imdb.num_classes, training=False,
                                 normalize=False)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=True)

        data_iter = iter(dataloader)

        stereoRCNN.eval()

        for i in tqdm(range(image_size)):

            if args.debug and i > args.debugnum:
                break

            data = next(data_iter)

            if list(data[0].shape) != [1, 3, 600, 1987] or \
               list(data[1].shape) != [1, 3, 600, 1987]:
                continue

            im_left_data.data.resize_(data[0].size()).copy_(data[0])
            im_right_data.data.resize_(data[1].size()).copy_(data[1])
            im_info.data.resize_(data[2].size()).copy_(data[2])
            gt_boxes_left.data.resize_(data[3].size()).copy_(data[3])
            gt_boxes_right.data.resize_(data[4].size()).copy_(data[4])
            gt_boxes_merge.data.resize_(data[5].size()).copy_(data[5])
            gt_dim_orien.data.resize_(data[6].size()).copy_(data[6])
            gt_kpts.data.resize_(data[7].size()).copy_(data[7])
            num_boxes.data.resize_(data[8].size()).copy_(data[8])

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

            patch_l = padding_l(patch)
            patch_r = padding_r(patch)

            # add patch to images
            im_left_data.data = torch.mul((1-mask_l), im_left_data.data) + \
                torch.mul(mask_l, patch_l)
            im_right_data.data = torch.mul((1-mask_r), im_right_data.data) + \
                torch.mul(mask_r, patch_r)

            rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, \
                kpts_prob, left_prob, right_prob, rpn_loss_cls, \
                rpn_loss_box_left_right, RCNN_loss_cls, RCNN_loss_bbox, \
                RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = \
                stereoRCNN(im_left_data, im_right_data, im_info,
                           gt_boxes_left, gt_boxes_right, gt_boxes_merge,
                           gt_dim_orien, gt_kpts, num_boxes)

            if args.save_feat_map:
                # test feature hook
                print("*" * 5 + "hook record features" + "*" * 5)
                print(module_name)
                print("*" * 5 + "hook record features" + "*" * 5)

            scores = cls_prob.data
            boxes_left = rois_left.data[:, :, 1:5]
            boxes_right = rois_right.data[:, :, 1:5]

            bbox_pred = bbox_pred.data
            box_delta_left = bbox_pred.new(bbox_pred.size()[1],
                                           4*len(imdb._classes)).zero_()
            box_delta_right = bbox_pred.new(bbox_pred.size()[1],
                                            4*len(imdb._classes)).zero_()

            for keep_inx in range(box_delta_left.size()[0]):
                box_delta_left[keep_inx, 0::4] = bbox_pred[0, keep_inx, 0::6]
                box_delta_left[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
                box_delta_left[keep_inx, 2::4] = bbox_pred[0, keep_inx, 2::6]
                box_delta_left[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

                box_delta_right[keep_inx, 0::4] = bbox_pred[0, keep_inx, 4::6]
                box_delta_right[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
                box_delta_right[keep_inx, 2::4] = bbox_pred[0, keep_inx, 5::6]
                box_delta_right[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

            box_delta_left = box_delta_left.view(-1, 4)
            box_delta_right = box_delta_right.view(-1, 4)

            dim_orien = bbox_pred_dim.data
            dim_orien = dim_orien.view(-1, 5)

            kpts_prob = kpts_prob.data
            kpts_prob = kpts_prob.view(-1, 4*cfg.KPTS_GRID)
            max_prob, kpts_delta = torch.max(kpts_prob, 1)

            left_prob = left_prob.data
            left_prob = left_prob.view(-1, cfg.KPTS_GRID)
            _, left_delta = torch.max(left_prob, 1)

            right_prob = right_prob.data
            right_prob = right_prob.view(-1, cfg.KPTS_GRID)
            _, right_delta = torch.max(right_prob, 1)

            box_delta_left = box_delta_left * \
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() + \
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_delta_right = box_delta_right * \
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() + \
                torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            dim_orien = dim_orien * \
                torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_STDS).cuda() + \
                torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_MEANS).cuda()

            box_delta_left = box_delta_left.view(1, -1, 4*len(imdb._classes))
            box_delta_right = box_delta_right.view(1, -1, 4*len(imdb._classes))
            dim_orien = dim_orien.view(1, -1, 5*len(imdb._classes))
            kpts_delta = kpts_delta.view(1, -1, 1)
            left_delta = left_delta.view(1, -1, 1)
            right_delta = right_delta.view(1, -1, 1)
            max_prob = max_prob.view(1, -1, 1)

            pred_boxes_left = \
                bbox_transform_inv(boxes_left, box_delta_left, 1)
            pred_boxes_right = \
                bbox_transform_inv(boxes_right, box_delta_right, 1)
            pred_kpts, kpts_type = \
                kpts_transform_inv(boxes_left, kpts_delta, cfg.KPTS_GRID)
            pred_left = \
                border_transform_inv(boxes_left, left_delta, cfg.KPTS_GRID)
            pred_right = \
                border_transform_inv(boxes_left, right_delta, cfg.KPTS_GRID)

            pred_boxes_left = clip_boxes(pred_boxes_left, im_info.data, 1)
            pred_boxes_right = clip_boxes(pred_boxes_right, im_info.data, 1)

            pred_boxes_left /= im_info[0, 2].data
            pred_boxes_right /= im_info[0, 2].data
            pred_kpts /= im_info[0, 2].data
            pred_left /= im_info[0, 2].data
            pred_right /= im_info[0, 2].data

            scores = scores.squeeze()
            pred_boxes_left = pred_boxes_left.squeeze()
            pred_boxes_right = pred_boxes_right.squeeze()

            pred_kpts = torch.cat((pred_kpts, kpts_type, max_prob,
                                   pred_left, pred_right), 2)
            pred_kpts = pred_kpts.squeeze()
            dim_orien = dim_orien.squeeze()

            img_path = roidb[i]['img_left']
            split_path = img_path.split('/')
            image_number = split_path[len(split_path)-1].split('.')[0]
            calib_path = img_path.replace("image_2", "calib")
            calib_path = calib_path.replace("png", "txt")
            calib = kitti_utils.read_obj_calibration(calib_path)
            label_path = calib_path.replace("calib", "label_2")
            lidar_path = calib_path.replace("calib", "velodyne")
            lidar_path = lidar_path.replace("txt", "bin")

            # save feature maps
            if args.save_feat_map:
                feat_out_dir = '{0}/{1}'.format(args.save_feat_path, image_number)

                if not os.path.isdir(feat_out_dir):
                    os.makedirs(feat_out_dir)

                for j in range(24):
                    np.save('{}/feat_out{}'.format(feat_out_dir, str(j)),
                            features_out_hook[j][0].cpu().numpy())

            im2show_left = np.copy(cv2.imread(roidb[i]['img_left']))
            im2show_right = np.copy(cv2.imread(roidb[i]['img_right']))

            # scale adversarial example size
            im_shape = im2show_left.shape
            # print("left shape: ", im_shape)
            im_size_min = np.min(im_shape[0:2])
            im_scale = float(375) / float(im_size_min)
            im2show_left = cv2.resize(im2show_left, None, None, fx=im_scale,
                                      fy=im_scale,
                                      interpolation=cv2.INTER_LINEAR)

            im2show_right = cv2.resize(im2show_right, None, None, fx=im_scale,
                                       fy=im_scale,
                                       interpolation=cv2.INTER_LINEAR)

            pointcloud = kitti_utils.get_point_cloud(lidar_path, calib)
            im_box = vis_utils.vis_lidar_in_bev(pointcloud,
                                                width=im2show_left.shape[0]*2)

            for j in range(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > eval_thresh).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)

                    cls_boxes_left = \
                        pred_boxes_left[inds][:, j * 4:(j + 1) * 4]
                    cls_boxes_right = \
                        pred_boxes_right[inds][:, j * 4:(j + 1) * 4]
                    cls_dim_orien = \
                        dim_orien[inds][:, j * 5:(j + 1) * 5]

                    cls_kpts = pred_kpts[inds]

                    cls_dets_left = torch.cat((cls_boxes_left,
                                               cls_scores.unsqueeze(1)), 1)
                    cls_dets_right = torch.cat((cls_boxes_right,
                                                cls_scores.unsqueeze(1)), 1)

                    cls_dets_left = cls_dets_left[order]
                    cls_dets_right = cls_dets_right[order]
                    cls_dim_orien = cls_dim_orien[order]
                    cls_kpts = cls_kpts[order]

                    keep = nms(cls_boxes_left[order, :], cls_scores[order],
                               cfg.TEST.NMS)
                    keep = keep.view(-1).long()
                    cls_dets_left = cls_dets_left[keep]
                    cls_dets_right = cls_dets_right[keep]
                    cls_dim_orien = cls_dim_orien[keep]
                    cls_kpts = cls_kpts[keep]

                    # optional operation, can check the regressed borderline
                    # keypoint using 2D box inference
                    infered_kpts = \
                        kitti_utils.infer_boundary(im2show_left.shape,
                                                   cls_dets_left.cpu().numpy())
                    infered_kpts = \
                        torch.from_numpy(infered_kpts).type_as(cls_dets_left)

                    for detect_idx in range(cls_dets_left.size()[0]):
                        if (cls_kpts[detect_idx, 4] -
                            cls_kpts[detect_idx, 3]) < 0.5 * \
                            (infered_kpts[detect_idx, 1] -
                             infered_kpts[detect_idx, 0]):
                            cls_kpts[detect_idx, 3:5] = \
                                infered_kpts[detect_idx]

                    im2show_left = \
                        vis_detections(im2show_left, imdb._classes[j],
                                       cls_dets_left.cpu().numpy(),
                                       vis_thresh, cls_kpts.cpu().numpy())
                    im2show_right = \
                        vis_detections(im2show_right, imdb._classes[j],
                                       cls_dets_right.cpu().numpy(),
                                       vis_thresh)

                    # read intrinsic
                    f = calib.p2[0, 0]
                    cx, cy = calib.p2[0, 2], calib.p2[1, 2]
                    bl = (calib.p2[0, 3] - calib.p3[0, 3]) / f

                    boxes_all = cls_dets_left.new(0, 5)
                    kpts_all = cls_dets_left.new(0, 5)
                    poses_all = cls_dets_left.new(0, 8)

                    for detect_idx in range(cls_dets_left.size()[0]):
                        if cls_dets_left[detect_idx, -1] > eval_thresh:
                            # based on origin image
                            box_left = \
                                cls_dets_left[detect_idx, 0:4].cpu().numpy()
                            box_right = \
                                cls_dets_right[detect_idx, 0:4].cpu().numpy()
                            kpts_u = cls_kpts[detect_idx, 0]
                            dim = cls_dim_orien[detect_idx, 0:3].cpu().numpy()
                            sin_alpha = cls_dim_orien[detect_idx, 3]
                            cos_alpha = cls_dim_orien[detect_idx, 4]
                            alpha = m.atan2(sin_alpha, cos_alpha)
                            status, state = \
                                box_estimator.solve_x_y_z_theta_from_kpt(
                                    im2show_left.shape, calib, alpha, dim,
                                    box_left, box_right,
                                    cls_kpts[detect_idx].cpu().numpy())

                            if status > 0:  # not faild
                                poses = im_left_data.data.new(8).zero_()
                                xyz = np.array([state[0], state[1], state[2]])
                                theta = state[3]
                                poses[0], poses[1], poses[2], poses[3], \
                                    poses[4], poses[5], poses[6], poses[7] = \
                                    xyz[0], xyz[1], xyz[2], float(dim[0]), \
                                    float(dim[1]), float(dim[2]), theta, alpha

                                boxes_all = torch.cat((
                                    boxes_all,
                                    cls_dets_left[detect_idx, 0:5].unsqueeze(0)
                                    ), 0)
                                kpts_all = torch.cat(
                                    (kpts_all,
                                     cls_kpts[detect_idx].unsqueeze(0)), 0)
                                poses_all = torch.cat(
                                    (poses_all, poses.unsqueeze(0)), 0)

                    if boxes_all.dim() > 0:
                        # solve disparity by dense alignment (enlarged image)
                        succ, dis_final = dense_align.align_parallel(
                            calib, im_info.data[0, 2], im_left_data.data,
                            im_right_data.data, boxes_all[:, 0:4], kpts_all,
                            poses_all[:, 0:7])

                        # do 3D rectify using the aligned disparity
                        for solved_idx in range(succ.size(0)):
                            if succ[solved_idx] > 0:  # succ
                                box_left = \
                                    boxes_all[solved_idx, 0:4].cpu().numpy()
                                score = boxes_all[solved_idx, 4].cpu().numpy()
                                dim = poses_all[solved_idx, 3:6].cpu().numpy()
                                state_rect, z = \
                                    box_estimator.solve_x_y_theta_from_kpt(
                                        im2show_left.shape, calib,
                                        poses_all[solved_idx, 7].cpu().numpy(),
                                        dim, box_left,
                                        dis_final[solved_idx].cpu().numpy(),
                                        kpts_all[solved_idx].cpu().numpy())
                                xyz = np.array(
                                    [state_rect[0], state_rect[1], z])
                                theta = state_rect[2]

                                if score > vis_thresh:
                                    im_box = vis_utils.vis_box_in_bev(
                                        im_box, xyz, dim, theta,
                                        width=im2show_left.shape[0]*2)
                                    im2show_left = \
                                        vis_utils.vis_single_box_in_img(
                                            im2show_left, calib, xyz, dim,
                                            theta)

                                # write result into txt file
                                kitti_utils.write_detection_results(
                                    result_dir, image_number, calib,
                                    box_left, xyz, dim, theta, score)


            if is_plot:
                im2show = np.concatenate((im2show_left, im2show_right), axis=0)
                im2show = np.concatenate((im2show, im_box), axis=1)

                plot_dir = os.path.join(result_dir, 'refer')
                if not os.path.isdir(plot_dir):
                    os.makedirs(plot_dir)

                cv2.imwrite(os.path.join(plot_dir, image_number+'.png'), im2show)

