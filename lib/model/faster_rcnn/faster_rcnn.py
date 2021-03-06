import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, pooling_size, teaching=False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.teaching = teaching
        self.pooling_size = pooling_size
        self.RCNN_rpn = _RPN(self.dout_base_model, teaching=self.teaching)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(self.pooling_size, self.pooling_size, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(self.pooling_size, self.pooling_size, 1.0 / 16.0)

        self.grid_size = self.pooling_size * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else self.pooling_size
        self.RCNN_roi_crop = _RoICrop()



    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        #print(im_data.shape)
        # feed base feature map tp RPN to obtain rois
        #todo modificato, adesso restituisce anche Ps e Rs (rpn_cls_score e rpn_bbox_pred)
        rois, rpn_loss_cls, rpn_loss_bbox, rpn_cls_score, rpn_bbox_pred, fg_bg_label, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining


        if self.training or self.teaching:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            # rpn_loss_bbox = 0
        '''
        roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        '''




        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic or self.teaching:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        #todo anche qui le loss L_hard per la cls   e L_s per la reg sono già calcolate
        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        if not (self.training or self.teaching):
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1) #reshape commentato per il calcolo della loss esterno

        # rpn_bbox_inside_weights(1,36,37,56)=outside
        RPN_mask = rpn_bbox_inside_weights, rpn_bbox_outside_weights

        # rpn_bbox_targets (1,36,37,56): 4 coordinate * 9 anchor per ciascun elemento della feature map
        # rpn_bbox_pred (1,36,37,56)
        # rpn_loss_box (int):
        RPN_reg = rpn_bbox_targets, rpn_bbox_pred, rpn_loss_bbox

        # rpn_cls_score (256,2): logits in uscita dalla strato convoluzionale senza calcolare softmax in RPN. Le probabilità le calcoliamo con softmax in loss.py
        # fg_back_ground_label (256 di 0,1): ground thruth-> back ground foreground
        # rpn_loss_cls (int)
        RPN_cls = rpn_cls_score, fg_bg_label, rpn_loss_cls

        # rois_inside_weights(256,4)=outside
        RCN_mask = rois_inside_ws, rois_outside_ws

        # roi (1,256,5): region of interest generate dal proposal layer (256)
        # rois_label (256):
        # bbox_pred (256,4)
        # rois_target (256,4)
        # RCNN_loss_bbox (int)
        RCN_reg = rois, rois_label, rois_target, bbox_pred, RCNN_loss_bbox

        # cls_score (256,21)
        # cls_prob (1,256,21)
        # RCNN_loss_cls(int)
        RCN_cls = cls_score, cls_prob, RCNN_loss_cls

        ###Losses:

        # Loss classification RPN: rpn_loss_cls
        # Loss regression_RCN : rpn_loss_bbox

        # Loss classification RCN: RCNN_loss_cls
        # Loss_regression RCN: RCNN_loss_bbox

        return RPN_mask, RPN_reg, RPN_cls, RCN_mask, RCN_reg, RCN_cls

        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,rpn_cls_score, rpn_bbox_pred, fg_bg_label, rpn_bbox_targets, \
        #       rpn_bbox_inside_weights, rpn_bbox_outside_weights, rois_target, rois_inside_ws, rois_outside_ws, cls_score

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
