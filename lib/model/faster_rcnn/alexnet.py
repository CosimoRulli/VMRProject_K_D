# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.faster_rcnn.alexnet_caffe import AlexNet_caffe
from collections import OrderedDict


class alexnet(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, teaching=False):
        self.dout_base_model = 256
        self.model_path = 'data/pretrained_model/alexnet_torch.pth'
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.teaching = teaching
        # todo parametrizzare
        self.n_frozen_layers = 10
        print("N_Frozen_layers: " + str(self.n_frozen_layers))

        _fasterRCNN.__init__(self, classes, class_agnostic, teaching)

    def _init_modules(self):
        alexnet = AlexNet_caffe()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k == "0.weight":
                    new_state_dict["features.0.weight"] = v
                if k == "0.bias":
                    new_state_dict["features.0.bias"] = v
                if k == "4.weight":
                    new_state_dict["features.4.weight"] = v
                if k == "4.bias":
                    new_state_dict["features.4.bias"] = v
                if k == "8.weight":
                    new_state_dict["features.8.weight"] = v
                if k == "8.bias":
                    new_state_dict["features.8.bias"] = v
                if k == "10.weight":
                    new_state_dict["features.10.weight"] = v
                if k == "10.bias":
                    new_state_dict["features.10.bias"] = v
                if k == "12.weight":
                    new_state_dict["features.12.weight"] = v
                if k == "12.bias":
                    new_state_dict["features.12.bias"] = v

                if k == "16.1.weight":
                    new_state_dict["classifier.0.weight"] = v
                if k == "16.1.bias":
                    new_state_dict["classifier.0.bias"] = v
                if k == "19.1.weight":
                    new_state_dict["classifier.3.weight"] = v
                if k == "19.1.bias":
                    new_state_dict["classifier.3.bias"] = v
                if k == "22.1.weight":
                    new_state_dict["classifier.6.weight"] = v
                if k == "22.1.bias":
                    new_state_dict["classifier.6.bias"] = v

            alexnet.load_state_dict(new_state_dict)

        alexnet.classifier = nn.Sequential(*list(alexnet.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(alexnet.features._modules.values())[:-1])

        for layer in range(self.n_frozen_layers):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        self.RCNN_top = alexnet.classifier

        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
