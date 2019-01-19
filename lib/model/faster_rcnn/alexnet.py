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
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb



'''
class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)
'''
class alexnet(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/alexnet-owt-4df8aa71.pth'
        self.dout_base_model = 256
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        #todo parametrizzare
        self.n_frozen_layers = 5

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        #vgg = models.vgg16()
        alexnet = models.alexnet()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            alexnet.load_state_dict({k:v for k,v in state_dict.items() if k in alexnet.state_dict()})

        alexnet.classifier = nn.Sequential(*list(alexnet.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(alexnet.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(self.n_frozen_layers):
          for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        #self.RCNN_top = alexnet.classifier
        #todo per provare, decommentare la line sopra una volta provato

        self.RCNN_top = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            #nn.Linear(4096, self.n_classes), -> non presente in alexnet.classifier, infatti messo dopo
        )
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
          self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
          self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)


    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7


