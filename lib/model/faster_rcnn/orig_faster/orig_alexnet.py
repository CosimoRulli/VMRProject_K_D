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
from model.faster_rcnn.orig_faster.orig_faster_rcnn import _orig_fasterRCNN
import pdb




model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class alexnet(_orig_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.dout_base_model = 256
        self.model_path = 'data/pretrained_model/alexnet-owt-4df8aa71.pth'

        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        #todo parametrizzare
        self.n_frozen_layers = 8
        print("N_Frozen_layers: "+str(self.n_frozen_layers))

        _orig_fasterRCNN.__init__(self, classes, class_agnostic)

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
        '''
        self.RCNN_base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=7),  #changed the padding to match dimensions with vgg16, 10 dava un errore
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )
        '''
        # Fix the layers before conv3:
        for layer in range(self.n_frozen_layers):
          for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
        ''''
        self.RCNN_top = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            #nn.Linear(4096, self.n_classes), -> non presente in alexnet.classifier, infatti messo dopo
        )
        '''
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


