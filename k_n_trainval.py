# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick

#  CUDA_VISIBLE_DEVICES=3 python k_n_trainval.py --dataset pascal_voc --net vgg16 --checksession 1 --checkepoch 6 --checkpoint 10021 --cuda --epochs 10
#  CUDA_VISIBLE_DEVICES=1 python trainval_net.py --dataset pascal_voc --net vgg16 --cuda
#  CUDA_VISIBLE_DEVICES=1 python test_net.py --dataset pascal_voc --net vgg16 --cuda  --checksession 1 --checkepoch 8 --checkpoint 10021

# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.alexnet import alexnet
from model.utils.loss import  compute_loss_regression, compute_loss_classification

from torchvision.transforms import ToTensor, ToPILImage, Resize

'''
def resize_images(im_batch, size):
    new_im_batch = torch.zeros([im_batch.shape[0], im_batch.shape[1], size[0], size[1]])
    for i in range(im_batch.shape[0]):
        im_pil = ToPILImage()(im_batch[0].cpu())
        im_pil = Resize(size)(im_pil)
        new_im_batch[0, :, :, :] = ToTensor()(im_pil)
    return new_im_batch.cuda()


from model.faster_rcnn.resnet import resnet

def print_tensor(tensor_2d):
    for i in range(tensor_2d.shape[0]):
        for j in range(tensor_2d.shape[1]):
            sys.stdout.write(str(round(tensor_2d[i][j].item() ,2))+ " ")
            sys.stdout.flush()
        print()
'''



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--t_net', dest='t_net',
                    help='vgg16',
                    default='vgg16', type=str)
  parser.add_argument('--s_net', dest='s_net',
                    help='alexnet',
                    default='alexnet', type=str)

  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

#config_loss_parameters
  parser.add_argument('--mu', dest='mu',
                      help='mu for Lcls loss', default=0.8, type=float)
  parser.add_argument('--lambda', dest='l',
                       help='lambda for both final rpn and rcn losses',default=1, type=float)
  parser.add_argument('--ni', dest='ni',
                       help='ni for regression losses', default=0.5, type=float)
  parser.add_argument('--m',dest='m',
                      help='m for regression losses  bound', default=0, type=float)
# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--t_checksession', dest='t_checksession',
                      help='teacher checksession to load model',
                      default=1, type=int)
  parser.add_argument('--t_checkepoch', dest='t_checkepoch',
                      help='teacher checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--t_checkpoint', dest='t_checkpoint',
                      help='teacher_checkpoint to load model',
                      default=0, type=int)

  parser.add_argument('--s_checksession', dest='s_checksession',
                      help='student checksession to load model',
                      default=1, type=int)
  parser.add_argument('--s_checkepoch', dest='s_checkepoch',
                      help='student checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--s_checkpoint', dest='s_checkpoint',
                      help='student_checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  '''
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  '''
  #  args.cfg_file = "student_cfgs/{}_ls.yml".format(args.net) if args.large_scale else "student_cfgs/{}.yml".format(args.net)

  args.cfg_file = "student_cfgs/{}.yml".format(args.s_net)
  print("File di configurazione della student")
  print(args.cfg_file)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  np.random.seed(cfg.RNG_SEED)

  #  torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.s_net + "/" + args.dataset
  print(" ")
  print("Output dir: "+str(output_dir))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  #TODO valutare num_workers da torch.utils  
  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  #  make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.

  if args.s_net == 'alexnet':
      student_net = alexnet(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  else:
      print("student network is not defined")
      pdb.set_trace()

  if args.t_net =='vgg16':
      teacher_net = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic, teaching=True)
  else:
      print("teacher network is not defined")
      pdb.set_trace()


  ##CREATE ARCHITECTURES
  teacher_net.create_architecture()
  student_net.create_architecture()

  #LOAD TEACHER NET

  input_dir = args.load_dir + "/" + args.t_net + "/" + args.dataset
  print(input_dir)
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    args.t_net+'_teacher_fast_rcnn_{}_{}_{}.pth'.format(args.t_checksession, args.t_checkepoch, args.t_checkpoint))


  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
      print(" ")
      print("Load name checkpoint: "+str(load_name))
      checkpoint = torch.load(load_name)
  else:
      checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

  teacher_net.load_state_dict(checkpoint['model'])


  if 'pooling_mode' in checkpoint.keys():
     cfg.POOLING_MODE = checkpoint['pooling_mode']



  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []

  for key, value in dict(student_net.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]


  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    teacher_net.cuda()
    student_net.cuda()


  if args.resume:
    load_name = os.path.join(output_dir,
      '{}_{}_student_net_{}_{}_{}.pth'.format(args.m, args.mu, args.s_checksession, args.s_checkepoch, args.s_checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    student_net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  #todo DataParallel è un modulo di pytorch https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
  #todo 'Divide i batch in sotto batch per eseguirli in parallelo su più GPUs che non è proprio quello che serve a noi'
  '''
  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)
  '''

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")
  #ASSIGN custom values to loss parameters
  mu = args.mu
  l = args.l
  ni = args.ni
  m = args.m
  print("LOSS PARAMETERS:")
  print("Mu: "+str(mu))
  print("l: " + str(l))
  print("ni: " + str(ni))
  print("m:" +str(m))

  for epoch in range(args.start_epoch, args.max_epochs + 1):

    student_net.train()
    teacher_net.eval()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)

    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      student_net.zero_grad()

      rois_t, cls_prob_t, bbox_pred_t, \
      rpn_loss_cls_t, rpn_loss_box_t, \
      RCNN_loss_cls_t, RCNN_loss_bbox_t, \
      rois_label_t, Z_t, R_t, fg_bg_label, \
      y_reg_t, iw_t, ow_t, rois_target_t, rois_inside_ws_t,\
      rois_outside_ws_t, rcn_cls_score_t = teacher_net(im_data, im_info, gt_boxes, num_boxes)

      cls_prob_t = cls_prob_t.detach()

      bbox_pred_t = bbox_pred_t.detach()
      Z_t = Z_t.detach()
      R_t = R_t.detach()
      rcn_cls_score_t= rcn_cls_score_t.detach()


      rois_s, cls_prob_s, bbox_pred_s, \
      rpn_loss_cls_s, rpn_loss_box_s, \
      RCNN_loss_cls_s, RCNN_loss_bbox_s, \
      rois_label_s, Z_s, R_s, _, y_reg_s, iw_s, ow_s,  rois_target_s, rois_inside_ws_s, \
      rois_outside_ws_s, rcn_cls_score_s = student_net(im_data, im_info, gt_boxes, num_boxes)


      
      L_hard = rpn_loss_cls_s
      loss_rpn_cls, loss_rpn_cls_soft = compute_loss_classification(Z_t, Z_s, mu, L_hard, fg_bg_label, T=1)
      loss_rpn_reg, loss_rpn_reg_soft = compute_loss_regression(rpn_loss_box_s, R_s, R_t, y_reg_s, y_reg_t, m=m, bbox_inside_weights_s=iw_s ,bbox_inside_weights_t= iw_t,bbox_outside_weights_s=ow_s,bbox_outside_weights_t=ow_t, ni=ni)
      torch.set_printoptions(threshold=10000)

      loss_rcn_cls, loss_rcn_cls_soft = compute_loss_classification(rcn_cls_score_t, rcn_cls_score_s,  mu ,RCNN_loss_cls_s, rois_label_t, T=1)
      loss_rcn_reg, loss_rcn_reg_soft = compute_loss_regression(RCNN_loss_bbox_s,bbox_pred_s, bbox_pred_t, rois_target_s,rois_target_t, m=m,  bbox_inside_weights_s=rois_inside_ws_s ,bbox_inside_weights_t= rois_inside_ws_t,bbox_outside_weights_s=rois_outside_ws_s,bbox_outside_weights_t=rois_outside_ws_t, ni=ni)

        
      loss = loss_rpn_cls+ l*loss_rpn_reg+ \
          loss_rcn_cls+ l*loss_rcn_reg

      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      #clip gradient avoids gradient explosion
      clip_gradient(student_net, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)
        '''
        if args.mGPUs:
            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_box = rpn_loss_box.mean().item()
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_box = RCNN_loss_bbox.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
        else:
        loss_rpn_reg = loss_rpn_reg.item()
        loss_rpn_cls = loss_rpn_cls.item()
        loss_rcn_cls = loss_rcn_cls.item()
        loss_rpn_reg = loss_rpn_reg.item()
        '''
       
        
        fg_cnt = torch.sum(rois_label_s.data.ne(0))
        bg_cnt = rois_label_s.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_reg: %.4f, rcn_cls: %.4f, rcn_reg %.4f" \
                      % (loss_rpn_cls, loss_rpn_reg, loss_rcn_cls, loss_rcn_reg))
        if args.use_tfboard:
          info = {
            'loss_rpn_cls_hard': L_hard,
            'loss_rpn_cls_soft': loss_rpn_cls_soft,
            'loss_rpn_cls': loss_rpn_cls,

            'loss_rpn_reg_hard': rpn_loss_box_s,
            'loss_rpn_reg_soft': loss_rpn_reg_soft,
            'loss_rpn_reg': loss_rpn_reg,

            'loss_rcn_cls_hard': RCNN_loss_cls_s,
            'loss_rcn_cls_soft': loss_rcn_cls_soft,
            'loss_rcn_cls':loss_rcn_cls,

            'loss_rcn_reg_hard':RCNN_loss_bbox_s,
            'loss_rcn_reg_soft':loss_rcn_reg_soft,
            'loss_rcn_reg': loss_rcn_reg,
            'loss': loss_temp,

          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    if(epoch !=0 and epoch%3==0):
        save_name = os.path.join(output_dir, '{}_{}_student_net_{}_{}_{}.pth'.format(args.m, args.mu, args.session, epoch, step))
        save_checkpoint({
          'session': args.session,
          'epoch': epoch + 1,
          'model': student_net.module.state_dict() if args.mGPUs else student_net.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()


