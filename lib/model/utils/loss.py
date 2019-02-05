import torch.nn.functional as F
import torch


def compute_loss_classification(Z_t, Z_s, mu, L_hard, y, T=1):
    #vettori di pesi
    wc = torch.where((y==0), 1.5*torch.ones(Z_t.shape[0]).cuda(), torch.ones(Z_s.shape[0]).cuda())

    Z_s = Z_s.double()
    Z_t = Z_t.double()

    P_t = F.softmax(Z_t /T, dim=1)
    P_s = F.softmax(Z_s /T, dim=1)


    P = torch.sum(P_t * torch.log(P_s), dim=1)# era P_s è e-10

    L_soft = -torch.mean(P*wc)

    L_cls = mu * L_hard + (1 - mu) * L_soft

    return L_cls, L_soft

def compute_loss_regression(smooth_l1_loss, Rs, Rt, y_reg_s, y_reg_t , m, bbox_inside_weights_s,bbox_inside_weights_t, bbox_outside_weights_s,bbox_outside_weights_t,ni):

  s_box_diff = Rs - y_reg_s
  t_box_diff = Rt - y_reg_t
  in_s_box_diff = bbox_inside_weights_s * s_box_diff
  in_t_box_diff = bbox_inside_weights_t * t_box_diff
  in_s_box_diff = in_s_box_diff * bbox_outside_weights_s
  in_t_box_diff = in_t_box_diff * bbox_outside_weights_t
  in_s_bd_quad = in_s_box_diff.pow(2)
  in_t_bd_quad = in_t_box_diff.pow(2)
  norm_s = in_s_bd_quad
  norm_t = in_t_bd_quad
  dim= range(1,len(in_s_box_diff.shape))
  for i in sorted(dim, reverse=True):
      norm_s = norm_s.sum(i)
      norm_t = norm_t.sum(i)
  zeros = torch.zeros(norm_s.shape).cuda()
  l_b = torch.where((norm_s + m <= norm_t), zeros, norm_s)
  l_reg =  smooth_l1_loss + ni * l_b.mean()
  return l_reg, l_b.mean(), norm_s.mean(), norm_t.mean()




