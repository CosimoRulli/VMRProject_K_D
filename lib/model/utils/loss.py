import torch.nn.functional as F
import torch
def compute_loss_rpn_cls(Z_t, Z_s, mu, L_hard,fg_bg_label, T = 1):
    #todo reimplementare le due loss come una unica funzione scrivendo wc come matrice ( mask)
    wc = fg_bg_label.float()*0.5 + 1
    '''
    for i in range (fg_bg_label.shape[0]):
        if(fg_bg_label[i]==1):
            wc[i] = 1.5
    '''
    P_t = F.softmax(Z_t / T, dim=1)
    P_s = F.softmax(Z_s / T, dim=1)
    P = torch.sum(P_t * torch.log(P_s), dim = 1)
    L_soft = - torch.sum(P*wc) / P_t.shape[0]
    #L_soft = - torch.sum(wc * P_t * torch.log(P_s))
    #print("Softmax_hand_crafted" + " " + str(L_soft))

    L_cls  =  mu * L_hard  + (1-mu) * L_soft
    return  L_cls

    # L_soft = F.cross


def compute_loss_rcn_cls(Z_t, Z_s, mu, L_hard,rois_label, T = 1):
    wc = torch.ones(Z_t.shape[1])
    wc[0] = 1.5
    P_t = F.softmax(Z_t /T, dim=1)
    P_s = F.softmax(Z_s /T, dim=1)
    P = torch.sum(P_t * torch.log(P_s), dim=1)
    L_soft = - torch.sum(P * wc) / P_t.shape[0]
    L_cls = mu * L_hard + (1 - mu) * L_soft
    return L_cls

def compute_loss_classification(Z_t, Z_s, mu, L_hard, y, T=1):
    #Date le y, devo costruire un vettore della stessa dimensione di Z_t (Z_s), dove gli elementi valgono 1.5 se gli elementi valgono 0 e 1 altrimenti
    wc = torch.where((y==0), 1.5*torch.ones(Z_t.shape[0]).cuda(), torch.ones(Z_s.shape[0]).cuda())
    #wc = torch.ones(Z_t.shape).cuda()
    P_t = F.softmax(Z_t /T, dim=1)
    P_s = F.softmax(Z_s /T, dim=1)
    P = torch.sum(P_t * torch.log(P_s), dim=1)
    L_soft = -torch.mean(P*wc)
    L_cls = mu * L_hard + (1 - mu) * L_soft
    return L_cls

def compute_loss_regression(smooth_l1_loss, Rs, Rt, y_reg, m, l, bbox_inside_weights, bbox_outside_weights ,ni=0.5):
  #batch_size = Rs.shape[0]
  s_box_diff = Rs - y_reg
  t_box_diff = Rt - y_reg
  in_s_box_diff = bbox_inside_weights * s_box_diff
  in_t_box_diff = bbox_inside_weights * t_box_diff
  in_s_box_diff = in_s_box_diff * bbox_outside_weights
  in_t_box_diff = in_t_box_diff * bbox_outside_weights
  in_s_bd_quad = in_s_box_diff.pow(2)
  in_t_bd_quad = in_t_box_diff.pow(2)
  norm_s = in_s_bd_quad
  norm_t = in_t_bd_quad
  dim= range(1,len(in_s_box_diff.shape) )
  for i in sorted(dim, reverse = True):
      norm_s = norm_s.sum(i)
      norm_t = norm_t.sum(i)
  cuda0 = torch.device('cuda:0')
  zeros = torch.zeros(norm_s.shape,device = cuda0)
  l_b = torch.where((norm_s + m <= norm_t), zeros, norm_s)
  l_reg =  smooth_l1_loss + ni * l_b.mean()
  return l_reg


'''
def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

if __name__ == '_main_':
    bbox_pred_s= torch.Tensor(2, 256, 4)
    bbox_pred_t= torch.Tensor(2, 256, 4)
    bbox_pred_s[0] = 1
    bbox_pred_s[1] = 2
    bbox_pred_t[0] = 0.5
    bbox_pred_t[1] = 1
    rois_target = torch.ones(256, 4)
    RCN_loss_box = 1
    a = compute_loss_rcn_regression(smooth_l1_loss=RCN_loss_box, Rs=bbox_pred_s, Rt=bbox_pred_t, y_reg=rois_target, m=-3, l=2 )
    print(a)
'''



