import torch.nn.functional as F
import torch

'''
def compute_loss_rpn_cls(Z_t, Z_s, mu, L_hard,fg_bg_label, T = 1):
    #todo reimplementare le due loss come una unica funzione scrivendo wc come matrice ( mask)
    wc = fg_bg_label.float()*0.5 + 1
    
    for i in range (fg_bg_label.shape[0]):
        if(fg_bg_label[i]==1):
            wc[i] = 1.5
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
'''
def compute_loss_classification(Z_t, Z_s, mu, L_hard, y, T=1):
    #Date le y, devo costruire un vettore della stessa dimensione di Z_t (Z_s), dove gli elementi valgono 1.5 se gli elementi valgono 0 e 1 altrimenti
    wc = torch.where((y==0), 1.5*torch.ones(Z_t.shape[0]).cuda(), torch.ones(Z_s.shape[0]).cuda())
    #wc = torch.ones(Z_t.shape).cuda()
    Z_s.double()
    Z_t.double()

    P_t = F.softmax(Z_t /T, dim=1)
    P_s = F.softmax(Z_s /T, dim=1)
    #print("P_t : ", str(P_t))
    #print("P_s : ", str(P_s))
    if (len(P_s.nonzero()) != len(P_s)*P_s.shape[1]):
        print("ATTENZIONE UNO ZERO")
        print("P_s: ")
        for j in range (len(P_s)):
            print(P_s[j][:])
        print("Z_s:")
        for j in range (len(Z_s)):
            print(Z_s[j][:])
        print("Z_t:")
        for j in range(len(Z_t)):
            print(Z_t[j][:])

    P = torch.sum(P_t * torch.log(P_s + 1e-10), dim=1)
    #P = torch.sum(P_t * torch.log(P_s), dim=1)
    #print("torch.log(P_s + 1e-10): ", torch.log(P_s + 1e-10))
    L_soft = -torch.mean(P*wc)
    #L_soft = -torch.mean(P)
    L_cls = mu * L_hard + (1 - mu) * L_soft
    # L_cls = L_hard
    #print("Lsoft : ", str(L_soft))
    #print("Lhard : ", str(L_hard))
    return L_cls, L_soft

def compute_loss_regression(smooth_l1_loss, Rs, Rt, y_reg_s, y_reg_t , m, bbox_inside_weights_s,bbox_inside_weights_t, bbox_outside_weights_s,bbox_outside_weights_t,ni):
  #batch_size = Rs.shape[0]
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

'''

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



