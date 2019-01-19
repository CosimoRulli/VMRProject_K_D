import torch.nn.functional as F
import torch
def compute_loss_rpn_cls(Z_t, Z_s, mu, L_hard,fg_bg_label, T = 1):

    wc = torch.tensor([1.5,1])



    #L_soft = F.cross_entropy(Z_t / T, Z_s, wc) #non si può utilizzare, si aspetta che Z_s sia un vettore di interi
    #print("Softmax_torch"+" "+ str(L_soft))
    #handcrafted

    #wc = torch.ones(fg_bg_label.shape[0], dtype = torch.float)

    wc = fg_bg_label.float()*0.5 + 1
    '''
    for i in range (fg_bg_label.shape[0]):
        if(fg_bg_label[i]==1):
            wc[i] = 1.5
    '''
    P_t = F.softmax(Z_t / T, dim=1)
    P_s = F.softmax(Z_s / T, dim=1)
    P = torch.sum(P_t * torch.log(P_s), dim = 1)
    L_soft = - torch.sum(P*wc)
    #L_soft = - torch.sum(wc * P_t * torch.log(P_s))
    #print("Softmax_hand_crafted" + " " + str(L_soft))

    L_cls  =  mu * L_hard  + (1-mu) * L_soft
    return  L_cls

    # L_soft = F.cross




