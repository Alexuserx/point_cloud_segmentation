import torch
import numpy as np
import torch.nn.functional as F 


def calculate_sem_IoU(pred_np, seg_np, visual=False, num_classes=3):
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all 

def cal_loss(pred, gold, weight, smoothing=False, ignore_index=255):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(
            pred, gold, reduction='mean', weight=weight,
            ignore_index=ignore_index)

    return loss