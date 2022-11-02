import torch
import torch.nn as nn
from loss_repo import *

def couple_name(names):
    name = ''
    for i in names:
        name += i + "+"
    name = name[:-1]
    return name

def get_loss_func_list(names):
    loss_list = []
    for name in names:
        loss_list.append(deal_key(name))
    return loss_list

class Loss(nn.Module):
    def __init__(self, loss, parameters):
        super(Loss, self).__init__()
        self.weight = parameters
        self.names = loss
        self.name = couple_name(self.names)
        self.loss_function = get_loss_func_list(self.names)
        # print(self.loss_function)

    def forward(self, pred, labels):
        loss_tensor = torch.zeros([len(self.loss_function), 2], device=pred.device)
        if isinstance(pred, (list, tuple)):
            labelses = [labels for _ in range(len(pred))]
            labels = torch.cat(labelses, dim=0)
            pred = torch.cat(pred, dim=0)
            for i, (w, f) in enumerate(zip(self.weight, self.loss_function), 0):
                    loss_tensor[i, :] = w * f(pred, labels)
        else:
            for i, (w, f) in enumerate(zip(self.weight, self.loss_function), 0):
                    loss_tensor[i, :] = w * f(pred, labels)

        all_loss = loss_tensor[:, 0]
        if len(self.loss_function) >= 2:
            f1_part = all_loss[0]
            loss = all_loss.sum()
            loss_rate = f1_part / loss
        else:
            loss_rate = torch.tensor(0)
            loss = all_loss.sum()

        func_rate = loss_tensor[:, 1]

        return loss, func_rate, loss_rate

# def pre_check(pred,lab):
#     if pred.shape == lab.shape:
#         return True
#     else:
#         raise ValueError
#
#
# def CE_sum_native_weight_0_1(pred, lab):
#     pre_check(pred, lab)
#     eps = 1e-6
#     threshold = 0
#     lambda_ = 1
#     pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=lambda_)
#     POS = weight_pos * (-pred_pos.log()).sum()
#     NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
#     cross_entropy = (POS + NEG) / pred.shape[0]
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor
#
# def CE_sum_native_weight_17_1(pred, lab):
#     pre_check(pred, lab)
#     eps = 1e-6
#     threshold = 0.17
#     lambda_ = 1
#     pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=lambda_)
#     POS = weight_pos * (-pred_pos.log()).sum()
#     NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
#     cross_entropy = (POS + NEG) / pred.shape[0]
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor
#
# def CE_sum_native_weight_17_11(pred, lab):
#     pre_check(pred, lab)
#     eps = 1e-6
#     threshold = 0.17
#     lambda_ = 1.1
#     pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=lambda_)
#     POS = weight_pos * (-pred_pos.log()).sum()
#     NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
#     cross_entropy = (POS + NEG) / pred.shape[0]
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor
#
#
# def ForCE_sum_native_weight_0_1(logits, labels):
#     pre_check(logits, labels)
#     loss_tensor = 0
#     for i, (_logit, _label) in enumerate(zip(logits, labels)):
#         loss_tensor += CE_sum_native_weight_0_1_one(_logit, _label)
#     return loss_tensor / len(logits)
#
# def CE_sum_native_weight_0_1_one(pred, lab):
#     eps = 1e-6
#     threshold = 0
#     lambda_ = 1
#     pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=lambda_)
#     POS = weight_pos * (-pred_pos.log()).sum()
#     NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
#     cross_entropy = (POS + NEG) / pred.shape[0]
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor
#
# def ForCE_sum_native_weight_17_1(logits, labels):
#     pre_check(logits, labels)
#     loss_tensor = 0
#     for i, (_logit, _label) in enumerate(zip(logits, labels)):
#         loss_tensor += CE_sum_native_weight_17_1_one(_logit, _label)
#     return loss_tensor / len(logits)
#
# def CE_sum_native_weight_17_1_one(pred, lab):
#     eps = 1e-6
#     threshold = 0.17
#     lambda_ = 1
#     pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=lambda_)
#     POS = weight_pos * (-pred_pos.log()).sum()
#     NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
#     cross_entropy = (POS + NEG) / pred.shape[0]
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor
#
# def ForCE_sum_native_weight_17_11(logits, labels):
#     pre_check(logits, labels)
#     loss_tensor = 0
#     for i, (_logit, _label) in enumerate(zip(logits, labels)):
#         loss_tensor += CE_sum_native_weight_17_11_one(_logit, _label)
#     return loss_tensor / len(logits)
#
# def CE_sum_native_weight_17_11_one(pred, lab):
#     eps = 1e-6
#     threshold = 0.17
#     lambda_ = 1.1
#     pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=lambda_)
#     POS = weight_pos * (-pred_pos.log()).sum()
#     NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
#     cross_entropy = (POS + NEG) / pred.shape[0]
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor




# def neighborhood_loss_sum(pred, lab):
#     pre_check(pred, lab)
#     eps = 1e-6
#     pred_pos = pred[lab > 0].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     weight_pos, weight_neg = get_weight(pred, lab, threshold=0, weight=1)
#     outer_lab = outerlook(lab, 0)
#     neighborhood_lab = outer_lab
#     neighbor_weight = neighborhood_lab[lab == 0]
#
#     a = 1
#     b = 1
#     neg_weight = (a * neighbor_weight - b).exp()
#     # neg_weight = (1 - neighbor_weight)
#     # w_anotation = lab[lab > 0]
#     # pos_weight = (a * w_anotation).exp()
#     # pos_weight = torch.tan(1.5 * w_anotation) * 5
#     pos_weight = 1
#
#     POS = (pos_weight * -pred_pos.log()).sum()
#     NEG = (neg_weight * -(1.0 - pred_neg).log()).sum()
#     cross_entropy = (POS + NEG)/pred.shape[0]
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#
#     return loss_tensor
#
# def neighborhood_loss_mean(pred, lab):
#     pre_check(pred, lab)
#     eps = 1e-6
#     pred_pos = pred[lab > 0].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     outer_lab = outerlook(lab, 0)
#     neighborhood_lab = outer_lab
#     neighbor_weight = neighborhood_lab[lab == 0]
#     a = 1
#     b = 1
#     neg_weight = (a * neighbor_weight - b).exp()
#     # neg_weight = (1 - neighbor_weight)
#     # w_anotation = lab[lab > 0]
#     # pos_weight = (a * w_anotation).exp()
#     pos_weight =1
#     # print(w_anotation.mean(), w_anotation.max())
#     # print(neighbor_weight.mean(), neighbor_weight.max())
#     # pos_weight = torch.tan(1.5 * w_anotation)
#     POS = (pos_weight * -pred_pos.log()).mean()
#     NEG = (neg_weight * -(1.0 - pred_neg).log()).mean()
#     cross_entropy = POS + NEG
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor
#
# def outerlook(lab,px):
#     _,_,h,w = lab.shape
#     crop_lab = lab[:,:,px:h-px,px:w-px]
#     copy_lab = torch.clone(crop_lab)
#     copy_lab[copy_lab > 0] = 1
#     pooling_lab1 = F.avg_pool2d(copy_lab,kernel_size=16, stride=16)
#     pooling_lab1 = pooling_lab1/pooling_lab1.max()
#
#     return F.interpolate(input=pooling_lab1, size=(h, w), mode='bilinear', align_corners=False)
#
# def cross_entropy_native_sum(pred, lab):
#     pre_check(pred, lab)
#     loss_tensor = torch.zeros([1, 2]).cuda()
#     eps = 1.0e-6
#     pred_pos = pred[lab > 0].clamp(eps, 1.0 - eps)
#     pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#     POS = (-pred_pos.log()).sum()
#     NEG = (-(1.0 - pred_neg).log()).sum()
#     cross_entropy = POS + NEG
#     rate = POS / (POS + NEG)
#     loss_tensor[0, 0] = cross_entropy / pred.shape[0]
#     loss_tensor[0, 1] = rate
#     return loss_tensor















# def innerlook(lab,px):
#     _,_,h,w = lab.shape
#     m = nn.ReplicationPad2d(px)
#     padding_lab = m(lab)
#     return F.interpolate(input=padding_lab, size=(h, w), mode='bilinear', align_corners=False)

# def cross_entropy_sum(pred, lab):
#     loss_tensor = torch.zeros([1, 2]).cuda()
#     eps = 1.0e-6
#     pred = pred.clamp(eps, 1.0 - eps)
#     POS = -(lab * (pred.log())).sum()
#     NEG = -((1.0 - lab) * ((1.0 - pred).log())).sum()
#     rate = POS / (POS + NEG)
#     cross_entropy = POS + NEG
#     loss_tensor[0, 0] = cross_entropy / pred.shape[0]
#     loss_tensor[0, 1] = rate
#     return loss_tensor
#
# def cross_entropy_sum_rate(pred, lab, threshold=0.17):
#     loss_tensor = torch.zeros([1, 2]).cuda()
#     eps = 1.0e-6
#     pred = pred.clamp(eps, 1.0 - eps)
#     count_pos = pred[lab >= threshold].size()[0]
#     count_neg = pred[lab == 0.0].size()[0]
#     total = count_neg + count_pos
#     weight_pos = count_neg / total
#     weight_neg = (count_pos / total)
#     POS = -(lab * (pred.log())).sum()
#     NEG = -((1.0 - lab) * ((1.0 - pred).log())).sum()
#     rate = POS / (POS + NEG)
#     cross_entropy = weight_pos * rate * POS + NEG * (1-rate) * weight_neg
#     loss_tensor[0, 0] = cross_entropy / pred.shape[0]
#     loss_tensor[0, 1] = rate
#     return loss_tensor
#
# def cross_entropy_native_sum(pred, lab):
#     if pred.shape == lab.shape:
#         loss_tensor = torch.zeros([1, 2]).cuda()
#         eps = 1.0e-6
#         pred_pos = pred[lab > 0].clamp(eps, 1.0 - eps)
#         pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#         POS = (-pred_pos.log()).sum()
#         NEG = (-(1.0 - pred_neg).log()).sum()
#         cross_entropy = POS + NEG
#         rate = POS / (POS + NEG)
#         loss_tensor[0, 0] = cross_entropy / pred.shape[0]
#         loss_tensor[0, 1] = rate
#     else:
#         raise NameError
#
#     return loss_tensor
#
# def cross_entropy_w_anotation(pred, lab):
#     if pred.shape == lab.shape:
#         eps = 1e-6  # 1e-6 is the good choise if smaller than 1e-6, it may appear NaN
#         pred_pos = pred[lab > 0].clamp(eps, 1.0 - eps)
#         pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#         w_anotation = lab[lab > 0]
#         POS = (-pred_pos.log() * w_anotation).mean()
#         NEG = (-(1.0 - pred_neg).log()).mean()
#         cross_entropy = POS + NEG
#         rate = POS / (POS + NEG)
#         loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     else:
#         raise NameError
#     return loss_tensor
#
# def cross_entropy_w_anotation_pre(logits, labels):
#     loss_tensor = 0
#     for i, (_logit, _label) in enumerate(zip(logits, labels)):
#         loss_tensor += cross_entropy2_with_weight(_logit, _label)
#     return loss_tensor / len(logits)
#
# def cross_entropy2_with_weight(logits, labels):
#     logits = logits.view(-1)
#     labels = labels.view(-1)
#     eps = 1e-6  # 1e-6 is the good choise if smaller than 1e-6, it may appear NaN
#     pred_pos = logits[labels > 0].clamp(eps, 1.0 - eps)
#     pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)
#     w_anotation = labels[labels > 0]
#     POS = (-pred_pos.log() * w_anotation).mean()
#     NEG = (-(1.0 - pred_neg).log()).mean()
#     cross_entropy = POS + NEG
#     rate = POS / (POS + NEG)
#     loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     return loss_tensor
#
# def Negtive_weight_loss(pred, lab):
#     if pred.shape == lab.shape:
#         eps = 1e-6  # 1e-6 is the good choise if smaller than 1e-6, it may appear NaN
#         pred_pos = pred[lab > 0].clamp(eps, 1.0 - eps)
#         pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
#         weight_neg = generate_weight(lab, )
#
#         w_anotation = lab[lab > 0]
#         POS = (-pred_pos.log() * w_anotation).mean()
#         NEG = (-(1.0 - pred_neg).log()).mean()
#         cross_entropy = POS + NEG
#         rate = POS / (POS + NEG)
#         loss_tensor = torch.stack([cross_entropy, rate], dim=0)
#     else:
#         raise NameError
#
# def Dice_like(pred, lab):
#     if pred.shape == lab.shape:
#         loss_tensor = torch.zeros([1, 2]).cuda(device)
#         eps = 1.0e-6
#         numerator = lab.sum()
#         denominator = ((1-lab)*pred).sum() + eps
#         fraction = numerator / denominator
#         loss_tensor[0, 0] = fraction
#         loss_tensor[0, 1] = numerator
#     else:
#         raise NameError
#     return loss_tensor
#
# def dice(pred, lab):
#     if pred.shape == lab.shape:
#         loss_tensor = torch.zeros([1, 2]).cuda(device)
#         eps = 1e-6
#         dice = ((pred * lab).sum() * 2 + eps) / (pred.sum() + lab.sum() + eps)
#         dice_loss = dice.pow(-1)
#         loss_tensor[0, 0] = dice_loss
#         loss_tensor[0, 1] = pred.sum()
#     else:
#         raise NameError
#     return loss_tensor
#
# def generate_weight(lab, kernel):
#     blur = K.filters.filter2d(lab, kernel)
#     return blur[lab == 0]
#
# def temp_plt(temp):
#     temp = temp.cpu().detach().numpy().transpose((0, 2, 3, 1))
#     temp = temp[0]
#     cv2.imwrite('./0.jpg',temp * 255)