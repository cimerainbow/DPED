import torch
from model_repo.utils import get_weight
import torch.nn.functional as F
global loss_list
loss_list = []

def deal_key(str):
    str_list = str.split('_')  # 分割字符，str, num
    if str_list[0] in loss_list:
        pass
    else:
        raise ValueError
    # 处理参数
    for i in range(len(str_list)):
        if i == 1:
            str_list[i] = float(str_list[i]) / 100
        elif i == 2:
            str_list[i] = float(str_list[i])
            if str_list[i] == 1:
                str_list[i] = 1.0
            else:
                str_list[i] = float(str_list[i]) / 10
        elif i == 3:
            if str_list[i][0] == 'K':
                str_list[i] = int(str_list[i][1:])
        elif i == 4:
            if str_list[i] == 'F':
                str_list[i] = False
            elif str_list[i] == 'T':
                str_list[i] = True
            else:
                raise ValueError

    # 返回函数
    # if str_list[0] == 'CEO':
    #     func = eval(str_list[0])
    #     return func(float(str_list[1]), float(str_list[2]))
    # elif str_list[0] == 'CEM':
    #     func = eval(str_list[0])
    #     return func(float(str_list[1]), float(str_list[2]))
    # # elif str_list[0] == 'ForCEM':
    # #     func = eval(str_list[0])
    # #     return func(float(str_list[1]), float(str_list[2]))
    # elif str_list[0] == 'NM':
    #     func = eval(str_list[0])
    #     return func(float(str_list[1]), float(str_list[2]), int(str_list[3]), bool(str_list[4]))
    # # elif str_list[0] == 'ForNM':
    # #     func = eval(str_list[0])
    # #     return func(int(str_list[1]), int(str_list[2]))
    # elif str_list[0] == 'NS':
    #     func = eval(str_list[0])
    #     return func(float(str_list[1]), float(str_list[2]), int(str_list[3]), bool(str_list[4]))
    # # elif str_list[0] == 'ForNS':
    # #     func = eval(str_list[0])
    # #     return func(int(str_list[1]), int(str_list[2]))
    # elif str_list[0] == 'LPCB':
    #     func = eval(str_list[0])
    #     return func()
    # 返回函数
    print('当前损失函数为：', str_list)

    if 'For' in str_list[0]:
        # print(str_list[0][3:])
        return Forfunc(*str_list[1:], func=str_list[0][3:])
    else:
        func = eval(str_list[0])
        return func(*str_list[1:])


def loss_register(func):
    loss_list.append(func.__name__)
    loss_list.append('For'+func.__name__)
    def decorator(*args, **kwargs):
        return func(*args, **kwargs)
    return decorator

def pre_check(pred,lab):
    if pred.shape == lab.shape:
        return True
    else:
        raise ValueError



def Forfunc(threshold=0, _lambda=1, func=''):
    func = eval(func)(threshold=threshold, _lambda=_lambda)
    def inerfunc(logits, labels):
        pre_check(logits, labels)
        loss_tensor = 0
        for i, (_logit, _label) in enumerate(zip(logits, labels)):
            loss_tensor += func(_logit, _label)
        return loss_tensor / len(logits)
    return inerfunc

@loss_register
def CEO(threshold = 0, _lambda=1):
    def CEO_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
        pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        POS = weight_pos * (-pred_pos.log()).sum()
        NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
        cross_entropy = (POS + NEG) / pred.shape[0]
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return CEO_inerfunc

@loss_register
def CEwoW(threshold = 0, _lambda=1):
    def CEO_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
        pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
        POS = (-pred_pos.log()).sum()
        NEG = (-(1.0 - pred_neg).log()).sum() / _lambda

        out_lab = outerlook(lab, 2, True)
        temp = (out_lab - lab).sigmoid()
        struct_loss = - (0.5 - temp).log().sum()

        cross_entropy = (POS + NEG + struct_loss) / pred.shape[0]
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return CEO_inerfunc


@loss_register
def CEM(threshold = 0, _lambda=1):
    def CEM_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
        pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        POS = weight_pos * (-pred_pos.log()).mean()
        NEG = weight_neg * (-(1.0 - pred_neg).log()).mean()
        cross_entropy = (POS + NEG)
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return CEM_inerfunc


def outerlook(lab, kernel_size, native):
    _,_,h,w = lab.shape
    copy_lab = torch.clone(lab)
    if native == False:
        copy_lab[copy_lab > 0] = 1
    pooling_lab1 = F.avg_pool2d(copy_lab, kernel_size=kernel_size, stride=kernel_size)
    pooling_lab1 = F.interpolate(input=pooling_lab1, size=(h, w), mode='bilinear', align_corners=False)
    # outer_weight = pooling_lab1[lab == 0]
    # return outer_weight
    return pooling_lab1

def handel_weight(out_weight):
    return out_weight + torch.stack(out_weight)
@loss_register
def NM(threshold = 0, _lambda=1, kernel_size=4, native=True):
    def NM_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
        pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
        out_weight = outerlook(lab, kernel_size, native)
        out_weight = handel_weight(out_weight)
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        POS = weight_pos * (-pred_pos.log()).mean()
        NEG = weight_neg * (out_weight * -(1.0 - pred_neg).log()).mean()
        cross_entropy = (POS + NEG)
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return NM_inerfunc


@loss_register
def NS(threshold = 0, _lambda=1, kernel_size=4, native=True):
    def NS_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
        pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
        out_weight = outerlook(lab, kernel_size, native)
        out_weight = handel_weight(out_weight)
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        POS = weight_pos * (-pred_pos.log()).sum()
        NEG = weight_neg * (out_weight * -(1.0 - pred_neg).log()).sum()
        cross_entropy = (POS + NEG) / pred.shape[0]
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return NS_inerfunc

@loss_register
def LPCB(threshold=0, _lambda=1):
    def inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold].clamp(eps, 1.0 - eps)
        pred_neg = pred[lab == 0].clamp(eps, 1.0 - eps)
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        POS = weight_pos * (-pred_pos.log()).sum()
        NEG = weight_neg * (-(1.0 - pred_neg).log()).sum()
        cross_entropy = (POS + NEG) / pred.shape[0]
        rate = POS / (POS + NEG)
        dice = ((pred ** 2).sum() + (lab ** 2).sum())/(2 * (pred * lab).sum())
        loss = cross_entropy + 1.0e-3 * dice
        loss_tensor = torch.stack([loss, rate], dim=0)
        return loss_tensor
    return inerfunc


# none_sigmoid
@loss_register
def nsCEO(threshold=0, _lambda=1):
    def CEO_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold]
        pred_neg = pred[lab == 0]
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        POS = weight_pos * ((1.0 + (-1 * pred_pos).exp()).log()).sum()
        NEG = weight_neg * ((1.0 + pred_neg.exp()).log()).sum()
        cross_entropy = (POS + NEG) / pred.shape[0]
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return CEO_inerfunc

@loss_register
def nsCEPS(threshold=0, _lambda=1):
    def CEO_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold]
        pred_neg = pred[lab == 0]
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        temp = pred_pos.sigmoid()
        POS = weight_pos * ((1.0 + (-(pred_pos - temp)).exp()).log()).sum()
        NEG = weight_neg * ((1.0 + (pred_neg).exp()).log()).sum()
        cross_entropy = (POS + NEG) / pred.shape[0]
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return CEO_inerfunc

@loss_register
def nsCELS(threshold=0, _lambda=1):
    def CEO_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        eps = 1e-6
        pred_pos = pred[lab > threshold]
        pred_neg = pred[lab == 0]
        weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        temp = lab[lab > threshold]
        POS = weight_pos * ((1.0 + (-(pred_pos - temp)).exp()).log()).sum()
        NEG = weight_neg * ((1.0 + (pred_neg).exp()).log()).sum()
        cross_entropy = (POS + NEG) / pred.shape[0]
        rate = POS / (POS + NEG)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return CEO_inerfunc

def PR_func(pred, lab, threshold):
    pred_pos = pred[lab > threshold]
    lab_pos = lab[lab > threshold]
    pred_neg = pred[lab == 0]
    numerator = (pred_pos * lab_pos).sum()  #
    Recall = numerator / lab_pos.sum()
    Precision = numerator / pred_pos.sum()
    return Recall, Precision

@loss_register
def DMSE(threshold=0, _lambda=1):
    def CEO_inerfunc(pred, lab, threshold=threshold, _lambda=_lambda):
        pre_check(pred, lab)
        Recall, Precision = PR_func(pred, lab, threshold)
        # weight_pos, weight_neg = get_weight(pred, lab, threshold=threshold, weight=_lambda)
        cross_entropy = ((Recall - Precision) ** 2) - (Recall + Precision - 2) + _lambda * pred.mean()
        rate = 2 * (Recall * Precision) / (Recall + Precision)
        loss_tensor = torch.stack([cross_entropy, rate], dim=0)
        return loss_tensor
    return CEO_inerfunc

# @loss_register
# def ForCEO(threshold=0, _lambda=1):
#     def CEO_inerfunc(logits, labels, threshold=threshold, _lambda=_lambda):
#         pre_check(logits, labels)
#         loss_tensor = 0
#         func = CEO(threshold=threshold, _lambda=_lambda)
#         for i, (_logit, _label) in enumerate(zip(logits, labels)):
#             loss_tensor += func(_logit, _label)
#         return loss_tensor / len(logits)
#     return CEO_inerfunc
# @loss_register
# def ForCEM(threshold=0, _lambda=1):
#     def CEM_inerfunc(logits, labels, threshold=threshold, _lambda=_lambda):
#         pre_check(logits, labels)
#         loss_tensor = 0
#         func = CEM(threshold=threshold, _lambda=_lambda)
#         for i, (_logit, _label) in enumerate(zip(logits, labels)):
#             loss_tensor += func(_logit, _label)
#         return loss_tensor / len(logits)
#     return CEM_inerfunc


if __name__ == '__main__':
    print(loss_list)
    deal_key('NM_0_1_K2_T')