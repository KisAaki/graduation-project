from torch import nn
import torch.nn.functional as F
import torch

#4.17, 加入 focal_loss

# IOU LOSS 损失函数
def _iou(pred, target):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:]*pred[i,:,:])
        Ior1 = torch.sum(target[i,:,:]) + torch.sum(pred[i,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()
    
    def forward(self, pred, target):

        return _iou(pred, target)


# WBCE LOSS 损失函数
class Weighed_Bce_Loss(nn.Module):
    def __init__(self):
        super(Weighed_Bce_Loss, self).__init__()

    def forward(self, x, label):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        label = label.view(-1, 1, label.shape[1], label.shape[2])
        label_t = (label == 1).float()
        label_f = (label == 0).float()
        p = torch.sum(label_t) / (torch.sum(label_t) + torch.sum(label_f))
        w = torch.zeros_like(label)
        w[label == 1] = p
        w[label == 0] = 1 - p
        loss = F.binary_cross_entropy(x, label, weight=w)
        return loss

# classification Loss 损失函数
class Cls_Loss(nn.Module):
    def __init__(self):
        super(Cls_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.binary_cross_entropy(x, label)

        return loss


class S_Loss(nn.Module):
    def __init__(self):
        super(S_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.smooth_l1_loss(x, label)
        return loss

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()

        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_wbce = Weighed_Bce_Loss()
        self.loss_cls = Cls_Loss()
        self.loss_s = S_Loss()
        self.loss_i = IOU()
        self.loss_focal = focal_loss()
        self.w_wbce = 1
        self.w_cls = 1
        self.w_smooth = 1
        self.w_iou = 1

    def forward(self, x, label, x_cls, label_cls):
        m_loss = self.loss_wbce(x, label) * self.w_wbce
        c_loss = self.loss_cls(x_cls, label_cls) * self.w_cls
        s_loss = self.loss_s(x, label) * self.w_smooth
        iou_loss = self.loss_i(x, label) * self.w_iou
        focal_loss = self.loss_focal(x, label) * self.w_focal
        loss = m_loss + c_loss + s_loss + iou_loss + focal_loss

        return loss, m_loss, c_loss, s_loss, iou_loss, focal_loss
