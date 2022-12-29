import torch
import torch.nn as nn
import math
import numpy as np


eps = 1e-10  #
grid_size = 16  #


class YOLOLoss(nn.Module):
    def __init__(self, in_h=288, in_w=384, stride=16, label_smooth=0.005, neg_smooth=0, reg_smooth=0.5, b_ciou=True, b_cuda=True, b_bce=True, b_fea_loss=True, b_fea_len_loss=False, c1=0.1, c2=10):
        super(YOLOLoss, self).__init__()
        #
        self.L2 = nn.MSELoss(reduction='sum')
        self.grid_size = stride  #
        self.reg_smooth = reg_smooth  #
        self.smooth = label_smooth  #
        self.in_h = in_h  #
        self.in_w = in_w  #
        self.b_cuda = b_cuda
        self.b_bce = b_bce  #
        self.b_fea_loss = b_fea_loss
        self.b_fea_len_loss = b_fea_len_loss  #
        self.c1 = c1
        self.c2 = c2
        if neg_smooth == -1:
            neg_smooth = label_smooth  #
        self.neg_smooth = neg_smooth
        self.b_ciou = b_ciou  #

    def forward(self, obj, reg, gt_rect_mat, gt_obj_mat, gt_obj_n_mat, obj_features):
        # obj batch × 1 × grid_h × grid_w  -》  obj batch × grid_h × grid_w × 1
        # reg batch × 4 × grid_h × grid_w  -》  obj batch × grid_h × grid_w × 4
        obj = obj.permute(0, 2, 3, 1).contiguous()
        reg = reg.permute(0, 2, 3, 1).contiguous()
        obj_features = obj_features.permute(0, 2, 3, 1).contiguous()  # batch × grid_h × grid_w × feature_c

        # loss_ciou = 1 - self.box_ciou(reg  * gt_obj_mat, gt_rect_mat * gt_obj_mat)
        # loss_ciou = loss_ciou.unsqueeze(dim=-1)
        b, h, w, c = gt_obj_mat.shape  # c = 1

        valid_rect = gt_obj_mat.expand(b, h, w, c * 4)  # rect， b, h, w, 1 -》 b, h, w, 4

        reg_valid = torch.tensor([1/grid_size, 1/grid_size, 1/grid_size, 1/grid_size],
                                 dtype=torch.float32, requires_grad=True)
        if self.reg_smooth > 0:
            reg_valid = reg_valid * self.reg_smooth  #
        reg_valid = reg_valid.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        reg_valid = reg_valid.expand(b, h, w, c*4)  # b, h, w, 4
        reg_valid = torch.pow(reg_valid, 2)  #

        if self.b_cuda:
            valid_rect = valid_rect.cuda()
            reg = reg.cuda()
            gt_rect_mat = gt_rect_mat.cuda()
            obj = obj.cuda()
            gt_obj_n_mat = gt_obj_n_mat.cuda()
            gt_obj_mat = gt_obj_mat.cuda()
            reg_valid = reg_valid.cuda()

        if self.b_fea_loss:
            bg_features = torch.sigmoid(obj_features) * gt_obj_n_mat  #
            bg_features = bg_features.view(-1, bg_features.shape[-1])  #
            tar_features = torch.sigmoid(obj_features) * gt_obj_mat
            tar_features = tar_features.view(-1, tar_features.shape[-1])  #
            bg_feature_eve = bg_features.sum(0) / torch.sum(gt_obj_n_mat) - 0.5
            tar_feature_eve = tar_features.sum(0) / torch.sum(gt_obj_mat) - 0.5
            feature_loss_1 = torch.sum(bg_feature_eve * tar_feature_eve)
            feature_loss_2 = torch.sqrt(torch.sum(torch.pow(bg_feature_eve, 2)))
            feature_loss_3 = torch.sqrt(torch.sum(torch.pow(tar_feature_eve, 2)))
            feature_loss = feature_loss_1 / (feature_loss_2 * feature_loss_3)  # cos loss -1 1
        else:
            feature_loss = torch.tensor(0)
            feature_loss_1 = 0
            feature_loss_2 = 0
            feature_loss_3 = 0

        if self.b_fea_len_loss:
            fea_len_loss = ( (torch.pow(feature_loss_2, 2)) + (torch.pow(feature_loss_3, 2)) ) / (2 * feature_loss_1)
        else:
            fea_len_loss = 0
        if self.b_ciou:
            # ciou is better than L2
            loss_ciou = self.box_ciou(reg * valid_rect, gt_rect_mat * valid_rect)
            loss_ciou = (1 - loss_ciou) * valid_rect[:, :, :, 0]
            if self.reg_smooth > 0:
                #
                loss_mse = torch.pow((reg * valid_rect - gt_rect_mat * valid_rect), 2)
                loss_mse, _ = torch.max(loss_mse, dim=3)
                reg_pos = loss_mse - reg_valid[:,:,:,0]  #
                reg_pos[reg_pos < 0] = 0
                reg_pos[reg_pos > 0] = 1
                loss_ciou = loss_ciou * reg_pos
                loss_ciou = torch.sum(loss_ciou) / torch.sum(valid_rect[:, :, :, 0] * reg_pos + eps)
            else:
                loss_ciou = torch.sum(loss_ciou) / torch.sum(valid_rect[:, :, :, 0] + eps)
        else:
            loss_ciou = torch.pow((reg * valid_rect - gt_rect_mat * valid_rect), 2)
            reg_pos = loss_ciou > reg_valid  #
            loss_ciou = torch.sum(loss_ciou * reg_pos) / (torch.sum(gt_obj_mat) + eps)

        # If a rect's confidence is higher than the threshold and it's a successful detection,
        # the confidence loss of this rect will be ignored.
        pos_y = (obj <= (1 - self.smooth))
        pos_n = (obj >= self.neg_smooth)  #
        if self.b_bce:
            loss_conf1 = self.BCELoss(obj * gt_obj_mat * pos_y, gt_obj_mat * gt_obj_mat * pos_y)
            loss_conf2 = self.BCELoss(obj * gt_obj_n_mat * pos_n, gt_obj_mat * gt_obj_n_mat * pos_n)
        else:
            if self.smooth > 0:
                loss_conf1 = self.L2(obj * gt_obj_mat * pos_y, gt_obj_mat * gt_obj_mat * pos_y)
                loss_conf2 = self.L2(obj * gt_obj_n_mat * pos_n, gt_obj_mat * gt_obj_n_mat * pos_n)
            else:
                loss_conf1 = self.L2(obj * gt_obj_mat, gt_obj_mat * gt_obj_mat)
                loss_conf2 = self.L2(obj * gt_obj_n_mat, gt_obj_mat * gt_obj_n_mat)
        loss_conf = self.c1 * torch.sum(loss_conf1) / (torch.sum(gt_obj_mat) + eps) + \
                    self.c2 * torch.sum(loss_conf2) / (torch.sum(gt_obj_n_mat) + eps)

        #
        if self.b_fea_loss:
            if self.b_fea_len_loss:
                loss = loss_ciou * 0.1 + loss_conf + feature_loss * 0.1 + fea_len_loss * 0.05  #
            else:
                loss = loss_ciou * 0.1 + loss_conf + feature_loss * 0.1
        else:
            loss = loss_ciou * 0.1 + loss_conf
        '''
        For this special task, better ciou makes less sense than accuracy (accurate confidence).
        So coe of ciou is set as 0.1.
        
        '''
        # loss = loss_ciou + loss_conf
        # loss = loss_conf
        return loss, loss_ciou, loss_conf, feature_loss


    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)  # 避免置信度无解
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_ciou(self, b1, b2):
        """
        batch × h × w × 4
        ciou batch × h × w × 1
        """

        # codes here are adopted from others

        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        #
        intersect_mins = torch.max(b1_mins, b2_mins)  #
        intersect_maxes = torch.min(b1_maxes, b2_maxes)  #
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))  #
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  #
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]  #
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area  #
        iou = intersect_area / torch.clamp(union_area, min=1e-6)  #
        #
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)  #

        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)  #
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou