import time
import numpy as np
import torch
import cv2
import os


grid_size = 16
eps = 1e-10


# save res
def MMRFF_Save_Res(file_name, res, in_h=-1, in_w=-1, b_stretch=True, b_min0=True):
    if type(res) is not np.ndarray:
        res = np.array(res.detach().cpu())

    dim = res.ndim
    if dim == 3:
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        ch, _, _ = res.shape
        assert(ch == 1)
        cur_res = res[0, :, :]
    else:
        cur_res = res

    assert(len(cur_res.shape) == 2)

    if b_stretch:
        if b_min0:
            min_v = 0
        else:
            min_v = np.min(cur_res)
        max_v = np.max(cur_res)
        cur_res = (cur_res - min_v) / (max_v - min_v + eps)

    if in_h != -1 and in_w != -1:
        cur_res = cv2.resize(cur_res, (in_w, in_h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(file_name, cur_res * 255)


def MMRFF_int2str_keep(data, min_num=0):
    data_str = str(data)
    while data_str.__len__() < min_num:
        data_str = '0' + data_str
    return data_str


def MMRFF_batch_save_res(file_name, res, in_h=-1, in_w=-1, b_stretch=True, b_min0=True):
    if type(res) is not np.ndarray:
        res = np.array(res.detach().cpu())

    file_name = file_name.replace('.png', '').replace('jpg', '')

    ch, h, w = res.shape
    for i in range(ch):
        tmp = res[i, :, :]
        res_file_name = file_name + MMRFF_int2str_keep(i, 2) + '.png'
        MMRFF_Save_Res(res_file_name, tmp, in_h, in_w, b_stretch, b_min0)


def MMRFF_batch_save_fea_list(save_path, tail_path, tail_name, fea_list, in_h=-1, in_w=-1, b_stretch=True, b_min0=True):
    cur_save_path = os.path.join(save_path, tail_path)
    if not os.path.isdir(cur_save_path):
        os.mkdir(cur_save_path)

    for i, tmp_fea in enumerate(fea_list):
        tmp_fea_name = os.path.join(cur_save_path, tail_name + MMRFF_int2str_keep(i, 2))
        tmp_fea = tmp_fea[0, :, :, :]
        MMRFF_batch_save_res(tmp_fea_name, tmp_fea, in_h=in_h, in_w=in_w,  b_stretch=b_stretch, b_min0=b_min0)


def MMRFF_rect_toX1X2Y1Y2(rect):
    x1 = rect[0] - rect[2] / 2
    x2 = rect[0] + rect[2] / 2
    y1 = rect[1] - rect[3] / 2
    y2 = rect[1] + rect[3] / 2
    res = [x1, x2, y1, y2]
    res = np.round(res)
    return res


def MMRFF_rect_toX1X2Y1Y2_t(rect):
    x1 = rect[0] - rect[2] / 2
    x2 = rect[0] + rect[2] / 2
    y1 = rect[1] - rect[3] / 2
    y2 = rect[1] + rect[3] / 2
    res = torch.tensor([x1, x2, y1, y2])
    return res


def b_rect_overlap(rect1, rect2, conf1, conf2, fusion_style=2):
    r_x1 = np.max((rect1[0], rect2[0]))  # x1
    r_y1 = np.max((rect1[2], rect2[2]))  # y1
    r_x2 = np.min((rect1[1], rect2[1]))  # x2
    r_y2 = np.min((rect1[3], rect2[3]))  # y2

    b_overlap = True
    rect_res = rect2
    conf_res = conf2
    if r_x1 > r_x2 or r_y1 > r_y2:
        b_overlap = False
    if b_overlap:
        # IOU > 0
        if fusion_style == 1:
            r_x1 = np.min((rect1[0], rect2[0]))  # x1
            r_y1 = np.min((rect1[2], rect2[2]))  # y1
            r_x2 = np.max((rect1[1], rect2[1]))  # x2
            r_y2 = np.max((rect1[3], rect2[3]))  # y2
            rect_res = [r_x1, r_y1, r_x2, r_y2]  #
        elif fusion_style == 2:
            #
            if conf1 > conf2:
                rect_res = rect1
            else:
                rect_res = rect2
        else:
            rect_res = None
        conf_res = np.max((conf1, conf2))
    return b_overlap, rect_res, conf_res


def b_rect_overlap_t(rect1, rect2, conf1, conf2, fusion_style=2):
    r_x1 = torch.max(rect1[0], rect2[0])  # x1
    r_y1 = torch.max(rect1[2], rect2[2])  # y1
    r_x2 = torch.min(rect1[1], rect2[1])  # x2
    r_y2 = torch.min(rect1[3], rect2[3])  # y2

    b_overlap = True
    rect_res = rect2
    conf_res = conf2
    if r_x1 > r_x2 or r_y1 > r_y2:
        b_overlap = False
    if b_overlap:
        # IOU > 0
        if fusion_style == 1:
            r_x1 = torch.min(rect1[0], rect2[0])  # x1
            r_y1 = torch.min(rect1[2], rect2[2])  # y1
            r_x2 = torch.max(rect1[1], rect2[1])  # x2
            r_y2 = torch.max(rect1[3], rect2[3])  # y2
            rect_res = torch.tensor([r_x1, r_y1, r_x2, r_y2])  #
        elif fusion_style == 2:
            #
            if conf1 > conf2:
                rect_res = rect1
            else:
                rect_res = rect2
        else:
            rect_res = None
        conf_res = torch.max(conf1, conf2)
    return b_overlap, rect_res, conf_res


def check_list_same(d_list, c_list, cur_rect, cur_conf, fusion_style=2):
    b_overlap = False
    for i, rect in enumerate(d_list):
        b_overlap, rect_res, conf_res = b_rect_overlap(cur_rect, rect, cur_conf, c_list[i], fusion_style=fusion_style)
        if b_overlap:
            d_list[i] = rect_res
            c_list[i] = conf_res
            break

    if len(d_list) == 0 or not b_overlap:
        d_list.append(cur_rect)
        c_list.append(cur_conf)
    return d_list, c_list


def MMRFF_Get_LastRes_coe3(obj, reg, img_w, img_h, conf_th=0.5, fusion_style=2):
    if type(obj) is not np.ndarray:
        obj = np.array(obj.detach().cpu())
    if type(reg) is not np.ndarray:
        reg = np.array(reg.detach().cpu())
    in_h = img_h // grid_size
    in_w = img_w // grid_size

    detect_rect = list()  # x1 x2 y1 y2
    detect_conf = list()  #

    for i_h in range(in_h):
        for i_w in range(in_w):
            if obj[0, i_h, i_w] > conf_th:
                cur_rect = reg[:, i_h, i_w]  #
                cur_rect[2] = cur_rect[2] * grid_size * 3 - grid_size #  w
                cur_rect[3] = cur_rect[3] * grid_size * 3 - grid_size #  h
                cur_rect[0] = cur_rect[0] * grid_size + i_w * grid_size  # x
                cur_rect[1] = cur_rect[1] * grid_size + i_h * grid_size  # y
                cur_rect = MMRFF_rect_toX1X2Y1Y2(cur_rect)  # x1x2y1y2
                detect_rect, detect_conf = check_list_same(detect_rect, detect_conf, cur_rect, obj[0, i_h, i_w],
                                                           fusion_style=fusion_style)
    return detect_rect, detect_conf


def MMRFF_Get_LastRes(obj, reg, img_w, img_h, conf_th=0.5, fusion_style=2):
    if type(obj) is not np.ndarray:
        obj = np.array(obj.detach().cpu())
    if type(reg) is not np.ndarray:
        reg = np.array(reg.detach().cpu())
    in_h = img_h // grid_size
    in_w = img_w // grid_size

    detect_rect = list()  # x1 x2 y1 y2
    detect_conf = list()  #

    for i_h in range(in_h):
        for i_w in range(in_w):
            if obj[0, i_h, i_w] > conf_th:
                cur_rect = reg[:, i_h, i_w]  #
                cur_rect[2] = cur_rect[2] * grid_size  #  w
                cur_rect[3] = cur_rect[3] * grid_size  #  h
                cur_rect[0] = cur_rect[0] * grid_size + i_w * grid_size  # x
                cur_rect[1] = cur_rect[1] * grid_size + i_h * grid_size  # y
                cur_rect = MMRFF_rect_toX1X2Y1Y2(cur_rect)  # x1x2y1y2
                detect_rect, detect_conf = check_list_same(detect_rect, detect_conf, cur_rect, obj[0, i_h, i_w],
                                                           fusion_style=fusion_style)
    return detect_rect, detect_conf


def MMRFF_Get_LastRes_t_coe3(obj, reg, img_w, img_h, conf_th=0.5, fusion_style=2):
    p1 = torch.where(obj > conf_th)
    v_obj = obj[:, p1[1], p1[2]]
    v_reg = reg[:, p1[1], p1[2]]

    tar_num = v_obj.shape[1]
    detect_rect = list()  # x1 x2 y1 y2
    detect_conf = list()  #

    for i in range(tar_num):
        cur_rect = v_reg[:, i]
        cur_rect[2] = cur_rect[2] * grid_size * 3 - grid_size  #  w
        cur_rect[3] = cur_rect[3] * grid_size * 3 - grid_size #  h
        cur_rect[0] = cur_rect[0] * grid_size + p1[2][i] * grid_size  # x
        cur_rect[1] = cur_rect[1] * grid_size + p1[1][i] * grid_size  # y
        detect_rect, detect_conf = check_list_same_t(detect_rect, detect_conf, cur_rect, obj[0, p1[1][i], p1[2][i]],
                                                   fusion_style=fusion_style)
    return detect_rect, detect_conf


def MMRFF_Get_LastRes_t(obj, reg, img_w, img_h, conf_th=0.5, fusion_style=2):
    p1 = torch.where(obj > conf_th)
    v_obj = obj[:, p1[1], p1[2]]
    v_reg = reg[:, p1[1], p1[2]]

    tar_num = v_obj.shape[1]
    detect_rect = list()  # x1 x2 y1 y2
    detect_conf = list()  #

    for i in range(tar_num):
        cur_rect = v_reg[:, i]
        cur_rect[2] = cur_rect[2] * grid_size  #  w
        cur_rect[3] = cur_rect[3] * grid_size  #  h
        cur_rect[0] = cur_rect[0] * grid_size + p1[2][i] * grid_size  # x
        cur_rect[1] = cur_rect[1] * grid_size + p1[1][i] * grid_size  # y
        # 仅执行x1x2y1y2的模式
        x1 = cur_rect[0] - cur_rect[2] / 2
        y1 = cur_rect[1] - cur_rect[3] / 2
        x2 = x1 + cur_rect[2]
        y2 = y1 + cur_rect[3]
        cur_rect = torch.tensor([x1, x2, y1, y2])
        detect_rect, detect_conf = check_list_same_t(detect_rect, detect_conf, cur_rect, obj[0, p1[1][i], p1[2][i]],
                                                   fusion_style=fusion_style)
    return detect_rect, detect_conf


def check_list_same_t(d_list, c_list, cur_rect, cur_conf, fusion_style=2):
    b_overlap = False
    for i, rect in enumerate(d_list):
        # if overlapped
        b_overlap, rect_res, conf_res = b_rect_overlap_t(cur_rect, rect, cur_conf, c_list[i], fusion_style=fusion_style)
        if b_overlap:
            #  rect_res, conf_res cur_rect
            d_list[i] = rect_res
            c_list[i] = conf_res
            break

    if len(d_list) == 0 or not b_overlap:
        d_list.append(cur_rect)
        c_list.append(cur_conf)
    return d_list, c_list


def MMRFF_Get_IOU(de_x1, de_x2, de_y1, de_y2, gt_x1, gt_x2, gt_y1, gt_y2, b_more1=False):
    union_x1 = np.max([de_x1, gt_x1])
    union_y1 = np.max([de_y1, gt_y1])
    union_x2 = np.min([de_x2, gt_x2])
    union_y2 = np.min([de_y2, gt_y2])
    if union_x1 >= union_x2 or union_y1 >= union_y2:
        return 0
    if b_more1:
        union_area = (union_x2 - union_x1 + 1) * (union_y2 - union_y1 + 1)
        unit_s1 = (de_x2 - de_x1 + 1) * (de_y2 - de_y1 + 1)
        unit_s2 = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
    else:
        union_area = (union_x2 - union_x1) * (union_y2 - union_y1)
        unit_s1 = (de_x2 - de_x1) * (de_y2 - de_y1)
        unit_s2 = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    whole_area = unit_s1 + unit_s2 - union_area
    iou = float(union_area) / whole_area
    return iou


def MMRFF_convert_gt2x1x2y1y1(gt_rect, gt_pos):
    cx = (gt_rect[0] + gt_pos[0]) * grid_size  # grid_size=16
    cy = (gt_rect[1] + gt_pos[1]) * grid_size
    gt_w = gt_rect[2] * grid_size
    gt_h = gt_rect[3] * grid_size
    gt_x1 = cx - gt_w / 2
    gt_x2 = cx + gt_w / 2
    gt_y1 = cy - gt_h / 2
    gt_y2 = cy + gt_h / 2
    gt_x1 = round(gt_x1)  #
    gt_x2 = round(gt_x2)
    gt_y1 = round(gt_y1)
    gt_y2 = round(gt_y2)
    return cx, cy, gt_x1, gt_x2, gt_y1, gt_y2


def b_maxPower_in_d_rect(de_x1, de_x2, de_y1, de_y2, gt_x1, gt_x2, gt_y1, gt_y2, img):
    if type(img) is not np.ndarray:
        res = np.array(img.detach().cpu())

    part_img = img[gt_y1:gt_y2, gt_x1:gt_x2]
    max_v = part_img.max()
    max_count = 0
    max_in = 0
    for r in range(gt_y1, gt_y2):
        for c in range(gt_x1, gt_x2):
            dv = img[r, c] - max_v
            if abs(dv) < eps:
                max_count = max_count + 1
                if de_y1 <= r < de_y2 and de_x1 <= c < de_x2:
                    max_in = max_in + 1

    coe = max_in / max_count
    if coe >= 0.5:
        return True
    return False


def b_gt_overlap(gt_rect, gt_pos, d_rect, check_style=3, img=None):
    de_x1 = d_rect[0]
    de_x2 = d_rect[1]
    de_y1 = d_rect[2]
    de_y2 = d_rect[3]

    cx, cy, gt_x1, gt_x2, gt_y1, gt_y2 = MMRFF_convert_gt2x1x2y1y1(gt_rect, gt_pos)

    iou = MMRFF_Get_IOU(de_x1, de_x2, de_y1, de_y2, gt_x1, gt_x2, gt_y1, gt_y2)

    if check_style == 1:
        # AP3p25
        b1 = de_x1 <= cx < de_x2 and de_y1 <= cy < de_y2
        dx = (de_x1 + de_x2) / 2
        dy = (de_y1 + de_y2) / 2
        dis = (dx-cx)**2 + (dy-cy)**2
        b2 = dis <= 9.01
        if b1 and b2 and iou >= 0.25:
            return True
        else:
            return False
    elif check_style == 2:
        # AP50
        if iou >= 0.5:
            return True
        else:
            return False
    elif check_style == 3:
        # AP3p + pixel
        b1 = de_x1 <= cx < de_x2 and de_y1 <= cy < de_y2
        dx = (de_x1 + de_x2) / 2
        dy = (de_y1 + de_y2) / 2
        dis = (dx - cx) ** 2 + (dy - cy) ** 2
        b2 = dis <= 25.01
        if b1 and b2:
            if img is not None:
                #
                b_max_ok = b_maxPower_in_d_rect(de_x1, de_x2, de_y1, de_y2, gt_x1, gt_x2, gt_y1, gt_y2, img)
                return b_max_ok, dis  #
            else:
                return True  #
        else:
            return False
    elif check_style == 4:
        b1 = de_x1 <= cx < de_x2 and de_y1 <= cy < de_y2
        dx = (de_x1 + de_x2) / 2
        dy = (de_y1 + de_y2) / 2
        dis = (dx - cx) ** 2 + (dy - cy) ** 2
        b2 = dis <= 9.01
        if b1 and b2:
            return True
        else:
            return False
    elif check_style == 5:
        b1 = de_x1 <= cx < de_x2 and de_y1 <= cy < de_y2
        dx = (de_x1 + de_x2) / 2
        dy = (de_y1 + de_y2) / 2
        dis = (dx - cx) ** 2 + (dy - cy) ** 2
        b2 = dis <= 9.01
        if (b1 and b2) or iou >= 0.5:
            return True
        else:
            return False


def MMRFF_evaluate(gt_rect, gt_pos, detect_rect, detect_conf, img_w, img_h, check_set=(1, 2, 3, 4, 5)):
    if type(gt_rect) is not np.ndarray:
        gt_rect = np.array(gt_rect.detach().cpu())
    if type(gt_pos) is not np.ndarray:
        gt_pos = np.array(gt_pos.detach().cpu())

    tmp_valid_tar = np.where(gt_rect > 0)
    tmp_valid_tar = np.unique(tmp_valid_tar[0])
    tar_num = len(tmp_valid_tar)  #
    detect_num = detect_conf.__len__()  #

    detect_status = np.zeros([detect_num, 1+len(check_set)])  #

    #
    # gt_status = np.zeros([tar_num, 2])
    gt_status = np.zeros([tar_num, 1+len(check_set)])

    for j in range(tar_num):
        if gt_rect[j, 0] == -img_w and gt_rect[j, 1] == -img_h:
            break

        for k in range(detect_num):
            # Overlapped boxes have been removed before 'MMRFF_evaluate', confidence can be set here.
            detect_status[k, 0] = detect_conf[k]  # 20220504 the confidence should be set at first for the last sorting

            for c_i in range(len(check_set)):
                if detect_status[k, 1 + c_i] > 0:
                    continue
                if gt_status[j, 1 + c_i] > 0:
                    continue
                if b_gt_overlap(gt_rect[j, :], gt_pos[j, :], detect_rect[k], check_style=check_set[c_i]):
                    detect_status[k, 1 + c_i] = 1  # the detection is a correct detection
                    gt_status[j, 0] = detect_conf[k]  # can be removed
                    gt_status[j, 1 + c_i] = 1  # the target is picked out successfully
                else:
                    if detect_status[k, 1 + c_i] != 1 and not detect_status[k, 0]:
                        detect_status[k, 0] = detect_conf[k]  # can be removed

    for i in range(detect_num):
        detect_status[i, 0] = detect_conf[i]
    return detect_status, gt_status


def MMRFF_Show_Save_Res_Features(gt_rect, gt_pos, detect_rect, obj,
                              img, fea_list, res_list, glo_list,
                              save_path, file_name,
                              save_setting=(1, 0, 0, 0, 0),
                              b_gt=True, bshow=False):
    if bshow:
        win_ori = 'win'
        win_heat = 'heat'
    else:
        win_ori = None
        win_heat = None

    if type(img) is not np.ndarray:
        img = np.array(img.detach().cpu())
    if type(obj) is not np.ndarray:
        obj = np.array(obj.detach().cpu())
    if b_gt:
        if type(gt_rect) is not np.ndarray:
            gt_rect = np.array(gt_rect.detach().cpu())
        if type(gt_pos) is not np.ndarray:
            gt_pos = np.array(gt_pos.detach().cpu())
    if img.ndim == 3:
        img = img[0, :, :]
    img_h, img_w = img.shape

    cur_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    file_name = file_name[0:len(file_name) - 4]  # .png

    if save_setting[0] > 0:
        if b_gt:
            tar_num = gt_rect.shape[0]
            for j in range(tar_num):
                cur_gt_rect = gt_rect[j, :]
                if cur_gt_rect[0] == 0 and cur_gt_rect[1] == 0 and cur_gt_rect[2] == 0 and cur_gt_rect[3] == 0:
                    break
                cur_gt_pos = gt_pos[j, :]
                cx, cy, gt_x1, gt_x2, gt_y1, gt_y2 = MMRFF_convert_gt2x1x2y1y1(cur_gt_rect, cur_gt_pos)
                pt1 = (gt_x1, gt_y1)
                pt2 = (gt_x2 - 1, gt_y2 - 1)
                pt1 = np.array(pt1) - 2
                pt2 = np.array(pt2) + 2
                cv2.rectangle(cur_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 0, 255), thickness=1)

        for i in range(len(detect_rect)):
            cur_rect = detect_rect[i]
            pt1 = (cur_rect[0], cur_rect[2])
            pt2 = (cur_rect[1]-1, cur_rect[3]-1)
            pt1 = np.array(pt1) - 3
            pt2 = np.array(pt2) + 3
            cv2.rectangle(cur_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), thickness=1)
            if bshow:
                cv2.imshow(win_ori, cur_img)
                cv2.waitKey(1)

    if bshow:
        cv2.imshow(win_ori, cur_img)
        cv2.waitKey(1)
        cur_res = cv2.resize(obj[0, :, :], (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_GRAY2RGB)
        cv2.putText(cur_res, file_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(255, 0, 0))
        cv2.imshow(win_heat, cur_res)
        cv2.waitKey(1)
    else:
        result_file = os.path.join(save_path, file_name + '.png')
        cv2.imwrite(result_file, cur_img * 255)  # cur_img  0-1
    # heat map
    if save_setting[1] > 0:
        # obj 1/16
        sub_file_path = os.path.join(save_path, file_name + 'heat.png')
        MMRFF_Save_Res(sub_file_path, obj[0, :, :], in_h=img_h, in_w=img_w, b_stretch=False, b_min0=True)


    if save_setting[2] or save_setting[3] or save_setting[4]:
        sub_save_path = os.path.join(save_path, file_name)
        if not os.path.isdir(sub_save_path):
            os.mkdir(sub_save_path)
        MMRFF_batch_save_fea_list(sub_save_path, 'fea', 'fea', fea_list,
                               in_h=img_h, in_w=img_w, b_stretch=True, b_min0=True)
        MMRFF_batch_save_fea_list(sub_save_path, 'res', 'res', res_list,
                               in_h=img_h, in_w=img_w, b_stretch=True, b_min0=True)
        MMRFF_batch_save_fea_list(sub_save_path, 'glo', 'glo', glo_list,
                               in_h=img_h, in_w=img_w, b_stretch=True, b_min0=True)
        result_file = os.path.join(sub_save_path, file_name + '.png')
        cv2.imwrite(result_file, cur_img*255)  # cur_img  0-1
        sub_file_path = os.path.join(sub_save_path, file_name + 'heat.png')
        MMRFF_Save_Res(sub_file_path, obj[0, :, :], in_h=img_h, in_w=img_w, b_stretch=False, b_min0=True)


def save_obj_features(obj_bg_files, obj_tar_files, obj_features, gt_pos, gt_rect):
    obj_features = obj_features.squeeze(0)
    fea_len, h, w = obj_features.shape
    bg_count = 0
    while bg_count < 4:
        row = int(np.random.rand() * h)
        col = int(np.random.rand() * w)
        b_bg = True
        for i in range(10):
            x = gt_pos[i, 0]
            y = gt_pos[i, 1]
            if gt_rect[i, 2] == 0 or gt_rect[i, 3] == 0:
                break  #
            if x == col and y == row:
                #
                b_bg = False
                break
        if b_bg:
            #
            bg_count = bg_count + 1
            features = obj_features[:, row, col]
            features = torch.sigmoid(features)
            features = np.array(features.detach().cpu())
            with open(obj_bg_files, 'a+') as f:
                for j in range(fea_len):
                    value = round(features[j] * 1000)
                    f.write(str(value))
                    f.write('\t')
                f.write('\n')

    for i in range(10):
        x = int(gt_pos[i, 0])
        y = int(gt_pos[i, 1])
        if gt_rect[i, 2] == 0 or gt_rect[i, 3] == 0:
            break  #
        features = obj_features[:, y, x]
        features = torch.sigmoid(features)
        features = np.array(features.detach().cpu())
        with open(obj_tar_files, 'a+') as f:
            for j in range(fea_len):
                value = round(features[j] * 1000)
                f.write(str(value))
                f.write('\t')
            f.write('\n')










