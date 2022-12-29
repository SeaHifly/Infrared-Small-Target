import math
import os
import cv2
import numpy as np
import PIL.Image as Image
import random
from tqdm import tqdm


grid_size = 16
eps = 1e-10
gray_unit = 1/255
unit_mat = np.float32([0, 0, 1])


def file_filter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp']:
        return True
    else:
        return False


def MMRFF_Infrared_Img_Read(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    return img


def MMRFF_Load_Truth(txt_name, img_h, img_w, max_tar_num=10):
    if os.path.isfile(txt_name):
        with open(txt_name, 'r') as f:
            data = f.read()
            data = data.split()
            tar_num = len(data) // 5
    else:
        tar_num = 0

    ori_data = np.ones([max_tar_num, 4]) * (-1)

    for i in range(tar_num):
        #
        cx = float(data[i * 5 + 1]) * img_w - 1
        cy = float(data[i * 5 + 2]) * img_h - 1
        iw = float(data[i * 5 + 3]) * img_w
        ih = float(data[i * 5 + 4]) * img_h
        x1 = cx - iw / 2
        x2 = cx + iw / 2
        y1 = cy - ih / 2
        y2 = cy + ih / 2
        x1 = round(x1)
        x2 = round(x2)
        y1 = round(y1)
        y2 = round(y2)

        x1 = np.max([x1, 0])
        y1 = np.max([y1, 0])
        x2 = np.min([x2, img_w])
        y2 = np.min([y2, img_h])
        ori_data[i, 0] = x1
        ori_data[i, 1] = y1
        ori_data[i, 2] = x2
        ori_data[i, 3] = y2

    ori_data = np.round(ori_data)
    return ori_data, tar_num


def MMRFF_Solve_img_hist_and_Th(img, min_cut=0.01, max_cut=0.001):
    im_hist = np.zeros([256, 1])
    img_h, img_w = img.shape
    for r in range(img_h):
        for c in range(img_w):
            g = img[r, c]
            g = int(g)
            im_hist[g] = im_hist[g] + 1

    whole_count = img_h * img_w
    count_min = whole_count * min_cut
    count_max = whole_count * max_cut
    th_min_v = 0
    tmp_min_count = 0
    for i in range(256):
        tmp_min_count = tmp_min_count + im_hist[i]
        if tmp_min_count >= count_min:
            th_min_v = np.max([i - 1, 0])  #
            break

    th_max_v = 255
    tmp_max_count = 0
    for i in range(256):
        tmp_max_count = tmp_max_count + im_hist[255 - i]
        if tmp_max_count >= count_max:
            th_max_v = np.min([255 - i + 1, 255])  #
            break

    return th_min_v, th_max_v, im_hist


# data aug -- img hist random
def MMRFF_Rand_Stretch(img, b_gray_rand):
    v1 = np.random.rand()
    v2 = np.random.rand()
    v1 = v1 * 2 * b_gray_rand - b_gray_rand
    v2 = v2 * 2 * b_gray_rand - b_gray_rand
    v2 = 1 - v2
    img = (img - v1) / (v2 - v1)
    img = np.clip(img, 0, 1)  #
    return img


# data aug -- img hist nonlinear
def MMRFF_img_rand_Nonlinear(img, coe=0.1):
    rand_c = (np.random.rand() - 0.5) / 0.5 * coe
    img = np.power(img, 1 + rand_c)
    return img


# data aug -- img flip and so on
def MMRFF_Flip_img_and_truth(img, ori_data, tar_num):
    img_h, img_w = img.shape
    flip_rand_count = int(np.random.rand() * 4)  #
    if (flip_rand_count & int(0b01)) > 0:
        img = np.flip(img, axis=1)  #
        for i in range(tar_num):
            tmp_x1 = img_w - ori_data[i, 0]  #
            tmp_x2 = img_w - ori_data[i, 2]  #
            ori_data[i, 0] = tmp_x2  # x1' = img_w - x2
            ori_data[i, 2] = tmp_x1  # x2’ = img_w - x1
    if (flip_rand_count & int(0b10)) > 0:
        img = np.flip(img, axis=0)  #
        for i in range(tar_num):
            tmp_y1 = img_h - ori_data[i, 1]  #
            tmp_y2 = img_h - ori_data[i, 3]
            ori_data[i, 1] = tmp_y2  # y1’ = img_h - y2
            ori_data[i, 3] = tmp_y1  # y2' = img_w - y1
    return img.copy(), ori_data


# x1y1x2y2   solve IOU
def MMRFF_Get_IOU_x1y1x2y2(rect1, rect2):
    x1 = np.max([rect1[0], rect2[0]])  #
    y1 = np.max([rect1[1], rect2[1]])
    x2 = np.min([rect1[2], rect2[2]])
    y2 = np.min([rect1[3], rect2[3]])
    if x1 >= x2 or y1 >= y2:
        return 0
    s0 = (x2-x1)*(y2-y1)  #
    s1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    s2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    iou = s0 / (s1 + s2 - s0)
    return iou


# get last training data
def MMRFF_Get_TargetRect(img, rect, tar_num, max_tar_num=10, b_train=True):
    #  rect x1y1x2y2
    # rect  max_tar_num * 4

    img_h, img_w = img.shape
    gt_pos = np.zeros([max_tar_num, 2])  #
    gt_rect = np.zeros([max_tar_num, 4])  #

    in_h = int(img_h / grid_size)
    in_w = int(img_w / grid_size)

    if b_train:
        mat_rect = np.zeros([in_h, in_w, 4], dtype=np.float32)
        mat_obj = np.zeros([in_h, in_w, 1], dtype=np.float32)
        mat_obj_n = np.ones([in_h, in_w, 1], dtype=np.float32)
    else:
        mat_rect = None
        mat_obj = None
        mat_obj_n = None

    for i in range(tar_num):
        if rect[i, 0] == 0 or rect[i, 1] == 0:
            continue
        if rect[i, 0] != -1:
            # x1y1x2y2
            cx = (rect[i, 0] + rect[i, 2]) / 2  #
            cy = (rect[i, 1] + rect[i, 3]) / 2
            cw = rect[i, 2] - rect[i, 0]  #
            ch = rect[i, 3] - rect[i, 1]

            x_i = cx // grid_size  #  x
            x_f = cx - x_i * grid_size  #   x

            y_i = cy // grid_size  #   y
            y_f = cy - y_i * grid_size  #   y

            gt_pos[i, 0] = x_i
            gt_pos[i, 1] = y_i
            gt_rect[i, 0] = x_f / grid_size
            gt_rect[i, 1] = y_f / grid_size
            gt_rect[i, 2] = cw / grid_size
            gt_rect[i, 3] = ch / grid_size

            if b_train:
                x_i = int(round(x_i))
                y_i = int(round(y_i))
                mat_rect[y_i, x_i, :] = gt_rect[i, :]  #
                mat_obj[y_i, x_i, 0] = 1  #
                #  -1 0 1
                for t_r in range(np.max([y_i-1, 0]), np.min([y_i+2, in_h])):
                    for t_c in range(np.max([x_i-1, 0]), np.min([x_i+2, in_w])):
                        mat_obj_n[t_r, t_c, 0] = 0

    return gt_rect, gt_pos, mat_rect, mat_obj, mat_obj_n


# load DATA from img_files  data num : min_cache_len ,-1 denotes all
def MMRFF_load_data(img_files, min_cache_len, root_path, b_gt, max_tar_num):
    cache_img = list()  #
    cache_ori = list()  #
    cache_tar = list()  #
    bar = tqdm(range(min_cache_len))
    for i in bar:
        bar.set_description("data loading")
        img_name = os.path.join(root_path, img_files[i])  #
        img = np.array(Image.open(img_name))  #
        img = img.astype(np.float32)
        img_s = img.shape
        if len(img_s) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cache_img.append(img)
        # 如果存在真值
        if b_gt:
            img_h, img_w = img.shape
            txt_name = img_name[0:len(img_name) - 4] + '.txt'  # txt名称
            ori_data, tar_num = MMRFF_Load_Truth(txt_name, img_h, img_w, max_tar_num)
            '''
            ori_data x1x2y1y2
            '''
            cache_ori.append(ori_data)
            cache_tar.append(tar_num)

    return cache_img, cache_ori, cache_tar


# load dataset
def MMRFF_load_dataset(img_files, data_len, min_cache_len, root_path, b_gt, max_tar_num):

    if data_len == min_cache_len:
        cache_img, cache_ori, cache_tar = MMRFF_load_data(img_files, min_cache_len, root_path,
                                                       b_gt, max_tar_num)
    else:
        copy_list = img_files.copy()
        random.shuffle(copy_list)
        cache_img, cache_ori, cache_tar = MMRFF_load_data(copy_list, min_cache_len, root_path,
                                                       b_gt, max_tar_num)
    return cache_img, cache_ori, cache_tar


def MMRFF_RestrictBoarder(in_data, max_data):
    in_data = np.min((in_data, max_data))  #
    out_data = np.max((in_data, 0))  # >= 0
    return out_data


def MMRFF_Restrict_data(ori_data, tar_num, img_h, img_w):
    for j in range(tar_num):
        if ori_data[j, 0] >= img_w or ori_data[j, 1] >= img_h or ori_data[j, 2] < 0 or ori_data[j, 3] < 0:
            ori_data[j, :] = 0  # target out of side
        ori_data[j, 0] = MMRFF_RestrictBoarder(ori_data[j, 0], img_w)
        ori_data[j, 1] = MMRFF_RestrictBoarder(ori_data[j, 1], img_h)
        ori_data[j, 2] = MMRFF_RestrictBoarder(ori_data[j, 2], img_w)
        ori_data[j, 3] = MMRFF_RestrictBoarder(ori_data[j, 3], img_h)
    return ori_data


def get_rand_shift_info(shift_num, max_size):
    if shift_num < 0:
        dst_x1 = 0
        dst_x2 = max_size + shift_num
        src_x1 = - shift_num
        src_x2 = max_size
    else:
        dst_x1 = shift_num
        dst_x2 = max_size
        src_x1 = 0
        src_x2 = max_size - shift_num
    return dst_x1, dst_x2, src_x1, src_x2


def avoid_boarder(value, max_v):
    res = np.min((value, max_v - 1))
    res = res // grid_size
    # return int(res)
    res1 = int(res)
    res2 = np.min((res1+1, max_v-1))
    res3 = np.max((res1-1, 0))
    return res3, res2+1


def get_bg_area(img, ori_data, tar_num):
    img_h, img_w = img.shape
    in_h = img_h // grid_size
    in_w = img_w // grid_size
    valid_grid = np.ones((in_h, in_w))
    for i in range(tar_num):
        try:  #
            min_r, max_r = avoid_boarder(ori_data[i, 1], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 0], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
            min_r, max_r = avoid_boarder(ori_data[i, 1], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 2], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
            min_r, max_r = avoid_boarder(ori_data[i, 3], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 0], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
            min_r, max_r = avoid_boarder(ori_data[i, 3], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 2], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
        except Exception as e:
            min_r, max_r = avoid_boarder(ori_data[i, 1], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 0], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
            min_r, max_r = avoid_boarder(ori_data[i, 1], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 2], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
            min_r, max_r = avoid_boarder(ori_data[i, 3], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 0], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
            min_r, max_r = avoid_boarder(ori_data[i, 3], img_h)
            min_c, max_c = avoid_boarder(ori_data[i, 2], img_w)
            valid_grid[min_r:max_r, min_c:max_c] = 0
    return valid_grid


# data aug -- grid patch
def rand_cut_grid(img, ori_data, tar_num, valid_grid, aug_num=3):
    img_h, img_w = img.shape
    in_h = img_h // grid_size
    in_w = img_w // grid_size
    aug_count = 0
    while True:
        rand_h = int(np.random.rand() * in_h)
        rand_w = int(np.random.rand() * in_w)
        if valid_grid[rand_h, rand_w] > 0:
            img[rand_h * grid_size:rand_h * grid_size + grid_size, rand_w * grid_size:rand_w * grid_size + grid_size] \
                = int(np.random.rand() * 255)
            aug_count = aug_count + 1
        if aug_count >= aug_num:
            break
    return img


# data aug -- grid patch paste
def rand_cut_mix(img, paste_grid, grid_r, grid_c, rand_size, valid_grid):
    img_h, img_w = img.shape
    in_h = img_h // grid_size
    in_w = img_w // grid_size
    while True:
        rand_h = int(np.random.rand() * rand_size) + 1
        rand_w = int(np.random.rand() * rand_size) + 1
        if (in_h - rand_h) < 0 or (in_w - rand_w) < 0 or (in_h - grid_r) < 0 or (in_w - grid_c)<0:
            continue  #
        rand_y1 = int(np.random.rand() * (in_h - np.max((rand_h, grid_r))))  #
        rand_x1 = int(np.random.rand() * (in_w - np.max((rand_w, grid_c))))  #
        b_suc = True
        for r in range(np.max((rand_h, grid_r))):
            for c in range(np.max((rand_w, grid_c))):
                if valid_grid[r + rand_y1, c + rand_x1] < 1:
                    #
                    b_suc = False
                    break
            if not b_suc:
                break

        if b_suc:
            grid_patch = img[rand_y1 * grid_size:rand_y1 * grid_size + grid_size * rand_h,
                         rand_x1 * grid_size:rand_x1 * grid_size + grid_size * rand_w].copy()
            img[rand_y1 * grid_size:rand_y1 * grid_size + grid_size * grid_r,
            rand_x1 * grid_size:rand_x1 * grid_size + grid_size * grid_c] = paste_grid.copy()
            grid_r = rand_h
            grid_c = rand_w
            break  #

    return img, grid_patch, grid_r, grid_c


# data aug size
def img_size_aug(img, ori_data, tar_num, img_coe):
    if abs(img_coe - 1) < 0.01:
        return img, ori_data, tar_num

    if tar_num > 0 and img_coe > 1:
        #
        max_tar_size = 0
        for i in range(tar_num):
            w = ori_data[i, 2] - ori_data[i, 0]
            h = ori_data[i, 3] - ori_data[i, 1]
            max_tar_size = np.max((max_tar_size, w))
            max_tar_size = np.max((max_tar_size, h))
        assert(max_tar_size > 0)
        img_coe = np.min((img_coe, grid_size / max_tar_size))  #

    img_h, img_w = img.shape
    res_h = int(img_h * img_coe)
    res_w = int(img_w * img_coe)
    for i in range(tar_num):
        ori_data[i, :] = np.round(ori_data[i, :] * img_coe)

    if img_coe > 1:
        #
        img_larger = cv2.resize(img, (res_w, res_h))
        rand_x1 = int(np.random.rand() * (res_w - img_w))
        rand_y1 = int(np.random.rand() * (res_h - img_h))
        res_img = img_larger[rand_y1:rand_y1+img_h, rand_x1:rand_y1+img_w].copy()
        #
        for i in range(tar_num):
            ori_data[i, 0] = ori_data[i, 0] - rand_x1
            ori_data[i, 2] = ori_data[i, 2] - rand_x1
            ori_data[i, 1] = ori_data[i, 1] - rand_y1
            ori_data[i, 3] = ori_data[i, 3] - rand_y1
            if ori_data[i, 0] >= img_w or ori_data[i, 1] >= img_h or ori_data[i, 2] < 0 or ori_data[i, 3] < 0:
                #
                ori_data[i, :] = -1
            else:
                #
                ori_data[i, 0] = np.max((ori_data[i, 0], 0))
                ori_data[i, 2] = np.max((ori_data[i, 2], 0))
                ori_data[i, 1] = np.min((ori_data[i, 1], img_w))
                ori_data[i, 3] = np.min((ori_data[i, 3], img_h))
    else:
        img_smaller = cv2.resize(img, (res_w, res_h))
        rand_x1 = int(np.random.rand() * (img_w - res_w))
        rand_y1 = int(np.random.rand() * (img_h - res_h))
        res_img = np.zeros((img_h, img_w))
        try:
            res_img[rand_y1:rand_y1+res_h, rand_x1:rand_x1+res_w] = img_smaller.copy()
        except Exception as e:
            res_img[rand_y1:rand_y1 + res_h, rand_x1:rand_x1 + res_w] = img_smaller.copy()
        #
        for i in range(tar_num):
            ori_data[i, 0] = ori_data[i, 0] + rand_x1
            ori_data[i, 2] = ori_data[i, 2] + rand_x1
            ori_data[i, 1] = ori_data[i, 1] + rand_y1
            ori_data[i, 3] = ori_data[i, 3] + rand_y1

    return res_img, ori_data, tar_num




