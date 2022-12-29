
# import cv2
# import numpy as np
# import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Data.MMRFF_SetBasicFun import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import PIL.Image as Image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class MMRFF_Dataset(Dataset):
    def __init__(self, path=None, b_gt_enable=True, grid=16, b_100=False, b_noise=0,
                 b_stretch=False, b_stretch_cut_min=0.01, b_stretch_cut_max=0.001, b_11pn=False,
                 b_gray_rand=0.0, Up_rand_tune=50, b_bad_pixel=0, b_rand_shift=False,
                 b_rand_grid_cut=0, b_cut_paste_mix=0, b_nonlinear=0,
                 b_flip_rand=0, b_rotate_rand=1, b_color_fill=True, fill_rand_level=15,
                 max_tar_num=10, b_train=False, batch_size=1, b_save=False, each_epoch_num=-1):

        assert (path is not None)
        assert(path is not None)

        self.b_save = b_save
        list_dir = os.listdir(path)  #
        list_dir.sort()
        img_files = list(filter(file_filter, list_dir))
        #
        if b_100:
            img_files = img_files[0:99]  # only for debug
        self.b_noise = b_noise  #

        self.data_len = img_files.__len__()   #
        self.b_gt_enable = b_gt_enable  #
        self.grid = grid  # 16×16

        self.b_stretch = b_stretch  #
        self.b_stretch_cut_min = b_stretch_cut_min  #
        self.b_stretch_cut_max = b_stretch_cut_max  #
        self.b_11pn = b_11pn  #

        self.b_gray_rand = b_gray_rand  #
        self.tune_count = 0  #
        self.Up_rand_tune = Up_rand_tune  #

        self.b_flip_rand = b_flip_rand  #
        self.b_rotate_rand = b_rotate_rand  #
        self.b_rand_shift = b_rand_shift  #
        self.b_rand_grid_cut = b_rand_grid_cut  #
        self.b_cut_paste_mix = b_cut_paste_mix  #
        if self.b_cut_paste_mix:
            self.paste_grid = np.zeros((grid_size, grid_size))
            self.paste_grid_r = 1  #
            self.paste_grid_c = 1
        else:
            self.paste_grid = None
        self.b_nonlinear = b_nonlinear  #

        self.b_color_fill = b_color_fill  #
        self.fill_rand_level = fill_rand_level  #
        self.batch_acc_count = 0  #
        self.max_tar_num = max_tar_num  #

        self.img_files = img_files  #
        self.root_path = path  #

        self.b_train = b_train  #

        self.b_bad_pixel = b_bad_pixel  #
        self.batch_size = batch_size  #

        self.bad_pixel_data = np.zeros([self.b_bad_pixel, 2])  #

        self.each_epoch_num = each_epoch_num
        if self.each_epoch_num == -1:
            self.each_epoch_num = self.data_len

    def __getitem__(self, index):

        img_name = os.path.join(self.root_path, self.img_files[index])  #
        try:
            img = np.array(Image.open(img_name))  #
        except Exception as e:
            img = np.array(Image.open(img_name))
        img = img.astype(np.float32)
        if len(img.shape) >= 3:
            img = img[:,:,0]
        img_h, img_w = img.shape
        if self.b_gt_enable:
            txt_name = img_name[0:len(img_name) - 4] + '.txt'  #
            ori_data, tar_num = MMRFF_Load_Truth(txt_name, img_h, img_w, self.max_tar_num)
        else:
            ori_data = None
            tar_num = 0

        if self.b_rand_grid_cut>0 or self.b_cut_paste_mix>0:
            valid_grid = get_bg_area(img, ori_data, tar_num)
        else:
            valid_grid = None

        if self.b_rand_grid_cut>0:
            #
            img = rand_cut_grid(img, ori_data, tar_num, valid_grid, aug_num=self.b_rand_grid_cut)

        if self.b_cut_paste_mix>0:
            img, grid_patch, g_r, g_c = rand_cut_mix(img, self.paste_grid, self.paste_grid_r, self.paste_grid_c, self.b_cut_paste_mix, valid_grid)
            self.paste_grid = grid_patch
            self.paste_grid_r = g_r
            self.paste_grid_c = g_c

        dst_img = np.zeros(img.shape)
        if self.b_rand_shift:
            #
            shift_x = int(np.random.rand() * 2 * grid_size - grid_size)
            shift_y = int(np.random.rand() * 2 * grid_size - grid_size)
            dst_x1, dst_x2, src_x1, src_x2 = get_rand_shift_info(shift_x, img_w)
            dst_y1, dst_y2, src_y1, src_y2 = get_rand_shift_info(shift_y, img_h)
            dst_img[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
            img = dst_img
            if tar_num > 0:
                ori_data[0:tar_num, 0] = ori_data[0:tar_num, 0] + shift_x
                ori_data[0:tar_num, 2] = ori_data[0:tar_num, 2] + shift_x
                ori_data[0:tar_num, 1] = ori_data[0:tar_num, 1] + shift_y
                ori_data[0:tar_num, 3] = ori_data[0:tar_num, 3] + shift_y
            for j in range(tar_num):
                if ori_data[j, 0] >= img_w or ori_data[j, 1] >= img_h or ori_data[j, 2] < 0 or ori_data[j, 3] < 0:
                    ori_data[j, :] = 0  # target out of side
                ori_data = MMRFF_Restrict_data(ori_data, tar_num, img_h, img_w)

        ori_img = img.copy()
        if self.b_stretch:
            max_v = np.max(img)
            min_v = np.min(img)
            img = (img - min_v) / (max_v - min_v)  #
            img[img > 1] = 1  #
            img[img <= 0] = 0  #
        else:
            img = img / 255

        #
        self.tune_count = self.tune_count + 1  #
        b_gray_rand = np.min([self.b_gray_rand, self.tune_count //
                              (self.Up_rand_tune + 0.000001) * self.b_gray_rand / 10])
        #
        if b_gray_rand > 0:
            img = MMRFF_Rand_Stretch(img, b_gray_rand)  #
        if self.b_nonlinear:
            img = MMRFF_img_rand_Nonlinear(img, self.b_nonlinear)

        if self.b_flip_rand > 0:
            # ori_data1 = ori_data.copy()
            img, ori_data = MMRFF_Flip_img_and_truth(img, ori_data, tar_num)

        for j in range(tar_num):
            if ori_data[j, 0] >= img_w or ori_data[j, 1] >= img_h or ori_data[j, 2] < 0 or ori_data[j, 3] < 0:
                ori_data[j, :] = 0  # target out of sid

        if self.b_noise > 0:
            rand_noise_coe = np.random.rand()
            if rand_noise_coe > 0.5:
                noise_map = np.random.normal(0, (rand_noise_coe-0.5)*2, img.shape)  #
                img = img + noise_map * self.b_noise
            img = np.clip(img, 0, 1)  #
            rand_gauss = np.random.rand()
            if rand_gauss > 0.5:
                img = cv2.GaussianBlur(img, (5, 5), (rand_gauss-0.5)*2)

        #
        if self.b_rotate_rand > 0:
            img[img <= eps] = 2*eps  #
            # codes here have been removed for some problems.

        if ori_data is None:
            # data error
            disp_str = 'data error'
            print(disp_str)
            img = np.ones(img.shape, dtype=np.float32) * np.random.rand()

        try:
            gt_rect, gt_pos, mat_rect, mat_obj, mat_obj_n = \
                MMRFF_Get_TargetRect(img, ori_data, tar_num, max_tar_num=self.max_tar_num, b_train=self.b_train)
        except Exception as e:
            gt_rect, gt_pos, mat_rect, mat_obj, mat_obj_n = \
                MMRFF_Get_TargetRect(img, ori_data, tar_num, max_tar_num=self.max_tar_num, b_train=self.b_train)

        # print(self.time_count_T)
        img = np.clip(img, 0, 1)  # -1  +1
        if self.b_11pn:
            img = img * 2 - 1
        img = np.array(img, dtype=np.float32)
        if self.b_train:
            return img, ori_data, gt_rect, gt_pos, mat_rect, mat_obj, mat_obj_n
        else:
            if self.b_save:
                return img, ori_data, gt_rect, gt_pos, self.img_files[index], ori_img
            else:
                return img, ori_data, gt_rect, gt_pos, self.img_files[index]  #

    def __len__(self):
        return self.each_epoch_num  #
