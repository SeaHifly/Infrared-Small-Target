import time
from collections import OrderedDict

import numpy as np
import torch
import math

from MMRFF_BasicOperate import *
from Data.MMRFF_SmallInfrared_Set8 import MMRFF_Dataset
from torch.utils.data import DataLoader
from Nets.MMRFF_SmallSE10_FL import YoloBody
from Loss.MMRFF_Small_Loss5_FL import YOLOLoss
from Result.MMRFF_Small_Res_F3 import *
import os
import datetime
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# It's revised from a version of YOLOv3
def compute_ap(recall, precision):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 计算AP值
    return ap, mrec, mpre


def parse_args():
    parser = argparse.ArgumentParser(description='MMRFF-Net')
    parser.add_argument('--th_v', type=float, default=0.05,
                        help='in the paper, 0.5 for speed evaluatiaon, 0.05 for AP3p25 evaluation for all regression based methods')
    parser.add_argument('--test_path', type=str, default=r'E:\XH_YOLO_SMALL\IST-E\Test',
                        help='the path of images to evaluate')
    parser.add_argument('--model_path', type=str, default='LD_SE_LG0511_Set8_FL1',
                        help='the pkl path')
    parser.add_argument('--model_id', type=int, default=750,
                        help='the id of the pkl to load ')
    parser.add_argument('--b_cuda', type=bool, default=True, help='')
    parser.add_argument('--b_show', type=bool, default=False, help='if the results are shown during the testing')
    parser.add_argument('--b_save', type=bool, default=False, help='if the results are saved')
    parser.add_argument('--b_gt_enable', type=bool, default=True, help='if GT files exist or not')
    parser.add_argument('--b_local', type=bool, default=True, help='LFs')
    parser.add_argument('--b_global', type=bool, default=True, help='GFs')
    parser.add_argument('--B1', type=int, default=2, help='CH')
    parser.add_argument('--B2', type=int, default=4, help='CH')
    parser.add_argument('--B3', type=int, default=4, help='CH')
    parser.add_argument('--B4', type=int, default=8, help='CH')
    parser.add_argument('--b_BLock_SE', type=bool, default=True, help='b_BLock_SE')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    '''   
    all settings are in consitent with the ones of training  
    only 'test_path' , 'model_path' , 'model_id' must be modified.
    '''

    b_train = False  # False for testing
    th_v = args.th_v  # usually 0.9 in detection and speed evalutation, 0.05 in evaluating AP3p25 or all methods
    # both yolo-series and ours in the paper. If you only set it as 0.5, the AP3p25 score will decline to 95~96
    # which is still much higher than other Seg-based methods in our experiments.

    # load test subset
    test_path = args.test_path  # data path
    model_path = args.model_path  # exact model path in 'model_results', a file path which will be crated automatically for each training
    model_id = args.model_id  # model id to load, int
    # a model's full path can be presented as: os.path.join(model_load_path, 'model_' + str(model_id) + '.pkl')

    '''
    AP3p25(in our paper): 
    'LD_SE_LG0511_SCL'
        96.96  IST-A ; 96.7 IST-B   
        set LD_SE_LG0511_SCL =  'LD_SE_LG0511_SCL',
        set model_id = 750,
        then you can load the pre-trained model.  
    More data augmentation are added in this version.    
    The latest experimental results of this released codes (before it's released) is shown below:   
    '20220525223909'  
    750epoch A96.68 B97.92  
    900epoch A96.65 B96.96
    20220527105257  trained with pre-loaded model from '20220525223909' at 750epoch
    900epoch A96.58 B97.00
    20220527110049 lower label smooth  label_smooth=neg_smooth=0.01
    750epoch A96.94 B96.65
    900epoch A96.86 B92.25
    20220527232155  reg_smooth=0.1  It seems that reg_smooth is a valuable training strategy
    Anyway, label values are inaccurate. 
    360epoch A95.63 B96.34 
    '''

    b_shuffle = False  # False for most cases.
    if b_shuffle:
        b_save_100 = True
    else:
        b_save_100 = False
    bGet_fea = False  # If it's True, all feature maps will be saved.

    b_save = args.b_save  # If it's True, all detection results will be saved in a file path in 'test_res_path'
    b_gt_enable = args.b_gt_enable  # If you are testing a image without gt, set it False.
    b_loss_count = False
    b_noise = False  # data aug
    root_path = os.path.abspath(os.curdir)
    model_save_path = check_and_create_path(root_path, 'model_results')
    model_load_path = os.path.join(model_save_path, model_path)
    test_res_path = create_time_path(model_load_path)

    b_cuda = args.b_cuda
    b_show = args.b_show  # show img or not during the testing

    # all settings are consitent with the ones of training
    b_100 = False
    b_global = args.b_global
    b_local = args.b_local
    Act_Setting = 0
    b5 = False
    ch_setting = (args.B1, args.B2, args.B3, args.B4, 1)
    obj_head = np.array((1, 1, 1, 1, 1))
    reg_head = np.array((1, 1, 1, 1, 1))
    b_left_BN = False
    bMaxAct = False
    b_Hsigmoid = False
    b_fea_decoupled = False
    b_BLock_SE = args.b_BLock_SE

    fusion_style = 2

    batch_size = 1
    each_epoch_num = -1

    b_stretch = True

    if device == 'cpu':
        b_cuda = False

    assert (model_load_path is not None)
    # create file path for potential saving
    test_save_path = os.path.join(test_res_path, 'test_res')
    if not os.path.isdir(test_save_path):
        os.mkdir(test_save_path)

    TestDataset = MMRFF_Dataset(path=test_path, b_gt_enable=True, grid=grid_size, b_100=b_100, b_noise=b_noise,
                             b_stretch=b_stretch, b_stretch_cut_min=0.01, b_stretch_cut_max=0.001, b_11pn=True,
                             b_gray_rand=0, Up_rand_tune=0, b_bad_pixel=0,
                             b_flip_rand=0, b_rotate_rand=0, b_color_fill=True, fill_rand_level=15,
                             max_tar_num=10, b_train=b_train, batch_size=batch_size,
                             b_save=b_save or b_show,  each_epoch_num=each_epoch_num)

    TestLoader = DataLoader(TestDataset, batch_size, drop_last=True, shuffle=b_shuffle, num_workers=0)

    Net = YoloBody(Act_Setting=Act_Setting, ch_setting=ch_setting,
                   b_global=b_global, b_local=b_local, b5=b5, b_left_BN=b_left_BN, bMaxAct=bMaxAct,
                   b_Hsigmoid=b_Hsigmoid, obj_head=obj_head, reg_head=reg_head,
                   b_BLock_SE=b_BLock_SE, b_cuda=b_cuda)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = YOLOLoss(label_smooth=0, b_cuda=b_cuda)
    if b_cuda:
        Net = Net.to(device)
        criterion = criterion.to(device)

    model_path = os.path.join(model_load_path, 'model_' + str(model_id) + '.pkl')
    if os.path.isfile(model_path) and model_id > 0:
        if not b_cuda:
            model_state = torch.load(model_path, map_location='cpu')
        else:
            model_state = torch.load(model_path)
        Net.load_state_dict(model_state['model_state_dict'])  # load the model
        print(model_path, '--model load suc')
    else:
        raise Exception('model load error')
    paras = list(Net.named_parameters())
    print(paras[0])

    show_num = len(TestLoader) / 20
    show_num = np.min((show_num, 100))
    show_num = int(show_num)
    best_consume_time = 1

    file_name = datetime.datetime.now()
    file_name = str(file_name)
    file_name = file_name.replace('-', '').replace(' ', '').replace(':', '').replace('.', '')
    test_record_file = os.path.join(test_res_path, str(model_id) + 'eval'  + '.txt')

    with open(test_record_file, 'w') as f:
        f.write('model' + str(model_id) + '\r\n')

    with torch.no_grad():
        t_sum = 0
        status_de = None
        status_gt = None

        cur_show_time = 0

        Net.eval()
        for t_num, all_data in enumerate(TestLoader):
            if b_save_100 and t_num>100:
                break
            if t_num % show_num == 0:
                print(str(t_num) + '/' + str(TestLoader.__len__()) + '---' +
                      str(t_sum / (t_num + 0.0000001)) + 's' +
                      '----' + str(t_num / (t_sum + 0.0000001)), '--------',
                      str(show_num / (cur_show_time + 0.0000001)) +
                      '--best_fps:' + str(1 / (best_consume_time + 0.0000001)))
                cur_show_time = 0

            if b_gt_enable:
                img = all_data[0]
                gt_rect = all_data[2]
                gt_pos = all_data[3]
            else:
                img = all_data[0]
                gt_rect = None
                gt_pos = None
                mat_rect = None
                mat_obj = None
                mat_obj_n = None

            if b_cuda:
                img = img.to(device)
                if b_gt_enable:
                    gt_pos = gt_pos.to(device)
                    gt_rect = gt_rect.to(device)

            img_name = all_data[4]
            if b_save or b_show:
                ori_img = all_data[5]
            else:
                ori_img = None

            _, img_h, img_w = img.shape

            in_h = np.ceil(img_h / grid_size) * grid_size
            in_w = np.ceil(img_w / grid_size) * grid_size
            in_img = torch.zeros((1, int(in_h), int(in_w)))
            in_img[:, 0:img_h, 0:img_w] = img
            if b_cuda:
                in_img = in_img.cuda()

            ##############################################################################################
            t1 = time.time()
            all_res = Net(in_img, bGet_fea=bGet_fea)
            detect_rect, detect_conf = MMRFF_Get_LastRes_t(all_res[0][0, :, :, :], all_res[1][0, :, :, :],
                                                      img_w, img_h, conf_th=th_v, fusion_style=fusion_style)
            t2 = time.time()
            ##############################################################################################
            obj = all_res[0]
            reg = all_res[1]

            # evaluating is based on numpy
            for dec_i in range(len(detect_rect)):
                detect_rect[dec_i] = np.array(detect_rect[dec_i].detach().cpu())
                detect_conf[dec_i] = np.array(detect_conf[dec_i].detach().cpu())

            t_cur = t2 - t1
            t_sum = t_sum + t_cur

            cur_show_time = cur_show_time + t_cur
            if best_consume_time > t_cur:
                best_consume_time = t_cur

            img = img.squeeze(dim=0)
            obj = obj.squeeze(dim=0)
            reg = reg.squeeze(dim=0)
            img_name = img_name[0]

            if b_gt_enable:
                gt_rect = gt_rect[0, :, :]
                gt_pos = gt_pos[0, :, :]
                d_s, g_s = MMRFF_evaluate(gt_rect, gt_pos, detect_rect,
                                       detect_conf, img_w, img_h, check_set=(1, 2, 3, 4, 5))
            else:
                d_s = None
                g_s = None

            if b_save and bGet_fea:
                fea_list = all_res[2]
                res_list = all_res[3]
                glo_list = all_res[4]

                ori_img = ori_img / 255
                MMRFF_Show_Save_Res_Features(gt_rect, gt_pos, detect_rect, obj,
                                          ori_img, fea_list, res_list, glo_list,
                                          test_save_path, img_name,
                                          save_setting=(1, 1, 1, 1, 1),
                                          b_gt=b_gt_enable, bshow=b_show)
            elif b_save:
                ori_img = ori_img / 255
                MMRFF_Show_Save_Res_Features(gt_rect, gt_pos, detect_rect, obj,
                                          ori_img, None, None, None,
                                          test_save_path, img_name,
                                          save_setting=(1, 1, 0, 0, 0),
                                          b_gt=b_gt_enable, bshow=b_show)
            elif b_show:
                ori_img = ori_img / 255
                MMRFF_Show_Save_Res_Features(gt_rect, gt_pos, detect_rect, obj,
                                          ori_img, None, None, None,
                                          test_save_path, img_name,
                                          save_setting=(1, 0, 0, 0, 0),
                                          b_gt=b_gt_enable, bshow=b_show)

            if g_s is not None:
                if status_gt is None:
                    status_gt = g_s
                else:
                    status_gt = np.concatenate((status_gt, g_s), axis=0)
            if d_s is not None:
                if status_de is None:
                    status_de = d_s
                else:
                    status_de = np.concatenate((status_de, d_s), axis=0)
                if detect_rect is not None:
                    for tmp_rect in detect_rect:
                        detect_pixel = (tmp_rect[1] - tmp_rect[0]) * (tmp_rect[3] - tmp_rect[2])

        if status_de is not None:
            eval_num = status_de.shape[1]
            len_d = status_de.shape[0]
            len_g = status_gt.shape[0]
            eval_res = np.zeros([eval_num, 3])
            sort_pos = np.argsort(status_de[:, 0])
            status_de_sort = status_de[sort_pos, :]

            for eval_i in range(1, eval_num):
                pre_conf = status_de_sort[:, eval_i]
                pre_conf = pre_conf[::-1]

                fpc = (1 - pre_conf).cumsum()
                tpc = pre_conf.cumsum()

                precision_curve = tpc / (tpc + fpc)
                recall_curve = tpc / (len_g + 1e-16)
                ap, mrec, mpre = compute_ap(recall_curve, precision_curve)

                recall = np.sum(pre_conf) / (len_g + eps)  # pre_conf 1 for suc detection while 0 for failure
                acc = np.sum(pre_conf) / (len_d + eps)

                eval_res[eval_i, 0] = ap
                eval_res[eval_i, 1] = recall
                eval_res[eval_i, 2] = acc

                # AP curve saving
                AP_file = os.path.join(test_res_path, 'MMRFF-' + str(eval_i) + '-ap.txt')
                unit_ap_data = np.concatenate(
                    (np.expand_dims(precision_curve, axis=0), np.expand_dims(recall_curve, axis=0)))
                np.savetxt(AP_file, unit_ap_data, fmt='%f', delimiter='\t')

            with open(test_record_file, 'a+') as f:
                # f.write(str(acc) + '\t' + str(recall) + '\t')
                # f.write('ap:' + str(ap) + '\r\n')
                for eval_i in range(eval_num):
                    data_str = str(eval_res[eval_i, 0]) + '\t' + str(eval_res[eval_i, 1]) + '\t' + str(eval_res[eval_i, 2]) + '\r\n'
                    f.write(data_str)

            print('model_load_epoch:' + str(model_id))
            print('th_v' + str(th_v))
            print('ap3p25:' + str(eval_res[1, 0]))
            print('acc:' + str(eval_res[1, 2]) + '------recall:' + str(eval_res[1, 1]))
            print('ap50:' + str(eval_res[2, 0]))
            print('acc:' + str(eval_res[2, 2]) + '------recall:' + str(eval_res[2, 1]))
            print('ap5p_max:' + str(eval_res[3, 0]))
            print('acc:' + str(eval_res[3, 2]) + '------recall:' + str(eval_res[3, 1]))
            print('ap3p:' + str(eval_res[4, 0]))
            print('acc:' + str(eval_res[4, 2]) + '------recall:' + str(eval_res[4, 1]))
            print('ap3p or AP50 :' + str(eval_res[5, 0]))
            print('acc:' + str(eval_res[5, 2]) + '------recall:' + str(eval_res[5, 1]))
        print('fps:' + str(TestLoader.__len__() / t_sum))
        print('max_fps:' + str(1 / (best_consume_time + 0.0000001)))
