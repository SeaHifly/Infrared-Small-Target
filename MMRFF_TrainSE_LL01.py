import numpy as np
import torch
import torch.optim as optim
from MMRFF_BasicOperate import *
from Data.MMRFF_SmallInfrared_Set8 import MMRFF_Dataset
from torch.utils.data import DataLoader
from Nets.MMRFF_SmallSE10_FL import YoloBody

from Loss.MMRFF_Small_Loss5_FL import YOLOLoss
from Result.MMRFF_Small_Res_F3 import *
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

eps = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(description='MMRFF-Net')
    parser.add_argument('--th_v', type=float, default=0.5,
                        help='in the paper, 0.5 for speed evaluatiaon, 0.05 for AP3p25 evaluation for all regression based methods')
    parser.add_argument('--train_path', type=str, default=r'E:\XH_YOLO_SMALL\IST-E\train',
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
    parser.add_argument('--epochs', type=int, default=780, help='Set as you like, 300 is OK')
    parser.add_argument('--batch_size', type=int, default=50, help='256 for RTX2080Ti')
    parser.add_argument('--b_100', type=bool, default=False,
                        help='Only 100 images will be load during the training if it is true. It is for debugging')
    parser.add_argument('--b_SCL', type=bool, default=True,
                        help='It seems that it is not effective loss function.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    '''
    It's an official codes of a version of MMRFF-Net.
    But something may be different from the version in our paper because of more modifications and debugging before it's released.
    Default settings are in high consistent with the version of the paper.
    
    The paper hasn't been accpeted by any journal up to 2022/12/31. 
    If anyone want to take the dataset for your own research, we'll upload a version to arxiv so that one can cite it.
    
    Many settings which need more evaluating are kept in this version such as feature decoupling, 
    regression label smooth strategy and so on.  
    
    Something like data aug are not mentioned in our paper.
    Before SCL is applied, data augs which are adopted by us can increase performance obivously.
    But in our last few experiments, there are few difference between the model trained with some augs and the model trained without some augs.
    Take care: If all augs are cut off, the performance will drop down. We just can't confirm the more useful aug parts.
    All augs that used to increase the performance in our memory are kept in this version. 
    Anyway, they can enrich the combinations of backgrounds and targets.
    '''
    args = parse_args()

    b_train = True  # True: training. False: testing
    # train_path = '/home/user/XH_Small_YOLO/IST-E/train'
    train_path = args.train_path
    # path creating
    root_path = os.path.abspath(os.curdir)
    model_save_path = check_and_create_path(root_path, 'model_results')
    model_save_time_path = create_time_path(model_save_path)

    b_cuda = True  #
    b_show = False  # usually False during training.

    b_100 = args.b_100  # only for debugging
    b_global = args.b_global  # if global features are adopted
    b_local = args.b_local   # if local features are adopted
    Act_Setting = 0  # 0:relu 1:relu6 2:swish
    b5 = False  # if fifth block is adopted.
    # The fifth block won't increase ap3p25 in IST-A. But in our last experiment, it increase ap3p25 by about 1.8% in IST-B
    ch_setting = (args.B1, args.B2, args.B3, args.B4, 1)  # ch setting. You can set it as you like.

    obj_head = np.array((1, 1, 1, 1, 1))  # still in testing, we are testing whether features can be decoupled to reduce more computation for cpu
    reg_head = np.array((1, 1, 1, 1, 1))  # still in testing, it isn't involved in the paper

    reg_smooth = 0.25  # It's a debatable label smooth method for training now.
    # We wish the regression won't be over-fitting due to inaccurate label by setting this.
    # However, its contribution need more experiments. (AP3p25 score declines a lot when it's 0.5)

    b_extend_center_Bias = 0  #

    b_left_BN = False  # not all layers are followed by a BN
    bMaxAct = False  #
    b_Hsigmoid = False
    b_BLock_SE = args.b_BLock_SE  # SE module
    img_w = 384  # img size
    img_h = 288  #

    b_shuffle = True  # data aug
    b_noise = 0  # data aug
    b_rand_grid_cut = 8  # data aug, number of grid patches to cut off during the training
    b_cut_paste_mix = 4  # data aug, max num of grid cells to cut and paste. Equal Max PIXEL NUMBER = 4 * grid_size = 32
    b_nonlinear = 0.05  # data aug, it need more investigation. image hist nonlinear

    b_fea_loss = args.b_SCL  # self-contrast loss , cos loss

    fusion_style = 2  # only the box of which the confidence is higher will be kept

    batch_size = args.batch_size  # batch size, 256 for 2080ti
    each_epoch_num = -1  # num of data you want to load for each epoch during training, -1 denotes all data.

    th_v = args.th_v  # confidence th, usually 0.05 for evaluating AP3p25 while 0.9 for testing speed.

    b_stretch = True  # stretch the img
    start_epoch = 0  # start epoch, If it's not zero,
    # potential existed model in './model_results' will be loaded before training.

    epochs = args.epochs  # end epoch num. We usually set it as 900.
    # Last few saved models won't vary a obvious margin in evaluation.
    # Indeed, we find that models after 400epochs score very close AP3p25 in most cases.

    epochs = epochs + 1
    learn_rate_ori = 0.00001 * batch_size  # Larger setting is not welcomed here in our work, maybe you can take a try.

    if device == 'cpu':
        b_cuda = False  # training on CPU

    if b_100:
        if batch_size > 10:
            batch_size = 10

    b_gt_enable = True  # must be True for training
    '''
    Many data augmentation methods are adopted here.
    Some have been proved useful. Some have been cut off.
    Maybe we will update it in the future.
    '''
    TrainDataset = MMRFF_Dataset(path=train_path, b_gt_enable=True, grid=grid_size, b_100=b_100, b_noise=b_noise,
                              b_stretch=b_stretch, b_stretch_cut_min=0.01, b_stretch_cut_max=0.001, b_11pn=True,
                              b_gray_rand=0.25, Up_rand_tune=50, b_bad_pixel=0, b_rand_shift=True,
                              b_rand_grid_cut=b_rand_grid_cut, b_cut_paste_mix=b_cut_paste_mix, b_nonlinear=b_nonlinear,
                              b_flip_rand=1, b_rotate_rand=0, b_color_fill=True, fill_rand_level=15,
                              max_tar_num=10, b_train=b_train, batch_size=batch_size, each_epoch_num=each_epoch_num)
    '''
    As for IST-A and IST-B, there are no more than 5 targets in single image though we set max_tar_num as 10.
    '''

    TrainLoader = DataLoader(TrainDataset, batch_size, drop_last=True, shuffle=b_shuffle, num_workers=0)  # num_workers=16
    print('Data Load Succ')

    Net = YoloBody(Act_Setting=Act_Setting, ch_setting=ch_setting,
                   b_global=b_global, b_local=b_local, b5=b5, b_left_BN=b_left_BN, bMaxAct=bMaxAct,
                   b_Hsigmoid=b_Hsigmoid, obj_head=obj_head, reg_head=reg_head,
                   b_BLock_SE=b_BLock_SE, b_cuda=b_cuda)
    Net = Net.to(device)

    criterion = YOLOLoss(in_h=img_h, in_w=img_w, stride=grid_size, label_smooth=0.01,
                         neg_smooth=0.01, reg_smooth=reg_smooth, b_ciou=True,
                         b_cuda=b_cuda, b_bce=True, b_fea_loss=b_fea_loss, c1=0.1, c2=10)
    '''
    For single image detection, higher c2 means higher accuracy 
    but lower recall in our previous work before SCL (b_fea_loss) is adopted.   
    Something may be different due to the contribution of SCL, though we keep a higher c2 before the codes are released.
    '''
    if b_cuda:
        criterion = criterion.cuda()

    optimizer = optim.Adam(Net.parameters(), lr=0.01)  # init LR

    model_path = os.path.join(model_save_path, 'model_' + str(start_epoch) + '.pkl')
    if os.path.isfile(model_path) and start_epoch > 0:
        model_state = torch.load(model_path)
        Net.load_state_dict(model_state['model_state_dict'])
        optimizer.load_state_dict(model_state['optimizer_state_dict'])
        print('load existed model')

    train_record_file = os.path.join(model_save_time_path, 'train_record.txt')
    if start_epoch == 0:
        with open(train_record_file, 'w') as f:
            f.write('sum_loss_all\t' + 'loss_ciou\t' + 'loss_conf\r\n')

    test_record_file = os.path.join(model_save_time_path, 'test_record.txt')
    if start_epoch == 0:
        with open(test_record_file, 'w') as f:
            f.write('sum_loss_all\t' + 'loss_ciou\t' + 'loss_conf\r\n')

    train_batch_num = TrainLoader.__len__()

    if b_extend_center_Bias > 0:
        extend_m = torch.ones([1, 1, 1, 1], dtype=torch.float32)

    time1 = time.time()

    for epoch in range(start_epoch+1, epochs):
        Net.train()
        learn_rate = learn_rate_ori
        lr = adjust_learning_rate(optimizer, epoch, epochs, start_lr=learn_rate)  # cos decay for lr

        sum_loss = 0  #
        sum_loss_ciou = 0
        sum_loss_conf = 0
        sum_loss_feas = 0

        status_de = None  #
        status_gt = None

        for t_num, (img, ori_data, gt_rect, gt_pos, mat_rect, mat_obj, mat_obj_n) in enumerate(TrainLoader):
            ##############################################################################
            img = img.to(device)
            optimizer.zero_grad()
            #
            all_res = Net(img, bGet_fea=False)  # batch × 1+4 × 18 × 24
            obj = all_res[0]
            reg = all_res[1]
            obj_features = all_res[2]  #
            # obj, reg, _, _, _, _, _, _ = Net(img)  # batch × 1+4 × 18 × 24
            # def forward(self, obj, reg, gt_rect_mat, gt_obj_mat, gt_obj_n_mat):
            # loss
            loss, loss_ciou, loss_conf, feature_loss = criterion(obj, reg, mat_rect, mat_obj, mat_obj_n, obj_features)

            loss.backward()  #
            optimizer.step()
            ##############################################################################

            time2 = time.time()
            #
            time_str = count_time_progress(t_num, epoch, start_epoch, epochs, train_batch_num, time1, time2)
            print('epoch:', epoch, 'PRO:', t_num, '/', len(TrainLoader) ,'loss:', loss.data, loss_ciou.data, loss_conf.data, feature_loss.data,
                  learn_rate / learn_rate_ori, lr, time_str)
            #
            sum_loss = sum_loss + loss
            sum_loss_ciou = sum_loss_ciou + loss_ciou
            sum_loss_conf = sum_loss_conf + loss_conf
            sum_loss_feas = sum_loss_feas + feature_loss
            #
            if (t_num > 1 and epoch % 30 > 0) or epoch < 700:
                continue
            batch_size, img_h, img_w = img.shape

            for tmp_i in range(batch_size):
                img_num = tmp_i + batch_size * t_num
                if img_num > 30:
                    continue

                detect_rect, detect_conf = MMRFF_Get_LastRes(obj[tmp_i, :, :, :], reg[tmp_i, :, :, :],
                                                          img_w, img_h, conf_th=th_v, fusion_style=fusion_style)
                d_s, g_s = MMRFF_evaluate(gt_rect[tmp_i, :, :], gt_pos[tmp_i, :,:], detect_rect,
                                       detect_conf, img_w, img_h, check_set=(1, 2, 3, 4, 5))
                MMRFF_Show_Save_Res_Features(gt_rect[tmp_i, :, :], gt_pos[tmp_i, :, :], detect_rect, obj,
                                          img[tmp_i, :, :], None, None, None,
                                          model_save_time_path, int2str(img_num) + '.png',
                                          save_setting=(1, 0, 0, 0, 0),
                                          b_gt=True, bshow=b_show)

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

        if status_de is not None:
            len_d = len(status_de)
            len_g = len(status_gt)
            detect_res_only = status_de[:, 1]
            gt_res_only = status_gt[:, 1]
            recall = np.sum(gt_res_only) / (len_g + eps)
            acc = np.sum(detect_res_only) / (len_d + eps)
        else:
            acc = 0
            recall = 0

        sum_loss = sum_loss / train_batch_num
        sum_loss_ciou = sum_loss_ciou / train_batch_num
        sum_loss_conf = sum_loss_conf / train_batch_num
        sum_loss_feas = sum_loss_feas / train_batch_num
        with open(train_record_file, 'a+') as f:
            f.write(str(epoch) + '\t' +
                    str(np.array(sum_loss.data.detach().cpu())) + '\t' + str(np.array(sum_loss_ciou.data.detach().cpu())) + '\t' +
                    str(np.array(sum_loss_conf.data.detach().cpu())) + '\t' + str(np.array(sum_loss_feas.data.detach().cpu())) + '\t'
                    + str(acc) + '\t' +
                    str(recall) + '\r\n')
        print('epoch:', epoch, 'loss:', sum_loss.data, sum_loss_ciou.data, sum_loss_conf.data, sum_loss_feas.data,
              learn_rate / learn_rate_ori, lr, acc, recall)

        if epoch % 30 == 0:
            Net.eval()
            model_name = os.path.join(model_save_time_path, 'model_' + str(epoch) + '.pkl')
            # torch.save(Net.state_dict(), model_name)  # Net.load_state_dict(torch.load(model_name))
            torch.save({
                'epoch': epoch,
                'model_state_dict': Net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_name)

    print(model_save_time_path)





















