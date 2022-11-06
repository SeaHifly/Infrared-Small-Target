import math
import os
import time
import numpy as np


eps = 1e-10


# lr
def adjust_learning_rate(opt, e, whole_e, start_lr=0.00001):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = start_lr * (1+math.cos(float(e)/whole_e*math.pi))
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr


#
def check_and_create_path(root_path, sub_name):
    sub_path = os.path.join(root_path, sub_name)
    if not os.path.isdir(sub_path):
        os.mkdir(sub_path)
    return sub_path


#
def create_time_path(root_path):
    #
    localtime = time.localtime(time.time())

    str_time_year = str(localtime.tm_year)

    str_time_month = str(localtime.tm_mon)
    if len(str_time_month) < 2:
        str_time_month = '0' + str_time_month

    str_time_day = str(localtime.tm_mday)
    if len(str_time_day) < 2:
        str_time_day = '0' + str_time_day

    str_time_hour = str(localtime.tm_hour)
    if len(str_time_hour) < 2:
        str_time_hour = '0' + str_time_hour

    str_time_minute = str(localtime.tm_min)
    if len(str_time_minute) < 2:
        str_time_minute = '0' + str_time_minute

    str_time_sec = str(localtime.tm_sec)
    if len(str_time_sec) < 2:
        str_time_sec = '0' + str_time_sec

    str_time_name = str_time_year + str_time_month + str_time_day + str_time_hour + str_time_minute + str_time_sec
    str_time_path = os.path.join(root_path, str_time_name)
    if not os.path.isdir(str_time_path):
        os.mkdir(str_time_path)
    return str_time_path


#
def count_time_progress(t_num, epoch, start_epoch, epochs, train_len, time1, time2):
    # TrainLoader.__len__() == train_len
    past_count = t_num + (epoch - start_epoch - 1) * train_len + 1
    left_count = (epochs - start_epoch - 1) * train_len - past_count
    past_time = (time2 - time1)
    left_time = past_time / (past_count + eps) * left_count
    past_h = past_time // 3600
    past_m = (past_time - past_h * 3600) // 60
    past_s = past_time - past_h * 3600 - past_m * 60
    past_h = np.int(past_h)
    past_m = np.int(past_m)
    past_s = np.int(past_s)
    left_h = left_time // 3600
    left_m = (left_time - left_h * 3600) // 60
    left_s = left_time - left_h * 3600 - left_m * 60
    left_h = np.int(left_h)
    left_m = np.int(left_m)
    left_s = np.int(left_s)
    time_str = 'past_time:' + str(past_h) + 'h' + str(past_m) + 'm' + str(past_s) + 's' + \
               'left_time:' + str(left_h) + 'h' + str(left_m) + 'm' + str(left_s) + 's'
    return time_str


def int2str(num, keep=6):
    res = str(num)
    while len(res)<keep:
        res = '0' + res
    return res
