import math
from config import *

AA_MAX = 180 * deg2rad
ATA_MAX = 180 * deg2rad


# 计算我方对敌方的角度优势、高度优势及速度优势
# 参数均为归一化之前的飞机状态参数
def angle_adv(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b = state
    return 0.5*(_angle_adv(aspect_angle, antenna_train_angle) - _angle_adv(pi - antenna_train_angle, pi - aspect_angle))

def angle_adv_half(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b = state
    if aspect_angle <= AA_MAX:
        adv_r = math.exp(-aspect_angle / AA_MAX)
    else:
        adv_r = 0
    if antenna_train_angle >= pi - ATA_MAX:
        adv_b = - math.exp(-(pi - antenna_train_angle) / ATA_MAX)
    else:
        adv_b = 0
    return adv_r + adv_b
def angle_adv_linear(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b = state
    return 1 - (aspect_angle + antenna_train_angle) / pi

def height_adv(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b = state
    if z_r - z_b > 1000:
        return 1.0
    elif z_r - z_b < -1000:
        return -1.0
    else:
        return (z_r - z_b) / 1000.0


def velocity_adv(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b = state
    return (v_r + 150 - v_b) / 250.0

def _angle_adv(aa, ata):
    if aa <= AA_MAX:
        adv_rr = math.exp(-aa / AA_MAX)
    else:
        adv_rr = 0

    if ata > ATA_MAX:
        # adv_b = - math.exp(-(pi - antenna_train_angle) / ATA_MAX)
        adv_rb = 0
    else:
        adv_rb = math.exp(-ata / ATA_MAX)
    return adv_rr + adv_rb