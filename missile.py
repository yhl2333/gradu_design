#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@file missile.py
@brief 导弹模型,导弹比例导引法
@version 0.1
@date 2022/9/12

@copyright Copyright (c) 2022
"""
#from matplotlib import pyplot as plt
import numpy as np
#import config

#from config import *


N = 4


class Missile:
    """空空导弹模型"""

    def __init__(self, num: int):
        self.num = num  # 编号
        self.ms_v = 400  # 导弹飞行速度大小
        # self.emitted = False  # 导弹是否已经发射
        # self.hit_tar = False  # 是否命中目标
        # self.sim_end = False  # 是否结束仿真
        # self.can_hit_tar = False  # 是否有能力命中目标,超过最大攻击距离就无法命中，低于不可逃逸距离就一定命中
        self.t = 0  # 仿真时间
        #self.t_list = []  # 仿真时间列表
        #self.max_t = 60  # 最大仿真时间
        self.dt = 0.25  # 比例导引仿真时间步长
        self.sim_ratio = int(0.25 / self.dt)  # 飞机仿真时间步长和导弹比例导引仿真步长的比
        self.sim_step = 0  # 仿真次数
        self.rm = np.array([0., 0., 0.])  # 导弹位置坐标
        self.vm = np.array([0., 0., 0.])  # 导弹速度
        self.rt = np.array([0., 0., 0.])  # 目标位置坐标
        self.vt = np.array([0., 0., 0.])  # 目标速度
        self.state_process = []  # 保存导弹飞行状态列表


    def reset(self):
        self.t = 0  # 仿真时间
        #self.t_list = []  # 仿真时间列表
        # self.emitted = False  # 导弹是否已经发射
        # self.hit_tar = False  # 是否命中目标
        # self.sim_end = False  # 是否结束仿真
        # self.can_hit_tar = False  # 是否有能力命中目标
        self.sim_step = 0  # 仿真时间
        self.rm = np.array([0., 0., 0.])  # 导弹位置坐标
        self.vm = np.array([0., 0., 0.])  # 导弹速度
        self.rt = np.array([0., 0., 0.])  # 目标位置坐标
        self.vt = np.array([0., 0., 0.])  # 目标速度
        self.state_process = []  # 保存导弹飞行状态列表



    @staticmethod
    def trans_state(state):
        """根据状态计算位置和速度，比例导引仿真过程需要敌我双方飞机的位置和三轴速度"""
        r = np.array(state[0:3])
        yaw, _, pitch = state[4:]
        v = state[3] * np.array([np.cos(pitch) * np.cos(yaw),
                                 np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        return r, v

    def emit_mis(self, enemy_state, ally_state):
        """发射导弹"""
        # if self.emitted:
        #     return
        # self.emitted = True
        # self.sim_end = False
        self.rt, self.vt = self.trans_state(enemy_state)
        self.rm, vm = self.trans_state(ally_state)
        #print(vm)
        vm1 = vm / np.linalg.norm(vm) * self.ms_v  # 导弹速度方向设置为发射时飞机的速度方向，大小为ms_v
        #print(vm1)
        r_m_t = self.rt - self.rm
        vm2 = r_m_t / np.linalg.norm(r_m_t) * self.ms_v  # 导弹速度方向设置为发射时敌我位置方向
        #self.vm = 0.2 * vm1 + 0.8 * vm2
        self.vm = 1 * vm1 + 0 * vm2
        #self.vm = 0.4 * vm1 + 0.6 * vm2
        self.t = 0
        #self.t_list.append(self.t)
        self.sim_step = 0
        state = np.concatenate([self.rt, self.vt, self.rm, self.vm])  # 目标状态和导弹状态
        self.state_process.append(state)  # 保存导弹飞行状态列表

    def p_guide_sim(self, enemy_state):
        """比例导引仿真"""
        # if not self.emitted:
        #     return
        # if self.sim_end:
        #     # print("sim_step = ", self.sim_step, "sim_time =", self.t)
        #     return
        self.rt, self.vt = self.trans_state(enemy_state)
        # if not self.can_hit_tar:
        #     self.rt += np.array([-2000, 2000, 2000])
        state = np.concatenate([self.rt, self.vt, self.rm, self.vm])  # 目标状态和导弹状态
        while True:
            k1 = self.dt * PGuidance.equ(self.t, state)
            k2 = self.dt * PGuidance.equ(self.t + 0.5 * self.dt, state + 0.5 * k1)
            k3 = self.dt * PGuidance.equ(self.t + 0.5 * self.dt, state + 0.5 * k2)
            k4 = self.dt * PGuidance.equ(self.t + self.dt, state + k3)
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.state_process.append(state)
            #self.t_list.append(self.t)
            # d = np.sqrt(np.square(state[0] - state[6])
            #             + np.square(state[1] - state[7])
            #             + np.square(state[2] - state[8]))
            self.sim_step += 1
            self.t += self.dt
            # if self.t > self.max_t:  # 导弹超过最大仿真时间仍未击中目标
            #     self.hit_tar = False
            #     self.rt, self.vt, self.rm, self.vm = state[0:3], state[3:6], state[6:9], state[9:]
            #     self.sim_end = True
            #     # print('超时未击中目标')
            #     break
            # if d < 10:  # 导弹击中目标
            #     self.hit_tar = True if self.can_hit_tar else False
            #     self.rt, self.vt, self.rm, self.vm = state[0:3], state[3:6], state[6:9], state[9:]
            #     self.sim_end = True
            #     # print(self.can_hit_tar, self.hit_tar)
            #     # if self.can_hit_tar and self.hit_tar:
            #     #     print('成功击中目标')
            #     # else:
            #     #     print('未击中目标')
            #     break
            if self.sim_step % self.sim_ratio == 0:
                self.rt, self.vt, self.rm, self.vm = state[0:3], state[3:6], state[6:9], state[9:]
                break

        self.vm = self.vm / np.linalg.norm(self.vm) * self.ms_v
        state = np.concatenate([self.rm, [self.ms_v], self.vm])
        #print(state)
        return state

class PGuidance:
    """比例导引法的辅助函数"""

    @staticmethod
    def equ(_, state):
        rt = np.array(state[0:3])
        vt = np.array(state[3:6])
        rm = np.array(state[6:9])
        vm = np.array(state[9:])
        am = PGuidance.p_guide(rt, vt, rm, vm)
        dotrt = vt
        dotvt = np.array([0, 0, 0])
        dotrm = vm
        dotvm = am
        out = np.concatenate((dotrt, dotvt, dotrm, dotvm), axis=0)
        return out

    @staticmethod
    def p_guide(rt, vt, rm, vm):
        r = rt - rm
        vr = vt - vm
        omega = np.cross(r, vr) / np.square(np.linalg.norm(r))
        vc = -np.linalg.norm(r * vr) * r / np.square(np.linalg.norm(r))
        am = N * np.cross(vc, omega)
        return am


# if __name__ == '__main__':
#     fighter_b = Fighter(kind='b')
#     fighter_r = Fighter(kind='r')
#     missile_r = Missile(num=0, target=fighter_b)
#     missile_r.reset()
#     missile_r.emit_mis(fighter_b.state, fighter_r.state)
#     fb_state = []
#     for i in range(100):
#         fighter_b.maneuver_discrete(action_id=0)
#         fb_state.append(fighter_b.state)
#         missile_r.p_guide_sim(fighter_b.state)
#         fighter_b.maneuver_discrete(action_id=4)
#         fb_state.append(fighter_b.state)
#         missile_r.p_guide_sim(fighter_b.state)
#
#     state_plt = np.concatenate(missile_r.state_process).reshape(-1, 12)
#     print(state_plt.shape)
#     print(missile_r.hit_tar)
#     print(missile_r.sim_step)
#     fig = plt.figure(figsize=(12, 7))
#
#     ax = plt.axes(projection='3d')
#     ax.plot(state_plt[:, 0], state_plt[:, 1], state_plt[:, 2], 'r-')
#     ax.plot(state_plt[:, 6], state_plt[:, 7], state_plt[:, 8], 'b-')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()

# 0.定常飞行； 1.加速； 2.减速； 3.左转弯； 4.右转弯； 5.拉起； 6.俯冲
