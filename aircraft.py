import numpy as np
from config import *
import math

V_MAX = 250
V_MIN = 150
ROLL_MAX = 78.5 * deg2rad
PITCH_MAX = 45 * deg2rad

Z_INIT = 6000
Z_MAX = 200000
Z_MIN = 1000
V_INIT = 200
V_M_INIT = 400
HEADING_INIT = 0
ROLL_INIT = 0
PITCH_INIT = 0

DIST_INIT_MIN = 2000
DIST_INIT_MAX = 100000  # 最大初始距离

dt = 0.25  # seconds

# action = [roll, nx, ny]
# 1.定常飞行； 2.加速； 3.减速； 4.左转弯； 5.右转弯； 6.拉起； 7.俯冲
action_lists = [[0, 0, 1], [0, 2, 1], [0, -2, 1],
                [-ROLL_MAX, 0, 5], [ROLL_MAX, 0, 5], [0, 0, 5], [0, 0, -5]]


class Aircraft(object):
    def __init__(self, state, config=None):
        if config is None:
            config = [ROLL_MAX, V_MAX, V_MIN, PITCH_MAX]
        roll_max, v_max, v_min, pitch_max = config
        x, y, z, v, heading, roll, pitch = state
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        self.heading = heading
        self.roll = roll
        self.pitch = pitch
        self.state = state

        self.roll_max = roll_max
        self.pitch_max = pitch_max
        self.v_max = v_max
        self.v_min = v_min

    def reset(self, state):
        """
        设置飞机状态
        :param state: 状态参量，7元组
        """
        x, y, z, v, heading, roll, pitch = state
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        self.heading = heading
        self.roll = roll
        self.pitch = pitch
        self.state = state

    def maneuver(self, action_id):
        """
        根据机动动作执行一个时间步的飞机状态更新
        :param action_id:
        :return: 新的状态
        """
        roll, nx, ny = action_lists[action_id]
        x, y, z, v, heading, _, pitch = self.state

        # print("[roll, nx, ny]:{0}, {1}, {2}".format(roll, nx, ny))

        # 位置及姿态更新微分方程
        dot_v = g * (nx - math.sin(pitch))
        dot_pitch = g * (ny * math.cos(roll) - math.cos(pitch)) / v
        dot_heading = g * ny * math.sin(roll) / (v * math.cos(pitch))

        dot_x = v * math.cos(pitch) * math.cos(heading)
        dot_y = v * math.cos(pitch) * math.sin(heading)
        dot_z = v * math.sin(pitch)

        v += dot_v * dt
        pitch += dot_pitch * dt
        heading += dot_heading * dt

        x += dot_x * dt
        y += dot_y * dt
        z += dot_z * dt

        v = self._clamp(self.v_min, self.v_max, v)
        pitch = self._clamp(-self.pitch_max, self.pitch_max, pitch)
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi

        self.reset([x, y, z, v, heading, roll, pitch])

        return self.state

    def _clamp(self, min_val, max_val, val):
        if val > max_val:
            val = max_val
        elif val < min_val:
            val = min_val
        return val
