from aircraft import *
import random
import advantage
from cache import *

class CombatEnv(object):
    def __init__(self, state_r=None, state_b=None):
        self.theta = None  # 敌方无人机相对我方的初始方位角（2D）
        if state_r is None:
            state_r = self._state_initialize(rand=False)
        if state_b is None:
            state_b = self._state_initialize(rand=True)
        self.aircraft_r = Aircraft(state_r)
        self.aircraft_b = Aircraft(state_b)

        # 虚拟对抗的敌机，用于做和真实无人机相同的动作
        # 敌机策略生成方法：敌机搜索7个动作，选取及时收益最大的动作
        # self.virtual_aircraft_b = Aircraft(state_b)

        # distance, aspect_angle, antenna_train_angle,  z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b
        # 状态表示为：距离，AA角，ATA角....
        self.state = []
        self.action_dim = 7
        self.state_dim = 11
        self.done = False
        self.total_steps = 0
        self.cache = Cache()


    # 初始化敌我无人机初始状态
    def _state_initialize(self, rand=False):
        # 引入theta为了使两机初始状态相对
        if self.theta is None:
            self.theta = random.uniform(-pi, pi)  # 敌方无人机相对我方的方位角（2D）
        if rand is False:
            x = 0
            y = 0
            z = Z_INIT
            v = V_INIT
            heading = self.theta
            roll = ROLL_INIT
            pitch = PITCH_INIT
            state = [x, y, z, v, heading, roll, pitch]
        else:
            if self.theta >= 0:
                heading = self.theta - pi
            else:
                heading = pi + self.theta
            distance_from_r = random.uniform(0.8 * DIST_INIT_MAX, DIST_INIT_MAX)  # 初始距离
            x = distance_from_r * math.cos(self.theta)
            y = distance_from_r * math.sin(self.theta)
            z = Z_INIT
            v = V_INIT
            roll = ROLL_INIT
            pitch = PITCH_INIT
            state = [x, y, z, v, heading, roll, pitch]
        return state

    def reset(self):
        """
        初始化环境，敌我无人机状态初始化
        :return: 初状态
        """
        self.done = False
        self.total_steps = 0
        self.cache.clear()

        state_r = self._state_initialize(rand=False)
        state_b = self._state_initialize(rand=True)

        self.aircraft_r.reset(state_r)
        self.aircraft_b.reset(state_b)
        self.cache.push_r_state(state_r)
        self.cache.push_b_state(state_b)

        # self.virtual_aircraft_b.reset(state_b)

        state = self._situation(self.aircraft_r, self.aircraft_b)
        state_norm = self._normalize(state)
        self.state = state_norm
        return self.state

    # 态势评估，由敌我无人机状态解算出距离、威胁角等
    def _situation(self, aircraft_r, aircraft_b):
        x_r, y_r, z_r, v_r, heading_r, roll_r, pitch_r = aircraft_r.state
        x_b, y_b, z_b, v_b, heading_b, roll_b, pitch_b = aircraft_b.state

        # 距离向量
        vector_d = np.array([x_b - x_r, y_b - y_r, z_b - z_r])
        # 敌我无人机的速度向量
        vector_vr = np.array([math.cos(pitch_r) * math.cos(heading_r),
                              math.sin(heading_r) * math.cos(pitch_r), math.sin(pitch_r)])
        vector_vb = np.array([math.cos(pitch_b) * math.cos(heading_b),
                               math.sin(heading_b) * math.cos(pitch_b), math.sin(pitch_b)])
        # AA角和ATA角计算，向量夹角
        aspect_angle = self._cal_angle(vector_vr, vector_d)
        antenna_train_angle = self._cal_angle(vector_vb, vector_d)
        # print("AA:{0}, ATA:{1}".format(aspect_angle, antenna_train_angle))

        distance = np.sqrt(np.sum(vector_d * vector_d))
        return [distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b]

    def _cal_angle(self, vector_vr, vector_d):
        dot_product = np.dot(vector_vr, vector_d)
        d_norm = np.sqrt(np.sum(vector_d * vector_d))
        angle = np.arccos(dot_product / (d_norm + 1e-5))
        return angle

    # 状态归一化，防止差异化过大
    def _normalize(self, state):
        distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b = state

        distance = (distance - 6000) / 4000.0
        aspect_angle = aspect_angle / pi
        antenna_train_angle = antenna_train_angle / pi
        v_r = (v_r - 250) / 50
        v_b = (v_b - 250) / 50
        z_r = (z_r - Z_MIN) / (Z_MAX - Z_MIN)
        z_b = (z_b - Z_MIN) / (Z_MAX - Z_MIN)
        pitch_r = pitch_r / PITCH_MAX
        pitch_b = pitch_b / PITCH_MAX
        roll_r = roll_r / ROLL_MAX
        roll_b = roll_b / ROLL_MAX

        return [distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b]

    # state是未归一化的！
    def _cal_reward(self, state, save=True):
        angle_reward = advantage.angle_adv(state)
        height_reward = advantage.height_adv(state)
        velocity_reward = advantage.velocity_adv(state)
        if save is True:
            self.cache.push_angle_adv(angle_reward)
            self.cache.push_height_adv(height_reward)
            self.cache.push_velocity_adv(velocity_reward)
            self.cache.push_reward(0.8*angle_reward + 0.1*height_reward + 0.1*velocity_reward)

        return 0.8*angle_reward + 0.1*height_reward + 0.1*velocity_reward

    def _enemy_ai(self):
        """
        敌机策略生成，滚动时域法，搜索7个动作中使我方无人机回报最小的动作执行
        :return:
        """
        # virtual_rewards = []
        # initial_state_b = self.virtual_aircraft_b.state
        # for i in range(self.action_dim):
        #     self.virtual_aircraft_b.maneuver(i)
        #     virtual_state = self._situation(self.aircraft_r, self.virtual_aircraft_b)
        #     virtual_rewards.append(self._cal_reward(virtual_state, save=False))
        #     # 模拟完一轮之后将状态复原
        #     self.virtual_aircraft_b.reset(initial_state_b)
        # # 选取使得敌方虚拟收益最小的动作
        # action = virtual_rewards.index(min(virtual_rewards))
        # 敌方固定策略？
        return 0


    def step(self, action):
        """
        执行状态的一个时间步的更新
        :param action: 执行动作
        :return: 下一状态（归一化后）、奖励、该幕是否结束
        """
        action_r = action
        action_b = self._enemy_ai()

        state_r = self.aircraft_r.maneuver(action_r)
        state_b = self.aircraft_b.maneuver(action_b)
        # print("b action is {0}, state is {1}".format(action_b, state_b))
        self.cache.push_r_state(state_r)
        self.cache.push_b_state(state_b)

        # self.virtual_aircraft_b.maneuver(action_b)

        state = self._situation(self.aircraft_r, self.aircraft_b)
        self.state = self._normalize(state)

        reward = self._cal_reward(state, save=True)
        self.total_steps += 1

        distance, z_r, aa, ata = state[0], state[3], state[1], state[2]
        # 超出近战范围或步长过大
        if self.done is False and (distance > DIST_INIT_MAX or self.total_steps >= 200):
            self.done = True
        # 距离过近，给出惩罚
        if distance < 1000:
            self.done = True
            reward = -5
        if aa*rad2deg < 30 and ata*rad2deg < 30:
            self.done = True
            reward = 20
        # if aa*rad2deg > 150 and ata*rad2deg > 150:
        #     self.done = True
        #     reward = -10
        return self.state, reward, self.done

    def get_cache(self):
        return self.cache
