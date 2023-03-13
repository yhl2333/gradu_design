from aircraft import *
import random
import advantage
from cache import *
from missile import*

class CombatEnv(object):
    def __init__(self, state_r=None, state_b=None):
        self.theta = None  # 敌方无人机相对我方的初始方位角（2D）
        if state_r is None:
            state_r = self._state_initialize(rand=False)
        if state_b is None:
            state_b = self._state_initialize(rand=True)
        self.aircraft_r = Aircraft(state_r)
        self.aircraft_b = Aircraft(state_b)
        self.missile1 = Missile(1)
        # 虚拟对抗的敌机，用于做和真实无人机相同的动作
        # 敌机策略生成方法：敌机搜索7个动作，选取及时收益最大的动作
        self.virtual_aircraft_b = Aircraft(state_b)
        # x_r, x_b, y_r, y_b, z_r, z_b, v_r, v_b, pitch_r, pitch_b, heading_r, heading_b
        # 状态表示为：距离，AA角，ATA角....
        self.state = []
        self.action_dim = 7
        self.state_dim = 12
        self.done = False
        self.total_steps = 0
        self.cache = Cache()


    # 初始化敌我无人机初始状态
    def _state_initialize(self, rand=False):
        # 引入theta为了使两机初始状态相对
        if self.theta is None:
            # self.theta = random.uniform(-pi, pi)  # 敌方无人机相对我方的方位角（2D）,在小角度出现？便于学习？
            self.theta = pi / 4
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
            self.x = 0  # distance_from_r * math.cos(self.theta)
            self.y = 20000  # distance_from_r * math.sin(self.theta)
            self.z = Z_INIT
            self.v = V_M_INIT
            self.roll = ROLL_INIT
            self.pitch = PITCH_INIT
            # heading = random.uniform(-pi, pi)
            # distance_from_r = random.uniform(0.4 * DIST_INIT_MAX, 0.5 * DIST_INIT_MAX)  # 初始距离
            # distance_from_r = 10000.0  # 固定距离？？ 便于学习？？
            #distance_from_r = 20000.0 / math.cos(self.theta)
            state = [self.y, self.y, self.z, self.v, heading, self.roll, self.pitch]
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
        # self.aircraft_b.reset(state_b)
        self.missile1.reset()
        self.missile1.emit_mis(state_r, state_b)
        self.cache.push_r_state(state_r)
        self.cache.push_b_state(state_b)

        self.virtual_aircraft_b.reset(state_b)
        state_b = [self.x, self.y, self.z, V_M_INIT, self.v*math.cos(self.pitch) * math.cos(-3*pi/4), self.v* math.sin(-3*pi/4) * math.cos(self.pitch), self.v*math.sin(self.pitch)]
        state_norm = self._normalize(state_r, state_b)
        self.state = state_norm
        return self.state

    # 状态归一化，防止差异化过大
    def _normalize(self, state_r, state_b):
        x_r, y_r, z_r, v_r, heading_r, roll_r, pitch_r = state_r
        x_b, y_b, z_b, v_m, v_x, v_y, v_z = state_b

        x_r = x_r / 10000.0
        x_b = x_b / 10000.0
        y_r = y_r / 10000.0
        y_b = y_b / 10000.0
        v_r = (v_r - 250) / 50
        v_x = v_x / V_INIT
        z_r = (z_r - Z_MIN) / (Z_MAX - Z_MIN)
        z_b = (z_b - Z_MIN) / (Z_MAX - Z_MIN)
        pitch_r = pitch_r / PITCH_MAX
        v_y = v_y/ V_INIT
        v_z = v_z/ V_INIT
        # roll_r = roll_r / ROLL_MAX
        # roll_b = roll_b / ROLL_MAX

        return [x_r, x_b, y_r, y_b, z_r, z_b, v_r, v_x, pitch_r, v_y, heading_r, v_z]

    # 态势评估，由敌我无人机状态解算出距离、威胁角等
    def _situation(self, state_r, state_b):
        x_r, y_r, z_r, v_r, heading_r, roll_r, pitch_r = state_r
        x_m, y_m, z_m, v_m, v_x, v_y, v_z = state_b

        # 距离向量
        vector_d = np.array([x_m - x_r, y_m - y_r, z_m - z_r])
        # 敌我无人机的速度向量
        vector_vr = np.array([math.cos(pitch_r) * math.cos(heading_r),
                              math.sin(heading_r) * math.cos(pitch_r), math.sin(pitch_r)])
        vector_mv = np.array([v_x / np.sqrt(np.square(v_x) + np.square(v_y) + np.square(v_z)),
                              v_y / np.sqrt(np.square(v_x) + np.square(v_y) + np.square(v_z)),
                              v_z / np.sqrt(np.square(v_x) + np.square(v_y) + np.square(v_z))])

        # AA角和ATA角计算，向量夹角
        # AA和ATA搞反了，和论文刚好相反
        aspect_angle = self._cal_angle(vector_vr, vector_d)
        antenna_train_angle = self._cal_angle(vector_mv, vector_d)
        # print("AA:{0}, ATA:{1}".format(aspect_angle, antenna_train_angle))

        distance = np.sqrt(np.sum(vector_d * vector_d))
        return [distance, aspect_angle, antenna_train_angle, z_r, z_m, v_r, v_m, pitch_r, pitch_r, roll_r, roll_r]

    def _cal_angle(self, vector_vr, vector_d):
        dot_product = np.dot(vector_vr, vector_d)
        d_norm = np.sqrt(np.sum(vector_d * vector_d))
        angle = np.arccos(dot_product / (d_norm + 1e-5))
        return angle

    def _cal_reward(self, situation, save=True):
        angle_reward = advantage.angle_adv(situation)
        height_reward = advantage.height_adv(situation)
        velocity_reward = advantage.velocity_adv(situation)
        if save is True:
            self.cache.push_angle_adv(angle_reward)
            self.cache.push_height_adv(height_reward)
            self.cache.push_velocity_adv(velocity_reward)
            self.cache.push_reward(0.7 * angle_reward + 0.2 * height_reward + 0.1 * velocity_reward)

        return 0.7 * angle_reward + 0.2 * height_reward + 0.1 * velocity_reward

    def _enemy_ai(self):
        """
        敌机策略生成，滚动时域法，搜索7个动作中使我方无人机回报最小的动作执行
        :return:
        """
        virtual_rewards = []
        initial_state_b = self.virtual_aircraft_b.state
        for i in range(self.action_dim):
            self.virtual_aircraft_b.maneuver(i)
            virtual_situation = self._situation(self.aircraft_r.state, self.virtual_aircraft_b.state)
            virtual_reward = 0.7 * advantage.angle_adv(virtual_situation) + 0.2 * advantage.height_adv(
                virtual_situation) + \
                             0.1 * advantage.velocity_adv(virtual_situation)
            virtual_rewards.append(virtual_reward)
            # 模拟完一轮之后将状态复原
            self.virtual_aircraft_b.reset(initial_state_b)
        # 选取使得敌方虚拟收益最小的动作
        action = virtual_rewards.index(min(virtual_rewards))
        return action
        # 敌方固定策略？
        # return 0

    def _enemy_ai_2(self):
        virtual_dis = []
        initial_state_b = self.virtual_aircraft_b.state
        for i in range(self.action_dim):
            self.virtual_aircraft_b.maneuver(i)
            virtual_situation = self._situation(self.virtual_aircraft_b.state, self.aircraft_r.state)
            distance = virtual_situation[0]
            virtual_dis.append(distance)
            # 模拟完一轮之后将状态复原
            self.virtual_aircraft_b.reset(initial_state_b)
        # 选取使得敌方虚拟收益最小的动作
        action = virtual_dis.index(min(virtual_dis))
        return action

    def _enemy_ai_expert(self):
        """
        专家系统策略
        :return:
        """
        state_r, state_b = self.aircraft_r.state, self.aircraft_b.state
        x_r, y_r, z_r, v_r, heading_r, roll_r, pitch_r = state_r
        x_b, y_b, z_b, v_b, heading_b, roll_b, pitch_b = state_b

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

        left_or_right = np.sign(vector_vr[0] * vector_vb[1] - vector_vr[1] * vector_vb[0])
        linear_adv = 1 - (aspect_angle + antenna_train_angle) / pi
        if random.random() >= 0.1:
            return self._chase(left_or_right)
        else:
            if z_r > z_b:
                return 5
            if z_r < z_b:
                return 6
            return 0

    def _escape(self, left_or_right):
        print("escape")
        if left_or_right > 0:
            return 3
        elif left_or_right < 0:
            return 4
        else:
            return 6

    def _chase(self, left_or_right):
        if left_or_right > 0:
            print("chase1")
            return 3
        elif left_or_right < 0:
            print("chase2")
            return 4
        else:
            print("chase3")
            return 6

    def step(self, action):
        """
        执行状态的一个时间步的更新
        :param action: 执行动作
        :return: 下一状态（归一化后）、奖励、该幕是否结束
        """
        action_r = action
        # action_b = self._enemy_ai()

        state_r = self.aircraft_r.maneuver(action_r)
        state_missile1 = self.missile1.p_guide_sim(state_r)
        # state_b = self.aircraft_b.maneuver(action_b)
        # print("b action is {0}, state is {1}".format(action_b, state_b))
        self.cache.push_r_action(action_r)
        # self.cache.push_b_action(action_b)
        self.cache.push_r_state(state_r)
        # self.cache.push_b_state(state_b)
        self.cache.push_missile1_state(state_missile1)
        # self.virtual_aircraft_b.maneuver(action_b)

        self.state = self._normalize(state_r, state_missile1)
        situation = self._situation(state_r, state_missile1)

        reward = self._cal_reward(situation, save=True)
        self.total_steps += 1
        # self.cache.save_combat_log(state_r,state_b,reward)

        distance, z_r, aa, ata = situation[0], situation[3], situation[1], situation[2]
        # 超出近战范围或步长过大
        if self.done is False and self.total_steps >= 250:
            self.done = True

        if distance > DIST_INIT_MAX or distance < 0:
            reward = -5
            self.cache.push_reward(reward)
            self.done = True

        #if aa * rad2deg < 30 and ata * rad2deg < 45:
        #    reward = 30
        #    self.cache.push_reward(reward)
        #    self.done = True

        # if aa*rad2deg > 145 and ata*rad2deg > 150:
        #     self.done = True
        #     reward = -30
        return self.state, reward, self.done

    def get_cache(self):
        return self.cache
