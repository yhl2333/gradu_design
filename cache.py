from config import *

# 对抗信息缓存，用于存储对抗过程的敌我状态、奖励优势信息
class Cache(object):
    def __init__(self):
        # self.combat_file = open('data/combat.log','w')
        self.aircraft_r_states = []
        self.aircraft_b_states = []
        self.r_actions = []
        self.b_actions = []
        self.rewards = []
        self.angle_adv = []
        self.height_adv = []
        self.velocity_adv = []
        self.missile1_states = []

    def clear(self):
        self.aircraft_r_states = []
        self.aircraft_b_states = []
        self.r_actions = []
        self.b_actions = []
        self.rewards = []
        self.angle_adv = []
        self.height_adv = []
        self.velocity_adv = []
        self.missile1_states = []

    # def save_combat_log(self, state_r, state_b, reward):
    #     self.combat_file.write(str(state_r + state_b + [reward]) + '\n')

    def push_r_action(self, action):
        self.r_actions.append(action)

    def push_b_action(self, action):
        self.b_actions.append(action)

    def push_r_state(self, state):
        self.aircraft_r_states.append(state)

    def push_b_state(self, state):
        self.aircraft_b_states.append(state)

    def push_missile1_state(self, state):
        self.missile1_states.append(state)

    def push_reward(self, reward):
        self.rewards.append(reward)

    def push_angle_adv(self, angle_adv):
        self.angle_adv.append(angle_adv)

    def push_height_adv(self, height_adv):
        self.height_adv.append(height_adv)

    def push_velocity_adv(self, velocity_adv):
        self.velocity_adv.append(velocity_adv)

    def get_r_states(self):
        return self.aircraft_r_states

    def get_b_states(self):
        return self.aircraft_b_states

    def get_rewards(self):
        return self.rewards

    def get_missile1_states(self):
        return  self.missile1_states

    def get_angle_adv(self):
        return self.angle_adv

    def get_height_adv(self):
        return self.height_adv

    def get_velocity_adv(self):
        return self.velocity_adv

    def get_r_actions(self):
        return self.r_actions

    def get_b_actions(self):
        return self.b_actions

    def is_empty(self):
        return len(self.aircraft_r_states) == 0