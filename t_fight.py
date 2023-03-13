from environment_new import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random

'''
我方随机策略对抗图绘制，用于验证敌方策略的有效性
'''

if __name__ == '__main__':
    env = CombatEnv()
    state = env.reset()
    for i in range(1000):
        action = random.randint(0, 6)
        s1, r, done = env.step(action)
        print(s1)
        if done is True:
            print("episode over, total steps:{0}".format(i))
            break
    cache = env.get_cache()
    r_states = cache.get_r_states()
    b_states = cache.get_b_states()
    rewards = cache.get_rewards()
    print("total rewards:{0}".format(sum(rewards)))
    angle_adv = cache.get_angle_adv()
    height_adv = cache.get_height_adv()
    velocity_adv = cache.get_velocity_adv()

    r_states = list(zip(*r_states))
    r_states_x = r_states[0]
    r_states_y = r_states[1]
    r_states_z = r_states[2]
    r_states_v = r_states[3]

    b_states = list(zip(*b_states))
    b_states_x = b_states[0]
    b_states_y = b_states[1]
    b_states_z = b_states[2]
    b_states_v = b_states[3]

    fig1 = plt.figure(1)
    ax = fig1.gca(projection='3d')
    ax.plot(r_states_x[:1], r_states_y[:1], r_states_z[:1], 'g', marker='o', markersize=10, label='start')
    ax.plot(b_states_x[:1], b_states_y[:1], b_states_z[:1], 'g', marker='o', markersize=10)
    for i in range(len(r_states_x)):
        ax.plot(r_states_x[0:i], r_states_y[0:i], r_states_z[0:i], 'r')
        ax.plot(b_states_x[0:i], b_states_y[0:i], b_states_z[0:i], 'b')
        plt.pause(0.05)
    ax.plot(r_states_x, r_states_y, r_states_z, 'r', label='aircraft_r')
    ax.plot(b_states_x, b_states_y, b_states_z, 'b', label='aircraft_b')

    ax.plot(r_states_x[-1:], r_states_y[-1:], r_states_z[-1:], 'black', marker='x', markersize=10, label='end')
    ax.plot(b_states_x[-1:], b_states_y[-1:], b_states_z[-1:], 'black', marker='x', markersize=10)

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('z(m)')
    plt.title('AirCombat')
    ax.legend(loc='upper right')

    fig2 = plt.figure(2)
    plt.subplot(221)
    plt.title("total advantage")
    plt.plot(rewards)
    plt.xlabel('steps')
    plt.ylabel('rewards')

    plt.subplot(222)
    plt.title("Real-time angle advantage")
    plt.plot(angle_adv)
    plt.xlabel('steps')
    plt.ylabel('angle advantage')

    plt.subplot(223)
    plt.title("Real-time height advantage")
    plt.plot(height_adv)
    plt.xlabel('steps')
    plt.ylabel('height advantage')

    plt.subplot(224)
    plt.title("Real-time velocity advantage")
    plt.plot(velocity_adv)
    plt.xlabel('steps')
    plt.ylabel('velocity advantage')
    plt.show()