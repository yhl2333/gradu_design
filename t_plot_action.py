from DQNAgent import *
from environment_new import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

'''
训练模型评估，读取训练好的数据进行评估绘图，动态三维对抗图，动态角度、高度、速度优势
'''
# 0.定常飞行； 1.加速； 2.减速； 3.左转弯； 4.右转弯； 5.拉起； 6.俯冲
DRAW_WITH_ADVANTAGE = True

if __name__ == '__main__':
    env = CombatEnv()
    agent = Agent(env)
    agent.load_model('model/airCom1199')
    cache = agent.test_result()
    r_actions = cache.get_r_actions()
    b_actions = cache.get_b_actions()

    plt.plot(r_actions, 'r', linewidth=1, label ='r_aircraft_acitons', linestyle='-')
    plt.plot(b_actions, 'b', linewidth=1, label ='b_aircraft_acitons', linestyle='--')
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('actions')
    plt.show()