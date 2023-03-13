from environment import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random

'''
    绘制7种不同动作示意图
'''
if __name__ == "__main__":
    x = 0
    y = 0
    z = Z_INIT
    v = V_INIT
    heading = 0
    roll = ROLL_INIT
    pitch = PITCH_INIT
    init_state = [x, y, z, v, heading, roll, pitch]

    actions_name = ['steady', 'accelerate', 'decelerate', 'turn left', 'turn right', 'pull up', 'pitchdown']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    states = []
    states.append(init_state)
    aircraft = Aircraft(init_state)
    for i in range(200):
        state = aircraft.maneuver(0)
        states.append(state)
    x = [state[0] for state in states]
    y = [state[1] for state in states]
    z = [state[2] for state in states]
    ax.plot(x, y, z, label='1')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show()
