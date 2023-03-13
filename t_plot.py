import matplotlib.pyplot as plt

'''
对比三种方法奖励图
'''
AVE_NUM = 30
AVE_NUM1 = 1000
ave_rewards = []
ave_rewards1 = []
rewards = []
losses = []
ave_loss = []
ave_Q = []
Q_sum = []
f = open('data/infoDQN.log', 'r')
f1 = open('data/infoDDQN.log', 'r')
f2 = open('data/infoDuel.log', 'r')
file_list = [f, f1, f2]
color = ['r', 'g', 'b']
for i, f in enumerate(file_list):
    for line in f.readlines()[1:100000]:
        line_split = line.split(' ')
        reward = float(line_split[3][13:])
        loss = float(line_split[4][6:-1])
        Q = float(line_split[6][6:-1])
        rewards.append(reward)
        losses.append(loss)
        Q_sum.append(Q)
        if len(rewards) > AVE_NUM:
            ave_rewards.append(sum(rewards[len(rewards)-AVE_NUM : len(rewards)]) / AVE_NUM)
            ave_loss.append(sum(losses[len(rewards) - AVE_NUM : len(rewards)]) / AVE_NUM)
            ave_Q.append(sum(Q_sum[len(rewards) - AVE_NUM : len(rewards)]) / AVE_NUM)
        if len(ave_rewards) > AVE_NUM1:
            ave_rewards1.append(sum(ave_rewards[len(ave_rewards)-AVE_NUM1 : len(ave_rewards)]) / AVE_NUM1)

    # plt.subplot(121)
    # plt.plot(rewards)
    # plt.plot(ave_rewards,'r', label='original', linewidth='0.05')
    plt.plot(ave_rewards1, color[i], linewidth='1.5')
    ave_rewards.clear()
    ave_rewards1.clear()

plt.xlabel('episodes')
plt.ylabel('total rewards')
plt.legend(['DQN', 'DDQN', 'DuelingDQN'])

plt.show()
f.close()
