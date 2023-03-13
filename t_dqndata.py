import matplotlib.pyplot as plt

'''
对训练过程奖励和损失绘图
'''
AVE_NUM = 100
ave_rewards = []
rewards = []
losses = []
ave_loss = []
ave_Q = []
Q_sum = []
f = open('data/info.log', 'r')
for line in f.readlines()[1:-1]:
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

plt.subplot(131)
# plt.plot(rewards)
plt.plot(ave_rewards)
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.legend(['total rewards', 'average rewards'])

plt.subplot(132)
# plt.plot(losses)
plt.plot(ave_loss)
plt.xlabel('episodes')
plt.ylabel('loss')

plt.subplot(133)
plt.plot(ave_Q)
plt.xlabel('episode')
plt.ylabel('AVE_Q')

plt.show()
f.close()
