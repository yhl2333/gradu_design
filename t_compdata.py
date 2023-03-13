import matplotlib.pyplot as plt

'''
对不同参数下训练结果进行比较
'''

AVE_NUM = 100
ave_rewards = [[],[],[]]
rewards = [[],[],[]]

f1 = open('data/info_lr1e-3', 'r')
f2 = open('data/info_lr1e-4', 'r')
f3 = open('data/info_lr1e-5', 'r')
file_list = [f1, f2, f3]

for i, f in enumerate(file_list):
    for line in f.readlines()[1:3000]:
        line_split = line.split(' ')
        reward = float(line_split[3][13:])
        rewards[i].append(reward)
        if len(rewards[i]) > AVE_NUM:
            ave_rewards[i].append(sum(rewards[i][len(rewards[i])-AVE_NUM : len(rewards[i])]) / AVE_NUM)

for i, f in enumerate(file_list):
    plt.plot(ave_rewards[i])
    plt.xlabel('episodes')
    plt.ylabel('rewards')

    file_list[i].close()
plt.legend(['lr=1e-3', 'lr=1e-4', 'lr=1e-5'])
plt.show()
