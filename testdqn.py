import torch
from gym import Env
import gym
import math
import random
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 500000
TARGET_UPDATE = 200
NUM_EPISODES = 20000
render = True

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Approximator(torch.nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=20):
        super(Approximator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.action_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplyMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trans:Transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Agent(object):
    def __init__(self, env:Env=None, capacity=1000000, hidden_dim=40):
        if env is None:
            raise Exception("agent should have an environment")
        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim

        self.policy_net = Approximator(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_net = Approximator(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

        self.memory = ReplyMemory(capacity)
        self.total_steps = 0
        self.state = None
        self.eps = EPS_START

    def select_action(self, state):
        self.eps = max(EPS_START - self.total_steps / EPS_DECAY * (EPS_START - EPS_END), EPS_END)
        self.total_steps += 1
        sample = random.random()
        if sample > self.eps:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long).to(device)

    # 测试模型时使用贪婪策略选择动作
    def select_action_greedy(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    # 执行梯度下降，返回损失值
    def _optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))  #使用*号可以将列表中的元素作为参数（即去除括号）
        #batch.state,batch.action等都是tuple，每个tuple有BATCH_SIZE个元素

        # mask用于处理非终止状态
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 取实际使用的动作对应的动作价值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #通过mask将非最终状态的value更新为 max q(s,a)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss  BATCH_SIZE*1
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # 梯度截断
        self.optimizer.step()

        this_loss = loss.item()
        return this_loss

    def learning(self):
        f = open('data/data1', 'w')
        for i_episode in range(NUM_EPISODES):
            self.state = self.env.reset()
            # unsqueeze(0)增加一维，便于直接输入神经网络，shape:1*4
            self.state = torch.from_numpy(self.state).float().unsqueeze(0).to(device)
            is_done = False
            step_in_episode = 0
            loss = 0.0
            total_rewards = 0.0
            while not is_done:
                s0 = self.state
                a0 = self.select_action(s0)
                s1, r1, is_done, info = self.env.step(a0.item())
                total_rewards += r1
                if is_done:
                    s1 = None
                else:
                    s1 = torch.from_numpy(s1).float().unsqueeze(0).to(device)
                r1 = torch.tensor([r1], device=device)
                self.memory.push(Transition(s0, a0, s1, r1))
                self.state = s1
                loss += self._optimize_model()
                step_in_episode += 1
                # 每TARGET_UPDATE步复制网络参数
                if self.total_steps % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            mean_loss = loss / step_in_episode
            print("episode:{0}; total step:{1}; total_reword:{2:.3f} loss:{3:.5f}; actual_eps:{4:.2f}" \
                  .format(i_episode + 1, step_in_episode, total_rewards, mean_loss, self.eps))
            f.write("episode:{0}; total step:{1}; total_reword:{2:.3f} loss:{3:.5f}; actual_eps:{4:.2f}\n" \
                  .format(i_episode + 1, step_in_episode, total_rewards, mean_loss, self.eps))

            # if i_episode % TARGET_UPDATE == 0:
            #     self.target_net.load_state_dict(self.policy_net.state_dict())

            # test code
            # if i_episode % 1000 == 0:
            #     self.test_result()

            # 每一千轮保存模型参数
            if (i_episode + 1) % 1000 == 0:
                self.save_model('model/cart' + str(i_episode))
        print("Complete")
        self.env.close()

        return

    def save_model(self, path):
        torch.save(self.policy_net, path)

    def load_model(self, path):
        self.policy_net = torch.load(path)

    # 使用贪心策略测试模型表现
    def test_result(self):
        for i in range(50):
            self.state = self.env.reset()
            # unsqueeze(0)增加一维，便于直接输入神经网络，shape:1*4
            self.state = torch.from_numpy(self.state).float().unsqueeze(0).to(device)
            total_rewards = 0
            is_done = False
            while not is_done:
                if render:
                    env.render()
                s0 = self.state
                a0 = self.select_action_greedy(s0)
                s1, r1, is_done, info = self.env.step(a0.item())
                total_rewards += r1
                s1 = torch.from_numpy(s1).float().unsqueeze(0).to(device)
                self.state = s1
            print("test episode {0}: total rewards:{1}".format(i, total_rewards))
        print("test over!")

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    # env = SimpleGridWorld()
    # directory = "/home/fxd/workspace/reinforce/video/cartpole"

    # env = gym.wrappers.Monitor(env, directory, force=True)
    agent = Agent(env)
    # agent.load_model('model/cart1999')
    # agent.test_result()
    print("Learning...")
    agent.learning()