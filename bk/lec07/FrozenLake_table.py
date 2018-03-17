import gym
import numpy as np
import numpy
import matplotlib.pyplot as plt

#创建FrozenLake的仿真环境
env = gym.make('FrozenLake-v0')
#初始化Q函数
Q =  np.zeros([env.observation_space.n, env.action_space.n])
#学习速率
alpha = .8
#回报衰减因子
gamma = .95
num_episodes = 2000
#累计回报列表
rList = []

for i in range(num_episodes):
    #获得初始状态
    s = env.reset()
    #新的episode总回报清零
    rAll = 0
    #新的episdoe没有结束
    d = False
    #新的episode状态数为0
    j = 0
    # env.render()
    while j < 99:
        j+=1
        #选择epsilon-greedy策略产生动作
        a = np.argmax(Q[s,:]+np.random.randn(1, env.action_space.n)*(1./(i+1)))
        #与环境进行交互，得到新的状态和回报
        s_next,r,d,_ = env.step(a)
        #利用时间差分方法更新行为值函数
        Q[s,a] = Q[s,a] +alpha *(r + gamma * np.max(Q[s_next,:])-Q[s,a])
        #将回报累加
        rAll += r
        #智能体推进一步
        s = s_next
        # print(s)
        if d == True:
            break
    rList.append(rAll)
print("Final Q-Table Value")
print(Q)
s = env.reset()
d = False
env.render()
while d == False:
    a = np.argmax(Q[s, :])
    s_next, r, d, _ = env.step(a)
    print(s,a)
    # print(d)
    # env.render()
    s = s_next



