import numpy as np
import random
#C:\Users\guoxi\AppData\Local\Programs\Python\Python36\Lib\site-packages\gym\envs\classic_control
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuanyang_env import YuanYangEnv

class Td_Qlearning:
    def __init__(self, yuanyang):
        #值函数的初始值
        self.qvalue=np.zeros((len(yuanyang.states),len(yuanyang.actions)))
        self.actions=yuanyang.actions
        self.yuanyang=yuanyang
        self.gamma = yuanyang.gamma
        self.learn_num=0
    #定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax=qfun[state,:].argmax()
        return self.actions[amax]
    #定义epsilon贪婪策略
    def epsilon_greedy_policy(self,qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.actions[amax]
        else:
            return self.actions[int(random.random() * len(self.actions))]
    #找到动作所对应的序号
    def find_anum(self,a):
        for i in range(len(self.actions)):
            if a==self.actions[i]:
                return i

    def qlearning(self,num_iter, alpha, epsilon):
        #大循环
        for iter in range(num_iter):
            #随机初始化状态
            s = yuanyang.reset()
            # s=0
            #随机选初始动作
            # a = self.actions[int(random.random()*len(self.actions))]
            a = self.epsilon_greedy_policy(self.qvalue,s,epsilon)
            t = False
            self.learn_num += 1
            count = 0
            while False==t and count < 30:
                #与环境交互得到下一个状态
                s_next, r, t = yuanyang.transform(s, a)
                # print(s)
                # print(s_next)
                a_num = self.find_anum(a)
                if t == True:
                    q_target = r
                else:
                    # 下一个状态处的最大动作，a1用greedy_policy
                    a1 = self.greedy_policy(self.qvalue, s_next)
                    a1_num = self.find_anum(a1)
                    # qlearning的更新公式TD(0)
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                    # 利用td方法更新动作值函数
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                # if self.learn_num % 100==0:
                #     yuanyang.bird_male_position = yuanyang.state_to_position(s)
                #     yuanyang.render()
                # #     print(s)
                #     print("第%d次学习的过程"%(self.learn_num))
                # time.sleep(1)
                # 转到下一个状态
                s = s_next
                #行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        return self.qvalue
if __name__=="__main__":
    yuanyang = YuanYangEnv()
    agent = Td_Qlearning(yuanyang)
    qvalue=agent.qlearning(num_iter=2000, alpha=0.1, epsilon=0.3)
    #打印学到的值函数
    print(qvalue)
    ##########################################
    #测试学到的策略
    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    # 将最优路径打印出来
    while flag:
        a = agent.greedy_policy(qvalue,s)
        print('%d->%s\t' % (s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_
    # print('optimal policy is \t')
    # print(policy_value.pi)
    # print('optimal value function is \t')
    # print(policy_value.v)
    # xx = np.linspace(0, len(policy_value.v) - 1, 101)
    # yy = policy_value.v
    # plt.figure()
    # plt.plot(xx, yy)
    # plt.show()
    # # 将值函数的图像显示出来
    # z = []
    # for i in range(100):
    #     z.append(1000 * policy_value.v[i])
    # zz = np.array(z).reshape(10, 10)
    # plt.figure(num=2)
    # plt.imshow(zz, interpolation='none')
    # plt.show()




        

