import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuanyang_env import *
from yuanyang_env import YuanYangEnv

class Td_lamda_Sarsa:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        self.lamda =0
        #值函数的初始值
        self.qvalue=np.zeros((len(yuanyang.states),len(yuanyang.actions)))
        #适合度轨迹
        self.eligibility_trace = np.zeros((len(yuanyang.states),len(yuanyang.actions)))
    #定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax=qfun[state,:].argmax()
        return self.yuanyang.actions[amax]
    #定义epsilon贪婪策略
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分 1-episilon
        if np.random.uniform() < 1 - epsilon:
            # 最优动作 a1
            return self.yuanyang.actions[amax]
        else:
            #a1的概率，e/4
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]
    #找到动作所对应的序号
    def find_anum(self,a):
        for i in range(len(self.yuanyang.actions)):
            if a==self.yuanyang.actions[i]:
                return i

    def sarsa(self, num_iter, alpha, epsilon):
        #外面的大循环，产生多次实验
        for iter in range(num_iter):
            # if epsilon>0.05:
            #     epsilon -= 1/num_iter
            # else:
            #     epsilon = 0.05
            epsilon = epsilon*0.99
            #适合度轨迹，跟一次实验的状态-动作流有关，每个episode，适合度轨迹都先清空
            self.eligibility_trace *= 0
            #随机初始化状态
            s = self.yuanyang.reset()
            # s=1
            #随机选初始动作
            a = self.yuanyang.actions[int(random.random()*len(self.yuanyang.actions))]
            # a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            t = False
            count = 0
            #小循环，s0-a0-s1-a1-s2
            #适合度轨迹，跟一次实验的状态-动作流有关
            while False==t and count < 30:
                #与环境交互得到下一个状态
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                if t == True:
                    q_target = r
                else:
                    # 下一个状态处的最大动作
                    a1 = self.epsilon_greedy_policy(self.qvalue, s_next, epsilon)
                    a1_num = self.find_anum(a1)
                    # qlearning的更新公式
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                #计算td误差
                td_error = q_target - self.qvalue[s, a_num]
                #更新适合度轨迹
                self.eligibility_trace[s, a_num] += 1
                #利用适合度轨迹更新值函数
                self.qvalue += alpha * td_error * self.eligibility_trace
                # # 利用td方法更新动作值函数
                # self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                #衰减适合度轨迹
                self.eligibility_trace *= self.gamma * self.lamda
                # YuanYangEnv2.bird_male_position = YuanYangEnv2.state_to_position(s)
                # YuanYangEnv2.render()
                # time.sleep(1)
                # 转到下一个状态
                s = s_next
                #行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        return self.qvalue
if __name__=="__main__":
    yuanyang = YuanYangEnv()
    agent = Td_lamda_Sarsa(yuanyang)
    qvalue=agent.sarsa(num_iter=5000, alpha=0.1, epsilon=0.1)
    #打印学到的值函数
    print(qvalue)
    ##########################################
    #测试学到的策略
    flag = 1
    s = 1
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
        if t == True or step_num > 200:
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




        

