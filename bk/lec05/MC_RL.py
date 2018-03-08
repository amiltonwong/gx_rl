import pygame
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from yuanyang_env import YuanYangEnv
class MC_RL:
    def __init__(self, yuanyang):
        #行为值函数的初始化
        self.qvalue = np.zeros((len(yuanyang.states),len(yuanyang.actions)))
        #次数初始化
        #n[s,a]=1,2,3?? 求经验平均时，q(s,a)=G(s,a)/n(s,a)
        self.n = 0.001*np.ones((len(yuanyang.states),len(yuanyang.actions)))
        self.actions = yuanyang.actions
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
        # self.gamma = 0.90
        self.learn_num = 0

    # 定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.actions[amax]
    #定义e-贪婪策略,蒙特卡罗方法，要评估的策略时e-greedy策略，产生数据的策略。
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        #概率部分
        if np.random.uniform() < 1- epsilon:
            #最优动作
            return self.actions[amax]
        else:
            return self.actions[int(random.random()*len(self.actions))]
    #找到动作所对应的序号
    def find_anum(self, a):
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i
    def mc_learning(self, num_iter, epsilon):
        #学习num_iter次
        for iter1 in range(num_iter):
            #采集状态样本
            s_sample = []
            #采集动作样本
            a_sample = []
            #采集回报样本
            r_sample = []
            #随机初始化状态
            s = self.yuanyang.reset()
            t = False
            step_num = 0
            #采集数据s0-a1-s1-a2-s2...terminate state
            #for i in range(5):
            while False == t or step_num < 30:
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                #与环境交互
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                #存储数据，采样数据
                s_sample.append(s)
                r_sample.append(r)
                a_sample.append(a_num)
                step_num+=1
                #转移到下一个状态，继续试验，s0-s1-s2
                s = s_next
            #从样本中计算累计回报,g(s_0) = r_0+gamma*r_1+gamma^2*r_2+gamma^3*r3
            g = 0.0
            #计算该序列的第一状态的累计回报
            for i in range(len(s_sample)-1, -1, -1):
                g *= self.gamma
                g += r_sample[i]
            #g=G(s1,a),开始算其他状态处的累计回报
            for i in range(len(s_sample)):
                #计算状态-行为对（s,a)的次数，s,a1...s,a2
                self.n[s_sample[i], a_sample[i]] += 1.0
                #利用增量式方法更新值函数
                self.qvalue[s_sample[i], a_sample[i]] = (self.qvalue[s_sample[i], a_sample[i]]*(self.n[s_sample[i],a_sample[i]]-1)+g)/ self.n[s_sample[i], a_sample[i]]
                g -= r_sample[i]
                g /= self.gamma
        return self.qvalue

if __name__=="__main__":
    yuanyang = YuanYangEnv()
    agent = MC_RL(yuanyang)
    qvalue=agent.mc_learning(num_iter=1000, epsilon=0.1)
    #打印学到的值函数
    print(qvalue)
    ##########################################
    #测试学到的策略
    flag = 1
    s = 2
    # print(policy_value.pi)
    step_num = 0
    # 将最优路径打印出来
    while flag:
        a = agent.greedy_policy(qvalue,s)
        print('%d->%s\t' % (s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(1)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_





