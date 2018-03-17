from __future__ import division
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
#创建环境
env = gym.make('FrozenLake-v0')

#设置一个图
tf.reset_default_graph()
#创建前向神经网络
inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)
#创建损失函数
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ-Qout))
#创建优化器
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#模型更新步，即权重更新步
updateModel = trainer.minimize(loss)

#训练网络
init = tf.global_variables_initializer()
gamma = 0.99
e = 0.1
num_episodes = 2000
#创建总回报和每步的回报
jList = []
rList = []
with tf.Session() as sess:
    #初始化变量,得到权重初始值
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            j+=1
            #利用epsilon-greedy策略产生动作
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #与环境交互，得到新的状态和回报
            s_next, r, d, _ = env.step(a[0])
            #得到下一个状态s_next处的最大行为值函数
            Q1 = sess.run(Qout, feed_dict={inputs1:np.identity(16)[s_next:s_next+1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + gamma * maxQ1
            #训练神经网络
            _, W1 = sess.run([updateModel,W], feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            #往前推进一步
            s = s_next
            if d == True:
                #衰减探索率
                e = 1./((i/50)+10)
                break
        jList.append(j)
        rList.append(rAll)
    plt.plot(rList)
    plt.show()
    #测试学到的策略
    for i in range(10):
        s = env.reset()
        env.render()
        d = False
        while d == False:
            #利用训练好的网络计算贪婪动作
            a = sess.run(predict, feed_dict={inputs1: np.identity(16)[s:s + 1]})
            s_next, r, d, _ = env.step(a[0])
            print(s, a,s_next)
            #推进一步
            s = s_next




