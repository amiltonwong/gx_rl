from __future__ import division
#######https://github.com/awjuliani/DeepRL-Agents###########
import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
import time
from gridworld import gameEnv
env = gameEnv(partial=False, size=5)
print(env.state)
plt.imshow(env.reset(), interpolation="nearest")
# plt.show()
class Qnetwork():
    def __init__(self, h_size):
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1,84,84,3])
        #第一层卷积
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4],padding='VALID',biases_initializer=None)
        #第二层卷积
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None)
        #第三层卷积
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        #第四层卷积
        self.conv4 = slim.conv2d(inputs=self.conv3,num_outputs=h_size, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)
        #从第四层卷积输出后，分成值函数和优势函数
        self.streamAC, self.streamVC = tf.split(self.conv4,2,3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        #优势函数的权重
        self.AW = tf.Variable(xavier_init([h_size//2, env.actions]))
        #值函数的权重
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        #组合优势函数和值函数，得到最终的行为-值函数
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        #下面，我们得到损失
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        #定义优化器
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        #定义训练函数
        self.updateModel = self.trainer.minimize(self.loss)
#下面的类允许我们存储经验，采样，并随机地训练网络
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    def add(self, experience):
        if len(self.buffer)+len(experience)>=self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer)-self.buffer_size)]=[]
        self.buffer.extend(experience)
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
    #处理数据状态
def processState(states):
    return np.reshape(states,[21168])
    #更新目标网络的参数
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder
#实际计算
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
#训练网络，设置所有训练参数
batch_size = 32   #每次训练需要多少经验
update_freq = 4   #执行一个训练步的频率
y = .99           #折扣因子
startE = 1        #开始时的随机行为
endE = 0.1        #最终的随机行为几率
annealing_step = 10000
num_episodes = 10000
pre_train_step = 10000
max_epLength = 50
load_model = False
path = "./dqn"
h_size = 512
tau = 0.001
###############开始训练网络
tf.reset_default_graph()
#需要训练的网络
mainQN = Qnetwork(h_size)
#目标网络
targetQN = Qnetwork(h_size)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#所有可更新变量
trainables = tf.trainable_variables()
#调用函数，来更新目标网络的权值
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()
#设置随机动作衰减率
e = startE
stepDrop = (startE-endE)/annealing_step
#创建列表来包含总的回报和每个episode的步
jList = []
rList = []
total_steps = 0
#创建存储目录
if not os.path.exists(path):
    os.makedirs(path)
with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        print(ckpt)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # saver.restore(sess, "./dqn\\model-1000.ckpt")
    for i in range(num_episodes):
        #一次实验的缓存
        episodeBuffer = experience_buffer()
        #重新设置环境，并获得第一个新观测
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        while j < max_epLength:
            j+=1
            #根据Q网络，按照epsilon-greedy策略选择动作
            if np.random.rand(1) < e or total_steps < pre_train_step:
                a = np.random.randint(0,4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})[0]
            #与环境交互一次
            s1, r, d = env.step(a)
            if i>100:
                plt.imshow(s1, interpolation="nearest")
            # scipy.misc.imshow(s1)
                plt.show()

            s1 = processState(s1)
            total_steps+=1
            #将经验[s,a,r,s1,d]保存到我们的经验回放池中
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
            #超过预训练步数后，进行实际训练
            if total_steps > pre_train_step:
                if e > endE:
                    e-=stepDrop
                if total_steps % (update_freq) ==0 :
                    #从缓存池中随机获取batch_size个数据,即s, a, r, s1, d
                    trainBatch = myBuffer.sample(batch_size)
                    #利用Double-DQN得到目标值,target = r + \gama*Q(s',argmax Q(s_t+1,a;\theta);\theta'),
                    #Q1 = argmax Q(s_t+1,a)
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    #是否为终止状态的指示因子
                    end_multiplier = -(trainBatch[:,4]-1)
                    doubleQ = Q2[range(batch_size),Q1]
                    #target = r + \gama*Q(s',argmax Q(s_t+1,a;\theta);\theta')
                    targetQ = trainBatch[:,2]+(y*doubleQ*end_multiplier)
                    #利用目标值更新网络参数
                    _=sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    #更新目标网络
                    updateTarget(targetOps,sess)
            rAll += r
            # 推进一步，到s1
            s = s1

            if d==True:
                break
        #将一次实验的数据存入总的缓存池myBuffer中
        myBuffer.add(episodeBuffer.buffer)
        #将一次实验的总步数存入列表中
        jList.append(j)
        #将一次实验的总得分
        rList.append(rAll)
        #周期地保存模型
        if i % 1000 ==0:
            saver.save(sess, path+'/model-'+str(i)+'.ckpt')
            print("Save Model")
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]),e)
    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of successful episodes:"+str(sum(rList)/num_episodes)+"%")




