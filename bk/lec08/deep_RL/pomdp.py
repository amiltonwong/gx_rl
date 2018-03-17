import numpy as np
####https://github.com/awjuliani/DeepRL-Agents
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from helper import *
from gridworld import gameEnv
env = gameEnv(partial=True, size=9)
class Qnetwork():
    def __init__(self, h_size, rnn_cell, myScope):
        #处理输入数据
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn  = tf.reshape(self.scalarInput, shape=[-1,84,84,3])
        #第一层卷积
        self.conv1 = slim.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None, scope=myScope+'_conv1')
        #第二层卷积
        self.conv2 = slim.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None, scope=myScope+'_conv2')
        #第三层卷积
        self.conv3 = slim.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None, scope=myScope+'_conv3')
        #第四层卷积
        self.conv4 = slim.convolution2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None, scope=myScope+'_conv4')
        #训练长度，即回溯多少帧数据
        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in,scope=myScope+'_rnn')
        #从循环神经网络中输出分成值函数和优势函数
        self.streamA, self.streamV = tf.split(self.rnn, 2,1)
        self.AW = tf.Variable(tf.random_normal([h_size//2, 4]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        self.salience = tf.gradients(self.Advantage, self.imageIn)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        #定义损失
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.maskA = tf.zeros([self.batch_size, self.trainLength//2])
        self.maskB = tf.ones([self.batch_size, self.trainLength//2])
        self.mask = tf.concat([self.maskA, self.maskB],1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
    def add(self, experience):
        if len(self.buffer)+1>=self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])

#设置训练参数
#每个训练步都多少个训练trace
batch_size = 4
#每个trace有多长的经验
trace_length = 8
#更新频率
update_freq = 5
y = .99
startE = 1
endE = 0.1
anneling_steps = 10000
num_episodes = 10000
pre_train_steps = 10000
load_model = False
path = "./drqn"
h_size = 512
max_epLength = 50
time_per_step = 1
summaryLength = 100
tau = 0.001

tf.reset_default_graph()
#定义LSTM包
cell = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)
mainQN = Qnetwork(h_size, cell, 'main')
targetQN = Qnetwork(h_size, cellT, 'target')
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=2)
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()
e = startE
stepDrop = (startE - endE)/anneling_steps
#创建列表来包括总的回报和每个episode的步骤
jList = []
rList = []
total_step = 0
#创建一个路径以保存我们的模型
if not os.path.exists(path):
    os.makedirs(path)
#写日志文件第一行
with open('./Center/log.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])
with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)
    for i in range(num_episodes):
        episodeBuffer = []
        #重置环境，并得到第一个新的观测
        sP = env.reset()
        s = processState(sP)
        d = False
        rAll = 0
        j = 0
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        while j < max_epLength:
            j+=1
            if np.random.rand(1)< e or total_step < pre_train_steps :
                state1 = sess.run(mainQN.rnn_state, feed_dict={mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state,mainQN.batch_size:1})
                a = np.random.randint(0,4)
            else:
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],\
                                     feed_dict={mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state,mainQN.batch_size:1})
                a = a[0]
            s1P, r, d = env.step(a)
            s1 = processState(s1P)
            total_step+=1
            episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
            if total_step > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                if total_step % (update_freq) == 0:
                    updateTarget(targetOps, sess)
                    #重新设置隐藏层的隐藏状态
                    state_train = (np.zeros([batch_size, h_size]),np.zeros([batch_size, h_size]))
                    trainBatch = myBuffer.sample(batch_size,trace_length)
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),\
                                                             mainQN.trainLength:trace_length, mainQN.state_in:state_train,\
                                                             mainQN.batch_size:batch_size})
                    Q2 = sess.run(targetQN.Qout, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),\
                                                            mainQN.trainLength:trace_length, mainQN.state_in:state_train,\
                                                            mainQN.batch_size:batch_size})
                    end_multiplier = -(trainBatch[:,4]-1)
                    doubleQ = Q2[range(batch_size * trace_length), Q1]
                    targetQ = trainBatch[:,2] + (y * doubleQ * end_multiplier)
                    #利用目标值来更新网络
                    sess.run(mainQN.updateModel,\
                             feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]/255.0), mainQN.targetQ:targetQ,\
                                        mainQN.actions:trainBatch[:,1], mainQN.trainLength:trace_length,\
                                        mainQN.state_in:state_train, mainQN.batch_size:batch_size})
            rAll += r
            s = s1
            sP = s1P
            state = state1
            if d== True:
                break
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = list(zip(bufferArray))
        myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)
        #周期地保存模型
        if i % 1000 == 0 and i!=0:
            saver.save(sess, path+'/model-'+str(i)+'.cptk')
            print("Saved Model")
        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print(total_step, np.mean(rList[-summaryLength:]),e)
            saveToCenter(i, rList, jList, np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
                         summaryLength, h_size, sess, mainQN, time_per_step)
    saver.save(sess, path+'/model-'+str(i)+'.cptk')
############testing the network####################
e = 0.01
num_episodes = 10000
load_model = True
path = './drqn'
h_size = 512
max_epLength = 50
time_per_step = 1
summaryLength = 100
tf.reset_default_graph()
cell = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)
mainQN = Qnetwork(h_size, cell,'main')
targetQN = Qnetwork(h_size, cellT,'target')
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=2)
#创建列表
jList = []
rList = []
total_step = 0
if not os.path.exists(path):
    os.makedirs(path)
#写日志文件第一行
with open('./Center/log.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])
with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)
    for i in range(num_episodes):
        episodeBuffer = []
        #重置环境，并得到第一个新的观测
        sP = env.reset()
        s = processState(sP)
        d = False
        rAll = 0
        j = 0
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        while j < max_epLength:
            j+=1
            if np.random.rand(1)< e or total_step < pre_train_steps :
                state1 = sess.run(mainQN.rnn_state, feed_dict={mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state,mainQN.batch_size:1})
                a = np.random.randint(0,4)
            else:
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],\
                                     feed_dict={mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state,mainQN.batch_size:1})
                a = a[0]
            s1P, r, d = env.step(a)
            s1 = processState(s1P)
            total_step+=1
            episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
            rAll += r
            s = s1
            sP = s1P
            state = state1
            if d == True:
                break
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = list(zip(bufferArray))
        myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)
        # 周期地保存模型
        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print(total_step, np.mean(rList[-summaryLength:]), e)
            saveToCenter(i, rList, jList, np.reshape(np.array(episodeBuffer), [len(episodeBuffer), 5]), \
                         summaryLength, h_size, sess, mainQN, time_per_step)
    saver.save(sess, path + '/model-' + str(i) + '.cptk')
    ############testing the network####################







