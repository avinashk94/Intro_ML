
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import time


# In[2]:


trainImages = mnist.train.images.T
testImages = mnist.test.images.T
trainLabels = mnist.train.labels.T
testLabels = mnist.test.labels.T
validationImages = mnist.validation.images.T
validationLabels = mnist.validation.labels.T
m = trainImages.shape[1]

batchSize = 100
printEvery = 5
nEpochs = 15
nClasses = 10
learningRate = 0.005 #IF used....


# In[3]:


x = tf.placeholder(tf.float32, [784,None])
t = tf.placeholder(tf.float32, [10,None])


# In[4]:


def convolve2D(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')


# In[5]:


def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1] ,strides=[1,2,2,1], padding='SAME')


# In[6]:


def neuralNetwork(x):
    lambdaa = 0.1
    nH1 = 512
    weights = {'hiddenL1':tf.Variable(tf.random_normal([nH1,784])),
               'hiddenL2':tf.Variable(tf.random_normal([10,nH1]))}
    
    biases = {'hiddenL1':tf.Variable(tf.random_normal([nH1,1])),
              'hiddenL2':tf.Variable(tf.random_normal([10,1]))}
    
    w_h1 = tf.summary.histogram("weights1",weights['hiddenL1'])
    w_h2 = tf.summary.histogram("weights2",weights['hiddenL2'])
    b_h1 = tf.summary.histogram("biases1",biases['hiddenL1'])
    b_h2 = tf.summary.histogram("biases2",biases['hiddenL2'])
    
    
    with tf.name_scope("a1") as scope:
        z1 = tf.matmul(weights['hiddenL1'],x) + biases['hiddenL1']
        a1 = tf.nn.relu(z1)
    
    with tf.name_scope("a2") as scope:
        z2 = tf.matmul(weights['hiddenL2'],a1) + biases['hiddenL2']
    return z2
    


# In[7]:


def logistiticRegression(x):
    W = tf.Variable(tf.random_normal([10,784]))
    b = tf.Variable(tf.zeros([10,1]))
    
    z = tf.matmul(W,x) + b
    return z


# In[6]:


def convolutionalNeuralNetwork(x):
    weights = {'conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'fullyC1':tf.Variable(tf.random_normal([7*7*64,1024])),
               'fullyC2':tf.Variable(tf.random_normal([1024,1024])),
               'out':tf.Variable(tf.random_normal([1024,nClasses]))}
    
    biases = {'conv1':tf.Variable(tf.random_normal([32])),
              'conv2':tf.Variable(tf.random_normal([64])),
              'fullyC1':tf.Variable(tf.random_normal([1024])),
              'fullyC2':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([nClasses]))}
    
    x = tf.transpose(x)
    
    Img = tf.reshape(x,[-1,28,28,1])
    
    conv1 = tf.nn.relu(convolve2D(Img,weights['conv1']) + biases['conv1'])
    
    conv1 = maxpool2d(conv1)
    print(conv1)
    
    conv2 = tf.nn.relu(convolve2D(conv1,weights['conv2']) + biases['conv2'])
#     print(conv2)
    conv2 = maxpool2d(conv2)
    print(conv2)
    
    conv2 = tf.reshape(conv2,[-1,7*7*64])
#     print(conv2)
    fcLayer1 = tf.nn.relu(tf.matmul(conv2,weights['fullyC1']) + biases['fullyC1'])
#     print(fcLayer1)
    fcLayer1 = tf.nn.dropout(fcLayer1,keepRate)
    fcLayer2 = tf.nn.relu(tf.matmul(fcLayer1,weights['fullyC2']) + biases['fullyC2'])
    output = tf.matmul(fcLayer2,weights['out']) + biases['out']
    print(output)
    
    return tf.transpose(output)


# In[7]:


def convolutionalNeuralNetwork2(x):
    weights = {'conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'conv2':tf.Variable(tf.random_normal([5,5,32,32])),
               'conv3':tf.Variable(tf.random_normal([5,5,32,64])),
               'conv4':tf.Variable(tf.random_normal([5,5,64,64])),
               'fullyC1':tf.Variable(tf.random_normal([7*7*64,1024])),
               'fullyC2':tf.Variable(tf.random_normal([1024,1024])),
               'out':tf.Variable(tf.random_normal([1024,nClasses]))}
    
    biases = {'conv1':tf.Variable(tf.random_normal([32])),
              'conv2':tf.Variable(tf.random_normal([32])),
              'conv3':tf.Variable(tf.random_normal([64])),
              'conv4':tf.Variable(tf.random_normal([64])),
              'fullyC1':tf.Variable(tf.random_normal([1024])),
              'fullyC2':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([nClasses]))}
    
    x = tf.transpose(x)
    
    Img = tf.reshape(x,[-1,28,28,1])
    
    conv1 = tf.nn.relu(convolve2D(Img,weights['conv1']) + biases['conv1'])
    print(conv1)
    conv2 = tf.nn.relu(convolve2D(conv1,weights['conv2']) + biases['conv2'])
    print(conv2)
    conv2 = maxpool2d(conv2)
    print(conv2)
    
    conv3 = tf.nn.relu(convolve2D(conv2,weights['conv3']) + biases['conv3'])
    print(conv3)
    conv4 = tf.nn.relu(convolve2D(conv3,weights['conv4']) + biases['conv4'])
    print(conv4)
    conv4 = maxpool2d(conv4)
    print(conv4)
    
    conv4 = tf.reshape(conv4,[-1,7*7*64])
    print(conv2)
    fcLayer1 = tf.nn.relu(tf.matmul(conv4,weights['fullyC1']) + biases['fullyC1'])
#     print(fcLayer1)
    fcLayer1 = tf.nn.dropout(fcLayer1,keepRate)
    fcLayer2 = tf.nn.relu(tf.matmul(fcLayer1,weights['fullyC2']) + biases['fullyC2'])
    output = tf.matmul(fcLayer2,weights['out']) + biases['out']
    print(output)
    
    return tf.transpose(output)


# In[8]:


def trainNetwork():
    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(y),labels=tf.transpose(t)))
        tf.summary.scalar("loss",loss)
    
    with tf.name_scope("training") as scope:
        optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    with tf.name_scope("accuracy") as scope:
        correct = tf.equal(tf.argmax(y),tf.argmax(t))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        
    init = tf.global_variables_initializer()
    sess = tf.Session()
    mergeSummary = tf.summary.merge_all()

    sess.run(init)
    summaryWriter = tf.summary.FileWriter('../../TFout/2', sess.graph)

    for epoch in range(nEpochs):
        error = 0.0
        for i in range(int(m/batchSize)):
            batch_xs, batch_ys = mnist.train.next_batch(batchSize)
            _, er, summaryStr = sess.run([optimizer,loss,mergeSummary],feed_dict={x:batch_xs.T,t:batch_ys.T})
            summaryWriter.add_summary(summaryStr, epoch*(int(m/batchSize)) + i)
            error += er
        if (epoch+1)%printEvery == 0:
            print('Loss in ',epoch+1,' epoch is ',error)

    prediction = tf.equal(tf.argmax(y),tf.argmax(t))
    accuracy = tf.reduce_mean(tf.cast(prediction,"float"))
    print("Accuracy Train:", sess.run(accuracy,{x: trainImages, t: trainLabels}))
    print("Accuracy validation:", sess.run(accuracy,{x: validationImages, t: validationLabels}))
    print("Accuracy Test:", sess.run(accuracy,{x: testImages, t: testLabels}))


# In[9]:


# y = logistiticRegression(x)
# nEpochs = 50
# trainNetwork()


# y = neuralNetwork(x)
# nEpochs = 15
# trainNetwork()

nEpochs = 2
keepRate = 0.7
start_time = time.time()
y = convolutionalNeuralNetwork(x)
trainNetwork()
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


nEpochs = 2
keepRate = 0.7
start_time = time.time()
y = convolutionalNeuralNetwork2(x)
trainNetwork()
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


nEpochs

