import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import os
from scipy import ndimage, misc
import glob

images = []
i = 0
labels = []

for root, dirnames, filenames in os.walk("proj3_images/Numerals/"):
    if dirnames!= []:
        dirrr = dirnames
    count = 0
    for filename in filenames:
        if ".png" in filename:
            count += 1
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="L")
            image_resized = misc.imresize(image, (28, 28))
            images.append(image_resized)
    if count != 0:
        lMid = np.zeros((count,10))
        lMid[:,int(dirrr[i])] = 1
        if labels == []:
            labels = lMid
        else:
            labels = np.vstack((labels,lMid))
        #         print(len(labels),len(labels[0]))
        i += 1

uspsImages = np.asarray(images)
uspsLabels = np.asarray(labels)
uspsImages = uspsImages/255
uspsImages = 1 - uspsImages
uspsImages = uspsImages.reshape((-1,784))
meanUSPSImg = np.mean(uspsImages,0)[:,np.newaxis]
uspsImages = uspsImages.T
uspsLabels = uspsLabels.T

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
lambdaa = 0.05 #Regularization

def convolve2D(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1] ,strides=[1,2,2,1], padding='SAME')

def logistiticRegression(x):
    W = tf.Variable(tf.random_normal([10,784]))
    b = tf.Variable(tf.zeros([10,1]))

    z = tf.matmul(W,x) + b
    regLoss = tf.nn.l2_loss(W)
    return z, regLoss

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

    regLoss = tf.nn.l2_loss(weights['hiddenL1']) + tf.nn.l2_loss(weights['hiddenL1'])
    return z2, regLoss

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

    conv2 = tf.nn.relu(convolve2D(conv1,weights['conv2']) + biases['conv2'])
    conv2 = maxpool2d(conv2)

    conv2 = tf.reshape(conv2,[-1,7*7*64])
    fcLayer1 = tf.nn.relu(tf.matmul(conv2,weights['fullyC1']) + biases['fullyC1'])
    fcLayer1 = tf.nn.dropout(fcLayer1,keepRate)
    fcLayer2 = tf.nn.relu(tf.matmul(fcLayer1,weights['fullyC2']) + biases['fullyC2'])
    output = tf.matmul(fcLayer2,weights['out']) + biases['out']

    regLoss = tf.nn.l2_loss(weights['conv1']) + tf.nn.l2_loss(weights['conv2']) + tf.nn.l2_loss(weights['fullyC1']) + tf.nn.l2_loss(weights['fullyC1']) + tf.nn.l2_loss(weights['out'])
    return tf.transpose(output), regLoss

def trainNetwork():
    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(y),labels=tf.transpose(t)) + lambdaa*regLoss)
        tf.summary.scalar("loss",loss)

    with tf.name_scope("training") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

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
    print("USPS Test Accuracy:", sess.run(accuracy,{x: uspsImages, t: uspsLabels}))

start_time = time.time()
y, regLoss = logistiticRegression(x)
nEpochs = 100
trainNetwork()
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
y, regLoss = neuralNetwork(x)
nEpochs = 25
trainNetwork()
print("--- %s seconds ---" % (time.time() - start_time))

nEpochs = 14
keepRate = 0.8
start_time = time.time()
y, regLoss = convolutionalNeuralNetwork(x)
trainNetwork()
print("--- %s seconds ---" % (time.time() - start_time))
