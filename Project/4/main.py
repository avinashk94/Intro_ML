import numpy as np
import tensorflow as tf
from sklearn import model_selection
from PIL import Image
import time
import pandas as pd
import os
from scipy import ndimage, misc
import glob
import pickle
import multiprocessing as mp

data = pd.read_csv('files/Anno/list_attr_celeba.txt', delim_whitespace = True, header=1)
df = data['Eyeglasses']
df = (df + 1)/2
d = np.eye(2)[df.values.astype(int)]

usingImages = 20000
nClasses = 2
shape1 = 178
shape2 = 218
printEvery = 1
batchSize = 100
lambdaa = 0.01

images = np.array([np.float32(np.array(Image.open("files/Img/img_align_celeba/"+str(fname)).resize((shape1, shape2))))/256 for fname in df.head(usingImages).index])
# with open('my.pickle', 'wb') as handle:
#     pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('my.pickle', 'rb') as handle:
#     images = pickle.load(handle)

X_train, X_test, y_train, y_test = model_selection.train_test_split(images, d[:usingImages], test_size=0.05, shuffle=False)
X_validate, X_test, y_validate, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.50, shuffle=False)

x = tf.placeholder(tf.float32, [None, shape2, shape1, 3])
t = tf.placeholder(tf.float32, [None,2])

def convolutionalNeuralNetwork(x):
    weights = {'conv1':tf.Variable(tf.random_normal([5,5,3,64])),
               'conv2':tf.Variable(tf.random_normal([5,5,64,128])),
               'conv3':tf.Variable(tf.random_normal([5,5,128,256])),
               'conv4':tf.Variable(tf.random_normal([5,5,256,256])),
               'fullyC1':tf.Variable(tf.random_normal([14*12*256,1024])),
               'fullyC2':tf.Variable(tf.random_normal([1024,1024])),
               'out':tf.Variable(tf.random_normal([1024,nClasses]))}

    biases = {'conv1':tf.Variable(tf.random_normal([64])),
              'conv2':tf.Variable(tf.random_normal([128])),
              'conv3':tf.Variable(tf.random_normal([256])),
              'conv4':tf.Variable(tf.random_normal([256])),
              'fullyC1':tf.Variable(tf.random_normal([1024])),
              'fullyC2':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([nClasses]))}

    conv1 = tf.nn.relu(tf.nn.conv2d(input=x, filter=weights['conv1'],strides=[1,1,1,1],padding='SAME')+ biases['conv1'])
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1] ,strides=[1,2,2,1], padding='SAME')
    print(conv1)

    conv2 = tf.nn.relu(tf.nn.conv2d(input=conv1, filter=weights['conv2'],strides=[1,1,1,1],padding='SAME') + biases['conv2'])
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1] ,strides=[1,2,2,1], padding='SAME')
    print(conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(input=conv2, filter=weights['conv3'],strides=[1,1,1,1],padding='SAME') + biases['conv3'])
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1] ,strides=[1,2,2,1], padding='SAME')
    print(conv3)

    conv4 = tf.nn.relu(tf.nn.conv2d(input=conv3, filter=weights['conv4'],strides=[1,1,1,1],padding='SAME') + biases['conv4'])
    conv4 = tf.nn.max_pool(conv4,ksize=[1,2,2,1] ,strides=[1,2,2,1], padding='SAME')
    print(conv4)

    conv4 = tf.reshape(conv4,[-1,14*12*256])
    print(conv4)
    fcLayer1 = tf.nn.relu(tf.matmul(conv4,weights['fullyC1']) + biases['fullyC1'])
    print(fcLayer1)
    fcLayer1 = tf.nn.dropout(fcLayer1,keepRate)
    fcLayer2 = tf.nn.relu(tf.matmul(fcLayer1,weights['fullyC2']) + biases['fullyC2'])
    output = tf.matmul(fcLayer2,weights['out']) + biases['out']
    print(output)

    regLoss = tf.nn.l2_loss(weights['conv1']) + tf.nn.l2_loss(weights['conv2']) + tf.nn.l2_loss(weights['conv3']) + tf.nn.l2_loss(weights['conv3']) + tf.nn.l2_loss(weights['fullyC1']) + tf.nn.l2_loss(weights['fullyC1']) + tf.nn.l2_loss(weights['out'])
    return output, regLoss

def trainNetwork():
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(y),labels=tf.transpose(t)) + lambdaa*regLoss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct = tf.equal(tf.argmax(y),tf.argmax(t))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    for epoch in range(nEpochs):
        error = 0.0
        for i in range(int(usingImages*0.8/batchSize)):
            xs = X_train[i*batchSize:(i+1)*batchSize]
            ys = y_train[i*batchSize:(i+1)*batchSize]
            _, er = sess.run([optimizer,loss],feed_dict={x:xs,t:ys})
            error += er
        if (epoch+1)%printEvery == 0:
            print('Loss in ',epoch+1,' epoch is ',error/(usingImages*8))

    prediction = tf.equal(tf.argmax(y),tf.argmax(t))
    accuracy = tf.reduce_mean(tf.cast(prediction,"float"))
    print("Accuracy validation:", sess.run(accuracy,{x: X_validate, t: y_validate}))
    print("Accuracy Test:", sess.run(accuracy,{x: X_test, t: y_test}))


nEpochs = 2
keepRate = 0.8
start_time = time.time()
y, regLoss = convolutionalNeuralNetwork(x)
print("Y::::::",y)
print(t)
trainNetwork()
print("--- %s seconds ---" % (time.time() - start_time))
