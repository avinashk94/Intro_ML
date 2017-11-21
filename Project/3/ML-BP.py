
# coding: utf-8

# In[1]:


import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[10]:


trainImages = mnist.train.images.T
testImages = mnist.test.images.T
trainLabels = mnist.train.labels.T
testLabels = mnist.test.labels.T
validationImages = mnist.validation.images.T
validationLabels = mnist.validation.labels.T
(nFeatures,m) = trainImages.shape

meanImg = np.mean(trainImages,1)[:,np.newaxis]
trainImages -= meanImg
testImages -= meanImg
validationImages -= meanImg


batchSize = 100
printEvery = 5
nEpochs = 50
nClasses = 10
learningRate = 0.0001 #IF used....
lambdaa = 0.05


W = np.random.randn(nClasses,nFeatures)
b = np.zeros(nClasses)


# In[11]:




# z = W.dot(trainImages) + b[:,np.newaxis]
# z -= np.max(z)
# zExp = np.exp(z)
# zExp.shape
# print(np.exp(np.max(z)))
# print(np.max(zExp))

# loss = -np.sum(np.multiply(trainLabels,(z-np.sum(zExp,0))),0)

# dz = (zExp/np.sum(zExp,0))-trainLabels
# dW = dz.dot(trainImages.T)
# db = np.sum(dz,1)
# np.max(z).shape

# a = zExp/np.sum(zExp,0)
# correct = np.sum(np.equal(np.argmax(a,0),np.argmax(trainLabels,0)))
# print(np.max(z,0).shape)


# In[12]:


def findOut(x):
    z1 = W.dot(x) + b[:,np.newaxis]
    z1 -= np.max(z1,0)
    zExp = np.exp(z1)
    a = zExp/np.sum(zExp,0)
    return a


# In[13]:


for _ in range(nEpochs):
    loss = 0.0
    #Forward....
    z = W.dot(trainImages) + b[:,np.newaxis]
    z -= np.max(z,0)
    zExp = np.exp(z)
    loss = -np.sum(np.multiply(trainLabels,(z-np.log(np.sum(zExp,0)))),0)
    a = zExp/np.sum(zExp,0)
#     print(np.equal(np.argmax(findOut(testImages),0),np.argmax(testLabels,0)).shape)
    corrert = np.sum(np.equal(np.argmax(findOut(testImages),0),np.argmax(testLabels,0)))/testLabels.shape[1]
    corrert2 = np.sum(np.equal(np.argmax(a,0),np.argmax(trainLabels,0)))/m
    print('Loss:',np.sum(loss),corrert,corrert2)
    
    #Backward
    dz = (zExp/np.sum(zExp,0))-trainLabels
    dW = dz.dot(trainImages.T)
    db = np.sum(dz,1)
    
    W = W - learningRate*dW - lambdaa*W
    b = b - learningRate*db

