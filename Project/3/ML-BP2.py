
# coding: utf-8

# In[1]:


import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[2]:


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
nEpochs = 100
nClasses = 10
learningRate = 0.00005 #IF used....
lambdaa = 0.01
nH1 = 512





# In[3]:


W1 = np.random.randn(nH1,nFeatures)
b1 = np.zeros(nH1)

W2 = np.random.randn(nClasses,nH1)
b2 = np.zeros(nClasses)



# z1 = W1.dot(trainImages) + b1[:,np.newaxis]
# a1 = np.maximum(z1,0,z1)

# z2 = W2.dot(a1) + b2[:,np.newaxis]
# z2 -= np.max(z2)
# z2Exp = np.exp(z2)
# a2 = np.maximum(z1,0,z1)

# loss = -np.sum(np.multiply(trainLabels,(z2-np.sum(z2Exp,0))),0)

# dz2 = (z2Exp/np.sum(z2Exp,0))-trainLabels
# dW2 = dz2.dot(a1.T)
# db2 = np.sum(dz2,1)

# da1 = W2.T.dot(dz2)
# dz1 = np.zeros_like(da1)
# dz1[da1>0] = 1

# dW1 = dz1.dot(trainImages.T)
# db1 = np.sum(dz1,1)
# print(dW1.shape,db1.shape)


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


# In[4]:


def findOut(x):
    z1 = W1.dot(x) + b1[:,np.newaxis]
    a1 = np.maximum(z1,0,z1)

    z2 = W2.dot(a1) + b2[:,np.newaxis]
    z2 -= np.max(z2,0)
    z2Exp = np.exp(z2)
    a = z2Exp/np.sum(z2Exp,0)
#     print(a.shape)
    return a


# In[ ]:


for _ in range(nEpochs):
    loss = 0.0
    #Forward....
    z1 = W1.dot(trainImages) + b1[:,np.newaxis]
#     a1 = np.maximum(z1,0,z1)
    a1 = np.maximum(z1,0)
    
    z2 = W2.dot(a1) + b2[:,np.newaxis]
    z2 -= np.max(z2,0)
    z2Exp = np.exp(z2)
#     z2Exp[z2Exp<=0] = 1e-10
    loss = -np.sum(np.multiply(trainLabels,z2-np.log(np.sum(z2Exp,0))),0) 
    
#     print(np.equal(np.argmax(findOut(testImages),0),np.argmax(testLabels,0)).shape)
    correct1 = np.sum(np.equal(np.argmax(findOut(trainImages),0),np.argmax(trainLabels,0)))/m
    correct2 = np.sum(np.equal(np.argmax(findOut(testImages),0),np.argmax(testLabels,0)))/testLabels.shape[1]
    correct3 = np.sum(np.equal(np.argmax(findOut(validationImages),0),np.argmax(validationLabels,0)))/validationLabels.shape[1]
    print('Loss:',np.sum(loss), correct1, correct2, correct3)
    
    #Backward
    dz2 = (z2Exp/np.sum(z2Exp,0))-trainLabels
#     print(np.sum(dz2))
    dW2 = dz2.dot(a1.T)
    db2 = np.sum(dz2,1)

    da1 = W2.T.dot(dz2)
    dz1 = np.zeros_like(da1)
    dz1[da1>0] = 1

    dW1 = dz1.dot(trainImages.T)
    db1 = np.sum(dz1,1)
    
    W2 = W2 - learningRate*dW2 - lambdaa*W2
    b2 = b2 - learningRate*db2
    W1 = W1 - learningRate*dW1 - lambdaa*W1
    b1 = b1 - learningRate*db1

