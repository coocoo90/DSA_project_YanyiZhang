__author__ = 'zhangyanyi'
import numpy as np
from pylab import *
import cPickle
import gzip
import os
import skimage.feature as features
class Layers:
    error=None
    inPut=None
    OutPut=None
    def __init__(self,size,isIn):
        self.size=size
        self.isIn=isIn

class ANN:

    def __init__(self,previous,current,weights=None):
        self.previous=previous
        self.current=current

        self.previousSize=previous.size+1

        self.currentSize=current.size

        if not weights:
           self.weights=np.random.random_sample([self.previousSize,self.currentSize])
        else:
            self.weights=weights


    def feed(self):

        inPut=self.previous.outPut

        inPut=np.append(inPut,[1])

        layerProduct=np.dot(self.weights.T,inPut)

        self.current.outPut=self.tanh(layerProduct)

        return self.current.outPut


    def bp(self,learningRate,result=None):

        outPut=self.current.outPut
        if result:
            curError=(outPut-result)*self.deriv(outPut)
            self.current.error=curError
        else:
            curError=self.current.error

        curError=np.array(curError)

        preOut=np.array(self.previous.outPut)
        preOut=np.append(preOut,[1])

        if not self.previous.isIn:
            preError=[]
            for i in range(len(preOut)):

                err=sum(curError*self.weights[i])*self.deriv(preOut[i])
                preError.append(err)
            preError=np.array(preError)

            preError=np.delete(preError,-1)
            self.previous.error=preError

        delta=np.dot(preOut[np.newaxis].T,curError[np.newaxis])
        self.weights-=delta*learningRate


    def tanh(self,x):
        return 1.7159 * np.tanh(2.0 * x / 3.0)

    def deriv(self, x):
        t = np.tanh(2.0 * x / 3.0) ** 2.0
        return 1.144 * (1 - t)

def load_mnist_dataset(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set

def transferNumToBinaryArray(num,size,a):

    for i in range(0,size):
        if i==num:
            a.append(1)
        else:
            a.append(0)

train_set, valid_set, test_set=load_mnist_dataset("mnist.pkl.gz")

inputLayer=Layers(784,True)
hidden1=Layers(196,False)
hidden2=Layers(49,False)
outPut=Layers(10,False)

network1=ANN(inputLayer,hidden1)
network2=ANN(hidden1,hidden2)
network3=ANN(hidden2,outPut)

iterations=4000
learningRate=0.001

oldTp=0
for i in range(iterations):
    total=0
    correct=0
    for j in range(len(train_set[0])):

        canny=features.canny(np.array(train_set[0][j]).reshape(28,28))
        inputLayer.outPut=np.array(np.array(canny).flatten())

        network1.feed()
        network2.feed()
        result=network3.feed()

        label=[]
        transferNumToBinaryArray(train_set[1][j],10,label)

        # print np.argmax(result)
        # print np.argmax(label)

        if np.argmax(result) == np.argmax(np.array(label)):
            correct+=1
            print "True!"
        total+=1
        TP_new=float(correct)/total
        print str(i )+"iteration "+str(j)+"sample "+"predicted: "+str(result)+" label: "+str(np.array(label))
        print "TP= "+str(TP_new)


        network3.bp(learningRate,label)
        network2.bp(learningRate)
        network1.bp(learningRate)





