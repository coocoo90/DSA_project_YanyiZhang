__author__ = 'zhangyanyi'
import cPickle
import os
import numpy as np
import gzip
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

def relu(z):
    return np.max([z, np.zeros(z.shape)], axis=0)

def transferNumToBinaryArray(num,size,a):
    
    for i in range(0,size):
        if i==num:
            a.append(1)
        else:
            a.append(0)

def Softmax(z):
    out = np.exp(z)
    sum_exp = sum(out)
    res = out/sum_exp
    return res

def relu_grad(z):
    index = z >= 0
    result = np.zeros(z.shape)
    result[index] = 1.0
    return result

class Layers:
    def __init__(self,size,isIn,isOut):
        self.Out=None
        self.isIn=isIn
        self.isOut=isOut
        self.size=size
        self.error=None
class ANN:
    def __init__(self,preLayer,currLayer,weight=None):
        self.preLayer=preLayer
        self.currLayer=currLayer
        self.bias=np.random.random_sample([currLayer.size])
        
        if not weight:
            l, h = self.sampleInterval(preLayer.size,currLayer.size)
            self.weights=np.random.uniform(low=l,high=h,size=[preLayer.size,currLayer.size])
        # self.weights=np.random.random_sample([preLayer.size,currLayer.size])
        else:
            self.weights=weight

def sampleInterval(self, prev, curr):
    d = (- 1.0) * np.sqrt(6.0 / (prev + curr))
        return [-d, d]
    def ff(self):
        preOut=self.preLayer.Out
        product=relu(np.dot(self.weights.T,preOut)+self.bias)
        
        if self.currLayer.isOut:
            self.currLayer.Out=Softmax(product)
        else:
            self.currLayer.Out= product
        return self.currLayer.Out

def bp(self,eta,labels=None):
    outPut=self.currLayer.Out
        error=None
        if self.currLayer.isOut:
            error=(outPut-labels)*relu_grad(outPut)
            self.currLayer.error=error
    else:
        error=self.currLayer.error
        
        if not self.preLayer.isIn:
            preError=(error*self.weights).sum()*relu_grad(self.preLayer.Out)
            self.preLayer.error=preError

delta=np.dot(self.preLayer.Out[np.newaxis].T,error[np.newaxis])
    self.weights-=delta*eta
        self.bias-=error*eta


inputLayer=Layers(784,True,False)
hiddenLayer1=Layers(196,False,False)
hiddenLayer2=Layers(49,False,False)
outPutLayer=Layers(10,False,True)

network1=ANN(inputLayer,hiddenLayer1)
network2=ANN(hiddenLayer1,hiddenLayer2)
network3=ANN(hiddenLayer2,outPutLayer)


train_set, valid_set, test_set=load_mnist_dataset("mnist.pkl.gz")
iterations=400
learningRate=0.0005
batchSize=50
oldTp=0

batchNum=1

for i in range(iterations):
    total=0
    correct=0
    
    for j in range(len(train_set[0])):
        
        # canny=features.canny(np.array(train_set[0][j]).reshape(28,28))
        inputLayer.Out=np.array(np.array(train_set[0][j]))
        
        network1.ff()
        network2.ff()
        result=network3.ff()
        
        label=[]
        transferNumToBinaryArray(train_set[1][j],10,label)
        
        # print np.argmax(result)
        # print np.argmax(label)
        
        if np.argmax(result) == np.argmax(np.array(label)):
            correct+=1
        # print "True!"
        total+=1
        TP_new=float(correct)/total
        print str(i )+"iteration "+str(j)+"sample "+"predicted: "+str(result)+" label: "+str(np.array(label))
        print "TP= "+str(TP_new)
        
        
        network3.bp(learningRate,label)
        network2.bp(learningRate)
        network1.bp(learningRate)
        batchNum+=1
    
    print str(i)+" th iteration"+" TP= "+str(TP_new)
