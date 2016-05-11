__author__ = 'zhangyanyi'
import numpy as np
import cPickle
import gzip
import os
class Layers:


    def __init__(self,size,isIn,bias):
        self.size=size
        self.isIn=isIn
        self.bias=bias
        self.error=None
        self.inPut=None
        self.OutPut=None


class NeuralNetwork:

    def __init__(self,preLayer,currentLayer,weights=None):
        self.preLayer=preLayer
        self.currentlayer=currentLayer
        self.deltas=[]

        if currentLayer.bias:
            self.preLayerSize=preLayer.size+1
        else:
            self.preLayerSize=preLayer.size
        self.currentLayerSize=currentLayer.size

        if not weights:
           l, h = self.sampleInterval(self.preLayerSize, self.currentLayerSize)
           self.weights=np.random.uniform(low=l,high=h,size=[self.preLayerSize,self.currentLayerSize])
        else:
            self.weights=weights

    def sampleInterval(self, prev, curr):
        d = (- 1.0) * np.sqrt(6.0 / (prev + curr))
        return [-d, d]

    def forwardProp(self):

        inPut=self.preLayer.outPut

        if self.currentlayer.bias:
            inPut=np.append(inPut,[1])

        layerProduct=np.dot(self.weights.T,inPut)

        self.currentlayer.outPut=self.leaky_relu(layerProduct)

        if self.currentlayer.size==10:
            self.currentlayer.outPut=self.Softmax(self.currentlayer.outPut)


        return self.currentlayer.outPut


    def backProp(self,learningRate,result=None):

        outPut=self.currentlayer.outPut
        if result:
            curError=(outPut-result)*self.leaky_relu_deriv(outPut)
            self.currentlayer.error=curError
        else:
            curError=self.currentlayer.error

        curError=np.array(curError)


        preOut=np.array(self.preLayer.outPut)
        preOut=np.append(preOut,[1])

        if not self.preLayer.isIn:
            preError=[]
            for i in range(len(preOut)):

                err=sum(curError*self.weights[i])*self.leaky_relu_deriv_single(np.array(preOut[i]))
                preError.append(err)
            preError=np.array(preError)

            preError=np.delete(preError,-1)
            self.preLayer.error=preError

        delta=np.dot(preOut[np.newaxis].T,curError[np.newaxis])
        self.deltas.append(delta)

        if batchNum%batchSize==0:
            for i in range(len(self.deltas)):
                bb=self.deltas[i]*learningRate
                self.weights-=bb
            self.deltas=[]


    def tanh(self,x):
        return 1.7159 * np.tanh(2.0 * x / 3.0)

    def deriv(self, x):
        t = np.tanh(2.0 * x / 3.0) ** 2.0
        return 1.144 * (1 - t)

    def ReLU(self,z):
        """the ReLU activation function"""
        return np.max([z, np.zeros(z.shape)], axis=0)


    def Softmax(self,z):
        """the softmax activation function for the output layer, best suitable for
           disjoint classes"""
        out = np.exp(z)
        sum_exp = sum(out)
        res = out/sum_exp
        return res

    def relu_grad(self,z):
        """the gradient of ReLU(z)"""
        index = z >= 0
        result = np.zeros(z.shape)
        result[index] = 1.0
        return result

    def leaky_relu(self,x):
        res=np.zeros(x.shape)
        for i in range(len(res)):
            if x[i]>0:
                res[i]=x[i]
            else:
                res[i]=-0.01*x[i]
        return res


    def leaky_relu_deriv(self,x):
        res=np.zeros(x.shape)
        for i in range(len(res)):
            if x[i]>0:
                res[i]=1
            else:
                res[i]=-0.01
        return res

    def leaky_relu_deriv_single(self,x):
        res=0
        if x>0:
            res=1
        else:
            res=-0.01
        return res

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

inputLayer=Layers(784,True,True)
hiddenLayer1=Layers(196,False,True)
hiddenLayer2=Layers(98,False,True)
hiddenLayer3=Layers(49,False,True)
outPutLayer=Layers(10,False,True)

network1=NeuralNetwork(inputLayer,hiddenLayer1)
network2=NeuralNetwork(hiddenLayer1,hiddenLayer2)
network3=NeuralNetwork(hiddenLayer2,hiddenLayer3)
network4=NeuralNetwork(hiddenLayer3,outPutLayer)

f=gzip.open("ann_MNIST_edges_dynamic_batch_leaky_relu2_4.pkl")
(network1.weights,network2.weights,network3.weights,network4.weights)=cPickle.load(f)
f.close()

iterations=400
learningRate=0.00005
batchSize=50
oldTp=0
total=0
correct=0
for j in range(len(test_set[0])):

    inputLayer.outPut=np.array(np.array(test_set[0][j]).flatten())

    network1.forwardProp()
    network2.forwardProp()
    network3.forwardProp()
    result=network4.forwardProp()

    label=[]
    transferNumToBinaryArray(test_set[1][j],10,label)

    # print str(np.argmax(result))+"  "+str(np.argmax(label))
    # print np.argmax(label)

    if np.argmax(result) == np.argmax(np.array(label)):
        correct+=1
        # print "True!"
    total+=1
    TP_new=float(correct)/total

print "accuracy="+str(TP_new)