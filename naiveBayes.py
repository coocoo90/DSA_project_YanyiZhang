__author__ = 'zhangyanyi'
import numpy as np
import time
import cPickle
import gzip
import os
import scipy.stats
class NaiveBayes():


    def __init__(self,labels,features):

        self.labelsNum=np.array(labels)
        self.featuresNum=np.array(features)
        self.cf=np.zeros((labels,features),dtype=np.float)
        self.std=np.zeros((labels,features),dtype=np.float)
        self.cy=np.zeros(labels,dtype=np.float)




    def train(self,datas,labels):
        global counter
        trainingDataNum=datas.shape[0]
        trainingLabelsNum=labels.shape[0]

        labels_all=np.zeros(self.labelsNum,dtype=np.float)

        for i in range(trainingLabelsNum):

            for j in range(self.labelsNum):
                counter+=1
                if labels[i]==j:
                    labels_all[j]+=1


        for i in range(self.labelsNum):

            sum=np.zeros(self.featuresNum,dtype=np.float)


            for j in range(trainingDataNum):
                counter+=1
                if labels[j]==i:
                    sum+=datas[j]

            self.cf[i] = sum / labels_all[i]

        for i in range(self.labelsNum):

            sum=np.zeros(self.featuresNum,dtype=np.float)

            for j in range(trainingDataNum):
                counter+=1
                if labels[j]==i:
                    sum+=(datas[j]-self.cf[i])**2

            self.std[i]=sum/labels_all[i]

        self.cy=labels_all/trainingDataNum

        return self.cy

    def prob(self,x,m,d):
        return 1.0 / (d * np.sqrt(2*np.pi)) * np.exp(-(x-m)**2/(2*d**2))


    def prediction(self,data):

        results=[]

        for i in range(self.labelsNum):

            p=0

            log_y=-np.log(self.cy[i])

            p+=log_y

            for j in range(self.featuresNum):
                epsiron = 1.0e-5
                if self.std[i][j] < epsiron:
                    p+=0
                else:
                    # p+=scipy.stats.norm(self.cf[i][j], self.std[i][j]).logpdf(data[j])
                    gaussian=self.prob(data[j],self.cf[i][j],self.std[i][j])
                    # print gaussian
                    # if gaussian==0:
                    #     pass
                    if gaussian==0:
                        # print data[j]
                        # print self.cf[i][j]
                        # print self.std[i][j]
                        # print "\n\n"
                        p+=-scipy.stats.norm(self.cf[i][j],self.std[i][j]).logpdf(data[j])

                    else:
                        p+=-np.log(gaussian)

            results.append(p)
            # print results
        return np.argmin(results)

def getInput(image,useFeature=False):
    if not useFeature:
        return np.array(image).flatten()
    else:
        # print np.array(image).shape
        return canny(np.array(image).reshape(28,28)).flatten()

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

counter=0
useFeature=False

train_set, valid_set, test_set = load_mnist_dataset("mnist.pkl.gz")
n_labels = 10 # 1,2,3,4,5,6,7,9,0
n_features = 784
mnist_model = NaiveBayes(n_labels, n_features)

# for i in range(len(train_set[0])):
#     train_set[0][i]*=255

trains=[]
for i in range(50000):
    trains.append(getInput(train_set[0][i],useFeature))
    # img=np.array(train_set[0][i]).reshape(28,28)
    # trains.append(canny(img).flatten())

starttime=time.time()
mnist_model.train(np.array(trains), train_set[1][:50000])
endtime=time.time()

print "time="+str(endtime-starttime)+"s"

test_data, labels = test_set
limit = 100
test_data, labels = test_data[:limit], labels[:limit]


results = np.arange(limit, dtype=np.int)

correct=0
for n in range(limit):
    features=getInput(test_data[n],useFeature)
    results[n] = mnist_model.prediction(np.array(features).flatten())
    if (results[n]==labels[n]):
        correct+=1

print "accuracy="+str(float(correct)/limit)
