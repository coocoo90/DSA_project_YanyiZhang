__author__ = 'zhangyanyi'
import subprocess, sys, time, random, math
import numpy as np
import gzip
import cPickle
import os



from collections import defaultdict
import sys
from numpy import *

class kNN(object):

    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

    def distance(self, img1, img2):
        img1=img1.reshape(28,28)
        img2=img2.reshape(28,28)
        distance = sum((img1[:][:] - img2[:][:]) ** 2)
        return distance


    def get_majority(self, votes):

        counter = np.zeros((10))
        for vote in votes:
            counter[vote] += 1
        max= counter.max()
        for i in range(10):
            if counter[i]==max:
                return i
    def predict(self, point):

        candidates = self.dataset[:]

        neighbors = []
        while len(neighbors) < self.k:
            distances = [self.distance(x[0], point) for x in candidates]

            best_distance = min(distances)
            index = distances.index(best_distance)
            neighbors.append(candidates[index])

            del candidates[index]

        prediction = self.get_majority([value[1] for value in neighbors])
        return prediction
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


dataset_size=500
ks = 5

train_set, valid, test_set= load_mnist_dataset("mnist.pkl.gz")

train = train_set[0][:dataset_size]
traint = train_set[1][:dataset_size]
test=[]
test.append(test_set[0][0:100])
test.append(test_set[1][0:100])
print test[0].shape
start = time.time()

dataset = []

for i in range(0, len(train)):
    dataset.append((train[i], traint[i]))


knn = kNN(dataset, ks)

predictions=[knn.predict(test[0][i]) for i in range(0, shape(test[0])[0])]

labels = asarray(test[1])
correct=0
total=0
for i in range(len(predictions)):
    if predictions[i]==labels[i]:
        correct+=1
    total+=1
accuracies=float(correct) / total

print 'k='+str(ks) + ': '+ str(100*accuracies) + '% accuracy'

end = time.time()
print "time="+str(end-start)+"s"
