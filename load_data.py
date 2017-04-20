import cPickle, gzip, numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import LogisticRegression

# For ploting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from itertools import compress
from numpy import genfromtxt

vesselnum = 5

print('load data...')
train_set = genfromtxt('/users/grad/xjiang/code/'+str(vesselnum)+'/trainoutput-99.csv', delimiter=',')
test_set = genfromtxt('/users/grad/xjiang/code/'+str(vesselnum)+'/testoutput-99.csv', delimiter=',')
valid_set = genfromtxt('/users/grad/xjiang/code/'+str(vesselnum)+'/valoutput-99.csv', delimiter=',')

def get_in_out(data):
    x = data[:,0:(data.shape[1]-2)]
    y = data[:, (data.shape[1]-2):(data.shape[1]-1)]
    y = y.astype(int)
    y = y.flatten()
    return (x,y)

train_x, train_y = get_in_out(train_set)
test_x, test_y = get_in_out(test_set)
valid_x, valid_y = get_in_out(valid_set)

train_set_x = theano.shared(numpy.array(train_x, dtype='float32'))
test_set_x = theano.shared(numpy.array(test_x, dtype='float32'))
valid_set_x = theano.shared(numpy.array(valid_x, dtype='float32'))
train_set_y = theano.shared(numpy.array(train_y, dtype='int32'))
test_set_y = theano.shared(numpy.array(test_y, dtype='int32'))
valid_set_y = theano.shared(numpy.array(valid_y, dtype='int32'))

print('data loaded...')
