from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import math
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import os
import inspect
import sys
import datetime
import cProfile
from enum import Enum
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
import threading
from tensorflow.python.platform import flags
from sklearn.metrics import confusion_matrix
from tensorflow.python.client import timeline

from customized_activations import maxout
from customized_rnncell import NewGRUCell

import Learning_rate
import Monitor
from param import RNNType
from MyThread import MyThread
from Log import Log
from Evaluate import evaluate_accuracy, evaluate_stat, evaluate_confusion
import Data
import Config

# for testing only
import cProfile


# check num of parameters
if len(sys.argv) < 2:
    dconfile = 'config.json'
elif (sys.argv[1].isdigit()):
    dconfile = 'config.json'
    test_task = int(sys.argv[1])  # speed up testing
else:
    dconfile = sys.argv[1]

logPath = './log/'
dataPath = './data/'

conf = Config.DataConfig(confile=dconfile)
task = conf.task

# overwrite testing task
try:
    test_task
except NameError:
    pass
else:
    conf.test_id = [test_task]

# this data are generated from create_npy.py
x_file = 'x_mobility_context.npy'
y_file = 'y_mobility_point.npy'
mmsi_file = 'mmsi_mobility_point.npy'

# selection of cell type
rnnType = RNNType.GRU_b

gpuMode = conf.useGPU
exp_seq_len = conf.truncated_seq_len
deep_output = False
use_dropout = False
weight_initializer = conf.weight_initializer
evaluate_freq = conf.evaluate_freq

bias_initializer = tf.random_uniform_initializer(0, 0.001)

if conf.activation == "maxout":
    rnnCell = NewGRUCell
    activation_function = tf.nn.tanh
else:
    rnnCell = tf.contrib.rnn.GRUCell
    if conf.activation == "sigmoid":
        activation_function = tf.nn.sigmoid
    elif conf.activation == "relu":
        activation_function = tf.nn.relu
    else:
        activation_function = tf.nn.tanh

lr = Learning_rate.Learning_rate(global_lr=0.001, decay_rate=0.999, decay_step=50)

# load data
x = np.load(dataPath + x_file)
y = np.load(dataPath+y_file)
mmsi = np.load(dataPath+mmsi_file)

# feature selection
def filter_features(x):
    print("warning: not all featuers are used")
    x = x[:, :, 0:40]
    return x

#x = filter_features(x)


def filter_classes(x, y, mmsi, cls):
    valid_index = np.concatenate([np.where(mmsi == i) for i in cls], axis=1)[0]

num_features = x.shape[2]

(x, y, mmsi) = Data.Data.reorganizeSeq(x, y, mmsi, exp_seq_len)

num_examples = x.shape[0]
unique_mmsi = np.unique(mmsi[0])
num_classes = len(np.unique(y))

test_vessel = conf.test_id
val_vessel = conf.val_id

if conf.testmode == "lobo":
    (train_index, test_index, valid_index) = Data.Data.splitDataset(mmsi[0], test_vessel, val_vessel)
elif conf.testmode == "random":
    (train_index, test_index, valid_index) = Data.Data.randomSplitDataset(mmsi[0], train_perc = conf.train_ratio, val_perc = conf.val_ratio)

print(train_index)

train_seq_len = mmsi[1][train_index]
test_seq_len = mmsi[1][test_index]
valid_seq_len = mmsi[1][valid_index]

num_class = np.unique(y).size

log = Log(task, logPath, num_class)
monitor = Monitor.Monitor(loss=True, num_class=num_class)

def encode_label(y):
    """encode label into a matrix based on the number of classes"""
    num_class = np.unique(y).size
    if num_class > 2: # multi-class
        lb = preprocessing.LabelBinarizer()
        lb.fit(range(num_class))
        labels = np.array([lb.transform(i) for i in y])
        #labels = lb.transform(y)
    else: # 2-class
    # the labels are stored in reserve in the numpy array
    # fishing is labeled 0
        Y0 = np.logical_not(y) * 1 # Y1 represents fishing
        Y1 = y # Y0 represents non-fishing
        labels = np.array([Y0, Y1])
        labels = labels.transpose(1,2,0) # dim: [example; length; classes]

    return labels

#labels = encode_label(y) # no need to encode y
labels = y
    
def get_all_data(conf):
    """generate data for all vessels"""
    early = mmsi[1]
    X = x.transpose((1, 0, 2))
    return (X, labels, early)

class VesselModel(object):
    """The vessel classification lstm model."""

    def __init__(self, config):
        self.num_threads = conf.num_threads
        self.hidden_size = conf.hidden_size
        self.learning_rate = conf.learning_rate
        self.num_layers = conf.num_layers
        self.num_epochs = conf.num_epochs
        self.batch_size = config.batch_size
        self.is_training = config.is_training
        self.is_validation = config.is_validation

        self.current_step = tf.Variable(0)
        # place holder for sequence that we will provide at runtime
        # batch size will be different for training and testing set
        self._input_data = tf.placeholder(tf.float32, [exp_seq_len, self.batch_size, num_features], name="input-data")

        # target for one batch
        self._targets = tf.placeholder(tf.int64, [self.batch_size, exp_seq_len], name = "y-target")

        # get the length of all training and test sequences
        if self.is_training:
            self.seq_len = exp_seq_len*self.batch_size #sum(train_seq_len)
        elif self.is_validation:
            self.seq_len = sum(valid_seq_len)
        else:
            self.seq_len = sum(test_seq_len)

        with tf.name_scope("lstm-cell") as scope:
            rnn_cell = self.get_rnn_cell()

        with tf.name_scope("multi-rnn-cell") as scope:
            cell = self.get_multi_rnn_cell(rnn_cell)

        # what timesteps we want to stop at, notice it's different for each batch
        self._early_stop = tf.placeholder(tf.int64, shape=[self.batch_size], name = "early-stop")

        self.set_initial_states(cell)

        #with tf.name_scope("dropout") as scope:
        #    if self.is_training and config.keep_prob < 1:
        #        self._input_data = tf.nn.dropout(self._input_data, config.keep_prob)

        outputs = []
        # Creates a recurrent neural network specified by RNNCell "cell
        # inputs for rnn needs to be a list, each item being a timestep. 
        # Args:
        #    cell: An instance of RNNCell.
        #    inputs: A length T list of inputs, each a tensor of shape
        #      [batch_size, cell.input_size].
        #    initial_state: (optional) An initial state for the RNN.  This must be
        #      a tensor of appropriate type and shape [batch_size x cell.state_size].
        #    dtype: (optional) The data type for the initial state.  Required if
        #      initial_state is not provided.
        #    sequence_length: Specifies the length of each sequence in inputs.
        #      An int32 or int64 vector (tensor) size [batch_size].  Values in [0, T).
        #    scope: VariableScope for the created subgraph; defaults to "RNN".
        #
        #  Returns:
        #    A pair (outputs, state) where:
        #      outputs is a length T list of outputs (one for each input)
        #      state is the final state
        with tf.name_scope("rnn-outputs") as scope:
            self.get_outputs(cell)

        self.valid_target = self.get_valid_sequence(tf.reshape(self._targets, [exp_seq_len * self.batch_size]), num_classes) # valid digit target
        self.lstm_output = self.valid_output

        if deep_output:
            with tf.name_scope("deep-output-layer") as scope:
                softmax_size = self.hidden_size * 2 if rnnType == RNNType.LSTM_b or rnnType == RNNType.GRU_b else self.hidden_size
                softmax_wout = tf.get_variable("softmax_w_deepout", [softmax_size, self.higher_hidden_size])
                softmaxb_dout = tf.get_variable("softmax_b_deepout", [self.higher_hidden_size])
                self.valid_output = tf.sigmoid(tf.matmul(self.valid_output, softmax_wout) + softmaxb_dout)
                if use_dropout:
                    self.valid_output = tf.nn.dropout(self.valid_output, keep_prob = 0.5)
                #softmax_wout2 = tf.get_variable("softmax_w_deepout2", [self.hidden_size, self.hidden_size])
                #softmaxb_dout2 = tf.get_variable("softmax_b_deepout2", [self.hidden_size])
                #self.valid_output = tf.matmul(self.valid_output, softmax_wout2) + softmaxb_dout2
                #if use_dropout:
                #    self.valid_output = tf.nn.dropout(self.valid_output, keep_prob = 0.5)

        with tf.name_scope("softmax-W") as scope:
            softmax_w = self.get_softmax_layer()
            self.w = softmax_w

        with tf.name_scope("softmax-b") as scope:
            softmax_b = tf.get_variable("softmax_b", [num_classes], initializer=bias_initializer)

        with tf.name_scope("softmax-predictions") as scope:
            self._predictions = tf.matmul(self.valid_output, softmax_w) + softmax_b
            self._prob_predictions = tf.nn.softmax(self._predictions)
            self.digit_predictions = tf.argmax(self._prob_predictions, axis=1)

        with tf.name_scope("confusion-matrix") as scope:
            self.confusion_matrix = tf.confusion_matrix(self.valid_target, self.digit_predictions)

        # Weighted cross-entropy loss for a sequence of logits (per example).
        # at: tensorflow/python/ops/seq2seq.py
        # Args:
        # logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        # targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        # weights: List of 1D batch-sized float-Tensors of the same length as logits.
        with tf.name_scope("seq2seq-loss-by-example") as scpoe:
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self._predictions],
                [self.valid_target],
                [tf.ones([int(self.getTensorShape(self.valid_target)[0])])])
            self._cost = tf.reduce_mean(self.loss)
            self._accuracy = tf.contrib.metrics.accuracy(self.digit_predictions, self.valid_target)

        # Add summary ops to collect data
        if conf.tensorboard:
            self.w_hist = tf.summary.histogram("weights", softmax_w)
            self.b_hist = tf.summary.histogram("biases", softmax_b)
            self.y_hist_train = tf.summary.histogram("train-predictions", self._predictions)
            self.y_hist_test = tf.summary.histogram("test-predictions", self._predictions)
            self.mse_summary_train = tf.summary.scalar("train-cross-entropy-cost", self._cost)
            self.mse_summary_test = tf.summary.scalar("test-cross-entropy-cost", self._cost)
    
        with tf.name_scope("optimization") as scope:
            self._train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self._cost, global_step=self.current_step)
            #self._train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self._cost, global_step=self.current_step)

    def get_rnn_cell(self):
        """Create rnn_cell based on RNN type"""
        if rnnType == RNNType.LSTM_b:
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True, use_peepholes=conf.peephole)
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True, use_peepholes=conf.peephole)
            return (lstm_cell_fw, lstm_cell_bw)
        elif rnnType == RNNType.LSTM_u:
            lstm_cell = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1, state_is_tuple=True, orthogonal_scale_factor=conf.init_scale, initializer = weight_initializer)
            return lstm_cell
        elif rnnType == RNNType.GRU:
            gru_cell = rnnCell(self.hidden_size, activation=activation_function)
            return gru_cell
        else:
            lstm_cell_fw = rnnCell(self.hidden_size, activation=activation_function)
            lstm_cell_bw = rnnCell(self.hidden_size, activation=activation_function)
            return (lstm_cell_fw, lstm_cell_bw)

    def get_multi_rnn_cell(self, rnn_cell):
        """Create multiple layers of rnn_cell based on RNN type"""
        if rnnType == RNNType.LSTM_b or rnnType == RNNType.GRU_b:
            (lstm_cell_fw, lstm_cell_bw) = rnn_cell
            cell_fw = tf.contrib.rnn.MultiRNNCell([rnnCell(self.hidden_size, activation=activation_function) for _ in range(self.num_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell([rnnCell(self.hidden_size, activation=activation_function) for _ in range(self.num_layers)])
            return (lstm_cell_fw, lstm_cell_bw)
        elif rnnType == RNNType.LSTM_u or rnnType == RNNType.GRU:
            cell = tf.contrib.rnn.MultiRNNCell([rnnCell(self.hidden_size, activation=activation_function) for _ in range(self.num_layers)])
            return cell

    def set_initial_states(self, cell):
        """set initial states based on RNN types"""
        # Initial state of the LSTM memory
        # If `state_size` is an int or TensorShape, then the return value is a
        # `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
        # If `state_size` is a nested list or tuple, then the return value is
        # a nested list or tuple (of the same structure) of `2-D` tensors with
        # the shapes `[batch_size x s]` for each s in `state_size`.
        if rnnType == RNNType.LSTM_b or rnnType == RNNType.GRU_b:
            (cell_fw, cell_bw) = cell
            self.initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            self.initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
        elif rnnType == RNNType.LSTM_u or rnnType == RNNType.GRU:
            self._initial_state = cell.zero_state(self.batch_size, tf.float32)

    def get_outputs(self, cell):
        """ get output tensor of the RNN"""
          # At: tensorflow/tensorflow/python/ops/rnn.py
          # Args:
          # Unlike `rnn`, the input `inputs` is not a Python list of `Tensors`.  Instead,
          # it is a single `Tensor` where the maximum time is either the first or second
          # dimension (see the parameter `time_major`).  The corresponding output is
          # a single `Tensor` having the same number of time steps and batch size.
          #
          # If time_major == False (default), this must be a tensor of shape:
          #    `[batch_size, max_time, input_size]`, or a nested tuple of such elements
          # If time_major == True, this must be a tensor of shape:
          #    `[max_time, batch_size, input_size]`, or a nested tuple of such elements
          #
          # Returns:
          # If time_major == False (default), this will be a `Tensor` shaped:
          #     `[batch_size, max_time, cell.output_size]`.
          # If time_major == True, this will be a `Tensor` shaped:
          #     `[max_time, batch_size, cell.output_size]`.
        if rnnType == RNNType.LSTM_b or rnnType == RNNType.GRU_b:
            (cell_fw, cell_bw) = cell
            self.outputs, self.state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self._input_data, sequence_length=self._early_stop, initial_state_fw=self.initial_state_fw, initial_state_bw=self.initial_state_bw, time_major=True, dtype='float32')
            output_fw, output_bw = self.outputs
            output_fw = tf.transpose(output_fw, perm=[1, 0, 2])
            output_bw = tf.transpose(output_bw, perm=[1, 0, 2])
            outputs = tf.concat(axis=2, values=[output_fw, output_bw])
            # Concatenates tensors along one dimension.
            # this will flatten the dimension of the matrix to [batch_size * num_steps, num_hidden_nodes]
            # However, this is not the true output sequence, since padding added a number of empty elements
            # Extra padding elements should be removed from the output sequence.
            # Here first concatenate all vessels into one long sequence, including paddings
            self.output = tf.reshape(tf.concat(axis=0, values=outputs), [exp_seq_len * self.batch_size, self.hidden_size*2])
            # Remove padding here
            self.valid_output = self.get_valid_sequence(self.output, self.hidden_size*2)
        elif rnnType == RNNType.LSTM_u or rnnType == RNNType.GRU:
            self.outputs, self.state = tf.nn.dynamic_rnn(cell, self._input_data, sequence_length=self._early_stop, initial_state=self._initial_state, time_major=True, dtype='float32')
            # This is a workaround with tf.reshape().
            # To make data with the same vessel continguous after reshape,
            # we need to transpose it first.
            outputs = tf.transpose(self.outputs, perm=[1, 0, 2])
            # Concatenates tensors along one dimension.
            # this will flatten the dimension of the matrix to [batch_size * num_steps, num_hidden_nodes]
            # However, this is not the true output sequence, since padding added a number of empty elements
            # Extra padding elements should be removed from the output sequence.
            # Here first concatenate all vessels into one long sequence, including paddings
            self.output = tf.reshape(tf.concat(axis=0, values=outputs), [exp_seq_len * self.batch_size, self.hidden_size])
            # Remove padding here
            self.valid_output = self.get_valid_sequence(self.output, self.hidden_size)

    def get_softmax_layer(self):
        if deep_output:
            softmax_w = tf.get_variable("softmax_w", [self.higher_hidden_size, num_classes])
        elif rnnType == RNNType.LSTM_b or rnnType == RNNType.GRU_b:
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size*2, num_classes])
        elif rnnType == RNNType.LSTM_u or rnnType == RNNType.GRU:
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, num_classes])
        return softmax_w

    def get_valid_sequence(self, seq, feature_size):
        """remove padding from sequences"""
        if self.is_training:
            stop = train_seq_len
        elif self.is_validation:
            stop = valid_seq_len
        else:
            stop = test_seq_len
        valid_sequence_list = []
        for i in range(self.batch_size):
            if len(tf.Tensor.get_shape(seq)) == 2:
                sub_seq = tf.slice(seq, [exp_seq_len*i, 0], [ stop[i], feature_size])
            else:
                sub_seq = tf.slice(seq, [exp_seq_len*i], [stop[i]])
            valid_sequence_list.append(sub_seq)
        valid_sequence = tf.concat(axis=0, values=valid_sequence_list)
        return valid_sequence


    def getTensorShape(this, tensor):
        return tf.Tensor.get_shape(tensor)

    @property
    def prob_predictions(self):
        return self._prob_predictions

    @property
    def input_data(self):
        return self._input_data

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def predictions(self):
        return self._predictions

    @property
    def early_stop(self):
        return self._early_stop

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def train_op(self):
        return self._train_op

    @property
    def final_state(self):
        return self._final_state

def test_model(sess, minibatch):
    # test and validate model
    if conf.test_mode:
        run_batch(sess, mtest, test_data, tf.no_op(), minibatch)

    t_train = MyThread(run_batch, (sess, m, train_data, tf.no_op(), minibatch))
    t_test = MyThread(run_batch, (sess, mtest, test_data, tf.no_op(), minibatch))
    t_val = MyThread(run_batch, (sess, mval, val_data, tf.no_op(), minibatch))

    t_train.start()
    t_test.start()
    t_val.start()

    t_train.join()
    result_train = t_train.get_result()
    t_test.join()
    result_test = t_test.get_result()
    t_val.join()
    result_val = t_val.get_result()
    
    result = result_train + result_test + result_val
    monitor.new(result, minibatch)
    return result
    

def run_batch(session, m, data, eval_op, minibatch):
    """Runs the model on the given data."""
    # prepare data for input
    x, y, e_stop = data
    epoch_size = x.shape[1] // m.batch_size

    # record results, keep results for each minibatch in list
    costs = []
    correct = []

    for batch in range(epoch_size):
        x_batch = x[:,batch*m.batch_size : (batch+1)*m.batch_size,:]
        y_batch = y[batch*m.batch_size : (batch+1)*m.batch_size,:]
        e_batch = e_stop[batch*m.batch_size : (batch+1)*m.batch_size]

        temp_dict = {m.input_data: x_batch}
        temp_dict.update({m.targets: y_batch})
        temp_dict.update({m.early_stop: e_batch})

        #m.learning_rate = lr.get_lr()

        # train the model
        if m.is_training and eval_op == m.train_op:

            _ = session.run([eval_op], feed_dict=temp_dict)

            print("minibatch {0}: {1}/{2}, lr={3:0.5f}\r".format(minibatch, batch, epoch_size,m.learning_rate),)
            lr.increase_global_step()
            # track stats every 10 minibatches
            if minibatch % evaluate_freq  == 0:
                result = test_model(session, minibatch) # recursive function
                log.write(result, minibatch)
            minibatch += 1
        # test the model
        else:
            cost, confusion, accuracy, _ = session.run([m.cost, m.confusion_matrix, m._accuracy, eval_op], feed_dict=temp_dict)

            # keep results for this minibatch
            costs.append(cost)
            correct.append(accuracy * sum(e_batch))
    
            # print test confusion matrix
            if not m.is_training and not m.is_validation:
                 print(confusion)
                 # output predictions in test mode
                 if conf.test_mode:
                     pred = session.run([m._prob_predictions], feed_dict=temp_dict)
                     pred = np.array(pred)
                     np.set_printoptions(threshold=np.nan)
                     print(pred.shape)
                     print(pred)
                     #results = np.column_stack((tar, pred))
                     #np.savetxt("results/prediction.result", pred)#, fmt='%.3f')
                     print("output target and predictions to file prediction.csv")
                     exit()
    
            if batch == epoch_size - 1:
                accuracy = sum(correct) / float(sum(e_stop))
                return(sum(costs)/float(epoch_size), accuracy)

    # training: keep track of minibatch number
    return(minibatch)

def getPredFileName(minibatch):
    """get the output of the prediction files"""
    return (logPath+str(test_vessel[0])+'/pred-'+task + str(minibatch)+'.csv')

def getLearnedParameters(param_name='model/bidirectional_rnn/fw/gru_cell/candidate/weights:0', filename='learned_embedding'):
    #print(tf.trainable_variables())
    var = [v for v in tf.trainable_variables() if v.name == param_name][0]
    x = var.eval()
    np.savetxt(filename, x)

def main(_):
    now = time.time()
    # get config
    train_conf = Config.TrainingConfig(is_training = True, is_validation = False, batch_size = conf.batch_size)
    test_conf = Config.TrainingConfig(is_training = False, is_validation = False, batch_size = len(test_index))
    valid_conf = Config.TrainingConfig(is_training = False, is_validation = True, batch_size = len(valid_index))

    # prepare all data to evaluate
    with tf.Session() as session:
        X_all, Y_all, e_stop_all = get_all_data(test_conf)

    # random shuffle, very important for stochastic gradient descent with minibatch
    np.random.shuffle(train_index)

    # specify training and test vessels
    X_train = X_all[:,train_index,:]
    y_train = Y_all[train_index,:]
    stop_train = e_stop_all[train_index]

    #print(X_train.shape)
    #(X_train, y_train, stop_train) = Data.Data.upsample((X_train, y_train, stop_train), cls=1, times=4)
    #print(X_train.shape)

    perm = np.random.permutation(X_train.shape[1])    
    X_train = X_all[:,perm,:]
    y_train = Y_all[perm,:]
    stop_train = e_stop_all[perm]

    X_test = X_all[:,test_index,:]
    y_test = Y_all[test_index,:]
    stop_test = e_stop_all[test_index]

    X_valid = X_all[:,valid_index,:]
    y_valid = Y_all[valid_index,:]
    stop_valid = e_stop_all[valid_index]

    # delete variables to save RAM
    del X_all
    del Y_all
    del e_stop_all
     
    global train_data
    train_data  = (X_train, y_train, stop_train)
    global test_data
    test_data = (X_test, y_test, stop_test)
    global val_data
    val_data = (X_valid, y_valid, stop_valid)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.intra_op_parallelism_threads=conf.num_threads
#    config.log_device_placement=True
    session = tf.Session(config=config)
    minibatch = 0


    with tf.Graph().as_default(), session as sess:
        tf.set_random_seed(0)


        if weight_initializer == "uniform":
          initializer = tf.random_uniform_initializer(0, conf.init_scale)
        elif weight_initializer == "orthogonal":
          initializer = tf.orthogonal_initializer(gain=conf.init_scale)
        else:
          print("Error: wrong weight initializer")
          exit()


        with tf.variable_scope("model", reuse=None, initializer=initializer):
            global m
            m = VesselModel(config=train_conf)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            global mtest
            mtest = VesselModel(config=test_conf)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            global mval
            mval = VesselModel(config=valid_conf)

        if conf.checkpoint or conf.restore:
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        if conf.tensorboard:
            global writer
            writer = tf.summary.FileWriter(logPath+"tf-logs", sess.graph_def)

        if not conf.restore:
            tf.global_variables_initializer().run()     #initialize all variables in the model
        else:
            saver.restore(sess, dataPath+task)
            print("Model restored.")

        # training
        for i in range(conf.num_epochs):
            print("running epoch {0}".format(i))
            minibatch = run_batch(sess, m, train_data, m.train_op, minibatch)

        # get best results
        best = monitor.getBest()
        log.write(best, monitor.minibatch)
        log.close()

        # save the model
        if conf.checkpoint:
            # Save the variables to disk
            save_path = saver.save(sess, dataPath+task)
            print("Model saved in file: %s" % save_path)

    later = time.time()
    difference = int(later - now)
    print('time elapsed: {:} seconds'.format(difference))

def prof(main=None):
    f = flags.FLAGS
    f._parse_flags()
    main = main or sys.modules['__main__'].main

    profile=cProfile.Profile()
    profile.run('main(sys.argv)')
    kProfile=lsprofcalltree.KCacheGrind(profile)
    kFile=open('profile','w+')
    kProfile.output(kFile)
    kFile.close()


if __name__ == "__main__":
    if not gpuMode:
        with tf.device('/cpu:0'):
            tf.app.run()
    else:
        tf.app.run()
