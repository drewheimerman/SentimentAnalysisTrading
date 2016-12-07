# Imports
import tensorflow as tf
import math
import numpy as np
import sklearn
from sklearn.model_selection import KFold
import sys
import yahoo_finance
import pandas as pd
import random
import yahoo_finance
import matplotlib.pyplot as pyplot

import logging
#Output logger

class Lg():
    def info(string):
        print(string)

    def error(string):
        print(string)

    def debug(string):
        print(string)

    def warn(string):
        print(string)

#logging.basicConfig(filename='./run.log', filemode='w',level=logging.DEBUG)
log = logging.getLogger(__name__)

import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

LOCAL = True

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

log = root
#log = Lg()

def main(_):
    sess = tf.InteractiveSession()
    # Constants

    global STEP_LENGTH
    STEP_LENGTH = 0.05 # alpha

    global TRAINING_ITERATIONS
    TRAINING_ITERATIONS = 10000

    global DATA_FILE
    log.info('Loading data file')
    DATA_FILE = '/Users/helpdesk/desktop/train.p'
    X, Y = load_data(DATA_FILE)
    if X is not None and Y is not None:
        log.info('Loaded training data')
    else:
        log.error('Could not load training data')
        sys.exit(0)


    global INPUT_VECTOR_LENGTH
    INPUT_VECTOR_LENGTH = len(X[0])
    global OUTPUT_VECTOR_LENGTH
    OUTPUT_VECTOR_LENGTH = 2
    global SIZE_PER_HIDDENLAYER
    SIZE_PER_HIDDENLAYER = [INPUT_VECTOR_LENGTH - 3 * i for i in range(1,3)] #Linear decrement - 10->(7->4)->2

    global NUM_FOLDS
    NUM_FOLDS = 5
    # Add different layer progression?



    x = tf.placeholder(tf.float32, shape=[None, INPUT_VECTOR_LENGTH], name='Inputs')

    #y_ = tf.placeholder(tf.float32, shape=(OUTPUT_VECTOR_LENGTH), name='Predicted_Outputs')

    y = tf.placeholder(tf.int64, shape=[None], name='Actual_Outputs')

    network, softmax = generate_network(x)
    y_=softmax


    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_, y, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    tf.scalar_summary(loss.op.name, loss)

    optimizer = tf.train.GradientDescentOptimizer(STEP_LENGTH)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    kf = CustomKFold(n_splits=NUM_FOLDS)
    log.info(('beginning neural network'))
    #for _ in range(TRAINING_ITERATIONS)
    j=0
    e_ins, e_outs = list(), list()
    for train, test in kf.split(X):
        j+= 1
        prev_acc = 0
        sess.run(tf.global_variables_initializer())
        x_train, y_train, x_test, y_test = X[train], Y[train], X[test], Y[test]
        e_in = list()
        e_out = list()
        for _ in range(TRAINING_ITERATIONS):
            #log.info('beginning epoch %s' % str(_))
            sess.run(train_op, feed_dict={x: x_train, y:y_train})
            correct_prediction = tf.equal(y, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            e_in.append(1 - sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
            e_out.append(1 - sess.run(accuracy, feed_dict={x: x_test,
                                            y: y_test}))
            if(_ % 10 == 0 and _ != 0):
                log.info('Fold %d: epoch %d: e_in, eout: %f, %f' %(j, _, e_in[-1], e_out[-1]))
                if((e_out[-1] == e_out[-2] and e_in[-1] == e_in[-2]) and _ > 50):
                    break
        e_ins.append(e_in)
        e_outs.append(e_out)
    nums = [i for i in range(61)]

    for i in range(len(e_ins)):
        pyplot.figure(i+1)
        pyplot.axis([0,60,0, 1])
        line1, line2 = pyplot.plot(nums, e_ins[i], nums, e_outs[i])
        pyplot.legend([line1,line2],['E_in', 'E_out'])
    pyplot.show()
class CustomKFold():

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, dataset):
        length, __ = dataset.shape
        nums = [i for i in range(length)]
        data  = np.array_split(nums, self.n_splits)
        for i in range(self.n_splits):
            temp  = np.delete(data, i)
            yield (np.hstack(temp.flat), np.array(data[i], dtype=int))





def generate_layer_input(input_layer, input_length, output_length):
    """
    Generate a layer for a normal NN of type 'tensor_type'

    Args:
        input_layer: layer to be used as input to this layer
        input_length: length of 'input_layer'
        output_length: length of the vector for this layer

    Returns:
        the product of 'input_layer' and a newly generated weight vector
        (initialized with a normal distribution), added to a bias vector

    """
    weights = tf.Variable(
        tf.truncated_normal([ input_length, output_length],
                            stddev=1.0 / math.sqrt(float(input_length))),
        name='weights', dtype=tf.float32)
    biases = tf.Variable(tf.zeros([output_length]),
                         name='biases', dtype=tf.float32)
    return tf.matmul(input_layer, weights) + biases

def generate_network(x, sizes=[5,2], tensor_type=tf.sigmoid):
    """

    generates a normally connected NN using tensors of type 'tensor_type', with the output as logits

    Args:
        x: input value placeholder
        sizes: the number of nodes per hidden layer, with the amount of hidden layers inferred from the length of the list

    Returns:
        tensor: network ready to be trained with output vector of size OUTPUT_VECTOR_LENGTH
    """
    first = None
    prev_layer = None
    layer = None
    for i in range(0, len(sizes)):
        with tf.name_scope('hidden_%d' % i):
            if i == 0:
                layer = tensor_type(generate_layer_input(x, INPUT_VECTOR_LENGTH, sizes[i]))
                first = layer
            else:
                layer = tensor_type(generate_layer_input(prev_layer, sizes[i-1], sizes[i]))
            prev_layer = layer

    with tf.name_scope('logits'):
        layer = generate_layer_input(prev_layer, sizes[-1], OUTPUT_VECTOR_LENGTH)

    prev_layer = layer
    w = tf.Variable(
        tf.truncated_normal([OUTPUT_VECTOR_LENGTH, OUTPUT_VECTOR_LENGTH],
        stddev=1.0 / math.sqrt(float(OUTPUT_VECTOR_LENGTH))),
        name='weights_softmax', dtype=tf.float32)
    b = tf.Variable(tf.zeros([OUTPUT_VECTOR_LENGTH]),
        name='biases_softmax', dtype=tf.float32)

    # Apply dropout?
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    softmax = tf.matmul(prev_layer, w) + b
    return first, softmax

def load_data(path):
    log.debug('Unpickling data from %s' % path)
    data = pd.read_pickle(path)
    log.debug('Unpickled data from %s' % path)

    aapl = None
    vals = list()
    stock_data = list()
    if LOCAL:
        aapl = pd.read_csv('./table.csv', index_col=0)
    else:
        aapl = yahoo_finance.Share('AAPL')
    log.debug('Stock downloader ready')
    for i in range(len(data.index)):
        date = transform_date(data.index[i])
        try:
            s = None
            if LOCAL:
                try:
                    s = aapl.loc[date]
                except KeyError:
#                    log.error('%s' % date)
                    continue
            else:
                log.debug('Donwloading %s %s' % ('AAPL', date))
                s = aapl.get_historical(date, date)[0]
                logging.debug('Donwloaded %s %s' % ('AAPL', date))
            stock_data.append(0 if s['Close'] > s['Open'] else 1)
        except yahoo_finance.YQLResponseMalformedError:
            continue
        row = data.iloc[i]
        vals.append(row.values)

    return np.array(vals, dtype=np.float32), np.array(stock_data, dtype=np.int32)

def transform_date(date):
    date = date.split('-')
    trim_zero = lambda x: x[1] if x[0] == '0' else x
    return (trim_zero(date[1]) + '/' + trim_zero(date[2]) + '/' + date[0][-2:])

if __name__ == '__main__':
    tf.app.run(main=main)
