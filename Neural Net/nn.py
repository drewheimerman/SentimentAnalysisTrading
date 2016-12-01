# Imports
import tensorflow as tf
import math
import numpy as np

def main():
    sess = tf.InteractiveSession()
    # Constants
    INPUT_VECTOR_LENGTH = 10
    OUTPUT_VECTOR_LENGTH = 2
    STEP_LENGTH = 0.5 # alpha
    SIZE_PER_HIDDENLAYER = [INPUT_VECTOR_LENGTH - 3 * i for i in range(1,3)] #Linear decrement - 10->(7->4)->2
    # Add different layer progression?

    x = tf.placeholder(tf.float32, shape=[None, INPUT_VECTOR_LENGTH])
    y_ = tf.placeholder(tf.float32, shape=(OUTPUT_VECTOR_LENGTH))

    y = tf.placeholder(tf.int32, shape=(OUTPUT_VECTOR_LENGTH))

    sess.run(tf.initialize_all_variables())

    network = generate_network(x)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(network, y, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    tf.scalar_summary(loss.op.name, loss)

    optimizer = tf.train.GradientDescentOptimizer(STEP_LENGTH)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

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
        tf.truncated_normal([input_length, output_length],
                            stddev=1.0 / math.sqrt(float(input_length))),
        name='weights')
    biases = tf.Variable(tf.zeros([output_length]),
                         name='biases')
    return tf.matmul(input_layer, weights) + biases

def generate_network(x, sizes=SIZE_PER_HIDDENLAYER, tensor_type=tf.sigmoid):
    """

    generates a normally connected NN using tensors of type 'tensor_type', with the output as logits

    Args:
        x: input value placeholder
        sizes: the number of nodes per hiddel layer, with the amount of hidden layers inferred from the length of the list

    Returns:
        tensor: network ready to be trained with output vector of size OUTPUT_VECTOR_LENGTH
    """
    prev_layer = None
    layer = None
    for i in range(0, len(sizes)):
        with tf.name_scope('hidden_%d' % i):
            if i == 0:
                layer = tensor_type(generate_layer_input(x, INPUT_VECTOR_LENGTH, sizes[i]))
            else:
                layer = tensor_type(generate_layer_input(prev_layer, sizes[i-1], sizes[i]))
            prev_layer = layer

    with tf.name_scope('logits'):
        layer = generate_layer_input(prev_layer, sizes[-1], OUTPUT_VECTOR_LENGTH)

    return layer
