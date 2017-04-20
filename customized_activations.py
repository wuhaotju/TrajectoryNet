import tensorflow as tf

def maxout(x):
  """maxout activation"""
  with tf.variable_scope('_maxout'):
    num_units = 3

    shape = x.get_shape().as_list()

    axis = -1
    dim = shape[axis]

    outputs = None
    for i in range(num_units):
      with tf.variable_scope(str(i)):
        W = tf.get_variable('W_%d' % i, (dim, dim))
        b = tf.get_variable('b_%d' % i, (dim,), initializer = tf.random_uniform_initializer(0,0.001))
        y = tf.matmul(x, W) + b
        if outputs is None:
          outputs = y
        else:
          outputs = tf.maximum(outputs, y)

    #outputs = tf.nn.tanh(outputs)

    return outputs
