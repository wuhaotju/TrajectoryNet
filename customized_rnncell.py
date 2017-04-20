# this code is modified based on tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import tensorflow as tf

class NewGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tanh, simple_gate=True):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self.simple_gate = simple_gate

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or "gru_cell"):
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        if self.simple_gate:
          gate=_linear(
                [inputs, state], 2 * self._num_units, True, 1.0, scope=scope)
        else:
          gate=self.maxout(inputs, state, 5, 0, scope=scope, output_size=2 * self._num_units)
        
        r, u = array_ops.split(
            gate,
            num_or_size_splits=2,
            axis=1)
        r, u = sigmoid(r), sigmoid(u)

      with vs.variable_scope("candidate"):
         c = self.maxout(inputs, r * state, 5, 0, scope=scope, output_size=self._num_units)

#         c = tf.nn.tanh(_linear([inputs, r * state],
#                                     self._num_units, True,
#                                     scope=scope))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

  def maxout(self, input1, input2, num_units, ini_value, output_size, scope=None):
    shape = input1.get_shape().as_list()
    dim = shape[-1]
    outputs = None
    for i in range(num_units):
      with tf.variable_scope(str(i)):
        y = self._activation(_linear([input1, input2],
                                output_size, True, ini_value,
                                scope=scope))
        if outputs is None:
          outputs = y
        else:
          outputs = tf.maximum(outputs, y)
    c = outputs
    return c



def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return nn_ops.bias_add(res, biases)
