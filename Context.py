import numpy as np
from itertools import islice
import sklearn
import os


class Context:
  @staticmethod
  def get_context(x, w=2, normalize=True):
    """w denotes window size on one side, the real window size is (w*2+1)"""

    # check if context exists
#    if os.path.isfile('contextdata.npy'):
#      print('loading context data from file')
#      return np.load('contextdata.npy')
#
    input_dim = x.shape

    if normalize:
      x = np.reshape(x, [input_dim[0]*input_dim[1],input_dim[2]])# for ease of normalization
      x = sklearn.preprocessing.normalize(x, norm='l2', axis=1)
      x = np.reshape(x, [input_dim[0], input_dim[1],input_dim[2]])

    # padding
    p = Context.pad(x, w)

    # extract context
    c = Context.slide(p, w)

#    np.save('contextdata.npy', c)
    
    return c

  @staticmethod
  def pad(x, w=2):
    if len(x.shape) == 2:
      length = x.shape[0]
    else:
      length = x.shape[1]
    repeats = [1]*length
    repeats[0] = repeats[-1] = w + 1
    if len(x.shape) == 2:
      p = np.repeat(x, repeats, axis=0)
    else:
      p = np.repeat(x, repeats, axis=1)
    return p

  @staticmethod
  def slide(x, w=2):
    if len(x.shape) == 3:
      n = x.shape[1] - w*2
    else:
      n = x.shape[0] - w*2
    width = w*2+1
    if len(x.shape) == 3:
      c = np.array([x[j, i:i+width, :].flatten()  for j in range(x.shape[0]) for i in range(n)])
      c = np.reshape(c, (x.shape[0], n,-1))
    elif len(x.shape) == 2:
      c = np.array([x[i:i+width, :].flatten() for i in range(n)])
      c = np.reshape(c, (c.shape[0], 2*w+1,-1))
    return c
