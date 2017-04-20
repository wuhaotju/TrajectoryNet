import matplotlib.pyplot as plt
import numpy as np

def plot_hist(weights):
  """show distribution of weights"""
  print(type(weights))
  weights = np.array(weights)
  weights = weights.flatten()
  plt.hist(weights)
  plt.show(block=False)
