from enum import Enum

class RNNType(Enum):
  LSTM_u = 1 # LSTM unidirectional
  LSTM_b = 2 # LSTM bidirectional
  GRU = 3 # GRU
  GRU_b = 4 # GRU, bidirectional
