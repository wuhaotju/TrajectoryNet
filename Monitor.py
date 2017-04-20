class Monitor:
  def __init__(self, loss=True, num_class=2):
    self._loss = loss
    self.num_class = num_class
    if loss:
      self._best = float("inf")
    else:
      self._best = -float("inf")
    self._is_best = False
    self.minibatch = 0
    self.best_rec = ()

  def new(self, result, minibatch):
    if self.num_class == 2:
      (cost_train, acc_train, sens_train, spec_train, prec_train, recall_train, auc_train, cost_test, acc_test, sens_test, spec_test, prec_test, recall_test, auc_test, cost_val, acc_val, sens_val, spec_val, prec_val, recall_val, auc_val) = result
    else:
      (cost_train, acc_train, cost_test, acc_test, cost_val, acc_val) = result

    self._is_best = False
    if self._loss:
      if self._best > cost_val:
        self._best = cost_val
        self._is_best = True
        self.best_rec = result
        self.minibatch = minibatch
    else:
      if self._best < cost_val:
        self._best= cost_val
        self._is_best = True
        self.best_rec = result
        self.minibatch = minibatch

  def isBest(self):
    return self._is_best

  def getBest(self):
    return self.best_rec
