class Log:
  def __init__(self, fname, path, num_class):
    self.log_name = fname
    if num_class == 2:
      self.mode = "detail"
      self.write = self.write_detail
    else:
      self.mode = "summary"
      self.write = self.write_summary
    self.logfile = open(path + fname + '.csv', 'w+')
    print(path + fname + '.csv')
    self.add_header()

  def add_header(self):
    if self.mode == "detail":
      self.logfile.write("iteration, trainLoss, valLoss, testLoss, trainAcc, valAcc, testAcc, trainSens, valSens, testSens, trainSpec, valSpec, testSpec, trainPrec, valPrec, testPrec, trainRecall, valRecall, testRecall, trainAUC, valAuc, testAUC\n")
    else:
      self.logfile.write("iteration, trainLoss, valLoss, testLoss, trainAcc, valAcc, testAcc\n")

  def write_detail(self, data, minibatch):
    (cost_train, acc_train, sens_train, spec_train, prec_train, recall_train, auc_train, cost_test, acc_test, sens_test, spec_test, prec_test, recall_test, auc_test, cost_val, acc_val, sens_val, spec_val, prec_val, recall_val, auc_val ) = data
    self.logfile.write("{0}, {1:0.3f}, {2:0.3f}, {3:0.3f}, {4:0.3f}, {5:0.3f}, {6:0.3f}, {7:0.3f}, {8:0.3f}, {9:0.3f}, {10:0.3f}, {11:0.3f}, {12:0.3f}, {13:0.3f}, {14:0.3f}, {15:0.3f}, {16:0.3f}, {17:0.3f}, {18:0.3f}, {19:0.3f}, {20:0.3f}, {21:0.3f}\n".format(minibatch, cost_train, cost_val, cost_test, acc_train,acc_val, acc_test, sens_train, sens_val, sens_test, spec_train, spec_val, spec_test, prec_train, prec_val, prec_test, recall_train, recall_val, recall_test, auc_train, auc_val, auc_test))
    self.logfile.flush()

  def write_summary(self, data, minibatch):
    (cost_train, acc_train, cost_test, acc_test, cost_val, acc_val) = data
    self.logfile.write("{0}, {1:0.3f}, {2:0.3f}, {3:0.3f}, {4:0.3f}, {5:0.3f}, {6:0.3f}\n".format(minibatch, cost_train, cost_val, cost_test, acc_train, acc_val, acc_test))
    self.logfile.flush()

  def close(self):
    self.logfile.close()
