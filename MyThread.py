import threading

class MyThread(threading.Thread):
  def __init__(self, func, args):
    threading.Thread.__init__(self)
    self.func = func
    self.args = args
    self.res = None

  def get_result(self):
    return self.res

  def run(self):
    self.res = apply(self.func, self.args)
