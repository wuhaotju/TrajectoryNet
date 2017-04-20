class Learning_rate:
  def __init__(self, global_lr, decay_rate, decay_step):
    self.global_lr = global_lr
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.local_lr = global_lr
    self.local_step = 1
    self.global_step = 0

  def reset_local_step(self):
    self.local_step = 1

  def increase_local_step(self):
    self.local_step += 1

  def increase_global_step(self):
    self.global_step += 1

  def get_lr(self):
    lr = self.global_lr * self.decay_rate ** (self.global_step // self.decay_step)
    #lr = self.global_lr * self.decay_rate ** ((self.global_step+self.local_step)  // self.decay_step)
    return(lr)
