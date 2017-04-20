import json

class DataConfig(object):
    def __init__(self, confile='config.json'):
        dconf = json.load(open(confile))
        self.test_id = dconf['test_id']
        self.val_id = dconf['val_id']
        self.truncated_seq_len = dconf["truncated_seq_len"]
        self.learning_rate = dconf["learning_rate"]
        self.batch_size = dconf["batch_size"]
        self.num_layers = dconf["num_layers"]
        self.num_epochs = dconf["num_epochs"]
        self.tensorboard = dconf["tensorboard"]
        self.init_scale = dconf["init_scale"]
        self.num_threads = dconf["num_threads"]
        self.hidden_size = dconf["hidden_size"]
        self.task = dconf["task"]
        self.useGPU = dconf["useGPU"]
        self.weight_initializer = dconf["weight_initializer"]
        self.evaluate_freq = dconf["evaluate_freq"]
        self.testmode = dconf["testmode"]
        self.checkpoint = dconf["checkpoint"]
        self.restore = dconf["restore"]
        self.activation = dconf["activation"]
        self.test_mode = dconf["test_mode"]

class TrainingConfig(object):
    """ Config for genetic purpose."""
    def __init__(self, is_training, is_validation, batch_size):
        self.is_training = is_training
        self.is_validation = is_validation
        self.batch_size = batch_size
