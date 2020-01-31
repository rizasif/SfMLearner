from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
# from SfMLearner import SfMLearner
from MarsLearner import MarsLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "data/train.txt", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.25, "Weight for smoothness")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 5, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 1000, "Image height")
flags.DEFINE_integer("img_width", 1000, "Image width")
flags.DEFINE_integer("seq_length", 1, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 2000, "Maximum number of training iterations") #200000
flags.DEFINE_integer("summary_freq", 50, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 100, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)") #5000
flags.DEFINE_integer("num_scales", 1, "currently set to 4")
flags.DEFINE_integer("num_source", 0, "should be seq_length-1")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
FLAGS = flags.FLAGS

# opt.num_source = opt.seq_length - 1
#         # TODO: currently fixed to 4
#         opt.num_scales = 4

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    sfm = MarsLearner()
    sfm.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()