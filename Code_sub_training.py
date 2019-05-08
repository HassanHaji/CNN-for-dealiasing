
from __future__ import division, print_function
import sys
sys.path.append('/home/Code_sub_Rad_AI/')
import CNN_rCS_4L
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
import os
plt.rcParams['image.cmap'] = 'gray'

## Specifying GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#Location of undersampled slice + time data sets for training
search_path_zp = "/home/Data_Training/ZeroFilled/"
#Location of XD-GRASP reconstructed slice + time data sets for training
search_path_rc = "/home/Data_Training/CSrecon/"

# Initializing network + optimizer
number_of_channels = 16
CNN_filter_space = 3
CNN_filter_time = 3
type_train_value = 'Training'
scale_factor = 1
batch=12
learning_rate_use = 0.005
decay_rate_use = 0.9

CNN_rCS_f16_L4 = CNN_rCS_4L.Sequential_network(depth = number_of_channels, 
                                               filter_size_space=CNN_filter_space, 
                                               fliter_size_time = CNN_filter_time, 
                                               type_train = type_train_value)

trainer = CNN_rCS_4L.Trainer(search_path_zp,search_path_rc,scale_factor, 
                             batch,CNN_rCS_f16_L4, optimizer='ADAM',
                             opt_kwargs=dict(learning_rate=learning_rate_use,
                                             decay_rate = decay_rate_use))

# Model path 
path_model = "/Model_keep/f16_L4/"
# Training network 
path_f16_L4 = trainer.train(path_model, training_iters=100, epochs=30, display_it=10, dropout=1.0)
