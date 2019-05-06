
## importing functions
from __future__ import division, print_function
import sys
sys.path.append('/home/damingshen/Code_sub_Rad_AI/')
import CNN_rCS_4L
import Load_slice_6RespState_batch
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow as tf
import os

plt.rcParams['image.cmap'] = 'gray'

## Specifying GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#Recon using CNN
#location of undersampled slice + time data sets for testing
search_path_zp = "/home/damingshen/Code_sub_Rad_AI/Testing/ZeroFilled/"
#location of XD-GRASP reconstructed slice + time data sets for testing
search_path_rc = "/home/damingshen/Code_sub_Rad_AI/Testing/CSrecon/"
#Each _zp.mat file contains the zero-filled (i.e. undersampled) images for a single slice + associated 6 respiratory time frames for a given testing patient 
#Each _rc.mat file contains the XD-GRASP (i.e. recostructed) a images for single slice + associated 6 respiratory time frames for a given testing patient 

data_images = Load_slice_6RespState_batch.Slice_Provider(search_path_zp,search_path_rc,
                                        zp_suffix="_zp.mat",rc_suffix="_rc.mat",scale_factor = 1)
x_valid=data_images.data[0]
y_valid=data_images.data[1]

#location of trained model 
path_model = "/Model_Location/f16_L4/model.ckpt"
CNN_rCS_f16_L4 = CNN_rCS_4L.Sequential_network(depth = 16, filter_size_space=3, fliter_size_time = 3,type_train = 'Testing')
prediction = CNN_rCS_f16_L4.predict(path_model, x_valid)
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13,5))
fig_num = 0
ax[0].imshow(x_valid[0,:,:,1,0], vmin=0,vmax=0.35)
ax[1].imshow(y_valid[0,:,:,1,0], vmin=0,vmax=0.35)
ax[2].imshow(prediction[0,:,:,1,0],vmin=0,vmax=0.35)

ax[0].set_title("Undersampled")
ax[1].set_title("XD-GRASP recon")
ax[2].set_title("CNN recon")

fig.tight_layout()
plt.show()

