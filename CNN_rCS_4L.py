
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
import tensorflow as tf
import sys
import Load_slice_6RespState_batch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# In this file we build our network, initialize our optimizer,implement training, and 
# enable network testing, based on parameters specified in main script (i.e. Code_sub_training, Code_sub_testing)  

def BuildNet(x,keep_prob, depth = 256, filter_size_space=3, fliter_size_time = 3, type_train = 'Training'):
    logging.info("Layers {layers}, depth {depth}, spatial filter size {filter_size_space}x{filter_size_space}, temporal filter size: {fliter_size_time}".format(layers=4,depth=depth, filter_size_space=filter_size_space,fliter_size_time=fliter_size_time))
   
    # Building the network a 4 hidden layer network

    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    nz = tf.shape(x)[3]
    
    # Specifying if network is used during training 
    if type_train == 'Training':
        check = True
    else:
        check = False
        
    channels = 1
    features = depth
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,nz,channels]))
    in_node1 = x_image
    x_shape = tf.shape(in_node1)
    
    #Initializing neural network weights    
    initial = tf.truncated_normal([filter_size_space, filter_size_space, fliter_size_time, 
                                   channels, features], stddev=0.1)
    w1 = tf.Variable(initial,name="w1")
    initial = tf.constant(0.1, shape=[features])
    b1 = tf.Variable(initial,name="b1")
    
    initial = tf.truncated_normal([filter_size_space, filter_size_space, fliter_size_time, 
                                   features, features], stddev=0.1)
    w2 = tf.Variable(initial,name="w2")
    initial = tf.constant(0.1, shape=[features])
    b2 = tf.Variable(initial,name="b2")
    
    initial = tf.truncated_normal([filter_size_space, filter_size_space, fliter_size_time, 
                                   features, features], stddev=0.1)
    w3 = tf.Variable(initial,name="w3")
    initial = tf.constant(0.1, shape=[features])
    b3 = tf.Variable(initial,name="b3")

    initial = tf.truncated_normal([filter_size_space, filter_size_space, fliter_size_time, 
                                   features, features], stddev=0.1)
    w4 = tf.Variable(initial,name="w4")
    initial = tf.constant(0.1, shape=[features])
    b4 = tf.Variable(initial,name="b4")
    
    initial = tf.truncated_normal([1, 1, 1, features, 1], stddev=0.1)
    w_all = tf.Variable(initial,name="w_all")
    
    #Creating neural network graphs    
    conv1 = tf.nn.conv3d(in_node1, w1,  strides = [1, 1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1,keep_prob)
    conv1_BM = tf.contrib.layers.batch_norm(conv1+ b1,decay = 0.9,updates_collections = None,is_training = check)
    tmp_img_1 = tf.nn.relu(conv1_BM)
    
    conv2 = tf.nn.conv3d(tmp_img_1, w2, strides = [1, 1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2,keep_prob)
    conv2_BM = tf.contrib.layers.batch_norm(conv2 + b2,decay = 0.9,updates_collections = None,is_training = check)
    tmp_img_2 = tf.nn.relu(conv2_BM)
    
    conv3 = tf.nn.conv3d(tmp_img_2, w3, strides = [1, 1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3,keep_prob)
    conv3_BM = tf.contrib.layers.batch_norm(conv3+b3,decay = 0.9,updates_collections = None,is_training = check)
    tmp_img_3 = tf.nn.relu(conv3_BM)
    
    conv4 = tf.nn.conv3d(tmp_img_3, w4, strides = [1, 1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.dropout(conv4,keep_prob)
    conv4_BM = tf.contrib.layers.batch_norm(conv4 + b4,decay = 0.9,updates_collections = None,is_training = check)
    tmp_img_4 = tf.nn.relu(conv4_BM)
    
    conv_all = tf.nn.conv3d(tmp_img_4, w_all, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    DNN_rCS =  conv_all 
    return DNN_rCS

# The Class Squential_network is tasked with 
    #1) Building network based on parameters set in main script(BuildNet)
    #2) Specifying the mean square error cost function (self.cost)
    #3) Allow for loading of previously trained model(restore) and network testing (predict) 
    
class Sequential_network(object):
    
    def __init__(self, cost_kwargs={}, **kwargs):
        
        tf.reset_default_graph()
        self.x = tf.placeholder("float", shape=[None, None, None, None,1],name="Inputs")
        self.y = tf.placeholder("float", shape=[None, None, None, None, 1],name="Outputs")
        self.keep_prob = tf.placeholder(tf.float32,name="Dropout")
        # Building network  
        network = BuildNet(self.x,self.keep_prob, **kwargs)
        self.network = network
        # Specifing cost function 
        self.cost = tf.reduce_mean(tf.square(self.y - network))
    
    def predict(self, model_path, x_test):
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            self.restore(sess, model_path)
            
            x_test.shape[0]
            x_test.shape[1]
            x_test.shape[2]
            x_test.shape[3]
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2],x_test.shape[3], 1))
            # Making predictions using trained network 
            prediction = sess.run(self.network, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
            
        return prediction
     
    def restore(self, sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)



 # The Class Squential_network is tasked with 
    #1) Initialized the ADAM optimizer (_Select_optimizer)
    #2) Initiates training (train) during which zero-filled and XD-GRASP reconstructions 
    #   are loaded from "search_path_zp" and "search_path_rc" file paths respectively
    #   and forward and backward projection is accomplished using tensorflow trainer 

class Trainer(object):
    
    def __init__(self,search_path_zp, search_path_rc,scale_factor, batch,
                 net, batch_size=1, optimizer="ADAM", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.search_path_zp = search_path_zp
        self.search_path_rc = search_path_rc
        self.scale_factor = scale_factor
        self.batch = batch

        
    
    def _Select_optimizer(self, cost, optimizer, training_iters, global_step):
        
        # Creating ADAM optimizer  
        if optimizer == 'ADAM':
            
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.005)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.9)
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                             global_step=global_step, 
                                                             decay_steps=training_iters,  
                                                             decay_rate=decay_rate, 
                                                             staircase=True)

            optimizer_chosen = tf.train.AdamOptimizer(learning_rate= learning_rate_node).minimize(cost,
                                                                                           global_step
                                                                                           =global_step)
            
        return optimizer_chosen,learning_rate_node
    
    
    def train(self, output_path, training_iters=100, epochs=10, display_it = 10, dropout=1.0):
            
            # Training the model  
            global_step = tf.Variable(0,name="global_step")
            self.optimizer_chosen,self.learning_rate_node = self._Select_optimizer(self.net.cost, 
                                                                                   self.optimizer, 
                                                                                   training_iters,
                                                                                   global_step)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            
            logging.info("Start Optimizatin")
            with tf.Session() as session:
                session.run(init)
                total_loss = 0
                for i in range(epochs*training_iters):
                    # loading data pairs for training  
                    
                    #Each _zp.mat file contains the zero-filled (i.e. undersampled) images 
                    #for a single slice + associated 6 nrespiratory time frames for a given training patient 

                    #Each _rc.mat file contains the XD-GRASP (i.e. recostructed) images
                    # images for single slice + associated 6 respiratory time frames for a given training patient 

                    data_images = Load_slice_6RespState_batch.Slice_Provider(search_path_zp=self.search_path_zp,
                                                            search_path_rc = self.search_path_rc,
                                                            zp_suffix="_zp.mat",rc_suffix="_rc.mat", 
                                                            scale_factor = self.scale_factor, batch_size= self.batch)
                    inpt=data_images.data[0]
                    outpt=data_images.data[1]
                    x_value = inpt
                    y_value = outpt
                    _, loss,lr =  session.run([self.optimizer_chosen,self.net.cost,self.learning_rate_node], 
                                              feed_dict = {self.net.x: x_value, self.net.y: y_value,self.net.keep_prob: dropout})

                    total_loss = loss + total_loss
                    
                    #logging the loss function 
                    if i  % display_it == 0 or i == 1:
                        logging.info("Iter {:}, Loss= {:.8f}".format(i,loss))
                    if i  % training_iters == 0 and i > 1:
                        total_loss_avg = total_loss/training_iters
                        logging.info("Iter {:},Average Loss= {:.8f},learning rate= {:.8f}".format(i,total_loss_avg,lr))
                        total_loss=0


                output_path2 = output_path + 'model.ckpt'   

                if not os.path.exists(output_path2):
                    os.makedirs(output_path2)
                    
                saver.save(session, output_path2)
            
            return output_path2


            

