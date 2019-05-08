from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
import os 
import random 
import hdf5storage
from scipy.interpolate import griddata
import scipy
import math
from PIL import Image


class Slice_Provider(object):
    """
        
    :param search_path_zp: path where coil combined zero-filled images are located 
    :param search_path_rc: path where coil combined XDGRASP images are located 
    :param zp_suffix: data suffix associated with zero-filled network input
    :param rc_suffix: data suffix associated with XDGRASP recon network label

    
    """
    
    def __init__(self, search_path_zp,search_path_rc, zp_suffix="_zp.mat", rc_suffix='rc.mat', scale_factor = 10,batch_size = 3):
      
        self.zp_suffix = zp_suffix
        self.rc_suffix = rc_suffix
        self.search_path_zp = search_path_zp
        self.search_path_rc = search_path_rc
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.data = self._extract_files()

    # For _extract_files we randomly grab multiple NC-MRA zero-filled and XD-GRASP reconstructed pairs. 
    # The number of pairs extracted is equal to specified batch size
    
    def _extract_files(self):
        ListofFiles_zp = os.listdir(self.search_path_zp)
        ListofFiles_zp_string = str(random.choice(ListofFiles_zp))
        filename_zp = ListofFiles_zp_string
        N_files_selected_zp = len(ListofFiles_zp_string)
        N_zp_suffix = len(self.zp_suffix)
        string_compare_zp = ListofFiles_zp_string[0:(N_files_selected_zp-N_zp_suffix)]  
        load_path_zp = self.search_path_zp + ListofFiles_zp_string

        mat = hdf5storage.loadmat(load_path_zp)
        mat_zp = np.float32(list(mat.values()))
        nx = mat_zp.shape[1]
        mat_zp3 = np.zeros([self.batch_size,nx,nx,6],'float32')
        mat_rc3 = np.zeros([self.batch_size,nx,nx,6],'float32')

        # Selecting zero-filled and XD-GRASP pairs 
        # while looping through batch size 
        for i in range(0,self.batch_size):
            ListofFiles_zp = os.listdir(self.search_path_zp)
            ListofFiles_rc = os.listdir(self.search_path_rc)
            
            # We randomly select zero-filled and XD-GRASP pairs 
            # in order to get batchs with samples that 
            # reflect the whole training population for each iteration
            
            ListofFiles_zp_string = str(random.choice(ListofFiles_zp))
            filename_zp = ListofFiles_zp_string
            N_files_selected_zp = len(ListofFiles_zp_string)
            N_zp_suffix = len(self.zp_suffix)
            string_compare_zp = ListofFiles_zp_string[0:(N_files_selected_zp-N_zp_suffix)]  
            N_files_total_rc = len(ListofFiles_rc)

            filename_recon = string_compare_zp + self.rc_suffix
            filenames = [] 
            filenames.append(ListofFiles_zp_string)
            filenames.append(filename_recon)
            
            load_path_zp = self.search_path_zp + ListofFiles_zp_string
            load_path_rc = self.search_path_rc + filename_recon

            mat = hdf5storage.loadmat(load_path_zp)
            mat_zp = np.float32(list(mat.values()))
            nx = mat_zp.shape[1]
            mat_zp2 = np.zeros([nx,nx,6],'float32')
            mat_rc2 = np.zeros([nx,nx,6],'float32')    
        
            mat = hdf5storage.loadmat(load_path_rc)
            mat_rc = np.float32(list(mat.values()))   
        
            mat_zp2 = mat_zp[0,:,:,:]
            mat_rc2 = mat_rc[0,:,:,:]

            mat_rc3[i,:,:,:] = mat_rc2
            mat_zp3[i,:,:,:] = mat_zp2
        
        # Originizing to meet tensorflow requirements
        inpt = np.expand_dims(mat_zp3[...,0:6],axis=4)*self.scale_factor
        outpt = np.expand_dims(mat_rc3[...,0:6],axis=4)*self.scale_factor
        return inpt,outpt

    def __call__(self):
        inpt, outpt = self._extract_files()
        return inpt, outpt

    
  
    
