# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:34:43 2026

@author: Franc
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
sys.path.append(r"C:\Users\Franc\Data\ExaTrack") # add exatrack directory to the system path
import exatrack_while_segment_infer_vars as exatrack
#import exatrack as exatrack
from glob import glob


exps = glob(r'C:\Users\Franc\Data\RodComplex\RodZ/*RodZ-IPTG*')
paths = glob(r'C:\Users\Franc\Data\RodComplex\RodZ/*RodZ-IPTG*/*.csv')

tracks = [np.arange(10)[:, None]*0.1*[[0,1]] + np.random.normal(0, 0.01, (10, 2)),
          np.arange(10)[:, None]*0.1*[[1,0]] + np.random.normal(0, 0.01, (10, 2))]

tracks, _, masks = exatrack.padding(tracks)
tracks = tf.constant(tracks[:,None, :, None, None, :nb_dims], dtype)

nb_states = 2
params = np.array([[np.log(0.01), np.log(0.0001), np.log(0.1), np.log(0.001), 1],
                   [np.log(0.03), np.log(0.15), np.log(0.1), np.log(0.005), 0]], dtype = dtype)

initial_params = np.array([[np.log(1)]]*nb_states, dtype = dtype) 

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*4.5

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 3 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

batch_size = len(tracks)

vary_params = True
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False
vary_transition_rates = True


