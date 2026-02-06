# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:39:25 2025

@author: Franc
"""

import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
import pandas as pd
from exatrack import anomalous_diff_transition, build_model, padding, read_table
import tensorflow as tf

'''
Test the 4 state model with state predictions on 30 ms data
'''

paths = glob(r'D:\Maria_DATA\ZM_data\Tracks\Low_density\*.csv')[:1]
dt = 0.03
(dt/0.1)**0.5

dtype = 'float64'
nb_states = 4
nb_dims = 2
'''
Initialization of the model.
The current model is not necessarly very robust to chosing wrong initial parameters.
One would ideally try the fitting with a few set of realistic initial parameters and 
pick the model that results in the maximal likelihood.

params: the row represent the different states, for each row: 
the first column represents the log of the localization error
the second column represents the log of the diffusion length (sqrt(2*D*t)),
the third column represents the anomalous factor (confinement factor of directed motion speed)
the fourth column represents the log of the change of the anomalous variable per step (diffusion lenght of the potential well in case of confinement and change of velocity in case of directed motion)

initial_params:
the first column represents the log of the spread of the initial particle position (spread of the field of view)
the second column currently has no use

transition_shapes and transition_rates: shape and rate parameters from on state (lines) to another (columns) according to a gamma distribution
for now the shape parameters are kept fixed
'''

# The 5th column is the model type indicator: 0 = confined motion, 1 = directed motion
params = tf.constant([[np.log(0.02), np.log(0.0025), 0.1, np.log(0.0001), 1],
                      [np.log(0.02), np.log(0.033), 0.05, np.log(0.00001), 1],
                      [np.log(0.02), np.log(0.08), 0.001, np.log(0.00001), 1],
                      [np.log(0.02), np.log(0.15), 0.001, np.log(0.00001), 1]], dtype = dtype)

initial_params = tf.constant([[np.log(22), np.log(0.05)],
                              [np.log(22), np.log(0.05)],
                              [np.log(22), np.log(0.05)],
                              [np.log(22), np.log(0.05)]], dtype = dtype)

transition_shapes = tf.ones((nb_states, nb_states), dtype = dtype)
transition_rates = tf.ones((nb_states, nb_states), dtype = dtype)*0.01

# initial_fractions: fraction of each state in logit scale (softmax to get fractions).
# Length is nb_states+1 because the last entry represents the mislinking state.
initial_fractions = np.ones((1, nb_states+1))
initial_fractions[0,-1] = -1

'''
Data loading and formating to the proper format
'''

# Inform the maximum track length used for the analysis (longer tracks are cut down to this track length)
track_len = 40

track_list, frames, track_ID_list, opt_metrics = read_table(paths, # path of the file to read or list of paths to read multiple files.
                                       lengths = np.arange(10,track_len+1), # number of positions per track accepted (take the first position if longer than max
                                       dist_th = np.inf, # maximum distance allowed for consecutive positions 
                                       frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                                       fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                                       colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                                       opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                                       remove_no_disp = True)

nb_tracks = len(track_list)

# The batch size can be reduced if you have memory issues (e.g. if the code crashes)
batch_size = 200 
nb_tracks = nb_tracks//batch_size*batch_size

all_tracks, all_frames, all_masks = padding(track_list[:nb_tracks], frames[:nb_tracks])
tracks = tf.constant(all_tracks[:,None, :, None, None, :nb_dims], dtype)

'''
Creation of the model
'''

sequence_length = 3 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 0.5 # maximum linking distance used for the linking algorithm
estimated_density = 0.1 # estimated density of the sample (number of counts per distance unit per frame)

model, pred_model = build_model(track_len, # maximum number of time points in the input tracks
                                nb_states, # Number of states of their model
                                params, # recurrent parameters of the model
                                initial_params, # initial parameters of the model
                                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                                initial_fractions,
                                batch_size = batch_size, # number of tracks analysed at the same time
                                nb_dims = nb_dims,
                                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                                estimated_density = estimated_density, # Estimated density of the sample.
                                )
model.summary()

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

class get_parameters(tf.keras.callbacks.Callback):
    def __init__(self, layer_name='params'):
        super(get_parameters, self).__init__()
        self.layer_name = layer_name  # Name of the layer whose weights you want to monitor
    
    def on_epoch_end(self, epoch, logs=None):
        # Retrieve the weights of the model
        weights = model.weights
        #params = tf.concat((model.weights[0][:, 2], np.exp(weights[0][:, 0]), np.exp(weights[0][:, 1]), [weights[3][0, 1]], [weights[3][1, 0]], [weights[4][0, 1]], [weights[4][1, 0]]), 0)
        params = {'anomalous factors': list(np.round(model.weights[0][:, 2],3)), 'Localization errors': list(np.round(np.exp(weights[0][:, 0]),3)), 'd': list(np.round(np.exp(weights[0][:, 1]), 3)), 'transition rates': list(np.round(weights[4], 3).reshape(nb_states**2)), 'Fractions': list(np.round(tf.math.softmax(model.weights[2][0]), 3))}
        # For Dense layers, the first item is the kernel, and the second is the bias (if any)
        
        # Print the weights (you can customize this part to show what you need)
        print(params)

class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, warmup_steps, peak_lr, decay_rate, decay_start):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.decay_rate = decay_rate
        self.decay_start = decay_start
    
    def __call__(self, step):
        # Linear warmup
        decay_step = tf.reduce_max([step-self.decay_start, 0])
        return self.peak_lr*(1-tf.math.exp(-step/self.warmup_steps))*tf.math.exp(-self.decay_rate*decay_step)

lr = WarmupLearningRateSchedule(10, 1/100, 0.005, 100) # learning rate schedule
adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/GPU:0'):
    history1 = model.fit((tracks, all_masks), tracks, epochs = 500, batch_size = batch_size, callbacks=[get_parameters()], shuffle=False, verbose = 1) #, callbacks  = [l_callback])

'''
Extracting and saving the model parameters (for now we keep the shape parameters fixed to 1)
'''

params = {'anomalous factors': model.weights[0][:, 2], 'Localization errors': np.exp(model.weights[0][:, 0]), 'd': np.exp(model.weights[0][:, 1]), 'transition rates': model.weights[4], 'q': np.exp(model.weights[0][:, 3]), 'transition shapes': model.weights[5]}

parameter_values = np.concatenate([params['Localization errors'][:, None], params['d'][:, None], params['anomalous factors'][:, None], params['q'][:, None],  params['transition rates']*(1-np.identity(nb_states)), params['transition shapes']], axis = 1)

rate_cols = []
shape_cols = []
for state in range(nb_states):
    rate_cols = rate_cols + ['transition rate (per step) to state %s'%state]
    shape_cols = shape_cols + ['transition shape (per step) to state %s'%state]

parameters = pd.DataFrame(parameter_values, columns = ['Localization errors', 'diffusion length', 'anomalous parameter', 'anomalous speed variation'] + rate_cols + shape_cols)

parameters.to_csv('Saved_parameters.csv')

'''
Computing the probabilities of each state
'''

state_preds = pred_model.predict((tracks, all_masks), batch_size = batch_size)

plt.figure(figsize = (10,10))

nb_rows = 5
lim = 0.7

for i in range(nb_rows):
    for j in range(nb_rows):
        track = all_tracks[i*nb_rows+j][:np.sum(all_masks[i*nb_rows+j]).astype(int)]
        track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
        
        preds = state_preds[i*nb_rows+j, :track.shape[0]]
        plt.plot(track[:, 0], track[:, 1], ':k')
        plt.scatter(track[:, 0], track[:, 1], c = preds[:,:3])
plt.gca().set_aspect('equal', adjustable='box')














