# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:23:28 2025

@author: franc

Analysis script for ExaTrack with classic fitting (no particular pattern of transitions and
no sharing of parameters between states.
"""

import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
import pandas as pd
from matplotlib.lines import Line2D
import tensorflow as tf

import sys
sys.path.append(r"D:\Density") # addapt the path to the path of the folder that contains the file exatrack 

from exatrack import read_table, padding, build_model, get_parameters, WarmupLearningRateSchedule, anomalous_diff_transition, MLE_loss, ExaTrack_2_DataFrame, correct_state_predictions_padding

dtype = 'float64'

'''
Data loading and formating to the proper format
'''

# indicate the path of the tracks in a csv format with polar coordinates
paths = glob(r'D:\Downloads\Divisome\PBP2B/*.csv')

# for this fitting, I only want the tracks of the movies 2 and 3
paths = paths[1:]

# Inform the maximum track length used for the analysis (longer tracks are cut down to this track length)
track_len = 40
nb_dims = 2
track_list, frame_list, track_ID_list, opt_metrics = read_table(paths, # path of the file to read or list of paths to read multiple files.
                                       lengths = np.arange(3,track_len+1), # number of positions per track accepted (take the first position if longer than max
                                       dist_th = np.inf, # maximum distance allowed for consecutive positions 
                                       frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                                       fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                                       colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                                       opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                                       remove_no_disp = True)


# The number of batches can be increased if you have memorie issues (e.g. if the code crashes)
# We just have to get rid of the last tracks as the batches need to have the same size.

nb_tracks = len(track_list)
nb_batches = 3

batch_size = nb_tracks//nb_batches

nb_tracks = batch_size*nb_batches
track_list = track_list[:nb_tracks]
frame_list = frame_list[:nb_tracks]
track_ID_list = track_ID_list[:nb_tracks]
for metric in opt_metrics:
    opt_metrics[metric] = opt_metrics[metric][:nb_tracks]

'''
# This function can be used to simulate tracks instead of loading tracks
all_tracks, all_states, all_masks = anomalous_diff_transition(max_track_len=100,
                              nb_tracks = 100,
                              LocErr=0.02, # localization error in x, y and z (even if not used)
                              Fs = np.array([0.3, 0.3, 0.4]),
                              Ds = np.array([0.0, 0, 0.25]),
                              nb_dims = 1, # can be 2D or 3D
                              velocities = np.array([0.01, 0.0, 0]),
                              angular_Ds = np.array([0.0, 0.0, 0]),
                              conf_forces = np.array([0.0, 0., 0]),
                              conf_Ds = np.array([0.0, 0.0, 0]),
                              conf_dists = np.array([0.0, 0.0, 0]),
                              transition_matrix = np.array([[0.00, 0.05, 0.02],
                                                            [0.05, 0.00, 0.02],
                                                            [0.1, 0.05,  0.]]),
                              shape_matrix = np.array([[0, 1, 1],
                                                       [1, 0, 1],
                                                       [1, 1, 0]]),
                              bleaching_rate = 1e-10,
                              LocErr_std = 0,
                              dt = 0.02,
                              field_of_view = np.array([10,10]),
                              nb_burning_steps=100,
                              nb_sub_steps = 10)
'''

all_tracks, all_frames, all_masks = padding(track_list[:nb_tracks], frame_list[:nb_tracks])

# We need to reshape the track to the shape used by the model, as following
tracks = tf.constant(all_tracks[:,None, :, None, None, :nb_dims], dtype)

# Hyperparameters required for the model
nb_obs_vars = 1 # The number of observed variables at each time point (do not modify this)
nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
nb_hidden_vars = 2
nb_gaussians = nb_obs_vars + nb_hidden_vars

'''
Initialization of the model.
The current model is not necessarly very robust to chosing wrong initial parameters.
One would ideally try the fitting with a few set of realistic initial parameters and 
pick the model that results in the maximal likelihood.

params: the rows represent the different states, for each row: 
The fifth column represent the type of model: 0 = confined model and 1 = directed model
The first column represents the log of the localization error
The second column represents the log of the diffusion length (sqrt(2*D*t)),
The third column represents either the log of the directed motion speed for a directed
model or the logit of the confinement factor.
The fourth column represents the log of the change of the anomalous variable per step
(diffusion lenght of the potential well in case of confinement and change of velocity 
in case of directed motion).

initial_params:
the first column represents the log of the spread of the initial particle position (spread of the field of view)
the second column currently has no use

transition_shapes and transition_rates:
    model parameters that inform about the shape and the rates of the transttions
    Assuming Gamma distributed transitions.
    transition_shapes: in log scale to ensure shape parameters in between 0 and inf
    transition_rates: To retrieve the true rate, apply the softmax(transition_rates)*true_shape
    
initial_fractions: fraction of each state in logit scale (use the softmax to retrieve the fractions)
NB: The number of fractions is nb_state +1 because we assume a mislinking state in additions to the
regular staes
'''

# PBP2B
nb_states = 4
params = np.array([[np.log(0.02), np.log(0.001), np.log(0.0005), np.log(0.0002), 1],
                   [np.log(0.02), np.log(0.01), np.log(0.001), np.log(0.0002), 1],
                   [np.log(0.02), np.log(0.0001), np.log(0.011), np.log(0.001), 1],
                   [np.log(0.02), np.log(0.2), np.log(0.001), np.log(0.00001), 1]], dtype = dtype)

'''
# Parameters used for the MreB data
nb_states = 4
params = np.array([[np.log(0.02), np.log(0.005), np.log(0.01), np.log(0.01), 1],
                   [np.log(0.02), np.log(0.005), np.log(0.05), np.log(0.01), 1],
                   [np.log(0.02), np.log(0.0001), np.log(0.15), np.log(0.15), 1],
                   [np.log(0.02), np.log(0.2), np.log(0.001), np.log(0.00001), 1]], dtype = dtype)
'''

if nb_states!=len(params):
    raise ValueError('The number of states (nb_states) and the length of the param array do not match.')

initial_params = np.array([[np.log(50), np.log(0.05)]]*nb_states, dtype = dtype) 

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*2.5
tf.math.softmax(transition_rates)

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1

'''
Creation of the model
'''

sequence_length = 10 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.01 # estimated density of the sample (number of counts per distance unit per frame)

model, pred_model = build_model(track_len, # maximum number of time points in the input tracks 
                                nb_states, # Number of states of their model
                                params, # recurrent parameters of the model
                                initial_params, # initial parameters of the model
                                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                                initial_fractions, 
                                batch_size, # number of tracks analysed at the same time
                                nb_dims = nb_independent_vars, # Number of dimensions of the tracks
                                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                                estimated_density = estimated_density, # Estimated density of the sample.
                                vary_transition_shapes = False, # we use False to prevent the fitting of the shape parameter
                                )

model.summary()

# compute the likelihood of the model before training
preds = model.predict((tracks, all_masks), batch_size = batch_size)
print('Initial likelihood:', MLE_loss(preds, preds))

# specify the model optimization parameters
lr = WarmupLearningRateSchedule(10, 1/100, 0.05, 300) # learning rate schedule
adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

# Run the model
with tf.device('/GPU:0'): # Use tf.device('/GPU:0') if you have access to a GPU
    history1 = model.fit((tracks, all_masks), tracks, epochs = 500, batch_size = batch_size, callbacks=[get_parameters()], shuffle=True, verbose = 1) #, callbacks  = [l_callback])

# Then, we can predict the states of the tracks
state_preds = pred_model.predict((tracks, all_masks), batch_size = batch_size)
#And we correct for the padding issues of pred_model
correct_state_predictions_padding(state_preds, all_masks, sequence_length)

# save the tracks with state predictions
data = ExaTrack_2_DataFrame(track_list[:nb_tracks], frame_list[:nb_tracks], track_ID_list[:nb_tracks], {}, state_preds, all_masks)

data.to_csv(r'Save\Path\DATA_with_state_preds.csv', index = False)

'''
Then, we can plot the trackss
'''
state_preds_3 = np.concatenate((state_preds[:,:,:1]+state_preds[:,:,1:2],state_preds[:,:,2:3],state_preds[:,:,3:4]),-1)

plt.figure(figsize = (10, 15))
#lim = 1 # MreB
lim = 0.5
nb_rows = 8
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        track = track_list[offset+i*nb_rows+j]
        cur_mask = all_masks[offset+i*nb_rows+j].astype(bool)
        cur_state_preds = state_preds_3[offset+i*nb_rows+j][cur_mask]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], 'k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cur_state_preds, s = 10)
plt.gca().set_aspect('equal', adjustable='box')


