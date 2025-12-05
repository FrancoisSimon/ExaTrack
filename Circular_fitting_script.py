# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:23:28 2025

@author: franc

Analysis script for ExaTrack with a pattern of transition between 2 directed states
sharing of parameters.
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

from exatrack import read_table, padding, build_abrupt_directed_motion_changes_model, get_parameters, WarmupLearningRateSchedule, anomalous_diff_transition, MLE_loss, ExaTrack_2_DataFrame, correct_state_predictions_padding

dtype = 'float64'

'''
Data loading and formating to the proper format
'''
# indicate the path of the tracks in a csv format with polar coordinates
paths = glob(r'D:\Downloads\Divisome\PBP2B_VerCINI\*polar.csv')

track_len = 40
nb_dims = 2
track_list_polar, frame_list, track_ID_list, opt_metrics = read_table(paths, # path of the file to read or list of paths to read multiple files.
                                       lengths = np.arange(3,track_len+1), # number of positions per track accepted (take the first position if longer than max
                                       dist_th = np.inf, # maximum distance allowed for consecutive positions 
                                       frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                                       fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                                       colnames = ['Rho_polar_um', 'Theta_polar_radians', 'Frame_no', 'Track_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                                       opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                                       remove_no_disp = True)

track_list = []
for polar_track in track_list_polar:
    tan_disps = (polar_track[1:, 0] + polar_track[:-1, 0])/2 * np.sin(polar_track[1:, 1] - polar_track[:-1, 1])
    tan_cum_disps = np.cumsum(np.concatenate((np.random.normal(0, 0.001, 1), tan_disps)))[:, None]
    track_list.append(tan_cum_disps)

# The number of batches can be increased if you have memorie issues (e.g. if the code crashes)
# We just have to get rid of the last tracks as the batches need to have the same size.

nb_tracks = len(track_list)
nb_batches = 1
batch_size = nb_tracks//nb_batches

nb_tracks = batch_size*nb_batches
track_list = track_list[:nb_tracks]
frame_list = frame_list[:nb_tracks]
track_ID_list = track_ID_list[:nb_tracks]
for metric in opt_metrics:
    opt_metrics[metric] = opt_metrics[metric][:nb_tracks]


all_tracks, all_frames, all_masks = padding(track_list[:nb_tracks], frame_list[:nb_tracks])

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

nb_dims = 1 # now, our tracks only have one dimension so we indicate it

# We need to reshape the track to the shape used by the model, as following
tracks = tf.constant(all_tracks[:,None, :, None, None, :nb_dims], dtype)

# Hyperparameters required for the model
nb_obs_vars = 1 # The number of observed variables at each time point (do not modify this)
nb_hidden_vars = 2
nb_gaussians = nb_obs_vars + nb_hidden_vars


# PBP2B
nb_states = 3

all_params = np.array([[np.log(0.04), np.log(0.001), np.log(0.011), np.log(0.0005), 1],
                       [np.log(0.04), np.log(0.05), np.log(0.001), np.log(0.00001), 0]], dtype = dtype)
abrupt_change_state = 0

vary_params = np.ones((len(all_params), 5))
#vary_params[0, 2] = 0
#vary_params[1, 2] = 0

all_initial_params = np.array([[np.log(50), np.log(0.05)]]*len(all_params), dtype = dtype) 

params = all_params
initial_params = all_initial_params

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*2.5
vary_transition_shapes = np.zeros(transition_shapes.shape)
tf.math.softmax(transition_rates)

initial_fractions = (np.random.rand(1, nb_states+1)+1) # This must normally have nb_states+1 states but we will add the initial fraction of the second directed state in the model as a fixed aprameter instead
initial_fractions[0,-1] = -1 # The last state corresponds to the misslinkings, which should have a low initial fraction
initial_fractions[0,1] = -20 # the state 1 corresponds to the second directed state so we set a low probability to favor the state 0 for the inital state of the tracks

vary_initial_fractions = np.ones(initial_fractions.shape)

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*4
transition_rates[2, 1] = -20
vary_transition_rates = np.ones(transition_rates.shape, dtype = dtype)
vary_transition_rates[2,1] = 0

vary_initial_params = np.ones(all_initial_params.shape)

sequence_length = 8 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.01 # estimated density of the sample (number of counts per distance unit per frame)

model, pred_model = build_abrupt_directed_motion_changes_model(track_len, # maximum number of time points in the input tracks 
                                nb_states, # Number of states of their model
                                all_params, # recurrent parameters of the model
                                all_initial_params, # initial parameters of the model
                                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                                initial_fractions, 
                                batch_size, # number of tracks analysed at the same time
                                nb_dims = nb_dims, # Number of dimensions of the tracks
                                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                                estimated_density = estimated_density, # Estimated density of the sample.
                                vary_transition_shapes = vary_transition_shapes,
                                abrupt_change_state = abrupt_change_state,
                                vary_initial_params = vary_initial_params,
                                vary_params = vary_params
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

state_preds = pred_model.predict((tracks, all_masks), batch_size = batch_size)
correct_state_predictions_padding(state_preds, all_masks, sequence_length)

# plotting with the state probabilities
plt.figure(figsize = (10, 15))
lim = 1 # MreB
#lim = 0.5
nb_rows = 8
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        track = track_list[offset+i*nb_rows+j]
        track = np.concatenate((np.arange(len(track))[:,None]*0.01, track), 1)
        cur_mask = all_masks[offset+i*nb_rows+j].astype(bool)
        cur_state_preds = state_preds[offset+i*nb_rows+j][cur_mask][:,:3]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], 'k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cur_state_preds[:,:3], s = 15)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 15, marker = 'x')

plt.gca().set_aspect('equal', adjustable='box')


# Plotting with the most likely states (nicer as more contrast but potentially missleading when the different states have similar probabilities)
arg_max_state_preds = np.argmax(state_preds, 2)
cs = np.array(['yellow', 'red', 'blue'])

plt.figure(figsize = (10, 15))
lim = 1 # MreB
#lim = 0.5
nb_rows = 8
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        track = track_list[offset+i*nb_rows+j]
        track = np.concatenate((np.arange(len(track))[:,None]*0.01, track), 1)
        cur_mask = all_masks[offset+i*nb_rows+j].astype(bool)
        cur_state_preds = arg_max_state_preds[offset+i*nb_rows+j][cur_mask]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], 'k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cs[cur_state_preds], s = 15)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 15, marker = 'x')

plt.gca().set_aspect('equal', adjustable='box')
plt.ylabel('position (cumulative tengential mouvements)')
plt.xlabel('time')





