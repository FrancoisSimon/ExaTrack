# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:40:39 2026

@author: Franc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:45:59 2026

@author: Franc

We removed the misslinking state from ExaTrack and modified the constraint function
so it can match any arbitrary state space model
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
import os
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except:
    # add the absolute path if you are running the script line by line
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)
#import exatrack_var_shape as exatrack
#import exatrack_var_shape as exatrack
import LSTM_exatrack as exatrack
from glob import glob

track_len = 60
nb_tracks = 500
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 1               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.16])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.05

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.5],   # State 0 -> State 1
                                [0.5, 0.00]]),
    shape_matrix=np.array([[0, 5],
                           [5, 0]]),
    LocErr_std = 0.0001,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.0001,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool),:,None]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

nb_obs_vars = 1
nb_hidden_vars = 2
nb_dims = 1
integration_variable_index = 1

nb_states = 2
initial_params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + nb_hidden_vars)*4 - 2
params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + 
                            nb_hidden_vars * nb_obs_vars + 2 * (nb_obs_vars + nb_hidden_vars) + 20)*4 - 2

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 2 * np.eye(nb_states, dtype='float64')
tf.math.softmax(transition_rates)/3
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0

vary_params = np.ones(params.shape)
vary_initial_params = np.ones(initial_params.shape)
vary_initial_fractions = np.ones(initial_fractions.shape)

vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = False
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)

nb_dims = 1
sequence_length = 5
max_linking_distance = 1
segment_length = 10
lstm_units=128
nb_lstm_layers=5
feature_hidden=32
nb_LocErr_dims=1
stop_gradient_features=False

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5,
                                    shuffle=False)

nb_batches = len(seq)

all_inputs, outputs = seq[0]
inputs = all_inputs[0]
LocErrs = all_inputs[1]
dts = all_inputs[2]
dtype = 'float64'
input_mask = tf.constant(all_inputs[3], dtype = dtype)
input_isfirst = tf.constant(all_inputs[4], dtype = dtype)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.001
nb_batches
epochs = 150
epoch_decay = 50
decay_threshold = epoch_decay*nb_batches
verbose = 1

# Compute the required decay rate to decay the learning rate by a factor decay_ratio
decay_ratio = 0.001 
decay_rate = - np.log(decay_ratio)/((epochs - epoch_decay) * nb_batches)


model, pred_model = exatrack.build_segment_model_LSTM(segment_length,
                             nb_states,
                             initial_fractions,
                             transition_rates,
                             transition_shapes,
                             batch_size,
                             reference_dt,
                             nb_dims=nb_dims,
                             sequence_length=sequence_length,
                             nb_hidden_vars=nb_hidden_vars,
                             nb_obs_vars=nb_obs_vars,
                             integration_variable_index=integration_variable_index,
                             lstm_units=lstm_units,
                             nb_lstm_layers=nb_lstm_layers,
                             feature_hidden=feature_hidden,
                             vary_initial_fractions=vary_initial_fractions,
                             vary_transition_shapes=vary_transition_shapes,
                             vary_transition_rates=vary_transition_rates,
                             nb_LocErr_dims=1,
                             stop_gradient_features=False)

device = '/CPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=0.1) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

plt.figure()
plt.plot(history.history['loss'])

weights = model.get_weights()

max_track_len = np.max([len(track) for track in track_list])
seq_full = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=max_track_len,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

_, pred_model = exatrack.build_segment_model_LSTM(max_track_len,
                             nb_states,
                             initial_fractions,
                             transition_rates,
                             transition_shapes,
                             batch_size,
                             reference_dt,
                             nb_dims=nb_dims,
                             sequence_length=sequence_length,
                             nb_hidden_vars=nb_hidden_vars,
                             nb_obs_vars=nb_obs_vars,
                             integration_variable_index=integration_variable_index,
                             lstm_units=lstm_units,
                             nb_lstm_layers=nb_lstm_layers,
                             feature_hidden=feature_hidden,
                             vary_initial_fractions=vary_initial_fractions,
                             vary_transition_shapes=vary_transition_shapes,
                             vary_transition_rates=vary_transition_rates,
                             nb_LocErr_dims=1,
                             stop_gradient_features=False)

pred_model.set_weights(weights)

LPs, preds, All_coefs, All_biases, All_LPs = pred_model.predict(seq_full)

tracks = np.concatenate([seq_full[i][0][0] for i in range(len(seq_full))], 0)
LocErrs = np.concatenate([seq_full[i][0][1] for i in range(len(seq_full))], 0)
time_steps = np.concatenate([seq_full[i][0][2] for i in range(len(seq_full))], 0)
masks = np.concatenate([seq_full[i][0][3] for i in range(len(seq_full))], 0)

hidden_estimates = exatrack.extract_hidden_variables_general(All_coefs, All_biases, All_LPs,
                                     nb_dims, sequence_length,
                                     ridge=1e-12, eps=1e-30)

hidden_estimates['collapsed_mean'].shape
'''
Plot tracks with state predictions
'''

colors = np.array([[1,0,0],
                   [0,0,1]])


plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(np.arange(len(track))*0.02 + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*0.02 + lim*j, track[:,0] , c = p@colors, s = 7)
plt.gca().set_aspect('equal', adjustable='box')


plt.figure(figsize = (15, 15))
plt.title('ExaTrack hidden state predictions')
lim = 6.2 # MreB
xlim = 0.08
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = hidden_estimates['collapsed_mean'][ID, mask.astype(bool)[1:], 0]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i]]
        p = preds[ID, mask.astype(bool)][1:]
        p = np.clip(p, 0, 1)
        
        plt.plot(np.arange(len(track))*xlim + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*xlim + lim*j, track[:,0] , c = p@colors, s = 7)
plt.gca().set_aspect('equal', adjustable='box')


plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 3, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')



'''
Test on bacteria tracking
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
import os
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except:
    # add the absolute path if you are running the script line by line
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)
#import exatrack_var_shape as exatrack
#import exatrack_var_shape as exatrack
import LSTM_exatrack as exatrack
from glob import glob

track_len = 200
nb_tracks = 500
reference_dt = 1                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 1               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.16])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.05
dtype = 'float64'

path = r"C:\Users\Franc\Downloads\bacteria_tracks.csv"

track_list, frame_list, track_IDs, opt_metrics = exatrack.read_table(path, # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(10, track_len+1), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions 
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True)
track_list[0]

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 30 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(track_list))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = track_list[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

# LocErr_list and dt_list can be set to None if they are assumed to be constant

track_list = [track[:,None,:] for track in track_list]
track_list[0].shape

LocErr_list = None
# LocErr_list = None
dt_list = [np.concatenate((frame_list[i][1:] - frame_list[i][:-1], [1])) for i in range(len(track_list))]

batch_size = 20

# Prepare parameters for a 4 state model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

nb_obs_vars = 1
nb_hidden_vars = 2
nb_dims = 2
integration_variable_index = 1

nb_states = 3
initial_params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + nb_hidden_vars)*4 - 2
params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + 
                            nb_hidden_vars * nb_obs_vars + 2 * (nb_obs_vars + nb_hidden_vars) + 20)*4 - 2

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 3 * np.eye(nb_states, dtype='float64')
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64') #+ np.log(2)#-7 + np.log
max_off_rate  = 1/3

(tf.eye(nb_states, dtype = dtype) * (1 - max_off_rate + max_off_rate*tf.math.softmax(transition_rates, axis = 1)) + (1-tf.eye(nb_states, dtype = dtype)) * ( max_off_rate*tf.math.softmax(transition_rates, axis = 1)) + 1e-7) * tf.math.exp(transition_shapes)


'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0

vary_params = np.ones(params.shape)
vary_initial_params = np.ones(initial_params.shape)
vary_initial_fractions = np.ones(initial_fractions.shape)

vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = True
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)

sequence_length = 10
max_linking_distance = 1
segment_length = 20
lstm_units=480
nb_lstm_layers=5
feature_hidden=480
nb_LocErr_dims=1
stop_gradient_features=False

H = nb_hidden_vars
S = nb_states
seq =  sequence_length
D =  nb_dims
pc = H * (seq * S) * H            # Prev_coefs  (H, N, seq*S, H)
pb = H * (seq * S) * D            # Prev_biases (H, N, seq*S, D)
lp = seq * S                      # LP
sl = seq * S                      # segment_len
gm = seq * S * S                  # gamma_dist_mean
gv = seq * S * S                  # gamma_dist_var
pc + pb + lp + sl + gm + gv

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5,
                                    shuffle=False)

nb_batches = len(seq)

all_inputs, outputs = seq[0]
inputs = all_inputs[0]
LocErrs = all_inputs[1]
dts = all_inputs[2]
dtype = 'float64'
input_mask = tf.constant(all_inputs[3], dtype = dtype)
input_isfirst = tf.constant(all_inputs[4], dtype = dtype)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.00005
nb_batches
epochs = 150
epoch_decay = 50
decay_threshold = epoch_decay*nb_batches
verbose = 1

# Compute the required decay rate to decay the learning rate by a factor decay_ratio
decay_ratio = 0.0001 
decay_rate = - np.log(decay_ratio)/((epochs - epoch_decay) * nb_batches)


model, pred_model = exatrack.build_segment_model_LSTM(segment_length,
                             nb_states,
                             initial_fractions,
                             transition_rates,
                             transition_shapes,
                             batch_size,
                             reference_dt,
                             nb_dims=nb_dims,
                             sequence_length=sequence_length,
                             nb_hidden_vars=nb_hidden_vars,
                             nb_obs_vars=nb_obs_vars,
                             integration_variable_index=integration_variable_index,
                             lstm_units=lstm_units,
                             nb_lstm_layers=nb_lstm_layers,
                             feature_hidden=feature_hidden,
                             vary_initial_fractions=vary_initial_fractions,
                             vary_transition_shapes=vary_transition_shapes,
                             vary_transition_rates=vary_transition_rates,
                             nb_LocErr_dims=1,
                             stop_gradient_features=False)

device = '/GPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=0.1) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

plt.figure()
plt.plot(history.history['loss'])

weights = model.get_weights()

max_track_len = np.max([len(track) for track in track_list])
seq_full = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=max_track_len,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

_, pred_model = exatrack.build_segment_model_LSTM(max_track_len,
                             nb_states,
                             initial_fractions,
                             transition_rates,
                             transition_shapes,
                             batch_size,
                             reference_dt,
                             nb_dims=nb_dims,
                             sequence_length=sequence_length,
                             nb_hidden_vars=nb_hidden_vars,
                             nb_obs_vars=nb_obs_vars,
                             integration_variable_index=integration_variable_index,
                             lstm_units=lstm_units,
                             nb_lstm_layers=nb_lstm_layers,
                             feature_hidden=feature_hidden,
                             vary_initial_fractions=vary_initial_fractions,
                             vary_transition_shapes=vary_transition_shapes,
                             vary_transition_rates=vary_transition_rates,
                             nb_LocErr_dims=1,
                             stop_gradient_features=False)

pred_model.set_weights(weights)

LPs, preds, All_coefs, All_biases, All_LPs = pred_model.predict(seq_full)

tracks = np.concatenate([seq_full[i][0][0] for i in range(len(seq_full))], 0)
LocErrs = np.concatenate([seq_full[i][0][1] for i in range(len(seq_full))], 0)
time_steps = np.concatenate([seq_full[i][0][2] for i in range(len(seq_full))], 0)
masks = np.concatenate([seq_full[i][0][3] for i in range(len(seq_full))], 0)

hidden_estimates = exatrack.extract_hidden_variables_general(All_coefs, All_biases, All_LPs,
                                     nb_dims, sequence_length,
                                     ridge=1e-12, eps=1e-30)

hidden_estimates['collapsed_mean'].shape
'''
Plot tracks with state predictions
'''

colors = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]])

plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 40.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool), 0, :]
        tracks.shape
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 3, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')


plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, ]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(np.arange(len(track))*0.02 + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*0.02 + lim*j, track[:,0] , c = p@colors, s = 7)
plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(np.arange(len(track))*0.02 + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*0.02 + lim*j, track[:,0] , c = p@colors, s = 7)
plt.gca().set_aspect('equal', adjustable='box')


plt.figure(figsize = (15, 15))
plt.title('ExaTrack hidden state predictions')
lim = 6.2 # MreB
xlim = 0.08
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = hidden_estimates['collapsed_mean'][ID, mask.astype(bool)[1:], 0]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i]]
        p = preds[ID, mask.astype(bool)][1:]
        p = np.clip(p, 0, 1)
        
        plt.plot(np.arange(len(track))*xlim + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*xlim + lim*j, track[:,0] , c = p@colors, s = 7)
plt.gca().set_aspect('equal', adjustable='box')


plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 3, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')


MCRolloutForecaster

'''
Comparison to exatrack fitting
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
import os
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except:
    # add the absolute path if you are running the script line by line
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)
#import exatrack_var_shape as exatrack
#import exatrack_var_shape as exatrack
import exatrack
from glob import glob

track_len = 60
nb_tracks = 200
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 1               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 1])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.05

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.5],   # State 0 -> State 1
                                [0.5, 0.00]]),
    shape_matrix=np.array([[0, 5],
                           [5, 0]]),
    LocErr_std = 0.0001,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.0001,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(1), np.log(0.01), np.log(0.01), np.log(0.0002), 1],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.001), 0]])
nb_states = len(params)

initial_params = np.array([[np.log(60)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 3 * np.eye(nb_states, dtype='float64')
transition_rates[0,0] = 2
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = True

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 1
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.05
nb_batches
epochs = 50
epoch_decay = 30
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
verbose = 1

model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                reference_dt,
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates,
                LocErr_type = 'Linear')

device = '/GPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

infer = exatrack.LSTMStateInference(pred_model, nb_states=nb_states,
                                    sequence_length=sequence_length, reference_dt=reference_dt)


'''
3 state model
'''

track_len = 60
nb_tracks = 200
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 1               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0, 0.16])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.05

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.3, 0.3, 0.4]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0, 0.3]),
    conf_Ds=np.array([0.0, 0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0, 0.0]),
    transition_matrix=np.array([[0.00, 0.5, 0.5],   # State 0 -> State 1
                                [0.5, 0.00, 0.5],
                                [0.5, 0.5, 0]]),
    shape_matrix=np.array([[0, 5, 5],
                           [5, 0, 5],
                           [5, 5, 0]]),
    LocErr_std = 0.0001,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.0001,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

nb_obs_vars = 1
nb_hidden_vars = 3
nb_dims = 1
integration_variable_index = 1

nb_states = 3
initial_params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + nb_hidden_vars)*4 - 2
params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + 
                            nb_hidden_vars * nb_obs_vars + 2 * (nb_obs_vars + nb_hidden_vars) + 20)*4 - 2

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 2 * np.eye(nb_states, dtype='float64')
tf.math.softmax(transition_rates)/3
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0

vary_params = np.ones(params.shape)
vary_initial_params = np.ones(initial_params.shape)
vary_initial_fractions = np.ones(initial_fractions.shape)

vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = False
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)

nb_dims = 1
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

all_inputs, outputs = seq[0]
inputs = all_inputs[0]
LocErrs = all_inputs[1]
dts = all_inputs[2]
dtype = 'float64'
input_mask = tf.constant(all_inputs[3], dtype = dtype)
input_isfirst = tf.constant(all_inputs[4], dtype = dtype)


current_constraint_function = constraint_function_arbitrary_KF(nb_hidden_vars, nb_obs_vars, integration_variable_index)
#self = current_constraint_function
current_constraint_function(params, initial_params, LocErrs, dts,
         nb_dims, reference_dt, 0, dtype)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.0003
nb_batches
epochs = 150
epoch_decay = 50
decay_threshold = epoch_decay*nb_batches
verbose = 1

# Compute the required decay rate to decay the learning rate by a factor decay_ratio
decay_ratio = 0.001 
decay_rate = - np.log(decay_ratio)/((epochs - epoch_decay) * nb_batches)


model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                reference_dt,
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                current_constraint_function = current_constraint_function,
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates,
                nb_LocErr_dims = 1,
                LocErr_type = 'Linear')

device = '/CPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=0.1) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

plt.figure()
plt.plot(history.history['loss'])

weights = model.get_weights()

max_track_len = np.max([len(track) for track in track_list])
seq_full = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=max_track_len,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

_, pred_model = exatrack.build_segment_model(max_track_len, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params = weights[0], # recurrent parameters of the model
                initial_params = weights[1], # initial parameters of the model
                transition_rates = weights[6], # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes = weights[7], # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions = weights[2],
                batch_size = batch_size, # number of tracks analysed at the same time
                reference_dt = reference_dt,
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                current_constraint_function = current_constraint_function,
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates,
                nb_LocErr_dims = 1,
                LocErr_type = 'Linear')

exatrack.get_model_params(model, track_segmentation=True)

LPs, preds, All_coefs, All_biases, All_LPs = pred_model.predict(seq_full)

tracks = np.concatenate([seq_full[i][0][0] for i in range(len(seq_full))], 0)
LocErrs = np.concatenate([seq_full[i][0][1] for i in range(len(seq_full))], 0)
time_steps = np.concatenate([seq_full[i][0][2] for i in range(len(seq_full))], 0)
masks = np.concatenate([seq_full[i][0][3] for i in range(len(seq_full))], 0)


'''
Plot tracks with state predictions
'''

colors = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]])


plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(np.arange(len(track))*0.02 + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*0.02 + lim*j, track[:,0] , c = p@colors, s = 7)
plt.gca().set_aspect('equal', adjustable='box')






'''
Inferring the hidden real positions and velocity vector
'''

motion_types = list(params[:, 4]) + [0]
position_mean, position_std, anomalous_mean, anomalous_std, mean_preds = exatrack.extract_smooth_hidden_variables(tracks, LocErrs, time_steps, masks, pred_model, batch_size, sequence_length, motion_types, reference_dt)

for track_ID in range(3):
    plt.figure()
    mask = masks[track_ID, 1:-1].astype(bool)
    plt.title('Example of refined positions, track %s'%track_ID)
    plt.errorbar(np.arange(len(position_mean[track_ID,mask,0])), position_mean[track_ID,mask,0] - np.mean(position_mean[track_ID,mask,0]), yerr = position_std[track_ID,mask,0])
    plt.errorbar(np.arange(len(position_mean[track_ID,mask,1])), position_mean[track_ID,mask,1] - np.mean(position_mean[track_ID,mask,1]), yerr = position_std[track_ID,mask,1])
    plt.xlim([-1, np.sum(mask)])
    plt.ylabel('Position')
    plt.xlabel('Time point')
    plt.legend(['x', 'y'])

track_ID = 0
directed_state_ID = 0
mask = masks[track_ID, 1:-1].astype(bool)
plt.figure()
plt.title('Velocity assuming directed state, track %s'%track_ID)
plt.plot((anomalous_mean[track_ID, mask,directed_state_ID, 0]**2 + anomalous_mean[track_ID,mask,directed_state_ID,1]**2)**0.5)
plt.xlabel('Time step')
plt.ylabel('Estimated velocity (um/time step)')

track_ID = 0
mask = masks[track_ID,].astype(bool)
plt.figure()
plt.title('state probabilities, track %s'%track_ID)
plt.plot(preds[track_ID,mask])
plt.xlabel('Time step')
plt.ylabel('Estimated velocity (um/time step)')
plt.legend(['State 0', 'State 1', 'Misslinking'])

# mean_preds gives better estimates at transition time points than preds (intermediate probability)
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 6
plt.title('State predictions with mean_preds')
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = mean_preds[ID, mask.astype(bool)][:,:-1]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

# Refined positions
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 6
plt.title('Refined particle positions')
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID, 1:-1]
        track = position_mean[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = mean_preds[:,1:-1][ID, mask.astype(bool)][:,:-1]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')





'''
Determining the number of states
'''

track_len = 50
nb_tracks = 50
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.000025, 0.25])

dt = 0.02
ds = (2*Ds*dt)**0.5 
velocity = 0.005

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition( 
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.3, 0.3, 0.4]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0, 0.2]),
    conf_Ds=np.array([0.0, 0, 0.0]),         # No diffusion of confinement center 
    conf_dists=np.array([0.0,0, 0.0]),
    transition_matrix=np.array([[0.00, 0, 0.02],   # State 0 -> State 2
                                [0.06, 0, 0.03],
                                [0.0, 0.05, 0]]),
    shape_matrix=np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]), 
    dt=dt, 
    dt_std = 0.001,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0, 
    bleaching_rate = 0.02) 

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 1 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(1), np.log(0.001), np.log(0.0001), np.log(0.0002), 1],
                   [np.log(1), np.log(0.001), np.log(0.1), np.log(0.0002), 0],
                   [np.log(1), np.log(0.001), np.log(0.005), np.log(0.0002), 1],
                   [np.log(1), np.log(0.003), np.log(0.1), np.log(0.001), 0],                   [np.log(0.025), np.log(0.01), np.log(0.01), np.log(0.002), 1],
                   [np.log(1), np.log(0.01), np.log(0.1), np.log(0.002), 0],
                   [np.log(1), np.log(0.05), np.log(0.05), np.log(0.002), 1],
                   [np.log(1), np.log(0.05), np.log(0.05), np.log(0.002), 0],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.0002), 1],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.0002), 0]])

nb_states = len(params)

initial_params = np.array([[np.log(60)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 3 * np.eye(nb_states, dtype='float64')
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')
tf.math.softmax(transition_rates, 1)

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.01
nb_batches
epochs = 50
epoch_decay = 13
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
verbose = 1
LocErr_type = 'Linear'

model_results = exatrack.get_number_of_states(track_list,
                         params,
                         initial_params,
                         transition_shapes,
                         transition_rates,
                         initial_fractions,
                         reference_dt,
                         dt_list = dt_list,
                         LocErr_list = LocErr_list,
                         nb_dims = 2,
                         sequence_length = 10,
                         max_linking_distance = 0.4,
                         estimated_density = 0.001,
                         epochs = epochs,
                         epoch_decay = epoch_decay,
                         learning_rate = 0.02,
                         decay_rate = 0.005,
                         batch_size = batch_size,
                         vary_params = True,
                         vary_initial_params = True,
                         vary_initial_fractions = True,
                         vary_transition_shapes = False,
                         vary_transition_rates = True,
                         device = device,
                         track_segmentation = True,
                         segment_length = 10,
                         LocErr_type = 'Linear')


log_likelihoods = [model_results[i]['log_likelihood'].numpy() for i in range(1, 11)]

model_results[3]

plt.figure()
plt.plot(np.arange(1, 11), log_likelihoods)
exatrack.equilibrium_distribution(np.array([[0.98, 0, 0.02],   # State 0 -> State 2
                            [0.06, 0.91, 0.03],
                            [0.0, 0.05, 0.95]]))


'''
Using Model_finder to determine if each state is rather confined or directed for a given number of states 
'''

track_len = 60
nb_tracks = 200
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 1])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.005

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.25],   # State 0 -> State 1
                                [0.25, 0.00]]),
    shape_matrix=np.array([[0, 5],
                           [5, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.002,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 50

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(1), np.log(0.01), np.log(0.01), np.log(0.0002), 0],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.001), 0]])
nb_states = len(params)

initial_params = np.array([[np.log(60)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 3 * np.eye(nb_states, dtype='float64')
transition_rates[0,0] = 2
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = True

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 20
epochs = 60

model, pred_model = exatrack.Model_finder(track_list,
                 reference_dt,
                 sequence_length,
                 nb_states, 
                 params,
                 initial_params,  
                 initial_fractions, 
                 transition_shapes, 
                 transition_rates, 
                 max_linking_distance, 
                 estimated_density, 
                 epochs, 
                 batch_size,
                 LocErr_list = LocErr_list,
                 dt_list = dt_list,
                 segment_length = 10,
                 learning_rate = 1/20,
                 decay_fraction = 0.2,
                 decay_rate = 0.002,
                 device = '/GPU:0', 
                 verbose = 1,       
                 shuffle = False,
                 vary_params = vary_params,
                 vary_initial_params = vary_initial_params,
                 vary_initial_fractions = vary_initial_fractions,
                 vary_transition_shapes = vary_transition_shapes,
                 vary_transition_rates = vary_transition_rates,
                 LocErr_type = 'Linear')