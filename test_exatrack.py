# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:30:22 2026

@author: Franc


#kjjgfsklgf
#Next to do: adjust the memory of the transition processes so it considers the uneven dts
#Parse the current coefficients, biases, scaling factors
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
import exatrack
#import exatrack as exatrack
from glob import glob

track_len = 100
nb_tracks = 500
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.25])
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
    transition_matrix=np.array([[0.00, 0.02],   # State 0 -> State 1
                                [0.05, 0.00]]),
    shape_matrix=np.array([[0, 1],
                           [1, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.01,
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
i=1
track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(0.02), np.log(0.01), np.log(0.01), np.log(0.0002), 1],
                   [np.log(0.02), np.log(0.1), np.log(0.1), np.log(0.001), 0]])
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
learning_rate = 0.005
nb_batches
epochs = 60
epoch_decay = 50
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005

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

device = '/CPU:0'
shuffle = True
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

'''
Then, we predict the state of motion of the different tracks
'''

weights = model.get_weights()

max_track_len = np.max([len(track) for track in track_list])
seq_full = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=max_track_len,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

model, pred_model = exatrack.build_segment_model(max_track_len, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params = weights[0], # recurrent parameters of the model
                initial_params = weights[1], # initial parameters of the model
                transition_rates = weights[7], # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes = weights[8], # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions = weights[2],
                batch_size = batch_size, # number of tracks analysed at the same time
                reference_dt = reference_dt,
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
                   [0,0,1]])

plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 0.8 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = preds[ID, mask.astype(bool)][:,:-1]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 3, marker = 'x')
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

# Refined position
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


