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
import exatrack_while_segment_infer_vars_locErr as exatrack
#import exatrack as exatrack
from glob import glob

track_len = 10
nb_tracks = 4
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.25])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.005

np.random.seed(42)
tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.0]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.02],   # State 0 -> State 1
                                [0.1, 0.00]]),
    shape_matrix=np.array([[0, 1],
                           [1, 0]]),
    LocErr_std = 0.00,
    dt=dt,
    dt_std = 0.00,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.000001)

tracks = np.array([[[9.53872732e+00, 7.33528811e+00],
        [9.44649316e+00, 7.28328291e+00],
        [9.55743938e+00, 7.24597236e+00],
        [9.56934807e+00, 7.22151187e+00],
        [9.52781156e+00, 7.25302225e+00],
        [9.53985438e+00, 7.27504347e+00],
        [9.53975161e+00, 7.24500297e+00],
        [9.58502616e+00, 7.27322404e+00],
        [9.55486483e+00, 7.25373512e+00],
        [9.54042770e+00, 7.28893906e+00]],

       [[1.69323616e+00, 6.69464726e-01],
        [1.89581215e+00, 4.93678808e-01],
        [1.77306379e+00, 4.77428996e-01],
        [1.80163187e+00, 4.93979986e-01],
        [1.71295794e+00, 4.07058053e-01],
        [1.73680609e+00, 3.67827147e-01],
        [1.78906379e+00, 4.26241493e-01],
        [1.76387307e+00, 2.30253995e-01],
        [1.63262056e+00, 6.78884168e-03],
        [1.67098117e+00, 1.07224036e-01]],

       [[2.79580666e+00, 5.43919436e+00],
        [2.83474660e+00, 5.44704832e+00],
        [2.80212378e+00, 5.42370035e+00],
        [2.83031491e+00, 5.45085738e+00],
        [2.81888768e+00, 5.42909518e+00],
        [2.81112598e+00, 5.41034611e+00],
        [2.85428471e+00, 5.46285693e+00],
        [2.84137551e+00, 5.45726467e+00],
        [2.85482995e+00, 5.42575350e+00],
        [2.85960666e+00, 5.47087851e+00]],

       [[4.93979118e+00, 5.18757692e+00],
        [4.98534216e+00, 5.19919555e+00],
        [5.02403675e+00, 5.18010631e+00],
        [4.98305226e+00, 5.17885499e+00],
        [5.02227345e+00, 5.19388358e+00],
        [4.99811347e+00, 5.19599233e+00],
        [5.01539350e+00, 5.20351830e+00],
        [4.92167420e+00, 5.07577807e+00],
        [4.92288080e+00, 5.05278358e+00],
        [4.93165307e+00, 5.08699756e+00]]])
all_states = np.array([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

# Refined position
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
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

tracks[0]
all_states[0]
track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 4

len(tracks[0])
tracks[0][all_masks[0].astype(bool)]
# Prepare parameters for a 4 states model
nb_states = 2

# Initialize with generic guesses
params = np.array([[np.log(0.02), np.log(0.002), np.log(0.002), np.log(0.0001), 1],
                   [np.log(0.02), np.log(0.080), np.log(0.100), np.log(0.0100), 0]], dtype='float64')

initial_params = np.array([[np.log(1.0)]]*nb_states, dtype='float64')

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
vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
nb_batches = len(tracks)//batch_size
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list, 
                                    dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.01
nb_batches
epochs = 100
epoch_decay = 80
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
np.exp(-20*64*0.001)

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
                vary_transition_rates = vary_transition_rates)


device = '/CPU:0'
shuffle = True
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

parameters = exatrack.get_model_params(model, track_segmentation = True)








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
import exatrack_while_segment_infer_vars_locErr as exatrack
#import exatrack as exatrack
from glob import glob

track_len = 50
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

np.random.seed(42)

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.3]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.02],   # State 0 -> State 1
                                [0.1, 0.00]]),
    shape_matrix=np.array([[0, 1],
                           [1, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.04,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.000001)

# Refined position
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
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 25

# Prepare parameters for a 4 states model
nb_states = 2

# Initialize with generic guesses
params = np.array([[np.log(0.02), np.log(0.00002), np.log(0.05), np.log(0.0001), 1],
                   [np.log(0.02), np.log(0.10), np.log(0.100), np.log(0.0100), 0]], dtype='float64')

initial_params = np.array([[np.log(1.0)]]*nb_states, dtype='float64')

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
vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
nb_batches = len(tracks)//batch_size
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list, 
                                    dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.01
nb_batches
epochs = 100
epoch_decay = 80
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
np.exp(-20*64*0.001)

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
                vary_transition_rates = vary_transition_rates)

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


preds = model.predict(seq)
likelihood = MLE_loss(preds, preds).numpy()
print(likelihood)

All_states, All_coefs, All_biases, All_LPs = pred_model.predict(seq)

All_LPs.shape

tf.math.logsumexp(All_LPs[0])

preds.shape
All_LPs.shape
max_LP =np.max(All_LPs, 2, keepdims = True)
reduced_LP = All_LPs - max_LP
pred = tf.math.log(np.sum(np.exp(reduced_LP), 2, keepdims = True)) + max_LP
pred = pred[:,:,0]

plt.figure()
plt.plot(pred[0])

segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list, 
                                    dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)


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
                vary_transition_rates = vary_transition_rates)

LPs, All_states, All_coefs, All_biases, All_LPs = pred_model.predict(seq)
max_LP =np.max(All_LPs, 2, keepdims = True)
reduced_LP = All_LPs - max_LP
pred = tf.math.log(np.sum(np.exp(reduced_LP), 2, keepdims = True)) + max_LP
pred = pred[:,:,0]


plt.plot(pred[0])
plt.plot(np.arange(9, 18), pred[batch_size])












'''
exatrack.equilibrium_distribution enables to compute the expected equilibrium distribution
assuming shape parameters fixed at 1. A more general function is required for Gamma-distributed lifetimes
'''
estimated_fractions = list(exatrack.equilibrium_distribution(parameters['transition rates']))
True_fractions = [pb/(pu+pb), pu/(pu+pb)]
print('***')
print('True fractions:', True_fractions)
print('Estimated fractions:', estimated_fractions)
print('***')
print('True diffusion length:', parameters['d'])
print('Estimated diffusion length:', ds)
print('***')
print('True velocity of state 0:', velocity)
print('Estimated velocity of state 0:', parameters['anomalous factors'][0])
print('***')
print('True binding rate:', pb)
print('Estimated binding rate:', parameters['transition rates'][1,0])
print('***')
print('True binding rate:', pu)
print('Estimated unbinding rate:', parameters['transition rates'][0,1])
print('***')

tracks, LocErrs, frames, masks = exatrack.padding(track_list, LocErr_list, frame_list = all_dts, batch_size = batch_size)


model, pred_model = exatrack.build_model(track_len, # maximum number of time points in the input tracks
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
                vary_transition_rates = vary_transition_rates)

device = '/GPU:0'
shuffle = True
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])




'''
State predictions:
As of now, the segmentation model (also called stateful model) does not enable state
predictions. We could improve the code so it does the prediction but I pefered to
just load a classic model and perform the state predictions with that model.
'''
weights = model.get_weights()

track_array = tracks
masks = all_masks
track_array = tf.constant(track_array[:,None, :, None, None, :nb_dims], dtype = 'float64')
#track_array.shape
_, pred_model = exatrack.build_model(track_array.shape[2], # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params = weights[0], # recurrent parameters of the model
                initial_params = weights[1], # initial parameters of the model
                transition_rates = weights[7], # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes = weights[8], # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions = weights[2],
                batch_size = batch_size, # number of tracks analysed at the same time
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates)
#tracks = track_list
#track_list = tracks
#tracks = track_array
preds, All_coefs, All_biases, All_LPs = pred_model.predict((track_array, masks), batch_size = batch_size)

plt.figure()
plt.imshow(np.arange(100).reshape((10,10)))

'''
Plot a selection of tracks with state predictions
WARNING: Use track = tracks[ID] if tracks is in the list of arrays format
and use track = tracks[ID, mask.astype(bool)] if tracks is in the padded array format (nb tracks, time frame, dim)
If tracks is in the following array format (nb tracks,1, time frame, 1, 1, nb dims), you can use:
track = tracks[ID, 0, mask.astype(bool), 0, 0]
'''

colors = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]])

plt.figure(figsize = (15, 15))
plt.title('State predictions')
lim = 1.5 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID] # if tracks is in the list of arrays format
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = preds[ID, mask.astype(bool)][:,:-1]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')


'''
Plot an overlap of all the tracks with state predictions
WARNING: Use track = tracks[ID] if tracks is in the list of arrays format
and use track = tracks[ID, mask.astype(bool)] if tracks is in the padded array format (nb tracks, time frame, dim)
If tracks is in the following array format (nb tracks,1, time frame, 1, 1, nb dims), you can use:
track = tracks[ID, 0, mask.astype(bool), 0, 0]
'''

colors = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]])

plt.figure(figsize = (15, 15))
plt.title('State predictions')
lim = 1.5 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    mask = masks[i]
    track = tracks[i] # if tracks is in the list of arrays format
    print(len(track))
    p = preds[i, mask.astype(bool)][:,:-1]
    plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
    plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
    plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')



'''
Inferring the hidden real positions and velocity vector
'''
motion_types = list(params[:, 4]) + [0]
position_mean, position_std, anomalous_mean, anomalous_std, mean_preds = exatrack.extract_smooth_hidden_variables(track_array, masks, pred_model, batch_size, sequence_length, motion_types)

for track_ID in range(3):
    plt.figure()
    mask = masks[track_ID, 1:-1].astype(bool)
    plt.title('Example of refined positions, track %s'%track_ID)
    plt.errorbar(np.arange(len(position_mean[track_ID,mask,0])), position_mean[track_ID,mask,0] - np.mean(position_mean[track_ID,mask,0]), yerr = position_std[track_ID,mask,0])
    plt.errorbar(np.arange(len(position_mean[track_ID,mask,1])), position_mean[track_ID,mask,1] - np.mean(position_mean[track_ID,mask,1]), yerr = position_std[track_ID,mask,1])
    plt.xlim([-1, 100])
    plt.ylabel('Position')
    plt.xlabel('Time point')
    plt.legend(['x', 'y'])

track_ID = 0
mask = masks[track_ID, 1:-1].astype(bool)
plt.figure()
plt.title('Velocity of the state 0, track %s'%track_ID)
plt.plot((anomalous_mean[track_ID,mask,0,0]**2 + anomalous_mean[track_ID,mask,0,1]**2)**0.5)
plt.xlabel('Time step')
plt.ylabel('Estimated velocity (um/time step)')
plt.plot(np.arange(98), 0.005*(1-all_states[:,1:-1][track_ID, mask]))

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
mean_preds.shape

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
mean_preds.shape
