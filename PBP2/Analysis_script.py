# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:04:58 2026

@author: Franc
"""

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import exatrack_while_segment as exatrack
#import exatrack as exatrack
from glob import glob


# TODO: Update this path to point to your RodComplex/RodZ data directory
data_dir = r'C:\Users\Franc\Data\RodComplex\RodZ'
exps = glob(os.path.join(data_dir, '*RodZ-IPTG*'))
paths = glob(os.path.join(data_dir, '*RodZ-IPTG*', '*.csv'))

tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(10, 101), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '.
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = False)

for i in range(len(tracks)):
    tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))

len(tracks)

ls = [len(track) for track in tracks]
np.mean(ls)
np.max(ls)

plt.figure(figsize = (15, 15))
lim = 0.5 # MreB
nb_rows = 14
offset = 220*2
for i in range(nb_rows):
    for j in range(nb_rows):
        track = tracks[offset+i*nb_rows+j]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

# %%

'''
Determining the number of states and the best parameters
'''

# Prepare parameters for a maximum of 8 states
max_states = 10

# Initialize with generic guesses
params = np.array([[np.log(0.025), np.log(0.003), np.log(0.003), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.003), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.003), np.log(0.003), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.03), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.03), np.log(0.01), np.log(0.01), 1],
                   [np.log(0.025), np.log(0.05), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.05), np.log(0.02), np.log(0.01), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.03), np.log(0.02), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.15), np.log(0.1), np.log(0.01), 1]], dtype='float64')

# Prepare parameters for a maximum of 8 states
'''
max_states = 4

# Initialize with generic guesses
params = np.array([[np.log(0.025), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.0001), np.log(0.0001), np.log(0.00001), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.15), np.log(0.1), np.log(0.01), 1]], dtype='float64')
'''
initial_params = np.array([[np.log(1.0)]]*max_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*max_states+[-5.0]], dtype='float64')

# Transition matrices
transition_rates = 4 * np.eye(max_states, dtype='float64')
transition_shapes = np.zeros((max_states, max_states), dtype='float64')

# Create vary masks to fix certain parameters
# vary_params: which recurrent parameters to optimize
vary_params = True
# vary_transition_shapes: fix shapes to 1 (exponential)
vary_transition_shapes = False
# Allow other parameters to vary
vary_initial_params = True
vary_initial_fractions = True
vary_transition_rates = True

epochs = 30
batch_size = 400
epoch_decay = 25
learning_rate = 0.01
decay_rate = 0.002
nb_batches = len(tracks)//batch_size
device = '/GPU:0'
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches*2))

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1

results = exatrack.get_number_of_states(tracks,
                     params,
                     initial_params,
                     transition_shapes,
                     transition_rates,
                     initial_fractions,
                     nb_dims,
                     sequence_length,
                     max_linking_distance,
                     estimated_density,
                     epochs = epochs,
                     epoch_decay = epoch_decay,
                     learning_rate = learning_rate,
                     decay_rate = decay_rate,
                     batch_size = batch_size,
                     vary_params = vary_params,
                     vary_initial_params = vary_initial_params,
                     vary_initial_fractions = vary_initial_fractions,
                     vary_transition_shapes = vary_transition_shapes,
                     vary_transition_rates = vary_transition_rates,
                     device = device,
                     track_segmentation = True,
                     segment_length = 10)

results.keys()
log_likelihoods = np.array([results[k]['log_likelihood'] for k in range(1, 11)])

plt.figure()
plt.plot(np.arange(1, 11), log_likelihoods)

save_model_selection_results(results, save_dir = r'C:\Users\Franc\Data\RodComplex\Results/RodZ_IPTG_100_10_2')

loaded_results = load_model_selection_results(r'C:\Users\Franc\Data\RodComplex\Results/RodZ_IPTG_100_10')
loaded_results[4]

results[4]
parameters = get_model_params(results['all_results'][5]['model'])









'''
3 state model analysis
'''

segment_length = 12
batch_size = 500
seq = exatrack.TrackSegmentSequence(tracks,
    batch_size=batch_size,
    segment_length=segment_length,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5)
nb_batches = len(seq)

dtype = 'float64'
device = '/GPU:0'
estimated_density = 0.01 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1

# We need to reshape the track to the shape used by the model, as following

nb_states = 3
params = np.array([#[np.log(0.015), np.log(0.001), np.log(0.00001), np.log(0.0000001), 1],
                   [np.log(0.015), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                   [np.log(0.015), np.log(0.03), np.log(0.1), np.log(0.002), 0],
                   [np.log(0.015), np.log(0.08), np.log(0.1), np.log(0.005), 0]], dtype = dtype)

initial_params = np.array([[np.log(1)]]*nb_states, dtype = dtype)

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*3

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 5 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

vary_params = True
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False
vary_transition_rates = True

model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates)

nb_batches = len(seq)

seq = exatrack.TrackSegmentSequence(tracks,
    batch_size=batch_size,
    segment_length=segment_length,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5)

#all_masks = masks
learning_rate = 0.01
epochs = 100
decay_threshold = int(0.7*nb_batches*epochs)
decay_rate = - np.log(0.001)/(0.3*nb_batches*epochs) #
device = '/GPU:0'
shuffle = False
verbose = 1

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])

preds = pred_model.predict((tracks, masks), batch_size = batch_size)
0.0005/0.06
0.014/0.000

import numpy as np
import pandas as pd

data = np.load(r'C:\Users\Franc\Data\Kinesin_minflux\fire\S-BIAD\608\S-BIAD608\Files\Astra_upload_files\04_HighATP_HighLaser_PFA\220113-183448_04_Minflux_642_L75_pho100_lp15_BGdis40k_1st2nd25k_hex3_dt300us_cfr09_dp1_.npy')

paths = glob(r'C:\Users\Franc\Data\Kinesin_minflux\fire\S-BIAD\608\S-BIAD608\Files\Astra_upload_files\07_K560N_U2OS\*.npy')
for path in paths:
    rows = []

    for event in data:
        if not event['vld']:
            continue  # skip invalid points

        itr = event['itr'][-1]  # final iteration

        row = {
            "track_id": event['tid'],
            "time": event['tim'],

            # position (meters → convert to nm)
            "x_um": itr['loc'][0] * 1e6,
            "y_um": itr['loc'][1] * 1e6,
            "z_um": itr['loc'][2] * 1e6,

            # precision
            "sigma_x_um": itr['ext'][0] * 1e6,
            "sigma_y_um": itr['ext'][1] * 1e6,
            "sigma_z_um": itr['ext'][2] * 1e6,

            # photons
            "photons": itr['eco'],
            "background": itr['fbg'],

            # quality
            "confidence": itr['cfr'],
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(r"C:\Users\Franc\Data\Kinesin_minflux\Tracks/" + path.rsplit('\\', 1)[1], index=False)



import scipy.io as sio
import numpy as np

# Load the .mat file into a Python dictionary
# Use 'r' before the string for raw string handling of file paths (especially on Windows)
mat_contents = sio.loadmat()

import mat73
data_dict = mat73.loadmat(r'C:\Users\Franc\Data\Kinesin_minflux\fire\S-BIAD\608\S-BIAD608\Files\Astra_upload_files\01_Neuron_HighATP_lowLaser_PFA_sml.mat')
len(data_dict)
data_dict.keys()
data_dict['saveloc'].keys()
data_dict['saveloc']['loc'].keys()
len(data_dict['saveloc']['loc']['xnm'])
data_dict['saveloc']['loc']['tid']
pd.DataFrame(data_dict['saveloc']['loc'])

model, pred_model = exatrack.build_model(segment_length, # maximum number of time points in the input tracks
                max_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates)

tracks, _, masks = padding(tracks[:40000])
tracks = tf.constant(tracks[:,None, :, None, None, :nb_dims], dtype)

len(tracks)
len(masks)
lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict((tracks, masks), batch_size = batch_size)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit((tracks, masks), tracks, epochs = epochs, batch_size = batch_size, callbacks=[get_parameters()], shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])


100/100 [==============================] - ETA: 0s - loss: -34.6754{'Model types': array(['Directed motion', 'Directed motion', 'Confined motion',
       'Directed motion'], dtype='<U15'), 'anomalous factors': [0.001, 0.001, 0.155, 0.005], 'Localization errors': [0.026, 0.014, 0.012, 0.023], 'd': [0.003, 0.003, 0.022, 0.071], 'transition rates': [0.776, 0.047, 0.161, 0.015, 0.06, 0.871, 0.065, 0.005, 0.188, 0.099, 0.664, 0.05, 0.013, 0.023, 0.195, 0.768], 'transition shapes': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'Fractions': [0.282, 0.301, 0.261, 0.156, 0.0]}

get_model_params(model)
model.weights
{'Model types': array(['Confined motion', 'Confined motion', 'Confined motion'],
       dtype='<U15'),
 'anomalous factors': <tf.Tensor: shape=(3,), dtype=float64, numpy=array([0.24466595, 0.0621081 , 0.00178778])>,
 'Localization errors': array([0.02914872, 0.00159614, 0.01618835]),
 'd': array([0.02398746, 0.08546698, 0.16293599]),
 'transition rates': <tf.Tensor: shape=(3, 3), dtype=float64, numpy=
 array([[9.31751074e-01, 6.60561079e-02, 2.19281816e-03],
        [1.39247639e-02, 8.36354697e-01, 1.49720539e-01],
        [7.05118041e-04, 2.78923933e-01, 7.20370949e-01]])>,
 'transition shapes': <tf.Tensor: shape=(3, 3), dtype=float64, numpy=
 array([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])>,
 'Fractions': <tf.Tensor: shape=(4,), dtype=float64, numpy=array([6.02270597e-02, 4.74831810e-01, 4.64914103e-01, 2.70279544e-05])>}
-70.05200958251953

Out[14]:
[<tf.Variable 'initial_layer_constraints/recurrence_variables:0' shape=(3, 5) dtype=float64, numpy=
 array([[-3.53534433, -3.73022428, -1.12726629, -5.58415354,  0.        ],
        [-6.44016471, -2.45962517, -2.71475827, -7.63083379,  0.        ],
        [-4.12346366, -1.81439784, -6.32498985, -1.90010762,  0.        ]])>,
 <tf.Variable 'initial_layer_constraints/initial_variables:0' shape=(3, 2) dtype=float64, numpy=
 array([[ 3.16625425, -2.99573227],
        [ 3.15371507, -2.99573227],
        [ 3.14854089, -2.99573227]])>,
 <tf.Variable 'initial_layer_constraints/Fractions:0' shape=(1, 4) dtype=float64, numpy=array([[-0.02230021,  2.0425387 ,  2.02143071, -7.73130556]])>,
 <tf.Variable 'initial_layer_constraints/max linking distance:0' shape=() dtype=float64, numpy=1.0>,
 <tf.Variable 'custom_rnn_layer/Transition rates:0' shape=(3, 3) dtype=float64, numpy=
 array([[ 4.53442105,  1.88785986, -1.5174571 ],
        [-0.92376287,  3.1716211 ,  1.45133878],
        [-3.78915201,  2.19117715,  3.14000434]])>,
 <tf.Variable 'custom_rnn_layer/Transition shape:0' shape=(3, 3) dtype=float64, numpy=
 array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])>]
track_array = tracks[:, 0, :, 0, 0].numpy()

# Plot tracks
cs = np.array([ [0,1,0], [1, 0, 0], [0,1,1], [0,0,1]])
cs = np.array([ [1,0,0], [0, 1, 0], [0,0,1]])
# dir, fixed conf, dir + dif, mov conf, diff

plt.figure(figsize = (15, 15))
lim = 2.5 # MreB
#lim = 0.5
nb_rows = 6
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        mask = masks[offset+i*nb_rows+j].astype(bool)
        track = track_array[offset+i*nb_rows+j, mask]
        cur_color = preds[offset+i*nb_rows+j][mask, :len(cs)]@cs
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)

        plt.scatter(track[:,0], track[:,1] , c = cur_color, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

cur_color.shape
track[:,1].shape
track.shape

'''
Analyse the EGCG condition
'''

tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths[6:], # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(30,81), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '.
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True)

# Remove tracks with mean quality lower than 200
new_tracks = []
new_frames = []
new_track_IDs = []
new_opt_metrics = {'QUALITY': []}
for i in range(len(tracks)):
    current_quality = opt_metrics['QUALITY'][i]
    if np.mean(current_quality)>200:
        new_tracks.append(tracks[i])
        new_frames.append(frames[i])
        new_track_IDs.append(track_IDs[i])
        new_opt_metrics['QUALITY'].append(current_quality)
tracks = new_tracks
frames = new_frames
track_IDs = new_track_IDs
opt_metrics = new_opt_metrics

tracks = tracks[:9000]
tracks, _, masks = padding(tracks)
tracks = tf.constant(tracks[:,None, :, None, None, :nb_dims], dtype)

nb_states = 3
params = np.array([[np.log(0.03), np.log(0.01), np.log(0.1), np.log(0.001), 0],
                   [np.log(0.03), np.log(0.05), np.log(0.1), np.log(0.002), 0],
                   [np.log(0.03), np.log(0.15), np.log(0.1), np.log(0.005), 0]], dtype = dtype)

initial_params = np.array([[np.log(100), np.log(0.05)]]*nb_states, dtype = dtype)

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*4.5

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 3 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

vary_params = True
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False
vary_transition_rates = True

model, pred_model = build_model(track_len, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates)

model.weights

#all_masks = masks
learning_rate = 0.05
epochs = 50
64*60
decay_threshold = 1000
decay_rate = 0.002
np.exp(-60*24*0.002)
10000/50
device = '/GPU:0'
shuffle = True
verbose = 1
24*100
lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict((tracks, masks), batch_size = batch_size)
likelihood = MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit((tracks, masks), tracks, epochs = epochs, batch_size = batch_size, callbacks=[get_parameters()], shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])

preds = pred_model.predict((tracks, masks), batch_size = batch_size)
track_array = tracks[:, 0, :, 0, 0].numpy()

# Plot tracks
cs = np.array([ [1,0,0], [0, 1, 0], [0,0,1]])
# dir, fixed conf, dir + dif, mov conf, diff

plt.figure(figsize = (15, 15))
lim = 2.5 # MreB
#lim = 0.5
nb_rows = 6
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        mask = masks[offset+i*nb_rows+j].astype(bool)
        track = track_array[offset+i*nb_rows+j, mask]
        cur_color = preds[offset+i*nb_rows+j][mask, :len(cs)]@cs
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cur_color, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')

plt.gca().set_aspect('equal', adjustable='box')

rates = np.array([[0.942, 0.058, 0.0],
         [0.037, 0.868, 0.095],
         [0.001, 0.334, 0.665]])/100
rates[np.arange(3), np.arange(3)] = 0
rates[np.arange(3), np.arange(3)] = 1-np.sum(rates, axis = 1)
Fs = np.ones(3)/3
for i in range(10000):
    Fs = Fs@rates

rates = np.array([[9.31751074e-01, 6.60561079e-02, 2.19281816e-03],
       [1.39247639e-02, 8.36354697e-01, 1.49720539e-01],
       [7.05118041e-04, 2.78923933e-01, 7.20370949e-01]])


get_model_params(model)
{'Model types': array(['Confined motion', 'Confined motion', 'Confined motion'],
       dtype='<U15'),
 'anomalous factors': <tf.Tensor: shape=(3,), dtype=float64, numpy=array([0.17849071, 0.2118589 , 0.78394504])>,
 'Localization errors': array([0.02687837, 0.0305368 , 0.00198493]),
 'd': array([0.01623294, 0.04844496, 0.13827311]),
 'transition rates': <tf.Tensor: shape=(3, 3), dtype=float64, numpy=
 array([[9.42048244e-01, 5.75822870e-02, 3.69469405e-04],
        [3.65364020e-02, 8.68326240e-01, 9.51373582e-02],
        [8.38214287e-04, 3.34254182e-01, 6.64907603e-01]])>,
 'transition shapes': <tf.Tensor: shape=(3, 3), dtype=float64, numpy=
 array([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])>,
 'Fractions': <tf.Tensor: shape=(4,), dtype=float64, numpy=array([2.17449388e-01, 5.23592802e-01, 2.58631754e-01, 3.26055630e-04])>}


[<tf.Variable 'initial_layer_constraints_2/recurrence_variables:0' shape=(3, 5) dtype=float64, numpy=
 array([[-3.61643328, -4.1207125 , -1.52660669, -5.51076385,  0.        ],
        [-3.48882278, -3.02732705, -1.31375664, -3.62708896,  0.        ],
        [-6.22217402, -1.97852451,  1.28880612, -9.75534503,  0.        ]])>,
 <tf.Variable 'initial_layer_constraints_2/initial_variables:0' shape=(3, 2) dtype=float64, numpy=
 array([[ 3.14814543, -2.99573227],
        [ 3.19385169, -2.99573227],
        [ 3.24500567, -2.99573227]])>,
 <tf.Variable 'initial_layer_constraints_2/Fractions:0' shape=(1, 4) dtype=float64, numpy=array([[ 1.10947949,  1.98822765,  1.28291862, -5.3931739 ]])>,
 <tf.Variable 'initial_layer_constraints_2/max linking distance:0' shape=() dtype=float64, numpy=1.0>,
 <tf.Variable 'custom_rnn_layer_2/Transition rates:0' shape=(3, 3) dtype=float64, numpy=
 array([[ 4.60246068,  1.8076192 , -3.24128315],
        [ 0.35553007,  3.52378849,  1.31254272],
        [-2.90627615,  3.08210707,  3.76985344]])>,
 <tf.Variable 'custom_rnn_layer_2/Transition shape:0' shape=(3, 3) dtype=float64, numpy=
 array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])>]


batch_size = 100
track_len = 30
tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths[6:], # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(5,track_len+1), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '.
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True)

# Remove tracks with mean quality lower than 200
new_tracks = []
new_frames = []
new_track_IDs = []
new_opt_metrics = {'QUALITY': []}
for i in range(len(tracks)):
    current_quality = opt_metrics['QUALITY'][i]
    if np.mean(current_quality)>200:
        new_tracks.append(tracks[i])
        new_frames.append(frames[i])
        new_track_IDs.append(track_IDs[i])
        new_opt_metrics['QUALITY'].append(current_quality)
tracks = new_tracks
frames = new_frames
track_IDs = new_track_IDs
opt_metrics = new_opt_metrics

print(len(tracks))

tracks = tracks[:len(tracks)//batch_size*batch_size]
tracks, _, masks = padding(tracks)
tracks = tf.constant(tracks[:,None, :, None, None, :nb_dims], dtype)

nb_states = 3
params = np.array([[np.log(0.03), np.log(0.01), np.log(0.1), np.log(0.001), 0],
                   [np.log(0.03), np.log(0.05), np.log(0.1), np.log(0.002), 0],
                   [np.log(0.03), np.log(0.15), np.log(0.05), np.log(0.005), 0]], dtype = dtype)

initial_params = np.array([[np.log(100), np.log(0.05)]]*nb_states, dtype = dtype)

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*4.5

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 6 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

vary_params = True
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False
vary_transition_rates = True

model, pred_model = build_model(track_len, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates)

model.weights

#all_masks = masks
learning_rate = 0.02
epochs = 50
64*60
decay_threshold = 2000
decay_rate = 0.001
np.exp(-20*64*0.001)
10000/50
device = '/GPU:0'
shuffle = True
verbose = 1

lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict((tracks, masks), batch_size = batch_size)
likelihood = MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit((tracks, masks), tracks, epochs = epochs, batch_size = batch_size, callbacks=[get_parameters()], shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])

preds = pred_model.predict((tracks, masks), batch_size = batch_size)
track_array = tracks[:, 0, :, 0, 0].numpy()

# Plot tracks
cs = np.array([ [1,0,0], [0, 1, 0], [0,0,1]])
# dir, fixed conf, dir + dif, mov conf, diff

plt.figure(figsize = (15, 15))
lim = 2.5 # MreB
#lim = 0.5
nb_rows = 6
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        mask = masks[offset+i*nb_rows+j].astype(bool)
        track = track_array[offset+i*nb_rows+j, mask]
        cur_color = preds[offset+i*nb_rows+j][mask, :len(cs)]@cs
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)

        plt.scatter(track[:,0], track[:,1] , c = cur_color, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')



'''
PBP2 number of states
'''

paths = glob(r'C:\Users\Franc\Data\RodComplex\PBP2/2020-12-18_PBP2-IPTG25-60ms/*.csv')

tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(5, 201), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '.
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True)

for i in range(len(tracks)):
    tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))
len(tracks)
plt.figure(figsize = (15, 15))
lim = 0.5 # MreB
nb_rows = 12
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        track = tracks[offset+i*nb_rows+j]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

len(tracks)

ls = [len(track) for track in tracks]
np.mean(ls)
np.max(ls)
np.mean(np.array(ls)==1)


plt.figure(figsize = (15, 15))
lim = 0.5 # MreB
nb_rows = 14
offset = 220*2
for i in range(nb_rows):
    for j in range(nb_rows):
        track = tracks[offset+i*nb_rows+j]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

# %%

'''
Determining the number of states and the best parameters
'''

# Prepare parameters for a maximum of 8 states
max_states = 10

# Initialize with generic guesses
params = np.array([[np.log(0.025), np.log(0.003), np.log(0.003), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.003), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.004), np.log(0.004), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.03), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.03), np.log(0.01), np.log(0.01), 1],
                   [np.log(0.025), np.log(0.05), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.05), np.log(0.02), np.log(0.01), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.03), np.log(0.02), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.15), np.log(0.1), np.log(0.01), 1]], dtype='float64')

# Prepare parameters for a maximum of 8 states
'''
max_states = 4

# Initialize with generic guesses
params = np.array([[np.log(0.025), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.0001), np.log(0.0001), np.log(0.00001), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.15), np.log(0.1), np.log(0.01), 1]], dtype='float64')
'''
initial_params = np.array([[np.log(1.0)]]*max_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*max_states+[-5.0]], dtype='float64')

# Transition matrices
transition_rates = 4 * np.eye(max_states, dtype='float64')
transition_shapes = np.zeros((max_states, max_states), dtype='float64')

# Create vary masks to fix certain parameters
# vary_params: which recurrent parameters to optimize
vary_params = True
# vary_transition_shapes: fix shapes to 1 (exponential)
vary_transition_shapes = False
# Allow other parameters to vary
vary_initial_params = True
vary_initial_fractions = True
vary_transition_rates = True

epochs = 60
batch_size = 400
epoch_decay = 50
learning_rate = 0.03
decay_rate = 0.005
nb_batches = len(tracks)//batch_size
device = '/GPU:0'
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches*2))

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1

results = exatrack.get_number_of_states(tracks,
                     params,
                     initial_params,
                     transition_shapes,
                     transition_rates,
                     initial_fractions,
                     nb_dims,
                     sequence_length,
                     max_linking_distance,
                     estimated_density,
                     epochs = epochs,
                     epoch_decay = epoch_decay,
                     learning_rate = learning_rate,
                     decay_rate = decay_rate,
                     batch_size = batch_size,
                     vary_params = vary_params,
                     vary_initial_params = vary_initial_params,
                     vary_initial_fractions = vary_initial_fractions,
                     vary_transition_shapes = vary_transition_shapes,
                     vary_transition_rates = vary_transition_rates,
                     device = device)

15*60*11/60/60

results.keys()
log_likelihoods = np.array([results[k]['log_likelihood'] for k in range(1, 11)])

plt.figure()
plt.plot(np.arange(1, 11), log_likelihoods)

save_model_selection_results(results, save_dir = r'C:\Users\Franc\Data\RodComplex\Results/PBP2_100_10')

loaded_results = load_model_selection_results(r'C:\Users\Franc\Data\RodComplex\Results/PBP2_100_10')
loaded_results[5]

results[5]
parameters = get_model_params(results[4]['model'])
results[10]['parameters']


'''
analysis for a fixed number of states
'''
exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2\*60ms*')
for exp in exps[:]:
    paths = glob(exp + '/*.csv')

    tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
                   lengths = np.arange(5, 301), # number of positions per track accepted (take the first position if longer than max
                   dist_th = np.inf, # maximum distance allowed for consecutive positions
                   frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                   fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '.
                   colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                   opt_colnames = ['QUALITY'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                   remove_no_disp = True)

    quality = opt_metrics['QUALITY']
    # Prepare parameters for a 4 states model
    nb_states = 3

    # Initialize with generic guesses
    params = np.array([[np.log(0.015), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                       #[np.log(0.015), np.log(0.001), np.log(0.002), np.log(0.0001), 1],
                       #[np.log(0.015), np.log(0.01), np.log(0.1), np.log(0.0001), 0],
                       [np.log(0.02), np.log(0.03), np.log(0.1), np.log(0.01), 0],
                       [np.log(0.02), np.log(0.08), np.log(0.1), np.log(0.01), 0]], dtype='float64')

    initial_params = np.array([[np.log(1.0)]]*nb_states, dtype='float64')

    # Equal initial fractions
    initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

    # Transition matrices
    transition_rates = 4 * np.eye(nb_states, dtype='float64')
    #transition_rates[0,0] = 5
    #transition_rates[1,1] = 3

    transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')
    tf.math.softmax(transition_rates, 1)

    # we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
    vary_params = np.ones(params.shape)
    #vary_params[:2, 0] = 0
    vary_initial_params = True
    vary_initial_fractions = True
    vary_transition_shapes = False

    # We prevent transitions between the two bound states to improve readability
    vary_transition_rates = np.ones(transition_rates.shape)
    #vary_transition_rates[:2, :2] = 0
    tf.math.softmax(transition_rates)
    batch_size = 400
    nb_batches = len(tracks)//batch_size
    device = '/GPU:0'

    estimated_density = 0.00001 # Negligible density
    nb_dims = 2
    sequence_length = 5
    max_linking_distance = 1
    segment_length = 10

    model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                    nb_states, # Number of states of their model
                    params, # recurrent parameters of the model
                    initial_params, # initial parameters of the model
                    transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                    transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                    initial_fractions,
                    batch_size, # number of tracks analysed at the same time
                    nb_dims = nb_dims, # Number of dimensions of the tracks
                    sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                    max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                    estimated_density = estimated_density, # Estimated density of the sample.
                    vary_params = vary_params,
                    vary_initial_params = vary_initial_params,
                    vary_initial_fractions = vary_initial_fractions,
                    vary_transition_shapes = vary_transition_shapes,
                    vary_transition_rates = vary_transition_rates)

    model.weights

    seq = exatrack.TrackSegmentSequence(tracks,
        batch_size=batch_size,
        segment_length=segment_length,
        min_segment_length=4,
        cutoff_batch_treshhold=0.5)

    nb_batches = len(seq)

    #all_masks = masks
    learning_rate = 0.01
    nb_batches
    epochs = 40
    epoch_decay = 32
    decay_threshold = epoch_decay*nb_batches
    decay_rate = 0.002
    np.exp(-20*64*0.001)
    10000/50
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

    model.save_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')

    weights = model.get_weights()
    model.load_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')

# Prepare parameters for a 3 state model
nb_states = 3

# Initialize with generic guesses
params = np.array([[np.log(0.015), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                   #[np.log(0.015), np.log(0.001), np.log(0.002), np.log(0.0001), 1],
                   #[np.log(0.015), np.log(0.01), np.log(0.1), np.log(0.0001), 0],
                   [np.log(0.02), np.log(0.03), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.02), np.log(0.08), np.log(0.1), np.log(0.01), 0]], dtype='float64')

initial_params = np.array([[np.log(1.0)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

# Transition matrices
transition_rates = 4 * np.eye(nb_states, dtype='float64')
#transition_rates[0,0] = 5
#transition_rates[1,1] = 3

transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')
tf.math.softmax(transition_rates, 1)

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
#vary_params[:2, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
batch_size = 400
nb_batches = len(tracks)//batch_size
device = '/GPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 10

model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates)


exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2\*PBP2-IPTG*60ms*')
all_params = []
for exp in exps[:]:
    model.load_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    params = get_model_params(model, track_segmentation = True)
    equ_fractions = exatrack.equilibrium_distribution(params['transition rates'])
    params['equilibrium fractions'] = equ_fractions
    all_params.append(params)
len(all_params)


exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2\*PBP2-A2250*60ms*')
all_params = []
for exp in exps[:]:
    model.load_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    params = get_model_params(model, track_segmentation = True)
    equ_fractions = exatrack.equilibrium_distribution(params['transition rates'])
    params['equilibrium fractions'] = equ_fractions
    all_params.append(params)

for params in all_params:
    print(params['equilibrium fractions'])

len(all_params)


    model.load_weigths(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')
08222973e
preds = model.predict(seq)

results[6]['parameters']

Epoch 24/60
150/150 [==============================] - ETA: 0s - loss: -55.3960{'Model types': array(['Directed motion', 'Confined motion', 'Confined motion',
       'Confined motion', 'Confined motion'], dtype='<U15'), 'anomalous factors': [0.0014, 0.2171, 0.0013, 0.3086, 0.0942], 'Localization errors': [0.013, 0.072, 0.02, 0.016, 0.019], 'd': [0.002, 0.0, 0.005, 0.031, 0.084], 'transition rates': [0.91, 0.001, 0.084, 0.004, 0.001, 0.002, 0.183, 0.468, 0.34, 0.008, 0.023, 0.023, 0.937, 0.016, 0.002, 0.005, 0.054, 0.02, 0.822, 0.1, 0.002, 0.0, 0.002, 0.116, 0.88], 'transition shapes': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'Fractions': [0.03, 0.014, 0.307, 0.193, 0.457, 0.0]}



exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2_2022\All_60ms\C*')
exp = exps[12]
for exp in exps[:]:
    paths = glob(exp + '/*.csv')

    tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
                   lengths = np.arange(5, 301), # number of positions per track accepted (take the first position if longer than max
                   dist_th = np.inf, # maximum distance allowed for consecutive positions
                   frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                   fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '.
                   colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                   opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                   remove_no_disp = True)

    frames[1][1:] -frames[1][:-1]
    for i in range(len(tracks)):
        tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))

    # Prepare parameters for a 4 states model
    nb_states = 3

    # Initialize with generic guesses
    params = np.array([[np.log(0.02), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                       #[np.log(0.02), np.log(0.001), np.log(0.1), np.log(0.0001), 0],
                       #[np.log(0.015), np.log(0.01), np.log(0.1), np.log(0.0001), 0],
                       [np.log(0.02), np.log(0.03), np.log(0.1), np.log(0.01), 0],
                       [np.log(0.02), np.log(0.08), np.log(0.1), np.log(0.01), 0]], dtype='float64')
    '''
    # directed state 1:
    array(['Directed motion', 'Directed motion', 'Confined motion',
           'Confined motion'], dtype='<U15'), 'anomalous factors': [0.0013, 0.0002, 0.1595, 0.282], 'Localization errors': [0.02, 0.02, 0.02, 0.02], 'd': [0.0, 0.031, 0.06, 0.101], 'anomalous variation': [1e-05, 1e-05, 0.00091, 0.00039], 'transition rates': [[0.959, 0.022, 0.007, 0.012], [0.032, 0.836, 0.129, 0.003], [0.003, 0.043, 0.831, 0.123], [0.009, 0.009, 0.324, 0.658]], 'transition shapes': [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 'Fractions': [0.122, 0.086, 0.5, 0.292, 0.0]}
    111/111 [==============================] - 40s 358ms/step - loss: -29.4061
    # Confined state 1:
    array(['Directed motion', 'Confined motion', 'Confined motion',
           'Confined motion'], dtype='<U15'), 'anomalous factors': [0.0011, 0.3405, 0.1222, 0.2898], 'Localization errors': [0.02, 0.02, 0.02, 0.02], 'd': [0.0, 0.031, 0.06, 0.101], 'anomalous variation': [1e-05, 0.00259, 0.00184, 0.00053], 'transition rates': [[0.956, 0.025, 0.005, 0.013], [0.034, 0.861, 0.098, 0.006], [0.003, 0.039, 0.837, 0.121], [0.01, 0.012, 0.31, 0.668]], 'transition shapes': [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 'Fractions': [0.119, 0.095, 0.492, 0.294, 0.0]}
    111/111 [==============================] - 38s 346ms/step - loss: -29.4276
    '''
    initial_params = np.array([[np.log(1.0)]]*nb_states, dtype='float64')

    # Equal initial fractions
    initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

    # Transition matrices

    # Transition matrices
    transition_rates = 4 * np.eye(nb_states, dtype='float64')
    transition_rates[0,0] = 5
    transition_rates[1,1] = 3
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
    batch_size = 400
    nb_batches = len(tracks)//batch_size
    device = '/GPU:0'

    estimated_density = 0.00001 # Negligible density
    nb_dims = 2
    sequence_length = 10
    max_linking_distance = 1
    segment_length = 10

    model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                    nb_states, # Number of states of their model
                    params, # recurrent parameters of the model
                    initial_params, # initial parameters of the model
                    transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                    transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                    initial_fractions,
                    batch_size, # number of tracks analysed at the same time
                    nb_dims = nb_dims, # Number of dimensions of the tracks
                    sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                    max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                    estimated_density = estimated_density, # Estimated density of the sample.
                    vary_params = vary_params,
                    vary_initial_params = vary_initial_params,
                    vary_initial_fractions = vary_initial_fractions,
                    vary_transition_shapes = vary_transition_shapes,
                    vary_transition_rates = vary_transition_rates)

    seq = exatrack.TrackSegmentSequence(tracks,
        batch_size=batch_size,
        segment_length=segment_length,
        min_segment_length=4,
        cutoff_batch_treshhold=0.5)

    nb_batches = len(seq)

    #all_masks = masks
    learning_rate = 0.01
    nb_batches
    epochs = 70
    epoch_decay = 60
    decay_threshold = epoch_decay*nb_batches
    decay_rate = 0.005
    np.exp(-20*64*0.001)

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

    model.save_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_2022_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')

    weights = model.get_weights()
    model.load_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')

    track_array, _, masks = exatrack.padding(tracks, batch_size = batch_size)
    track_array = tf.constant(track_array[:,None, :, None, None, :nb_dims], dtype = 'float64')

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

    preds = pred_model.predict((track_array, masks), batch_size = batch_size)

    colors = np.array([[1,0,0],
                       [1,1,0],
                       [0,1,1],
                       [0,0,1]])
    plt.figure(figsize = (15, 15))
    lim = 0.5 # MreB
    nb_rows = 15
    offset = 0
    for i in range(nb_rows):
        for j in range(nb_rows):
            mask = masks[offset+i*nb_rows+j]
            track = tracks[offset+i*nb_rows+j]
            print(len(track))
            track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
            p = preds[offset+i*nb_rows+j, mask.astype(bool)][:,:-1]
            plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
            plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
            plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
    plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize = (15, 15))
lim = 0.5 # MreB
nb_rows = 12
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        track = tracks[offset+i*nb_rows+j]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')


all_params = []
exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2_2022\All_60ms\C*')
exp = exps[1]
for exp in exps[:]:
    model.load_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_2022_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    params = exatrack.get_model_params(model, track_segmentation = True)
    equ_fractions = exatrack.equilibrium_distribution(params['transition rates'])
    params['equilibrium fractions'] = equ_fractions
    all_params.append(params)


for params, exp in zip(all_params, exps):
    print(exp.rsplit('\\')[-1])
    print(params['equilibrium fractions'])

for params, exp in zip(all_params, exps):
    print(exp.rsplit('\\')[-1])
    print(params['anomalous factors'], params['d'], params['q'])


for params, exp in zip(all_params, exps):
    print(exp.rsplit('\\')[-1])
    print(params['transition rates'])


# Prepare parameters for a 3 state model
nb_states = 3


# Initialize with generic guesses
params = np.array([[np.log(0.015), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                   #[np.log(0.015), np.log(0.001), np.log(0.002), np.log(0.0001), 1],
                   #[np.log(0.015), np.log(0.01), np.log(0.1), np.log(0.0001), 0],
                   [np.log(0.02), np.log(0.03), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.02), np.log(0.08), np.log(0.1), np.log(0.01), 0]], dtype='float64')

initial_params = np.array([[np.log(1.0)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

# Transition matrices
transition_rates = 4 * np.eye(nb_states, dtype='float64')
transition_rates[0,0] = 5
transition_rates[1,1] = 3
transition_rates[1,0] = -1
transition_rates[0,1] = -3

transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')
tf.math.softmax(transition_rates, 1)

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
#vary_params[:2, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
batch_size = 400
nb_batches = len(tracks)//batch_size
device = '/GPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 10

model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions,
                batch_size, # number of tracks analysed at the same time
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates)


exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2\*PBP2-IPTG*60ms*')
all_params = []
for exp in exps[:]:
    model.load_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    params = get_model_params(model, track_segmentation = True)
    equ_fractions = exatrack.equilibrium_distribution(params['transition rates'])
    params['equilibrium fractions'] = equ_fractions
    all_params.append(params)
len(all_params)


exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2\*PBP2-A2250*60ms*')
all_params = []
for exp in exps[:]:
    model.load_weights(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    params = get_model_params(model, track_segmentation = True)
    equ_fractions = exatrack.equilibrium_distribution(params['transition rates'])
    params['equilibrium fractions'] = equ_fractions
    all_params.append(params)

for params in all_params:
    print(params['equilibrium fractions'])

len(all_params)


    model.load_weigths(r'C:\Users\Franc\Data\RodComplex\Results\PBP2_3states/' + exp.rsplit('\\')[-1]  + '_weights.tf')
08222973e
preds = model.predict(seq)

results[6]['parameters']

array(['Directed motion', 'Directed motion', 'Confined motion',
       'Confined motion'], dtype='<U15'), 'anomalous factors': [0.0013, 0.0004, 0.1583, 0.2765], 'Localization errors': [0.02, 0.02, 0.02, 0.02], 'd': [0.0, 0.032, 0.06, 0.098], 'anomalous variation': [1e-05, 2e-05, 0.00161, 0.00065], 'transition rates': [[0.959, 0.021, 0.009, 0.012], [0.03, 0.844, 0.12, 0.006], [0.004, 0.039, 0.842, 0.116], [0.009, 0.011, 0.329, 0.651]], 'transition shapes': [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 'Fractions': [0.119, 0.084, 0.512, 0.285, 0.0]}
111/111 [==============================] - 36s 324ms/

exps = glob(r'C:\Users\Franc\Data\RodComplex\PBP2_2022\All_60ms\C5*')
exp = exps[1]
paths = glob(exp + '/*.csv')

tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(5, 301), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '.
               colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True)

frames[1][1:] -frames[1][:-1]
for i in range(len(tracks)):
    tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))

# Prepare parameters for a maximum of 10 states
max_states = 10

# Initialize with generic guesses
params = np.array([[np.log(0.025), np.log(0.003), np.log(0.003), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.003), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.004), np.log(0.004), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.03), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.03), np.log(0.01), np.log(0.01), 1],
                   [np.log(0.025), np.log(0.05), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.05), np.log(0.02), np.log(0.01), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.03), np.log(0.02), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.15), np.log(0.1), np.log(0.01), 1]], dtype='float64')

# Prepare parameters for a maximum of 8 states
'''
max_states = 4

# Initialize with generic guesses
params = np.array([[np.log(0.025), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
                   [np.log(0.025), np.log(0.0001), np.log(0.0001), np.log(0.00001), 1],
                   [np.log(0.025), np.log(0.1), np.log(0.1), np.log(0.01), 0],
                   [np.log(0.025), np.log(0.15), np.log(0.1), np.log(0.01), 1]], dtype='float64')
'''
initial_params = np.array([[np.log(1.0)]]*max_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*max_states+[-5.0]], dtype='float64')

# Transition matrices
transition_rates = 4 * np.eye(max_states, dtype='float64')
transition_shapes = np.zeros((max_states, max_states), dtype='float64')

# Create vary masks to fix certain parameters
# vary_params: which recurrent parameters to optimize
vary_params = True
# vary_transition_shapes: fix shapes to 1 (exponential)
vary_transition_shapes = False
# Allow other parameters to vary
vary_initial_params = True
vary_initial_fractions = True
vary_transition_rates = True

epochs = 60
batch_size = 400
epoch_decay = 50
learning_rate = 0.03
decay_rate = 0.005
nb_batches = len(tracks)//batch_size
device = '/GPU:0'
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches*2))

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1

results = exatrack.get_number_of_states(tracks,
                     params,
                     initial_params,
                     transition_shapes,
                     transition_rates,
                     initial_fractions,
                     nb_dims,
                     sequence_length,
                     max_linking_distance,
                     estimated_density,
                     epochs = epochs,
                     epoch_decay = epoch_decay,
                     learning_rate = learning_rate,
                     decay_rate = decay_rate,
                     batch_size = batch_size,
                     vary_params = vary_params,
                     vary_initial_params = vary_initial_params,
                     vary_initial_fractions = vary_initial_fractions,
                     vary_transition_shapes = vary_transition_shapes,
                     vary_transition_rates = vary_transition_rates,
                     device = device)

15*60*11/60/60

results.keys()
log_likelihoods = np.array([results[k]['log_likelihood'] for k in range(1, 11)])

plt.figure()
plt.plot(np.arange(1, 11), log_likelihoods)

save_model_selection_results(results, save_dir = r'C:\Users\Franc\Data\RodComplex\Results/PBP2_100_10')
