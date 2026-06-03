# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:52:48 2026

@author: Franc
"""


# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import os
# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except:
    # add the absolute path if you are running the script line by line
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)
import exatrack
from glob import glob
import random


paths = glob(r'C:\Users\Franc\Data\Kinesin_minflux\Tracks/*Minflux_642_L75_pho100_lp15_BGdis40k_hex3_dt300us_cfr09_dp1*')
reference_dt = 0.0013841750000054276

track_list, frame_list, track_IDs, opt_metrics = exatrack.read_table(paths[:], # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(100, 11000), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions 
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
               colnames = ['x_um', 'y_um', 'time', 'track_id'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = ['photons', 'sigma_x_um', 'sigma_y_um'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = False)

for i in range(len(track_list)):
    track_list[i] = track_list[i] - track_list[i][:1] + np.random.normal(0,1,(1,2))

dt_list = [np.concatenate(((frame[1:] - frame[:-1]), [reference_dt])) for frame in frame_list]
LocErr_list = opt_metrics['photons']


ls = [len(track) for track in track_list]
np.mean(ls)
np.max(ls)

plt.figure(figsize = (15, 15))
lim = 0.3 # MreB
nb_rows = 16
offset = 3*16**2
for i in range(nb_rows):
    for j in range(nb_rows):
        ID=offset+i*nb_rows+j
        track = track_list[ID]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 5)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
        plt.text(track[0,0],track[0,1], str(ID), fontsize = 10)
plt.gca().set_aspect('equal', adjustable='box')

# tracks identified as stepping directed from paths = glob(r'C:\Users\Franc\Data\Kinesin_minflux\Tracks/*Minflux_642_L75_pho100_lp15_BGdis40k_hex3_dt300us_cfr09_dp1*')
IDs = [63, 94, 124, 156, 155, 171, 159, 161, 233, 347, 310, 273, 466, 467, 464]
IDs = [ 466, 310, 161]
IDs = [ 466, 310]
IDs = [ 466]


track_list = [track_list[i] for i in IDs]
LocErr_list = [LocErr_list[i] for i in IDs]
dt_list = [dt_list[i] for i in IDs]
len(track_list)

track_list[0] = track_list[0][40:]
LocErr_list[0] = LocErr_list[0][40:]
dt_list[0] = dt_list[0][40:]

track_list[1] = track_list[1][160:-220]
LocErr_list[1] = LocErr_list[1][160:-220]
dt_list[1] = dt_list[1][160:-220]    

LocErr_list = None
LocErr_type = 'Linear'

# %%

'''
2 state model analysis
'''

segment_length = 30
batch_size = 2

seq = exatrack.TrackSegmentSequence(track_list, 
                                    LocErr_list,
                                    dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

dtype = 'float64'
device = '/CPU:0'
estimated_density = 0.01 # Negligible density
nb_dims = 2
sequence_length = 20
max_linking_distance = 1

# We need to reshape the track to the shape used by the model, as following

nb_states = 2
params = np.array([[np.log(0.02), np.log(0.0005), np.log(0.0001), np.log(0.000005), 1],
                   [np.log(0.02), np.log(0.01), np.log(0.01), np.log(0.0001), 1]], dtype = dtype)

initial_params = np.array([[np.log(1)]]*nb_states, dtype = dtype)

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_rates = np.eye(nb_states, dtype = dtype)*2


initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 10 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

vary_params = np.ones(params.shape)
#vary_params[2] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = np.ones(transition_shapes.shape)
vary_transition_shapes[1, 0] = 0
vary_transition_rates = True

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
                nb_LocErr_dims = 0,
                LocErr_type = LocErr_type)
    
exatrack.get_model_params(model, track_segmentation = True)['transition rates']

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.001
epochs = 500
decay_threshold = int(0.7*nb_batches*epochs)
decay_rate = - np.log(0.001)/(0.3*nb_batches*epochs) # 
device = '/CPU:0'
verbose = 1

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

-2099.9668{'Model types': array(['Directed motion', 'Directed motion'], dtype='<U15'), 'anomalous factors': [0.0, 0.0001], 'Localization errors': [0.029, 0.019], 'd': [0.001, 0.004], 'anomalous variation': [0.0, 0.00063], 'transition rates': [[0.795, 0.075], [0.941, 0.75]], 'transition shapes': [[1.0, 0.369], [3.756, 1.0]], 'Fractions': [0.834, 0.109, 0.057]}


weights = model.get_weights()
max_track_len = np.max([len(track) for track in track_list])

batch_size = 1

seq_full = exatrack.TrackSegmentSequence(track_list, 
                                    LocErr_list,
                                    dt_list,
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
                nb_LocErr_dims = 0,
                LocErr_type = LocErr_type)

exatrack.get_model_params(model, track_segmentation=True)

LPs, preds, All_coefs, All_biases, All_LPs = pred_model.predict(seq_full )

'''
Plot tracks with state predictions
'''
tracks = np.concatenate([seq_full[i][0][0] for i in range(len(seq_full))], 0)
LocErrs = np.concatenate([seq_full[i][0][1] for i in range(len(seq_full))], 0)
time_steps = np.concatenate([seq_full[i][0][2] for i in range(len(seq_full))], 0)
masks = np.concatenate([seq_full[i][0][3] for i in range(len(seq_full))], 0)

colors = np.array([[1,0,0],
                   [0,0,1]])

plt.figure(figsize = (20, 20))
lim = 0.05 # MreB
nb_rows = 2
IDs = random.sample(list(np.arange(len(tracks))), min(nb_rows**2, len(tracks)))
for i in range(nb_rows):
    for j in range(nb_rows):
        if i*nb_rows+j<len(tracks):
            ID = IDs[i*nb_rows+j]
            mask = masks[ID]
            track = tracks[ID]
            print(len(track))
            track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
            p = preds[ID, mask.astype(bool)][:,:-1]
            plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
            plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 6)
            plt.scatter(track[0,0], track[0,1] , c = 'k', s = 6, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

for ID in range(len(tracks)):
    
    plt.figure(figsize = (4, 3.5))
    plt.title('Track %s'%ID)
    mask = masks[ID]
    times = time_steps[ID, :-1]
    times = np.cumsum(times[mask.astype(bool)])
    track = tracks[ID, mask.astype(bool)]
    p = preds[ID, mask.astype(bool)][:,:-1]
    rel_track = track - 50*track[:1] + 50*track[-1:]
    dists = np.sqrt(np.sum(rel_track**2, 1))
    
    p = preds[ID, mask.astype(bool)][:,:-1]
    plt.plot(times, dists, ':k', alpha = 0.5)
    plt.scatter(times, dists, c = p@colors, s = 6)


    
colors = np.array([[1,0,0],
                   [0,0,1]])





'''
Quantitative analysis on 15 tracks
'''

paths = glob(r'C:\Users\Franc\Data\Kinesin_minflux\Tracks/*Minflux_642_L75_pho100_lp15_BGdis40k_hex3_dt300us_cfr09_dp1*')
reference_dt = 0.0013841750000054276

track_list0, frame_list0, track_IDs, opt_metrics = exatrack.read_table(paths[:], # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(100, 11000), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions 
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
               colnames = ['x_um', 'y_um', 'time', 'track_id'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = ['photons', 'sigma_x_um', 'sigma_y_um'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = False)

for i in range(len(track_list0)):
    track_list0[i] = track_list0[i] - track_list0[i][:1] + np.random.normal(0,1,(1,2))

dt_list0 = [np.concatenate(((frame[1:] - frame[:-1]), [reference_dt])) for frame in frame_list0]
LocErr_list = None

plt.figure(figsize = (15, 15))
lim = 0.3 # MreB
nb_rows = 16
offset = 15*16**2
for i in range(nb_rows):
    for j in range(nb_rows):
        ID=offset+i*nb_rows+j
        track = track_list0[ID]
        print(len(track))
        track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 5)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
        plt.text(track[0,0],track[0,1], str(ID), fontsize = 10)
plt.gca().set_aspect('equal', adjustable='box')

# tracks identified as stepping directed from paths = glob(r'C:\Users\Franc\Data\Kinesin_minflux\Tracks/*Minflux_642_L75_pho100_lp15_BGdis40k_hex3_dt300us_cfr09_dp1*')
# Tracks with stop and go motion that is difficult to clearly identify
IDs = [ 466, 310, 922, 161, 156, 155, 171, 124, 94, 233, 273, 467, 639, 681, 628, 795, 959, 926, 1003, 937, 953, 984, 965]
IDs = [ 466, 310, 922, 161                        , 233, 273, 467, 639, 681          , 959, 926, 1003, 937    ]

# Tracks with clear stop and go motion
IDs = [959, 960, 1028, 1042, 1289, 1668, 1853, 1824, 2068, 2498, 2783, 3185]
1504
3628 # Very straight track where we cant see stop and go motion
928
track_list = [track_list0[i] for i in IDs]
dt_list = [dt_list0[i] for i in IDs]
len(track_list)

track_list[0] = track_list[0][40:]
dt_list[0] = dt_list[0][40:]

track_list[1] = track_list[1][160:-220]
dt_list[1] = dt_list[1][160:-220]

track_list[2] = track_list[2][25:]
dt_list[2] = dt_list[2][25:]

LocErr_list = None
LocErr_type = 'Linear'
# %%

'''
2 state model analysis
'''segment_length
import gc

segment_length = 40
batch_size = 11

seq = exatrack.TrackSegmentSequence(track_list, 
                                    None,
                                    dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.01)

nb_batches = len(seq)

dtype = 'float64'
device = '/CPU:0'
estimated_density = 0.01 # Negligible density
nb_dims = 2
sequence_length = 20
max_linking_distance = 1

# We need to reshape the track to the shape used by the model, as following

nb_states = 3
params = np.array([[np.log(0.002), np.log(0.001), np.log(0.0001), np.log(0.000005), 1],
                   [np.log(0.002), np.log(0.005), np.log(0.01), np.log(0.0001), 1],
                   [np.log(0.002), np.log(0.016), np.log(0.01), np.log(0.001), 1]], dtype = dtype)

initial_params = np.array([[np.log(1)]]*nb_states, dtype = dtype)

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_shapes[0,1]  = np.log(1)
transition_rates = np.eye(nb_states, dtype = dtype)*2
transition_rates[0,2] = -10
tf.math.softmax(transition_rates, axis = 1)

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 10 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

vary_params = np.ones(params.shape)
vary_params[:, 1] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = np.ones(transition_shapes.shape)
vary_transition_shapes[1, 0] = 0
vary_transition_shapes[0, 2] = 0
#vary_transition_shapes[0, 1] = 0

vary_transition_rates = np.ones(transition_rates.shape)
vary_transition_rates[0,2] = 0

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
                nb_LocErr_dims = 0,
                LocErr_type = LocErr_type)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.05
epochs = 50
decay_fraction = 0.1
decay_threshold = int(decay_fraction*nb_batches*epochs)
decay_rate = - np.log(0.001)/((1-decay_fraction)*nb_batches*epochs) # 
device = '/CPU:0'
verbose = 1

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])


model_params = []

# model results using all the tracks
#with params = np.array([[np.log(0.002), np.log(0.0005), np.log(0.0001), np.log(0.000005), 1],
#                   [np.log(0.002), np.log(0.01), np.log(0.01), np.log(0.0001), 1],
#                   [np.log(0.002), np.log(0.02), np.log(0.01), np.log(0.001), 1]], dtype = dtype)
# batch_size = 12

'''
loss: -1240.9805{'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
      dtype='<U15'), 'anomalous factors': [0.0001, 0.0002, 0.0015], 'Localization errors': [0.002, 0.002, 0.001], 'd': [0.001, 0.005, 0.016], 'anomalous variation': [0.0, 0.00041, 0.00014], 'transition rates': [[0.906, 0.095, 0.0], [0.313, 0.671, 0.026], [0.031, 0.234, 0.695]], 'transition shapes': [[1.0, 1.006, 1.0], [1.0, 1.0, 1.621], [0.706, 0.897, 1.0]], 'Fractions': [0.845, 0.025, 0.116, 0.014]}
{'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
       dtype='<U15'),
 'anomalous factors': array([8.06786561e-05, 1.98205384e-04, 1.35142412e-03]),
 'Localization errors': array([0.00233132, 0.00210043, 0.0014445 ]),
 'd': array([0.00118656, 0.00461647, 0.01567   ]),
 'q': array([3.63241095e-07, 4.21684239e-04, 1.97987957e-04]),
 'transition rates': array([[9.03063303e-01, 9.15386109e-02, 2.62891301e-06],
        [3.14236762e-01, 6.70404237e-01, 2.39302988e-02],
        [3.52836042e-02, 2.50289995e-01, 6.94732615e-01]]),
 'transition shapes': array([[1.        , 0.94433598, 1.        ],
        [1.        , 1.        , 1.55803305],
        [0.73309758, 0.97336759, 1.        ]]),
 'Fractions': array([0.018146  , 0.84562157, 0.12227423, 0.0139582 ])}
''' 
for i in range(len(track_list)):
    
    cur_track_list = [track_list[i]]
    cur_dt_list = [np.concatenate((dt_list[i], [reference_dt]))]

    seq = exatrack.TrackSegmentSequence(cur_track_list, 
                                        None,
                                        cur_dt_list,
                                        batch_size=batch_size,
                                        segment_length=segment_length,
                                        min_segment_length=4,
                                        cutoff_batch_treshhold=0.01)
    
    nb_batches = len(seq)
    
    dtype = 'float64'
    device = '/CPU:0'
    estimated_density = 0.01 # Negligible density
    nb_dims = 2
    sequence_length = 20
    max_linking_distance = 1
    
    # We need to reshape the track to the shape used by the model, as following
    
    nb_states = 3
    params = np.array([[np.log(0.002), np.log(0.001), np.log(0.0001), np.log(0.000005), 1],
                       [np.log(0.002), np.log(0.005), np.log(0.01), np.log(0.0001), 1],
                       [np.log(0.002), np.log(0.016), np.log(0.01), np.log(0.001), 1]], dtype = dtype)
    
    initial_params = np.array([[np.log(1)]]*nb_states, dtype = dtype)
    
    transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
    transition_shapes[0,1]  = np.log(1)
    transition_rates = np.eye(nb_states, dtype = dtype)*2
    transition_rates[0,2] = -10
    tf.math.softmax(transition_rates, axis = 1)
    
    initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
    initial_fractions[0,-1] = -1
    sequence_length = 10 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
    max_linking_distance = 1 # maximum linking distance used for the linking algorithm
    estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)
    
    vary_params = np.ones(params.shape)
    vary_params[:, 1] = 0
    vary_initial_params = True
    vary_initial_fractions = True
    vary_transition_shapes = np.ones(transition_shapes.shape)
    vary_transition_shapes[1, 0] = 0
    vary_transition_shapes[0, 2] = 0
    #vary_transition_shapes[0, 1] = 0

    vary_transition_rates = np.ones(transition_rates.shape)
    vary_transition_rates[0,2] = 0
    
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
                    nb_LocErr_dims = 0,
                    LocErr_type = LocErr_type)
    
    nb_batches = len(seq)
    
    #all_masks = masks
    learning_rate = 0.05
    epochs = 50
    decay_fraction = 0.1
    decay_threshold = int(decay_fraction*nb_batches*epochs)
    decay_rate = - np.log(0.001)/((1-decay_fraction)*nb_batches*epochs) # 
    device = '/CPU:0'
    verbose = 1
    
    lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)
    
    preds = model.predict(seq)
    likelihood = exatrack.MLE_loss(preds, preds).numpy()
    print(likelihood)
    
    with tf.device(device):
        history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    
    model_params.append(exatrack.get_model_params(model, track_segmentation = True))
    
    track_length = len(cur_track_list[0])

    seq_long = exatrack.TrackSegmentSequence(cur_track_list, 
                                        None,
                                        cur_dt_list,
                                        batch_size=1,
                                        segment_length=track_length,
                                        min_segment_length=4,
                                        cutoff_batch_treshhold=0.01)
    
    weights = model.get_weights()
    
    del model, pred_model
    
    model, pred_model = exatrack.build_segment_model(track_length, # maximum number of time points in the input tracks
                    nb_states, # Number of states of their model
                    weights[0], # recurrent parameters of the model
                    weights[1], # initial parameters of the model
                    weights[7], # transition rates for each pair of states (gamma distributed transition lifetimes)
                    weights[8], # transition shapes for each pair of states (gamma distributed transition lifetimes)
                    weights[2],
                    batch_size=1, # number of tracks analysed at the same time
                    reference_dt= reference_dt,
                    nb_dims = nb_dims, # Number of dimensions of the tracks
                    sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                    max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                    estimated_density = estimated_density, # Estimated density of the sample.
                    vary_params = vary_params,
                    vary_initial_params = vary_initial_params,
                    vary_initial_fractions = vary_initial_fractions,
                    vary_transition_shapes = vary_transition_shapes,
                    vary_transition_rates = vary_transition_rates,
                    nb_LocErr_dims = 0,
                    LocErr_type = LocErr_type)
    
    LPs, preds, All_coefs, All_biases, All_LPs = pred_model.predict(seq_long)
    seq_long[0]
    colors = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])
    
    del model, pred_model

    plt.figure(figsize = (15, 15))
    shape = np.round(model_params[i]['transition shapes'][0, 1], 2)
    rate = np.round(model_params[i]['transition rates'][0, 1], 2)
    track = track_list[i]
    p = preds[0, :,:-1]
    track_len = len(track)
    plt.title('Track %s, shape: %s, \n rate: %s, nb time points: %s, '%(i, shape, rate, track_len))
    print(len(track))
    track = track - np.mean(track, 0, keepdims = True)
    plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
    plt.scatter(track[:,0], track[:,1] , c = p @ colors, s = 5)
    plt.xlim([-0.4, 0.4])
    plt.ylim([-0.4, 0.4])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_2D\track_%s.svg'%i)
    plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_2D\track_%s.png'%i)
    plt.close()
    
    plt.figure(figsize = (15, 10))
    plt.title('Track %s, shape: %s, \n rate: %s, nb time points: %s, '%(i, shape, rate, track_len))
    times = cur_dt_list[0][:-1]
    times = np.cumsum(times)
    track = track_list[i]
    p = preds[0, :, :-1]
    rel_track = track - 50*track[:1] + 50*track[-1:]
    dists = np.sqrt(np.sum(rel_track**2, 1))
    dists = dists - dists[0]
    plt.plot(times, dists, ':k', alpha = 0.5)
    plt.scatter(times, dists, c = p@colors, s = 6)
    plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_1D\track_%s.svg'%i)
    plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_1D\track_%s.png'%i)
    plt.close()

    tf.keras.backend.clear_session()
    gc.collect()

-2700.4587


lim = 0.3 # MreB
for i in range(len(track_list)):
    plt.figure(figsize = (5, 5))
    shape = np.round(model_params[i]['transition shapes'][0, 1], 2)
    rate = np.round(model_params[i]['transition rates'][0, 1], 2)
    track = track_list[i]
    track_len = len(track)
    plt.title('Track %s, shape: %s, \n rate: %s, nb time points: %s, '%(i, shape, rate, track_len))
    print(len(track))
    track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
    plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
    plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 5)
    plt.gca().set_aspect('equal', adjustable='box')



shape_params = []
for params in model_params:
    shape_params.append(params['transition shapes'][0,1])
shape_params = np.array(shape_params)

rate_params = []
for params in model_params:
    rate_params.append(params['transition rates'][0,1])
rate_params = np.array(rate_params)

shape_params = np.array([ 4.39028161,  1.89872803,  3.50890269,  1.39838934,  2.23814844,
        7.10220824,  1.54027764,  2.63609963, 10.96784763,  8.91213365,
        2.05784582,  4.37064062,  9.91915001])

rate_params = np.array([0.54677959, 0.15868275, 0.62218063, 0.20421315, 0.19512842,
       0.52280338, 0.07585416, 0.21598536, 0.77389182, 0.42485011,
       0.12592246, 0.33879949, 0.65558264])

shape_params**0.5/rate_params
shape_params/rate_params







segment_length = 60

i = 9

cur_track_list = [track_list[i]]
cur_dt_list = [np.concatenate((dt_list[i], [reference_dt]))]

seq = exatrack.TrackSegmentSequence(cur_track_list, 
                                    None,
                                    cur_dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.01)

nb_batches = len(seq)

dtype = 'float64'
device = '/CPU:0'
estimated_density = 0.01 # Negligible density
nb_dims = 2
sequence_length = 20
max_linking_distance = 1

# We need to reshape the track to the shape used by the model, as following

nb_states = 2
params = np.array([[np.log(0.002), np.log(0.001), np.log(0.0001), np.log(0.000005), 1],
                   [np.log(0.002), np.log(0.005), np.log(0.01), np.log(0.0001), 1]], dtype = dtype)

initial_params = np.array([[np.log(1)]]*nb_states, dtype = dtype)

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_shapes[0,1]  = np.log(1)
transition_rates = np.eye(nb_states, dtype = dtype)*2
tf.math.softmax(transition_rates, axis = 1)

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 10 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

vary_params = np.ones(params.shape)
vary_params[:, 1] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = np.ones(transition_shapes.shape)
vary_transition_shapes[1, 0] = 0
vary_transition_shapes[0, 1] = 0

vary_transition_rates = np.ones(transition_rates.shape)

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
                nb_LocErr_dims = 0,
                LocErr_type = LocErr_type)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.02
epochs = 150
decay_fraction = 0.5
decay_threshold = int(decay_fraction*nb_batches*epochs)
decay_rate = - np.log(0.001)/((1-decay_fraction)*nb_batches*epochs) # 
device = '/CPU:0'
verbose = 1

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])



weights = model.get_weights()
        
vary_transition_shapes = np.ones(transition_shapes.shape)
vary_transition_shapes[1, 0] = 0

model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                weights[0], # recurrent parameters of the model
                weights[1], # initial parameters of the model
                weights[7], # transition rates for each pair of states (gamma distributed transition lifetimes)
                weights[8], # transition shapes for each pair of states (gamma distributed transition lifetimes)
                weights[2],
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
                nb_LocErr_dims = 0,
                LocErr_type = LocErr_type)


lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history2 = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

np.exp(2843.287-2850.5437)

-3092.1211
np.exp(1673.6437-1677.9824)

-2701.3274











nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.001
epochs = 500
decay_threshold = int(0.7*nb_batches*epochs)
decay_rate = - np.log(0.001)/(0.3*nb_batches*epochs) # 
device = '/CPU:0'
verbose = 1

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])




# Then we change version to infer the real positions and the velocity vectors (we could have used this new version from the beginning)

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

pos_mean, anomalous_mean, position_std, anomalous_std = exatrack.extract_hidden_variables(All_coefs, All_biases, All_LPs, nb_dims = 2, sequence_length = sequence_length)


colors = np.array([[1,0,0],
                   [0,0,1]])

for ID in range(len(tracks)):
    
    plt.figure(figsize = (8, 3.5))
    plt.subplot(1, 2, 1)
    plt.title('Track %s'%ID)
    mask = masks[ID]
    times = time_steps[ID, :-1]
    times = np.cumsum(times[mask.astype(bool)])
    track = tracks[ID, mask.astype(bool)]
    p = preds[ID, mask.astype(bool)][:,:-1]
    plt.plot(times, track[:,0], ':k', alpha = 0.5)
    plt.scatter(times, track[:,0], c = p@colors, s = 6)
    plt.xlabel('Time in s')
    plt.ylabel('x position')
    
    plt.subplot(1, 2, 2)
    mask = masks[ID]
    times = time_steps[ID, :-1]
    times = np.cumsum(times[mask.astype(bool)])
    track = tracks[ID, mask.astype(bool)]
    p = preds[ID, mask.astype(bool)][:,:-1]
    plt.plot(times, track[:,1], ':k', alpha = 0.5)
    plt.scatter(times, track[:,1], c = p@colors, s = 6)
    plt.xlabel('Time in s')
    plt.ylabel('y position')
    
    plt.tight_layout()
    
    track_id,time,x_um,y_um,z_um,sigma_x_um,sigma_y_um,sigma_z_um,photons,background,confidence
146,1.0454987,3.2415220589888873,-3.3619286114125213,0.0,0.07559999999999999,0.07559999999999999,0.07559999999999999,149,188522114.3473571,0.751861572265625
146,1.048718825,3.260196295057092,-3.3757200050390526,0.0,0.07559999999999999,0.07559999999999999,0.07559999999999999,149,188522114.3473571,0.751861572265625
146,1.049192875,3.255396050824117,-3.3782296120260766,0.0,0.07559999999999999,0.07559999999999999,0.07559999999999999,143,188522114.3473571,0.6551513671875
146,1.062468525,3.254832854891855,-3.375996347937161,0.0,0.07559999999999999,0.07559999999999999,0.07559999999999999,253,188522114.3473571,0.762725830078125
146,1.063397675,3.259053435871411,-3.376519718294927,0.0,0.07559999999999999,0.07559999999999999,0.07559999999999999,186,188522114.3473571,0.7720947265625
146,1.063871775,3.2586273494831954,-3.375985160196582,0.0,0.07559999999999999,0.07559999999999999,0.07559999999999999,128,188522114.3473571,0.634765625

    
plt.gca().set_aspect('equal', adjustable='box')

    
'''
track 0: 1000 steps:
track 1: 150
track 3: 350
track 4: 50
track 5, 400


0,0,0,0,0,0,0,0,0,0
1,1,3,5,5,5,5
2,2,2,2,

'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

def plot_gamma_distribution(shape, scale=1.0):
    """
    Plot the gamma distribution for a given shape (k) and scale (θ) parameter.

    Args:
        shape (float): Shape parameter (k > 0)
        scale (float): Scale parameter (θ > 0), default is 1.0
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be positive.")

    # Generate x values
    x = np.linspace(0, gamma.ppf(0.999, a=shape, scale=scale), 1000)

    # Compute PDF and CDF
    pdf = gamma.pdf(x, a=shape, scale=scale)
    cdf = gamma.cdf(x, a=shape, scale=scale)

    # Stats
    mean = gamma.mean(a=shape, scale=scale)
    variance = gamma.var(a=shape, scale=scale)
    mode = (shape - 1) * scale if shape >= 1 else 0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.6))

    # --- PDF ---
    axes[0].plot(x, pdf, color="steelblue", linewidth=2.5, label="PDF")
    axes[0].set_xlabel("Lifetime")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    plt.tight_layout()
    plt.show()

plot_gamma_distribution(2, 1.305)