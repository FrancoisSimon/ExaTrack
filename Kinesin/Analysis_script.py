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
IDs = [959, 960, 1028, 1042, 1289, 1668, 1824, 2068, 2498]
1504
3628 # Very straight track where we cant see stop and go motion
928
track_list = [track_list0[i] for i in IDs]
dt_list = [dt_list0[i] for i in IDs]
len(track_list)


plt.figure(figsize = (15, 15))
lim = 0.3 # MreB
nb_rows = 3
for i in range(nb_rows):
    for j in range(nb_rows):
        ID=i*nb_rows+j
        if ID < len(track_list):
            track = track_list[ID]
            print(len(track))
            track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
            plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
            plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 5)
            plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
            plt.text(track[0,0],track[0,1], str(ID), fontsize = 10)
plt.gca().set_aspect('equal', adjustable='box')


'''
track_list[0] = track_list[0][40:]
dt_list[0] = dt_list[0][40:]

track_list[1] = track_list[1][160:-220]
dt_list[1] = dt_list[1][160:-220]

track_list[2] = track_list[2][25:]
dt_list[2] = dt_list[2][25:]
'''
LocErr_list = None
LocErr_type = 'Linear'
# %%

'''
2 state model analysis
'''
import gc

segment_length = 40
batch_size = 9

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
#vary_params[:, 1] = 0
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
learning_rate = 0.10
epochs = 100
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

exatrack.get_model_params(model, track_segmentation = True)

plt.figure()
plt.plot(history.history['loss'])

'''
fittings from clear dataset
{'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
       dtype='<U15'),
 'anomalous factors': array([3.44113281e-08, 6.44531011e-07, 1.48932947e-04]),
 'Localization errors': array([0.00230753, 0.00167367, 0.01431733]),
 'd': array([0.00116655, 0.00517776, 0.01265862]),
 'q': array([4.73489693e-09, 4.39901199e-04, 1.05365259e-04]),
 'transition rates': array([[9.19409694e-01, 1.90879198e-01, 2.47139486e-06],
        [3.33156223e-01, 6.66667044e-01, 7.92584198e-05],
        [4.02031481e-06, 1.11682184e-01, 6.66667663e-01]]),
 'transition shapes': array([[1.        , 2.36857696, 1.        ],
        [1.        , 1.        , 0.44770305],
        [0.77455778, 0.33505247, 1.        ]]),
 'Fractions': array([8.85822232e-01, 8.62838986e-09, 1.14177735e-01, 2.45966687e-08])}
'''

model_params = []
likelihoods = []
batch_size = 1
# model results using all the tracks
#with params = np.array([[np.log(0.002), np.log(0.0005), np.log(0.0001), np.log(0.000005), 1],
#                   [np.log(0.002), np.log(0.01), np.log(0.01), np.log(0.0001), 1],
#                   [np.log(0.002), np.log(0.02), np.log(0.01), np.log(0.001), 1]], dtype = dtype)
# batch_size = 12

'''
# fittings from unclear data
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
    for k in range(5):
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
        params = np.array([[np.log(0.00225), np.log(0.00106), np.log(0.0001), np.log(0.000000005), 1],
                           [np.log(0.00225), np.log(0.00520), np.log(0.00044), np.log(0.0000001), 1],
                           [np.log(0.00225), np.log(0.0140), np.log(0.0001), np.log(0.00001), 1]], dtype = dtype)
        
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
        
        vary_params = np.zeros(params.shape)
        #vary_params[:, 1] = 0
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
        learning_rate = 0.1
        epochs = 150
        decay_fraction = 0.4
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
        
        likelihood = - history.history['loss'][-1]
        
        model_params.append(exatrack.get_model_params(model, track_segmentation = True))
        likelihoods.append(likelihood)
        
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
        plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_2D\track_%s_rep%s.svg'%(i, k))
        plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_2D\track_%s_rep%s.png'%(i, k))
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
        plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_1D\track_%s_rep%s.svg'%(i, k))
        plt.savefig(r'C:\Users\Franc\Data\Kinesin_minflux\Figs\Labeled_Tracks_1D\track_%s_rep%s.png'%(i, k))
        plt.close()
    
        tf.keras.backend.clear_session()
        gc.collect()


[{'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([5.72727351e-06, 1.62368829e-03, 5.61524035e-04]),
  'Localization errors': array([0.00279858, 0.00031536, 0.01013876]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([3.50450139e-07, 1.13780795e-03, 4.00033555e-05]),
  'transition rates': array([[9.54174954e-01, 2.57421199e+00, 2.01638649e-06],
         [2.28504112e-01, 7.61046178e-01, 7.71218315e-02],
         [8.75936022e-01, 1.18577363e-03, 6.68743700e-01]]),
  'transition shapes': array([[ 1.        , 56.17688612,  1.        ],
         [ 1.        ,  1.        ,  7.38007269],
         [ 2.64785337,  2.65432447,  1.        ]]),
  'Fractions': array([1.22218522e-03, 1.22232103e-03, 9.97392556e-01, 1.62937655e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([2.41519419e-05, 1.14134110e-03, 1.02901155e-02]),
  'Localization errors': array([0.00264016, 0.00021911, 0.01300834]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([1.67165286e-05, 1.98980171e-05, 4.79561870e-04]),
  'transition rates': array([[9.07978675e-01, 6.66660700e-01, 2.58829397e-06],
         [3.01002105e-01, 6.94009808e-01, 4.34382540e-02],
         [1.41555921e-03, 1.31971451e-02, 6.98752888e-01]]),
  'transition shapes': array([[1.        , 7.24481288, 1.        ],
         [1.        , 1.        , 8.7078752 ],
         [0.14180454, 0.04530976, 1.        ]]),
  'Fractions': array([6.56439450e-03, 6.38261329e-03, 9.86173675e-01, 8.79317467e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([8.89310166e-05, 3.52630537e-04, 8.85450219e-03]),
  'Localization errors': array([0.00219403, 0.00084356, 0.00093848]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([9.68398954e-06, 3.66980660e-06, 2.92368894e-04]),
  'transition rates': array([[8.52620860e-01, 7.26181711e-01, 2.86312129e-06],
         [3.31980833e-01, 6.67754436e-01, 1.15296734e-03],
         [3.08003435e-03, 3.37002005e-03, 6.77459477e-01]]),
  'transition shapes': array([[1.        , 4.9273889 , 1.        ],
         [1.        , 1.        , 4.35031191],
         [0.07738973, 0.01191907, 1.        ]]),
  'Fractions': array([1.53345432e-03, 4.06360310e-03, 9.94195422e-01, 2.07521057e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([1.50567559e-05, 2.41889983e-03, 4.79856933e-02]),
  'Localization errors': array([0.00197789, 0.00118251, 0.00146278]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([7.93804333e-07, 7.65718228e-06, 1.37613828e-02]),
  'transition rates': array([[8.59965499e-01, 8.26811346e-01, 2.84630296e-06],
         [2.09313092e-01, 7.82188196e-01, 5.86842979e-02],
         [1.02905555e-02, 6.81804623e-03, 7.05522745e-01]]),
  'transition shapes': array([[1.        , 5.90444766, 1.        ],
         [1.        , 1.        , 6.90483796],
         [1.26375259, 0.02381146, 1.        ]]),
  'Fractions': array([4.92192152e-03, 4.92037270e-03, 9.89541261e-01, 6.16444967e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([6.88973388e-06, 1.70845358e-03, 4.00914266e-04]),
  'Localization errors': array([0.00208254, 0.00157713, 0.00605929]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([2.00314145e-07, 6.67420254e-06, 6.76070421e-05]),
  'transition rates': array([[9.71606165e-01, 2.75870379e-02, 1.65325263e-06],
         [3.06762133e-01, 6.70417440e-01, 5.75699444e-03],
         [9.10360171e-01, 2.90362202e-04, 6.67701356e-01]]),
  'transition shapes': array([[1.        , 0.97163179, 1.        ],
         [1.        , 1.        , 0.25227042],
         [2.80124041, 0.03969799, 1.        ]]),
  'Fractions': array([2.27057176e-03, 9.93590962e-01, 3.83046322e-03, 3.08003079e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([1.10247179e-05, 1.98438697e-04, 1.21320260e-03]),
  'Localization errors': array([0.00228109, 0.00104965, 0.01112661]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([2.02924716e-06, 6.62660880e-05, 1.05645917e-04]),
  'transition rates': array([[9.29901449e-01, 4.91194662e-01, 2.36780409e-06],
         [2.79206478e-01, 6.96343449e-01, 6.10634869e-02],
         [5.59373633e-01, 1.00723701e-04, 6.69470314e-01]]),
  'transition shapes': array([[1.        , 7.00740811, 1.        ],
         [1.        , 1.        , 2.49744605],
         [1.7504062 , 0.00918837, 1.        ]]),
  'Fractions': array([2.13312855e-03, 2.03089138e-03, 9.95547992e-01, 2.87987801e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([0.00026287, 0.00024409, 0.00218014]),
  'Localization errors': array([0.0019791 , 0.00102826, 0.02645864]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([2.26455513e-06, 1.17795773e-04, 5.28450965e-04]),
  'transition rates': array([[9.25888503e-01, 2.68983912e-03, 2.40801063e-06],
         [3.30648372e-01, 6.68590642e-01, 4.99684384e-03],
         [4.22785564e-03, 4.18499729e-03, 6.89063326e-01]]),
  'transition shapes': array([[1.        , 0.03629552, 1.        ],
         [1.        , 1.        , 6.5636902 ],
         [0.04359834, 0.01955934, 1.        ]]),
  'Fractions': array([3.57422816e-03, 2.54211243e-03, 9.93404545e-01, 4.79114438e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([1.02528298e-05, 4.74081107e-05, 8.83399925e-03]),
  'Localization errors': array([0.00226498, 0.00152683, 0.01133697]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([3.25793275e-07, 2.63539843e-06, 8.55039527e-03]),
  'transition rates': array([[9.05671089e-01, 7.84025282e-02, 2.60316089e-06],
         [3.33062327e-01, 6.66871084e-01, 4.35067616e-04],
         [4.78218609e-03, 5.90984133e-01, 6.67428183e-01]]),
  'transition shapes': array([[1.        , 0.83118147, 1.        ],
         [1.        , 1.        , 6.50431639],
         [0.56949552, 1.82304108, 1.        ]]),
  'Fractions': array([4.62784953e-04, 4.62946907e-04, 9.98943174e-01, 1.31093984e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([1.46647102e-05, 1.10619210e-04, 6.49323411e-03]),
  'Localization errors': array([2.21423292e-03, 1.02985683e-03, 6.50212363e-05]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([8.10721774e-07, 1.28451097e-05, 2.47697899e-03]),
  'transition rates': array([[8.94518765e-01, 2.76204764e+00, 2.68817903e-06],
         [3.32214491e-01, 6.67552679e-01, 1.35171257e-03],
         [5.21863098e-02, 3.61556139e+02, 6.87888510e-01]]),
  'transition shapes': array([[1.00000000e+00, 2.61857967e+01, 1.00000000e+00],
         [1.00000000e+00, 1.00000000e+00, 5.79812462e+00],
         [4.14455267e+00, 1.20711752e+03, 1.00000000e+00]]),
  'Fractions': array([0.00235887, 0.00237398, 0.00241065, 0.9928565 ])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([8.22948414e-06, 2.16800982e-04, 3.41655055e-04]),
  'Localization errors': array([0.00230061, 0.00214398, 0.00704754]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([5.05396764e-07, 5.27918525e-06, 2.01301301e-05]),
  'transition rates': array([[9.27162867e-01, 1.79461052e-01, 2.39768629e-06],
         [3.19469493e-01, 6.78189999e-01, 1.49523087e-02],
         [5.44347061e-02, 1.90908667e+02, 8.54247344e-01]]),
  'transition shapes': array([[1.00000000e+00, 2.46393854e+00, 1.00000000e+00],
         [1.00000000e+00, 1.00000000e+00, 6.38767080e+00],
         [5.15101462e+00, 1.41220103e+03, 1.00000000e+00]]),
  'Fractions': array([1.61980890e-03, 1.61856630e-03, 9.96547541e-01, 2.14083991e-04])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([3.56655445e-05, 3.95770136e-05, 1.95685053e-04]),
  'Localization errors': array([2.59809343e-03, 1.69752101e-03, 4.16087964e-05]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([6.79651353e-07, 4.49168725e-04, 4.88243249e-04]),
  'transition rates': array([[9.13156979e-01, 2.15962093e-01, 2.53872944e-06],
         [3.32690680e-01, 6.67147058e-01, 1.08203907e-03],
         [4.48922183e-04, 9.31874642e-01, 6.76869657e-01]]),
  'transition shapes': array([[1.        , 2.4868741 , 1.        ],
         [1.        , 1.        , 6.65617428],
         [1.41768549, 2.88672328, 1.        ]]),
  'Fractions': array([3.49523116e-04, 4.12967988e-04, 5.00448575e-04, 9.98737060e-01])},
 {'Model types': array(['Directed motion', 'Directed motion', 'Directed motion'],
        dtype='<U15'),
  'anomalous factors': array([1.45725462e-04, 4.94594290e-05, 1.13359514e-04]),
  'Localization errors': array([0.00230237, 0.00066357, 0.00984926]),
  'd': array([0.00116, 0.00518, 0.013  ]),
  'q': array([1.22063156e-07, 3.28991024e-06, 4.79939241e-04]),
  'transition rates': array([[9.31107648e-01, 8.92846228e-01, 2.35237308e-06],
         [2.25098196e-01, 7.74851358e-01, 3.42991162e-07],
         [5.43655774e-02, 5.50616749e-06, 6.69430800e-01]]),
  'transition shapes': array([[1.00000000e+00, 1.29604057e+01, 1.00000000e+00],
         [1.00000000e+00, 1.00000000e+00, 6.75897402e-03],
         [1.65935146e-01, 1.87410760e-03, 1.00000000e+00]]),
  'Fractions': array([1.56633119e-04, 1.49876400e-04, 9.99672813e-01, 2.06773649e-05])}]
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

np.mean(shape_params)

rate_params = []
for params in model_params:
    rate_params.append(params['transition rates'][0,1])
rate_params = np.array(rate_params)

shape_params = np.array([5.61768861e+01, 7.24481288e+00, 4.92738890e+00, 5.90444766e+00,
       9.71631793e-01, 7.00740811e+00, 3.62955241e-02, 8.31181468e-01,
       2.61857967e+01, 2.46393854e+00, 2.48687410e+00, 1.29604057e+01])

rate_params = np.array([2.57421199e+00, 6.66660700e-01, 7.26181711e-01, 8.26811346e-01,
       2.75870379e-02, 4.91194662e-01, 2.68983912e-03, 7.84025282e-02,
       2.76204764e+00, 1.79461052e-01, 2.15962093e-01, 8.92846228e-01])

shape_params**0.5/rate_params
shape_params/rate_params




segment_length = 60

i = 0
cur_track_list = [track_list[i]]
cur_dt_list = [np.concatenate((dt_list[i], [reference_dt]))]

len(cur_track_list[0])

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
sequence_length = 40
max_linking_distance = 1

# We need to reshape the track to the shape used by the model, as following

nb_states = 3

params = np.array([[np.log(0.0023), np.log(0.00116), np.log(0.0001), np.log(0.000005), 1],
                   [np.log(0.0023), np.log(0.00518), np.log(0.0005), np.log(0.00001), 1],
                   [np.log(0.0023), np.log(0.013), np.log(0.0001), np.log(0.00001), 1]], dtype = dtype)

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

vary_params = np.zeros(params.shape)
vary_params[:, 1] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = np.zeros(transition_shapes.shape)
vary_transition_shapes[1, 0] = 0
vary_transition_shapes[0, 2] = 0
vary_transition_shapes[0, 1] = 0

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
epochs = 100
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

vary_transition_shapes[0, 1] = 1

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


lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history2 = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

-1682.005


history.history['loss']
history2.history['loss']
-1870.1246
-1870.08





segment_length = 40

i = 0

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
params = np.array([[np.log(0.002), np.log(0.00116), np.log(0.0001), np.log(0.000005), 1],
                   [np.log(0.002), np.log(0.00518), np.log(0.01), np.log(0.0001), 1]], dtype = dtype)

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
learning_rate = 0.1
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


lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile = False)

preds = model.predict(seq)
likelihood = exatrack.MLE_loss(preds, preds).numpy()
print(likelihood)

with tf.device(device):
    history2 = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])


-189.2708


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