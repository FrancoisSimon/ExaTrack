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

# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
sys.path.append(r"C:\Users\Franc\Data\ExaTrack") # add exatrack directory to the system path
import exatrack_while_segment as exatrack
from glob import glob
import random

paths = glob(r'C:\Users\Franc\Data\Kinesin_minflux\Tracks/*Minflux_642_L75_pho100_lp15_BGdis40k_hex3_dt300us_cfr09_dp1*')

tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths[:], # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(100, 11000), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions 
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
               colnames = ['x_um', 'y_um', 'time', 'track_id'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = ['photons', 'sigma_x_um', 'sigma_y_um'], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = False)

for i in range(len(tracks)):
    tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))

ls = [len(track) for track in tracks]
np.mean(ls)
np.max(ls)

plt.figure(figsize = (15, 15))
lim = 0.3 # MreB
nb_rows = 16
offset = 0
for i in range(nb_rows):
    for j in range(nb_rows):
        ID=offset+i*nb_rows+j
        track = tracks[ID]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1, len(track))), s = 5)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 10, marker = 'x')
        plt.text(track[0,0],track[0,1], str(ID), fontsize = 10)
plt.gca().set_aspect('equal', adjustable='box')

# Must identify a few directed tracks with stepping motion from the above plot before proceeding to the next steps
IDs = [0,1,2]

tracks = [tracks[i] for i in IDs]
len(tracks)


# %%

'''
2 state model analysis
'''

segment_length = 20
batch_size = 2
seq = exatrack.TrackSegmentSequence(tracks,
    batch_size=batch_size,
    segment_length=segment_length,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5)
nb_batches = len(seq)

dtype = 'float64'
device = '/CPU:0'
estimated_density = 0.01 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1

# We need to reshape the track to the shape used by the model, as following

nb_states = 2
params = np.array([[np.log(0.002), np.log(0.002), np.log(0.0001), np.log(0.00005), 1],
                   [np.log(0.003), np.log(0.005), np.log(0.002), np.log(0.0001), 1]], dtype = dtype)

initial_params = np.array([[np.log(1)]]*nb_states, dtype = dtype) 

transition_shapes = np.zeros((nb_states, nb_states), dtype = dtype)
transition_shapes[0,1]=np.log(3)
transition_rates = np.eye(nb_states, dtype = dtype)*3
transition_rates[0,0] = 1
transition_rates[1,1] = 1

initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
initial_fractions[0,-1] = -1
sequence_length = 5 # sequence length to allow without forcing fusion of sequences, the higher the better but the more computationally demanding
max_linking_distance = 1 # maximum linking distance used for the linking algorithm
estimated_density = 0.0001 # estimated density of the sample (number of counts per distance unit per frame)

vary_params = np.ones(params.shape)
#vary_params[2] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = True
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
exatrack.get_model_params(model, track_segmentation = True)['transition rates']

nb_batches = len(seq)

seq = exatrack.TrackSegmentSequence(tracks,
    batch_size=batch_size,
    segment_length=segment_length,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5)

#all_masks = masks
learning_rate = 0.01
epochs = 500
decay_threshold = int(0.7*nb_batches*epochs)
decay_rate = - np.log(0.001)/(0.3*nb_batches*epochs) # 
device = '/CPU:0'
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

#params = eget_model_params(model, track_segmentation = True)
'''
{'Model types': array(['Directed motion', 'Directed motion'], dtype='<U15'),
 'anomalous factors': <tf.Tensor: shape=(2,), dtype=float64, numpy=array([1.10854021e-07, 4.15030453e-03])>,
 'Localization errors': array([0.0028289 , 0.00156922]),
 'd': array([0.0013981 , 0.00260727]),
 'q': array([3.06995005e-08, 6.55117415e-03]),
 'transition rates': <tf.Tensor: shape=(2, 2), dtype=float64, numpy=
 array([[0.68707004, 0.58577119],
        [1.3054926 , 0.34647765]])>,
 'transition shapes': <tf.Tensor: shape=(2, 2), dtype=float64, numpy=
 array([[1.        , 1.87189234],
        [1.99762505, 1.        ]])>,
 'Fractions': <tf.Tensor: shape=(3,), dtype=float64, numpy=array([1.21754144e-07, 4.07451053e-10, 9.99999878e-01])>}
'''
print('lifetimes:', params['transition shapes']/params['transition rates'])

weights = model.get_weights()

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

# Then we change version to infer the real positions and the velocity vectors (we could have used this new version from the beginning)
import exatrack_while_segment_infer_vars as exatrack

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

preds, All_coefs, All_biases, All_LPs = pred_model.predict((track_array, masks), batch_size = batch_size)

pos_mean, anomalous_mean = exatrack.extract_hidden_variables(All_coefs, All_biases, All_LPs, nb_dims = 2)

def plot_track(ID, tracks, preds, pos_mean, anomalous_mean, masks, colors=None, nb_states=3, dt=1):
    if colors is None:
        colors = np.array([[1, 0, 0], [0, 0, 1]])[:nb_states]
    
    #ID = random.randint(0, len(tracks) - 1)
    mask = masks[ID].astype(bool)
    track = tracks[ID]
    p = preds[ID, mask][:, :-1]
    
    refined = pos_mean[ID][mask[1:]]
    p_refined = preds[ID, mask][1:, :-1]
    anomalous_vect = anomalous_mean[ID, mask[1:]]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 1) Raw track with state labels
    ax = axes[0]
    ax.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
    ax.scatter(track[:, 0], track[:, 1], c=p @ colors, s=15)
    ax.scatter(track[0, 0], track[0, 1], c='k', s=30, marker='x', zorder=5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Raw track (ID={ID}, len={len(track)})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Store limits from first plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 2) Refined track with state labels (same limits as raw)
    ax = axes[1]
    ax.plot(refined[:, 0], refined[:, 1], ':k', alpha=0.5)
    ax.scatter(refined[:, 0], refined[:, 1], c=p_refined @ colors, s=15)
    ax.scatter(refined[0, 0], refined[0, 1], c='k', s=30, marker='x', zorder=5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title('Refined track')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 3) Velocity vs time
    ax = axes[2]
    disp_raw = np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1)) / dt
    disp_refined = np.sqrt(np.sum(np.diff(refined, axis=0)**2, axis=1)) / dt
    disp_ano = np.sqrt(np.sum(anomalous_vect**2, axis=1)) / dt
    anomalous_mean.shape
    
    t_raw = np.arange(len(disp_raw))
    t_refined = np.arange(len(disp_refined))
    
    ax.plot(t_raw, disp_raw, '-', color='gray', alpha=0.5, label='Raw displacements')
    #ax.scatter(t_raw, disp_raw, c=p[1:] @ colors, s=15, zorder=3)
    ax.plot(t_refined, disp_refined, '-', color=[0.9, 0.3, 0.3], alpha=0.5, label='Refined displacements (MAP)')
    #ax.scatter(t_refined, disp_refined, c=p_refined[1:] @ colors, s=15, zorder=3, marker='s')
    #plt.figure()
    #plt.scatter(t_refined, disp_ano[1:], c=p[2:], s=10, zorder=4)
    ax.plot(t_refined, disp_ano[1:], color = [0.2, 0.8, 0], label = 'underlying velocity (MAP estimate)')
    ax.legend()
    # 3) Velocity vs time
    ax = axes[3]
    
    t_raw = np.arange(len(disp_raw))
    t_refined = np.arange(len(disp_refined))
    
    #ax.scatter(t_raw, disp_raw, c=p[1:] @ colors, s=15, zorder=3)
    ax.plot(t_refined, disp_refined, '-', color=[0.9, 0.3, 0.3], alpha=0.5, label='Refined displacements (MAP)')
    #ax.scatter(t_refined, disp_refined, c=p_refined[1:] @ colors, s=15, zorder=3, marker='s')
    #plt.figure()
    #plt.scatter(t_refined, disp_ano[1:], c=p[2:], s=10, zorder=4)
    ax.plot(t_refined, disp_ano[1:], color = [0.2, 0.8, 0], label = 'Hidden velocity (MAP)')
    ax.set_ylim(0, np.quantile(disp_ano[1:], 0.9)*1.4)
    
    ax.set_title('Velocity')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Displacement / dt')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
for ID in range(len(tracks)):
    plot_track(ID, tracks, preds, pos_mean, anomalous_mean, masks, colors=None, nb_states=3, dt=1)


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
