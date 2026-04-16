# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:30:22 2026

@author: Franc
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# Import the ExaTrack module (ensure exatrack.py is in your path)
import os
import sys
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except:
    # add the absolute path if you are running the script line by line
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
    sys.path.append(rootdir) 
sys.path.insert(0, rootdir)
import exatrack_while_segment_infer_vars as exatrack
#import exatrack as exatrack
from glob import glob

exps = glob(rootdir + r'\PBP2\Datasets\C*')
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
    
    for i in range(len(tracks)):
        tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))
    
    # Prepare parameters for a 4 states model
    nb_states = 3
    
    # Initialize with generic guesses
    params = np.array([[np.log(0.02), np.log(0.001), np.log(0.001), np.log(0.0001), 1],
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
    
    model.save_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_weights.tf')

# Plot a random selection of tracks
for exp in exps[:]:
    model.load_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    weights = model.get_weights()
    
    tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
                   lengths = np.arange(5, 301), # number of positions per track accepted (take the first position if longer than max
                   dist_th = np.inf, # maximum distance allowed for consecutive positions 
                   frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                   fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                   colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                   opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                   remove_no_disp = True)
    
    for i in range(len(tracks)):
        tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))
    
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
                       [0,1,0],
                       [0,0,1]])
    plt.figure(figsize = (15, 15))
    lim = 0.6 # MreB
    nb_rows = 15
    IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
    for i in range(nb_rows):
        for j in range(nb_rows):
            ID = IDs[i*nb_rows+j]
            mask = masks[ID]
            track = tracks[ID]
            print(len(track))
            track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
            p = preds[ID, mask.astype(bool)][:,:-1]
            plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
            plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
            plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(rootdir + '/Results/Figs/Fig_random_labeled_tracks_' + exp.rsplit('\\')[-1] + '.png')
    plt.savefig(rootdir + '/Results/Figs/Fig_random_labeled_tracks_'  + exp.rsplit('\\')[-1] + '.svg')

# Plot a random selection of long tracks
for exp in exps[:]:
    model.load_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    weights = model.get_weights()
    
    paths = glob(exp + '/*.csv')
    tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
                   lengths = np.arange(100, 301), # number of positions per track accepted (take the first position if longer than max
                   dist_th = np.inf, # maximum distance allowed for consecutive positions 
                   frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                   fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                   colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                   opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                   remove_no_disp = True)
    
    for i in range(len(tracks)):
        tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))
    
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
                       [0,1,0],
                       [0,0,1]])
    plt.figure(figsize = (15, 15))
    lim = 0.6 # MreB
    nb_rows = 10
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
                plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
                plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(rootdir + '/Results/Figs/Fig_random_labeled_long_tracks_' + exp.rsplit('\\')[-1] + '.png')
    plt.savefig(rootdir + '/Results/Figs/Fig_random_labeled_long_tracks_' + exp.rsplit('\\')[-1] + '.svg')


All_params = []
exps = glob(rootdir + r'\PBP2\Datasets\C*')
exps = exps[:6] + exps[8:]

exp = exps[1]
for exp in exps[:]:
    model.load_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_weights.tf')
    params = exatrack.get_model_params(model, track_segmentation = True)
    equ_fractions = exatrack.equilibrium_distribution(params['transition rates'])
    params['equilibrium fractions'] = equ_fractions
    All_params.append(params)

labels = [exp.rsplit('\\')[-1] for exp in exps]

labels=['AV127',
 'AV127 +A22',
 'WT',
 'WT +A22',
 'WT +MEC',
 'Mut',
 'Mut + A22']

colors = plt.cm.tab10([float(exp.rsplit('\\')[-1][1])/8 for exp in exps])
colors[11:13] = colors[10]

groups = np.array([int(exp.rsplit('\\')[-1][1]) for exp in exps])
groups[9:11] = groups[8]

def compare_params(All_params, groups, labels=None):
    """
    Compare parameters across elements of all_params.
    Replicates sharing the same color are grouped into a single bar
    showing the mean, with individual dots and std-dev error bars.
    """
    n_runs = len(All_params)
    if labels is None:
        labels = [f"Run {i}" for i in range(n_runs)]

    unique_groups = np.unique(groups)
    # Use the first label of each group as the group label
    n_groups = len(unique_groups)

    all_states = ['State 0', 'State 1', 'State 2']
    n_states = len(all_states)
    param_keys = ["anomalous factors", "d", "q"]
    param_labels = ["Anomalous\nfactors", "d", "q"]

    # --- Helper to plot grouped bars with dots and error bars ---
    def _plot_grouped(ax, all_values_per_run):
        """
        all_values_per_run: list of scalar values, one per run.
        Groups by color, plots mean bar + std errorbar + individual dots.
        """
        means, stds, colors = [], [], []
        for gi, group in enumerate(unique_groups):
            vals = all_values_per_run[groups == group]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            colors.append(plt.cm.tab10(gi/10))
        x = np.arange(n_groups)
        ax.bar(x, means, color = colors, edgecolor="k", linewidth=0.5,
               zorder=1, width=0.6)
        ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='k',
                    capsize=4, linewidth=1.2, zorder=2)

        # Overlay individual dots (jittered)
        rng = np.random.default_rng(42)
        for gi, group in enumerate(unique_groups):
            dot_vals = all_values_per_run[groups == group]
            jitter = rng.uniform(-0.15, 0.15, size=len(dot_vals))
            ax.scatter(gi + jitter, dot_vals, color='k', s=18,
                       zorder=3, alpha=0.7, edgecolors='none')
            
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))

    # --- One figure per motion state, 3 subplot rows ---
    for state_idx, state_name in enumerate(all_states):
        fig, axes = plt.subplots(len(param_keys), 1, figsize=(6, 8), sharex=True)
        fig.suptitle(state_name, fontsize=14, fontweight="bold")

        for row, (key, ylabel) in enumerate(zip(param_keys, param_labels)):
            ax = axes[row]
            values = []
            for p in All_params:
                arr = np.asarray(p[key])
                values.append(arr[state_idx])
            values = np.array(values)
            _plot_grouped(ax, values)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_ylim([0,None])

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
    # --- Transition rates: 3x3 grid ---
    fig_tr, axes_tr = plt.subplots(n_states, n_states, figsize=(12, 8), sharex=True)
    fig_tr.suptitle("Transition Rates", fontsize=14, fontweight="bold")

    for from_s in range(n_states):
        for to_s in range(n_states):
            ax = axes_tr[from_s, to_s]
            values = []
            for p in All_params:
                tr = np.asarray(p["transition rates"])
                values.append(tr[from_s, to_s])
            values = np.array(values)
            _plot_grouped(ax, values)
            ax.set_ylim([0, None])
            if to_s == 0:
                ax.set_ylabel(f"From {all_states[from_s]}", fontsize=9)
            if from_s == 0:
                ax.set_title(f"To {all_states[to_s]}", fontsize=10)

    fig_tr.tight_layout(rect=[0, 0, 1, 0.93])

    # --- Equilibrium fractions ---
    fig_eq, axes_eq = plt.subplots(n_states, 1, figsize=(6, 6), sharex=True)
    fig_eq.suptitle("Equilibrium Fractions", fontsize=14, fontweight="bold")

    for state_idx, state_name in enumerate(all_states):
        ax = axes_eq[state_idx]
        values = []
        for p in All_params:
            arr = np.asarray(p["equilibrium fractions"])
            values.append(arr[state_idx])
        values = np.array(values)
        _plot_grouped(ax, values)
        ax.set_ylabel(state_name, fontsize=10)
        ax.set_ylim([0, None])

    fig_eq.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    
# Model fitting with fixed state parameters from WT 
exps = glob(rootdir + r'\PBP2\Datasets\C*')
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
    
    for i in range(len(tracks)):
        tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))
    
    # Prepare parameters for a 4 states model
    nb_states = 3
    
    # Initialize with generic guesses
    params = np.array([[np.log(0.02), np.log(0.002), np.log(0.0017/2**0.5), np.log(0.00003), 1],
                       #[np.log(0.02), np.log(0.001), np.log(0.1), np.log(0.0001), 0], 
                       #[np.log(0.015), np.log(0.01), np.log(0.1), np.log(0.0001), 0],
                       [np.log(0.02), np.log(0.05), np.log(0.12) - np.log(1-0.12), np.log(0.001), 0],
                       [np.log(0.02), np.log(0.095), np.log(0.24) - np.log(1-0.24), np.log(0.001), 0]], dtype='float64')
    
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
    vary_params = False
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
    
    model.save_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_fixed_params_weights.tf')


'''
Load the parameters from the fittings with fixed state parameters
'''
all_params = []
exps = glob(rootdir + r'\PBP2\Datasets\C*')
exp = exps[1]
for exp in exps[:]:
    model.load_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_fixed_params_weights.tf')
    params = exatrack.get_model_params(model, track_segmentation = True)
    equ_fractions = exatrack.equilibrium_distribution(params['transition rates'])
    params['equilibrium fractions'] = equ_fractions
    all_params.append(params)

labels = [exp.rsplit('\\')[-1] for exp in exps]

labels=['PBP2AV127_1',
 'PBP2AV127_2',
 'PBP2AV127_3',
 'PBP2AV127_A22_1',
 'PBP2AV127_A22_2',
 'PBP2AV127_A22_3',
 'PBP2_IPTG5_1',
 'PBP2_IPTG5_2',
 'PBP2_IPTG25_1',
 'PBP2_IPTG25_2',
 'PBP2_IPTG25_3',
 'PBP2_IPTG25_4',
 'PBP2_IPTG25_5',
 'PBP2_IPTG25_A22_1',
 'PBP2_IPTG25_A22_2',
 'PBP2_IPTG25_A22_3',
 'PBP2_IPTG25_A22_4',
 'PBP2_IPTG25_MEC_1',
 'PBP2_IPTG25_MEC_2',
 'PBP2_IPTG25_MEC_3']
'''
 'PBP2_Mut_1',
 'PBP2_Mut_2',
 'PBP2_Mut_3',
 'PBP2_Mut_A22_1',
 'PBP2_Mut_A22_2',
 'PBP2_Mut_A22_3']'''
colors = plt.cm.tab10([float(exp.rsplit('\\')[-1][1])/9 for exp in exps])
colors[11:13] = colors[10]

exp = exps[10]
# Plot a random selection of long tracks with the fixed parameters (and variable transition rates learned from each replicate)
for exp in exps[:]:
    model.load_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_fixed_params_weights.tf')
    weights = model.get_weights()
    
    paths = glob(exp + '/*.csv')
    tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
                   lengths = np.arange(100, 301), # number of positions per track accepted (take the first position if longer than max
                   dist_th = np.inf, # maximum distance allowed for consecutive positions 
                   frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                   fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                   colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                   opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                   remove_no_disp = True)
    
    for i in range(len(tracks)):
        tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))
    
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
                       [0,1,0],
                       [0,0,1]])
    plt.figure(figsize = (15, 15))
    lim = 0.6 # MreB
    nb_rows = 10
    plt.title('Long tracks with fixed params model')
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
                plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
                plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
    plt.gca().set_aspect('equal', adjustable='box')


# Plot a random selection of tracks with the fixed parameters (and variable transition rates learned from each replicate)
for exp in exps[:]:
    model.load_weights(rootdir + '/Results/' + exp.rsplit('\\')[-1]  + '_fixed_params_weights.tf')
    weights = model.get_weights()
    
    paths = glob(exp + '/*.csv')
    tracks, frames, track_IDs, opt_metrics = exatrack.read_table(paths, # path of the file to read or list of paths to read multiple files.
                   lengths = np.arange(5, 301), # number of positions per track accepted (take the first position if longer than max
                   dist_th = np.inf, # maximum distance allowed for consecutive positions 
                   frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
                   fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                   colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
                   opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
                   remove_no_disp = True)
    
    for i in range(len(tracks)):
        tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,2))
    
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
                       [0,1,0],
                       [0,0,1]])
    plt.figure(figsize = (15, 15))
    lim = 0.6 # MreB
    nb_rows = 10
    plt.title('Tracks with fixed params model')
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
                plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
                plt.scatter(track[0,0], track[0,1] , c = 'k', s = 8, marker = 'x')
    plt.gca().set_aspect('equal', adjustable='box')


