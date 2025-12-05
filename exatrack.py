# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:53:55 2025

@author: Franc

This script is derived from exatrack_alternate.py to allow the user to fix parameters.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN
#from tensorflow.python.keras.layers.recurrent import RNN

dtype = 'float64'
pi = tf.constant(np.pi, dtype = dtype)
minval = 1e-20

from matplotlib import pyplot as plt
from numba import njit, typed, prange, jit
from scipy.stats import gamma
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import pandas as pd
from glob import glob

from scipy.spatial.transform import Rotation as R
jit_compile = False

def anomalous_diff_transition(max_track_len=100,
                              nb_tracks = 100,
                              LocErr=0.02, # localization error in x, y and z (even if not used)
                              Fs = np.array([0., 1]),
                              Ds = np.array([0.0, 0.25]),
                              nb_dims = 2, # can be 2D or 3D
                              velocities = np.array([0.03, 0.0]),
                              angular_Ds = np.array([0.0, 0.0]),
                              conf_forces = np.array([0.0, 0.2]),
                              conf_Ds = np.array([0.0, 0.0]),
                              conf_dists = np.array([0.0, 0.0]),
                              transition_matrix = np.array([[0.00, 0.1],
                                                            [0.1, 0.00]]),
                              shape_matrix = np.array([[0, 1],
                                                       [1, 0]]),
                              bleaching_rate = 1e-10,
                              LocErr_std = 0,
                              dt = 0.02,
                              field_of_view = np.array([10,10]),
                              nb_burning_steps=100,
                              nb_sub_steps = 10):
        
    nb_states = len(velocities)
    if not np.all(np.array([len(Fs), len(Ds), len(velocities), len(angular_Ds), len(conf_forces), len(conf_Ds), len(conf_dists), len(transition_matrix)]) == nb_states):
        raise ValueError('Fs, Ds, velocities, angular_Ds, conf_forces, conf_Ds, conf_dists and transition_matrix must all be arrays of the same lenght (one element per state)')
    # diff + persistent motion + elastic confinement
    
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
    '''
    state_dists = []
    for state in range(nb_states):
        density_function = (1-transition_matrix[state,state])*(transition_matrix[state,state])**np.arange(max_track_len)
        CPD = [density_function[0]]
        for l in range(1, max_track_len):
            CPD += [CPD[-1] + density_function[l]]
        CPD[-1] = 1
        state_dists += [CPD]
    state_dists = np.array(state_dists)
    '''
    CPD_transition_mat = np.zeros((nb_states, nb_states, nb_burning_steps + max_track_len*nb_sub_steps*10))
    for state0 in range(nb_states):
        for state1 in range(nb_states):
            if state0 != state1:
                p = gamma.cdf(np.arange(nb_burning_steps + max_track_len*nb_sub_steps*10), shape_matrix[state0, state1], scale = 1/transition_matrix[state0, state1]*nb_sub_steps)
                p[-1] = 1
                CPD_transition_mat[state0, state1] = p
            else:
                CPD_transition_mat[state0, state1,-1] = 1
    
    '''
    When tracks appear in the field of view, they have already experienced time, therefor the first lifetime is not gamma distributed
    (except if shape = 1), if the shape is very high for instance the initial distribution will be uniform between 1 and the mean lifetime.
    The initial lifetime L=l follows a distribution that depends on the actual gamma-distributed lifetime G=g and the time the particle
    has already been in the current state prior the first step F=f.
    l+f=g
    P(f) = sum over f of P(g)P(g-f|g) with P(g-f|g) a uniform distribution from 1 to g
    '''
    
    all_tracks = np.zeros((nb_tracks, max_track_len, nb_dims))
    all_states = np.zeros((nb_tracks, max_track_len))
    all_masks = np.zeros((nb_tracks, max_track_len))
    
    for k in range(nb_tracks):
        if bleaching_rate/nb_sub_steps>0.00000001:
            track_len = min(max_track_len, np.random.geometric(p=bleaching_rate))
        else:
            track_len = max_track_len
        
        initial_positions = np.random.rand(nb_dims)*field_of_view
        track = []
        
        next_state = np.argmin(np.random.rand()>cum_Fs)
        states = []
        
        n = 0
        while n < nb_burning_steps*nb_sub_steps:
            state = next_state
            transitions = np.argmin(CPD_transition_mat[state, :] < np.random.rand(nb_states)[:, None], axis=1)
            next_state = np.argmin(transitions)
            current_segment_length = np.min(transitions)
            n += current_segment_length
        transitions[next_state] = n - nb_burning_steps*nb_sub_steps
        
        # in case the segment finishes at the very end of the burning step, we need to define a fresh segment
        if transitions[next_state] == 0:
            state = next_state
            transitions = np.argmin(CPD_transition_mat[state, :] < np.random.rand(nb_states)[:, None], axis=1)

        while len(track) < track_len*nb_sub_steps:
            if len(track)>0: # we must shorten the first lifetime as the track has already been its in initial state for a random periode of time before the start of the track 
                transitions = np.argmin(CPD_transition_mat[state, :] < np.random.rand(nb_states)[:, None], axis=1)
            l = np.min([np.min(transitions), track_len*nb_sub_steps - len(track)])
            D, velocity, angular_D, conf_force, conf_D, conf_dist = (Ds[state], velocities[state], angular_Ds[state], conf_forces[state], conf_Ds[state], conf_dists[state])
            
            if nb_dims < 3:
                segment = anomalous_diff_2D(track_len=l+1,
                                         LocErr=0, # localization error in x, y and z (even if not used)
                                         D = D,
                                         velocity = velocity/nb_sub_steps,
                                         angular_D = angular_D,
                                         conf_force = conf_force/nb_sub_steps,
                                         conf_D = conf_D,
                                         conf_dist = conf_dist,
                                         LocErr_std = LocErr_std,
                                         dt = dt/nb_sub_steps,
                                         nb_sub_steps = 1,
                                         initial_positions = initial_positions)
                
                segment = segment[:,:nb_dims]
                
            elif nb_dims == 3:
                segment = anomalous_diff_3D(track_len=l+1,
                                         LocErr=0, # localization error in x, y and z (even if not used)
                                         D = D,
                                         velocity = velocity/nb_sub_steps,
                                         angular_D = angular_D,
                                         conf_force = conf_force/nb_sub_steps,
                                         conf_D = conf_D,
                                         conf_dist = conf_dist,
                                         LocErr_std = LocErr_std,
                                         dt = dt/nb_sub_steps,
                                         nb_sub_steps = 1,
                                         initial_positions = initial_positions)
            else:
                raise ValueError('The number of dimensions must be 1, 2 or 3')
            
            track += list(segment[:-1])
            states += [state]*l
            
            initial_positions = segment[-1]
            state = np.argmin(transitions)
        
        #print(track_len, nb_dims)
        track = np.array(track)[::nb_sub_steps] + np.random.normal(0, LocErr, (track_len, nb_dims))
        states = np.array(states)[::nb_sub_steps]
        
        all_tracks[k,:track_len] = track
        all_states[k,:track_len] = states
        all_masks[k,:track_len] = 1
        
    return all_tracks, all_states, all_masks

@njit
def anomalous_diff_2D(track_len=20,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           D = 0.05,
                           velocity = 0,
                           angular_D = 0.0,
                           conf_force = 0.0,
                           conf_D = 0.0,
                           conf_dist = 0.0,
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           initial_positions = [0,0]):
    nb_dims = 2
    
    conf_sub_force = conf_force / nb_sub_steps
    sub_dt = dt / nb_sub_steps
     
    positions = np.zeros((track_len * nb_sub_steps, nb_dims))
    
    positions[0] = initial_positions
    disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
   
    anchor_positions = np.random.normal(0, np.sqrt(2*conf_D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
    anchor_positions[0] = positions[0] + np.random.normal(0,conf_dist, nb_dims)
   
    for i in range(1, len(anchor_positions)):
        anchor_positions[i] += anchor_positions[i-1]
   
    d_angles = np.random.normal(0, 1, ((track_len) * nb_sub_steps)-1) * (2*angular_D*sub_dt)**0.5
    angles = np.zeros((track_len * nb_sub_steps-1))
    angles[0] = np.random.rand()*2*np.pi
    for i in range(1, len(d_angles)):
        angles[i] = angles[i-1] + d_angles[i]
    
    for i in range(len(positions)-1):
        angle = angles[i-1]
        pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
        positions[i+1] = positions[i] + pesistent_disp + disps[i]
        positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
    
    final_track = np.zeros((track_len, nb_dims))
    for i in range(track_len):
        final_track[i] = positions[i*nb_sub_steps]
    
    if LocErr>0:
        final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
    return final_track

def anomalous_diff_3D(track_len=20,
                           LocErr=0.02, # localization error in x, y and z (even if not used)
                           D = 0.05,
                           velocity = 0,
                           angular_D = 0.0,
                           conf_force = 0.0,
                           conf_D = 0.0,
                           conf_dist = 0.0,
                           LocErr_std = 0,
                           dt = 0.02,
                           nb_sub_steps = 10,
                           initial_positions = [0,0,0]):
    
    nb_dims = 3

    conf_sub_force = conf_force / nb_sub_steps
    sub_dt = dt / nb_sub_steps
     
    positions = np.zeros((track_len * nb_sub_steps, nb_dims))
    
    positions[0] = initial_positions
    disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
   
    anchor_positions = np.random.normal(0, np.sqrt(2*conf_D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
    anchor_positions[0] = positions[0] + np.random.normal(0,conf_dist, nb_dims)
   
    for i in range(1, len(anchor_positions)):
        anchor_positions[i] += anchor_positions[i-1]
    
    pesistent_displacements = simulate_3D_rotational_diffusion(track_len * nb_sub_steps - 1, velocity/nb_sub_steps, angular_D, sub_dt)
    
    for i in range(len(positions)-1):
        pesistent_disp = pesistent_displacements[i]
        positions[i+1] = positions[i] + pesistent_disp + disps[i]
        positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
    
    final_track = np.zeros((track_len, nb_dims))
    for i in range(track_len):
        final_track[i] = positions[i*nb_sub_steps]
    
    if LocErr>0:
        final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
    return final_track

def simulate_3D_rotational_diffusion(nb_steps, velocity, D_r, dt):
    
    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(2 * np.random.rand() - 1)
    vx = np.sin(phi) * np.cos(theta)
    vy = np.sin(phi) * np.sin(theta)
    vz = np.cos(phi)
    v = [vx, vy, vz]
    v = np.array(v) / np.linalg.norm(v)  # Ensure v is a unit vector
    vs = [v]

    sigma_theta = np.sqrt(2 * D_r * dt)
    for _ in range(nb_steps-1):
        # Generate small random rotations
        dtheta = np.random.normal(0, sigma_theta, size=3)
        
        # Convert the random angles into a rotation
        rotation = R.from_rotvec(dtheta)
        
        v = rotation.apply(v)
        
        v = v / np.linalg.norm(v)
        vs.append(v)
    return np.array(vs) * velocity

def generate_movie(track_list, time_list, state_list, average_photon_number, average_background, emission_std, max_time, pixel_dims, pixel_size):
    movie = np.random.poisson(average_background, size = [max_time]+list(pixel_dims)).astype('int16')
    nb_counts = 0
    for track, times in zip(track_list, time_list):
        for pos, time in zip(track, times):
            
            pixel_pos = pos / pixel_size
            nb_photons = np.random.poisson(average_photon_number)
            
            movie = emit_photons(pixel_pos, nb_photons, movie, time, emission_std)
            nb_counts += 1
    return movie, nb_counts

@njit
def emit_photons(pixel_pos, nb_photons, movie, time, emission_std, pixel_dims):
    for k in range(nb_photons):
        photon_pos_x = int(np.random.normal(pixel_pos[0], emission_std))
        photon_pos_y = int(np.random.normal(pixel_pos[1], emission_std))
        if photon_pos_x < pixel_dims[0] and photon_pos_y < pixel_dims[1] and photon_pos_x >= 0 and photon_pos_y >= 0:
            movie[time, photon_pos_x, photon_pos_y] = movie[time, photon_pos_x, photon_pos_y] + 1
    return movie

def padding(track_list, frame_list):
    '''
    If tracks have multiple lengths, we need to homogenize the shape to the longest track lengths using padding and mask the state updates of the padding
    
    This function takes a list of tracks as imput and returns a padded array of tracks and the corresponding padding mask.
    
    fitting_type: can be either 'All', 'Directed', 'Confined' or 'Brownian' depending on the type of motion you want to analyse.
    'All': works for all types of motion but requires tracks of at least 5 time steps
    'Directed': works for both directed and Brownian types of motion.
    'Confined': works for both confined and Brownian types of motion.
    'Brownian': only works for Brownian motion.
    If your tracks are all 5 time points or more you can ignore the `fitting_type` argument.
    '''
    start_len = 1
    max_len = 0
    for track in track_list:
        if track.shape[0] > max_len:
            max_len = track.shape[0]
    nb_tracks = len(track_list)
    padded_tracks = np.zeros((nb_tracks, max_len, track_list[0].shape[1]), dtype = track_list[0].dtype)
    padded_frames =  np.zeros((nb_tracks, max_len), dtype = frame_list[0].dtype)
    mask = np.zeros((nb_tracks, max_len), dtype = track.dtype)

    for i, track in enumerate(track_list):
        if track.shape[0]>=start_len:
            frames = frame_list[i]
            cur_len = track.shape[0]
            padded_tracks[i, :cur_len] = track
            mask[i, :cur_len] = 1
            padded_frames[i, :cur_len] = frames
        else:
            raise Warning('The minimal track length supported is 2 time points. Tracks of 1 time point were discarded.')
    
    return padded_tracks, padded_frames, mask

@tf.function(jit_compile=jit_compile)
def log_gaussian(top, variance=tf.constant(1, dtype = dtype)):
    return - 0.5*tf.math.log(2*pi*variance) - top**2/(2*variance)


@tf.function(jit_compile=jit_compile)
def norm_log_gaussian(top):
    return - 0.5*(tf.math.log(2*pi) + top**2)

def read_table(paths, # path of the file to read or list of paths to read multiple files.
               lengths = np.arange(4,40), # number of positions per track accepted (take the first position if longer than max
               dist_th = np.inf, # maximum distance allowed for consecutive positions 
               frames_boundaries = [-np.inf, np.inf], # min and max frame values allowed for peak detection
               fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
               colnames = ['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True):
    
    if type(paths) == str or type(paths) == np.str_:
        paths = [paths]
    
    tracks = []
    frames = []
    track_IDs = []
    opt_metrics = {}
    for m in opt_colnames:
        opt_metrics[m] = []
        
    for path in paths:
        
        if fmt == 'csv':
            data = pd.read_csv(path, sep=',')
        elif fmt == 'pkl':
            data = pd.read_pickle(path)
        else:
            data = pd.read_csv(path, sep = fmt)
        
        if not (type(colnames[3]) == str or type(colnames[3]) == np.str_):
            None_ID = (data[colnames[3]] == 'None') + pd.isna(data[colnames[3]])
            data = data.drop(data[np.any(None_ID,1)].index)
                
            new_ID = data[colnames[3][0]].astype(str)
            
            for k in range(1,len(colnames[3])):
                new_ID = new_ID + '_' + data[colnames[3][k]].astype(str)
            data['unique_ID'] = new_ID
            colnames[3] = 'unique_ID'        
        try:
            # in this case, peaks without an ID are assumed alone and are added a unique ID, only works if ID are integers
            None_ID = (data[colnames[3]] == 'None' ) + pd.isna(data[colnames[3]])
            max_ID = np.max(data[colnames[3]][(data[colnames[3]] != 'None' ) * (pd.isna(data[colnames[3]]) == False)].astype(int))
            data.loc[None_ID, colnames[3]] = np.arange(max_ID+1, max_ID+1 + np.sum(None_ID))
        except:
            None_ID = (data[colnames[3]] == 'None' ) + pd.isna(data[colnames[3]])
            data = data.drop(data[None_ID].index)
        
        data = data[colnames + opt_colnames]
        
        zero_disp_tracks = 0
            
        try:
            for ID, track in data.groupby(colnames[3]):
                
                track = track.sort_values(colnames[2], axis = 0)
                track_mat = track.values[:,:4].astype('float64')
                dists2 = (track_mat[1:, :2] - track_mat[:-1, :2])**2
                if remove_no_disp:
                    if np.mean(dists2==0)>0.05:
                        continue
                dists = np.sum(dists2, axis = 1)**0.5
                
                if track_mat[0, 2] >= frames_boundaries[0] and track_mat[0, 2] <= frames_boundaries[1] : #and np.all(dists<dist_th):
                    if not np.any(dists>dist_th):
                        
                        if np.any([len(track_mat)]*len(lengths) == np.array(lengths)):
                            l = len(track)
                            tracks.append(track_mat[:, 0:2])
                            frames.append(track_mat[:, 2])
                            track_IDs.append(track_mat[:, 3])
                            for m in opt_colnames:
                                opt_metrics[m].append(track[m].values)
                        
                        elif len(track_mat) > np.max(lengths):
                            l = np.max(lengths)
                            tracks.append(track_mat[:l, 0:2])
                            frames.append(track_mat[:l, 2])
                            track_IDs.append(track_mat[:l, 3])
                            for m in opt_colnames:
                                opt_metrics[m].append(track[m].values[:l]) 
        
        except :
            print('problem with file :', path)
    
    if zero_disp_tracks and not remove_no_disp:
        print('Warning: some tracks show no displacements. To be checked if normal or not. These tracks can be removed with remove_no_disp = True')
    return tracks, frames, track_IDs, opt_metrics

def ExaTrack_2_DataFrame(track_list, frame_list, track_ID_list, opt_metrics, state_preds, all_masks):
    nb_rows = np.sum(all_masks).astype(int)
    nb_dims = track_list[0].shape[1]
    track_array = np.zeros((nb_rows, nb_dims))
    frame_array = np.zeros((nb_rows,1))
    nb_states = state_preds.shape[-1]
    opt_metrics_array = np.zeros((nb_rows, len(opt_metrics.keys())))
    state_pred_array = np.zeros((nb_rows, nb_states))
    opt_colnames = list(opt_metrics.keys())
    track_ID_array = np.zeros((nb_rows, 1))
    idx = 0
    for i in range(len(track_list)):
        track_length = np.sum(all_masks[i]).astype(int)
        track_array[idx:idx+track_length] = track_list[i]
        frame_array[idx:idx+track_length] = frame_list[i][:,None]
        state_pred_array[idx:idx+track_length] = state_preds[i][all_masks[i].astype(bool)]
        track_ID_array[idx:idx+track_length] = track_ID_list[i][:,None]
        for j, opt_colname in enumerate(opt_colnames):
            opt_metrics_array[idx:idx+track_length, j] = opt_metrics[opt_colname][i]
        idx += track_length

    data = np.concatenate((track_array, frame_array, track_ID_array, state_pred_array, opt_metrics_array), axis = 1)
    state_names = []
    for s in range(nb_states-1):
        state_names.append('STATE_%s'%s)
    state_names.append('STATE_MISLABELED')
    columns = ['POSITION_X', 'POSITION_Y', 'POSITION_Z'][:nb_dims] + ['FRAME', 'TRACK_ID'] + state_names + opt_colnames

    data = pd.DataFrame(data, columns = columns)
    return data

def correct_state_predictions_padding(state_preds, all_masks, sequence_length):
    max_length = state_preds.shape[1]
    for i in range(len(state_preds)):
        current_mask = all_masks[i]
        track_length = np.sum(current_mask).astype(int)        
        if track_length <= sequence_length:
            state_preds[i, :track_length] =  state_preds[i, -track_length:]
        elif track_length<max_length:
            state_preds[i, track_length - sequence_length:track_length] =  state_preds[i, -sequence_length:]
        
        state_preds[i, track_length:] = 0

def RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2, coef_index, nb_dims = 1):
    '''
    Basic function of the method to simplify a product of two Gaussians that both depend on
    a hidden variable of index `coef_index` into one gaussian that depend on this variable
    and one Gaussian that is independent of this variable. 
    Here, the 2 Gaussians that depend on a linear combination of hidden variables are characterized
    by the coefficients associated with each hidden variables in the linear combination and a
    biais vector.
    
    Parameters
    ----------
    current_hidden_var_coefs_1 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the current hidden variables (time step i) for the first Gaussian.
    current_hidden_var_coefs_2 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the current hidden variables (time step i) for the second Gaussian.
    next_hidden_var_coefs_1 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the next hidden variables (time step i+1) for the first Gaussian.
    next_hidden_var_coefs_2 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the next hidden variables (time step i+1) for the second Gaussian.
        DESCRIPTION.
    biases_1 : tensor of shape (nb_tracks, nb_states, nb_dims)
        Biases of the first Gaussian. 
    biases_2 : tensor of shape (nb_tracks, nb_states, nb_dims)
        Biases of the second Gaussian.  
    coef_index : integer
        index of the hidden variable to simplify
    nb_dims : integer
        Number of independent dimensions (e.g. spatial dimensions when considering they do
        not influence each other).

    Returns
    -------
    LogConstant : tensor of shape (nb_tracks, nb_states)
        Log of a constant that needs to be added to the probability to ensure equality after
        the changes of variables.
    current_coefs3 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the first new Gaussian for the current step i (independent of the hidden variable of index coef_index).
    current_coefs4 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the second new Gaussian for the current step i.
    next_coefs3 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the first new Gaussian for the next step i+1.
    next_coefs4 : tensor of shape (nb_tracks, nb_states, nb_hidden_variables)
        coefficients of the second new Gaussian for the next step i+1.
    biases3 : tensor of shape (nb_tracks, nb_states, nb_dims).
        biases of the first new Gaussian.
    biases4 : tensor of shape (nb_tracks, nb_states, nb_dims).
        biases of the second new Gaussian.
    '''
    
    C1 = current_hidden_var_coefs_1[:,:, coef_index:coef_index+1] + tf.random.normal([1,1,1], 0, 1e-20, dtype = dtype)
    C2 = current_hidden_var_coefs_2[:,:, coef_index:coef_index+1] + tf.random.normal([1,1,1], 0, 1e-20, dtype = dtype)
    
    current_coefs1 = tf.math.divide_no_nan(current_hidden_var_coefs_1, C1)
    current_coefs2 = tf.math.divide_no_nan(current_hidden_var_coefs_2, C2)
    
    next_coefs1 = tf.math.divide_no_nan(next_hidden_var_coefs_1, C1)
    next_coefs2 = tf.math.divide_no_nan(next_hidden_var_coefs_2, C2)
    biases1 = tf.math.divide_no_nan(biases_1, C1[:,:])
    biases2 = tf.math.divide_no_nan(biases_2, C2[:,:])
    
    var1 = 1./(C1**2 + tf.random.normal([1,1,1], 0, 1e-100, dtype = dtype))
    var2 = 1./(C2**2 + tf.random.normal([1,1,1], 0, 1e-100, dtype = dtype))
    
    var3 = var1 + var2
    std3 = var3**0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3
    biases3 = (biases1 - biases2)/std3[:,:]
    
    var4 = var1 * var2 / var3
    std4 = var4**0.5
    current_coefs4 = (current_coefs1*var2 + current_coefs2*var1)/(var3*std4)
    next_coefs4 = (next_coefs1*var2 + next_coefs2*var1)/(var3*std4)
    
    biases4 = (biases1*var2[:,:] + biases2*var1[:,:])/(var3*std4)[:,:]
    
    LogConstant = -nb_dims*tf.math.log(tf.math.abs(C1*C2*std4*std3))[:,:,0]
    return LogConstant, current_coefs3, current_coefs4, next_coefs3, next_coefs4, biases3, biases4

'''
current_hidden_var_coefs = current_hidden_var_coefs_cp
next_hidden_var_coefs = next_hidden_var_coefs_cp
biases = biases_cp
kept_next_hidden_var_coefs = kept_next_hidden_var_coefs_cp
kept_biases = kept_biases_cp
'''
@tf.function(jit_compile=jit_compile)
def intermediate_RNN_function(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases, nb_dims):
    
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)
    
    current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2 = current_hidden_var_coefs_cp[ID_1], current_hidden_var_coefs_cp[ID_2], next_hidden_var_coefs_cp[ID_1], next_hidden_var_coefs_cp[ID_2], biases_cp[ID_1], biases_cp[ID_2]
    LogConstant, current_coefs3, current_coefs4, next_coefs3, next_coefs4, biases3, biases4 = RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2, coef_index, nb_dims)
    
    current_hidden_var_coefs_cp[ID_1] = tf.identity(current_coefs3)
    current_hidden_var_coefs_cp[ID_2] = tf.identity(current_coefs4)
    next_hidden_var_coefs_cp[ID_1] = tf.identity(next_coefs3)
    next_hidden_var_coefs_cp[ID_2] = tf.identity(next_coefs4)
    biases_cp[ID_1] = tf.identity(biases3)
    biases_cp[ID_2] = tf.identity(biases4)
    LC += LogConstant
    
    return tf.stack(current_hidden_var_coefs_cp), tf.stack(next_hidden_var_coefs_cp), tf.stack(biases_cp), LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases

@tf.function(jit_compile=jit_compile)
def final_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases, nb_dims):
    
    current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases = intermediate_RNN_function(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases, nb_dims)
    
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs_cp)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs_cp)
    biases_cp = tf.unstack(biases_cp)
    
    LC += - nb_dims * tf.math.log(tf.abs(current_hidden_var_coefs_cp[ID_2][:,:,coef_index])) # we must first normalize the integrated variable, log_gaussian(xs*coefs_matrix[2], 1) == log_gaussian(xs*coefs_matrix[2]/a, 1/a**2) - np.log(a)
    
    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    
    nb_gaussians += -1
    
    return tf.stack(current_hidden_var_coefs_cp), tf.stack(next_hidden_var_coefs_cp), tf.stack(biases_cp), LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases

@tf.function(jit_compile=jit_compile)
def no_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases, nb_dims):
    
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)
    
    LC += - nb_dims * tf.math.log(tf.abs(current_hidden_var_coefs_cp[ID_2][:,:,coef_index])) # we must first normalize the integrated variable, log_gaussian(xs*coefs_matrix[2], 1) == log_gaussian(xs*coefs_matrix[2]/a, 1/a**2) - np.log(a)
    
    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    
    nb_gaussians += -1
    
    biases_cp = tf.cast(tf.reshape(tf.stack(biases_cp), [len(biases_cp)]+biases.shape[1:]), dtype = dtype) # we need to explicitely assign the biase shape to avoid issues in `new_LCs = tf.reduce_sum(norm_log_gaussian(tf.cast(tf.stack(biases_cp), dtype = dtype)), axis = 3)` at the final step when the biase tensor is empty
    return tf.stack(current_hidden_var_coefs_cp), tf.stack(next_hidden_var_coefs_cp), biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases

@tf.function(jit_compile=jit_compile)
def final_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases, nb_dims):
    
    next_hidden_var_coefs_cp, current_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases = intermediate_RNN_function(next_hidden_var_coefs, current_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases, nb_dims)
    
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs_cp)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs_cp)
    biases_cp = tf.unstack(biases_cp)
    
    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)
    
    kept_next_hidden_var_coefs_cp = tf.unstack(kept_next_hidden_var_coefs)
    kept_biases_cp = tf.unstack(kept_biases)
    
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    
    nb_gaussians += -1
    
    return tf.stack(next_hidden_var_coefs_cp), tf.stack(current_hidden_var_coefs_cp), tf.stack(biases_cp), LC, nb_gaussians, tf.stack(kept_next_hidden_var_coefs_cp), tf.stack(kept_biases_cp)

@tf.function(jit_compile=jit_compile)
def no_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs, biases, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs, kept_biases, nb_dims):
    
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)
    
    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)
    
    kept_next_hidden_var_coefs_cp = tf.unstack(kept_next_hidden_var_coefs)
    kept_biases_cp = tf.unstack(kept_biases)
    
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    
    nb_gaussians += -1
    
    biases_cp = tf.cast(tf.reshape(tf.stack(biases_cp), [len(biases_cp)]+biases.shape[1:]), dtype = dtype) # we need to explicitely assign the biase shape to avoid issues in `new_LCs = tf.reduce_sum(norm_log_gaussian(tf.cast(tf.stack(biases_cp), dtype = dtype)), axis = 3)` at the final step when the biase tensor is empty
    
    return tf.stack(next_hidden_var_coefs_cp), current_hidden_var_coefs, biases_cp, LC, nb_gaussians, tf.stack(kept_next_hidden_var_coefs_cp), tf.stack(kept_biases_cp)


@tf.function(jit_compile=jit_compile)
def RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                           next_hidden_var_coefs,
                           biases,
                           sequence_phase_1,
                           sequence_phase_2,
                           nb_dims,
                           dtype = 'float64'): # False by default, set to true when aiming to compute the scaling factor
    '''
    We first integrate over the current hidden variables. To do so, we use RNN_gaussian_product
    to reduce the number of gaussians that depend on the current hidden variable to 1. Once this
    is done, we can simply remove the last gaussian.
    '''
    
    current_hidden_var_coefs_cp = tf.identity(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.identity(next_hidden_var_coefs)
    biases_cp = tf.identity(biases)
    
    kept_next_hidden_var_coefs_cp, kept_biases_cp = [[],[]]
    
    nb_gaussians = len(biases_cp)
    
    nb_hidden_variables = current_hidden_var_coefs_cp[0].shape[-1]
    
    LC = tf.constant(0, shape = current_hidden_var_coefs_cp[0].shape[:2], dtype = dtype)

    for f, s in zip(sequence_phase_1[0], sequence_phase_1[1]):
        print('1...')
        
        coef_index, ID_1, ID_2 = s
        current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp = f(current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp, nb_dims)

    '''
    Once the integration is done, all the current_hidden_var_coefs_cp are 0 and we 
    have nb_gaussians - nb_hidden_variables variables left. If that number is higher than 
    nb_hidden_variables, we have redundancies that we can eliminate. To eliminate them 
    we can perform RNN_gaussian_product inverting current_hidden_var_coefs_cp and next_hidden_var_coefs_cp
    on the nb_remaining_gaussians - nb_hidden_variables + 1 first gaussians to set the
    nb_remaining_gaussians - nb_hidden_variables next hidden variables to 0 and obtain 
    a final number of gaussians equal to nb_hidden_variables
    '''
    for f, s in zip(sequence_phase_2[0][:], sequence_phase_2[1][:]):
        print('2...')
        coef_index, ID_1, ID_2 = s
        next_hidden_var_coefs_cp, current_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp = f(next_hidden_var_coefs_cp, current_hidden_var_coefs_cp, biases_cp, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp, nb_dims)

    new_LCs = tf.reduce_sum(norm_log_gaussian(biases_cp), axis = 3)
    LC += tf.math.reduce_sum(new_LCs, 0)
    
    Next_coefs = tf.stack(kept_next_hidden_var_coefs_cp[::-1])
    Next_biases = tf.stack(kept_biases_cp[::-1])
    
    return Next_coefs, Next_biases, LC

@tf.function(jit_compile=jit_compile)
def transition_RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                           next_hidden_var_coefs,
                           biases,
                           transition_sequence,
                           nb_dims,
                           dtype = 'float64'): # False by default, set to true when aiming to compute the scaling factor
    '''
    We first integrate over the current hidden variables. To do so, we use RNN_gaussian_product
    to reduce the number of gaussians that depend on the current hidden variable to 1. Once this
    is done, we can simply remove the last gaussian.
    '''
    current_hidden_var_coefs_cp = tf.identity(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.identity(next_hidden_var_coefs)
    biases_cp = tf.identity(biases)
    
    kept_next_hidden_var_coefs_cp, kept_biases_cp = [[],[]]
    
    nb_gaussians = len(biases_cp)
    
    nb_hidden_variables = current_hidden_var_coefs_cp[0].shape[-1]
    
    LC = tf.constant(0, shape = current_hidden_var_coefs_cp[0].shape[:2], dtype = dtype)
    
    for f, s in zip(transition_sequence[0], transition_sequence[1]):
        print('1...')
        coef_index, ID_1, ID_2 = s
        current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp = f(current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp, nb_dims)

    Next_coefs = current_hidden_var_coefs_cp
    Next_biases = biases_cp
    
    return Next_coefs, Next_biases, LC

def get_all_sequences(sequence_length, nb_states):
    '''
    produces a matrix of the possible sequences of states
    '''
    Bs_ID = np.arange(nb_states**sequence_length)
    all_sequences = np.zeros((nb_states**sequence_length, sequence_length), int)
    
    for k in range(all_sequences.shape[1]):
        cur_row = np.mod(Bs_ID,nb_states**(k+1))
        Bs_ID = (Bs_ID - cur_row)
        all_sequences[:,k] = cur_row//nb_states**k
    all_sequences = all_sequences[:, ::-1]
    return all_sequences

class Initial_layer_constraints(tf.keras.layers.Layer):
    def __init__(
        self,
        nb_states,
        nb_gaussians,
        nb_obs_vars,
        nb_hidden_vars,
        params,
        initial_params,
        initial_fractions,
        max_linking_distance,
        constraint_function,
        vary_params = None,
        vary_initial_params = None,
        vary_initial_fractions = None,
        sequence_length = 3,
        **kwargs):
        
        super().__init__(**kwargs)
        
        dtype = self.dtype
        
        if type(vary_params) == type(None):
            vary_params = np.ones(params.shape, dtype = dtype)
        
        if type(vary_initial_params) == type(None):
            vary_initial_params = np.ones(initial_params.shape, dtype = dtype)
        
        if type(vary_initial_fractions) == type(None):
            vary_initial_fractions = np.ones(initial_fractions.shape, dtype = dtype)
        
        self.nb_states = nb_states
        self.nb_gaussians = nb_gaussians
        self.nb_obs_vars = nb_obs_vars
        self.nb_hidden_vars = nb_hidden_vars
        self.params = params
        self.initial_params = initial_params
        self.initial_fractions = initial_fractions
        self.constraint_function = constraint_function
        self.sequence_length = sequence_length
        self.max_linking_distance = max_linking_distance
        self.vary_params = vary_params
        self.vary_initial_params = vary_initial_params
        self.vary_initial_fractions = vary_initial_fractions
        
        initial_sequence_phase_1, initial_sequence_phase_2, recurrent_sequence_phase_1, recurrent_sequence_phase_2, final_sequence_phase_1, transition_sequence = get_sequences(params, initial_params, constraint_function, nb_gaussians, nb_hidden_vars, dtype)
        
        self.initial_sequence_phase_1 = initial_sequence_phase_1
        self.initial_sequence_phase_2 = initial_sequence_phase_2
        self.recurrent_sequence_phase_1 = recurrent_sequence_phase_1
        self.recurrent_sequence_phase_2 = recurrent_sequence_phase_2
        self.transition_sequence = transition_sequence
        self.final_sequence_phase_1 = final_sequence_phase_1
    
    def build(self, input_shape):
        dtype = self.dtype
        '''
        param_vars = tf.Variable(params,  dtype = dtype, name = 'recurrence_variables', constraint=lambda w: tf.where(tf.greater_equal(w, -1), w, 0.0000001))
        initial_param_vars = tf.Variable(initial_params,  dtype = dtype, name = 'initial_variables', constraint=lambda w: tf.where(tf.greater_equal(w, 0), w, 0.0000001))
        initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
        initial_fractions[0,-1] = -1
        '''
        self.param_vars = tf.Variable(self.params,  dtype = dtype, name = 'recurrence_variables')
        self.initial_param_vars = tf.Variable(self.initial_params,  dtype = dtype, name = 'initial_variables', trainable = True)
        self.max_linking_distance_param = tf.Variable(self.max_linking_distance, dtype = dtype, name = 'max linking distance', trainable = False)
        initial_fractions = self.initial_fractions
        self.initial_fractions = tf.Variable(initial_fractions, dtype = dtype, name = 'Fractions', trainable = True)
    
    def call(self, inputs):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        inputs = transposed_inputs
        '''
        
        nb_tracks = inputs.shape[2]
        nb_hidden_vars = self.nb_hidden_vars
        dtype = self.dtype
        constraint_function = self.constraint_function
        
        param_vars = self.param_vars
        initial_param_vars = self.initial_param_vars
        nb_states = self.nb_states
        max_linking_distance = self.max_linking_distance_param
        vary_params = self.vary_params
        vary_initial_params = self.vary_initial_params
        initial_fractions = tf.math.softmax(self.initial_fractions)
        vary_initial_fractions = self.vary_initial_fractions
        
        nb_dims = inputs.shape[-1]
        
        param_vars = vary_params * param_vars + (1 - vary_params) * tf.stop_gradient(param_vars)
        initial_param_vars = vary_initial_params * initial_param_vars + (1 - vary_initial_params) * tf.stop_gradient(initial_param_vars)
        initial_fractions = vary_initial_fractions * initial_fractions + (1 - vary_initial_fractions) * tf.stop_gradient(initial_fractions)
        
        # optional function taht can be modified to duplicate states
        param_vars, initial_param_vars, initial_fractions = self.duplicate_states(param_vars, initial_param_vars, initial_fractions)

        # We add the mislinking state:
        param_vars = tf.concat((param_vars, [[param_vars[-1][0], tf.math.log(tf.cast(max_linking_distance, dtype = dtype)), -15., tf.math.log(tf.cast(0.00001, dtype = dtype)), 0]]), axis = 0)
        initial_param_vars = tf.concat((initial_param_vars, [initial_param_vars[-1]]), axis = 0)
        nb_states = nb_states + 1 
        
        hidden_var_coefs, obs_var_coefs, Gaussian_stds, biases, initial_hidden_var_coefs, initial_obs_var_coefs, initial_Gaussian_stds, initial_biases, transition_hidden_var_coefs, transition_Gaussian_stds, transition_biases, integration_variable_index = constraint_function(param_vars, initial_param_vars, nb_dims, dtype)
        
        hidden_var_coefs = hidden_var_coefs/Gaussian_stds
        obs_var_coefs = obs_var_coefs/Gaussian_stds
        biases = biases/Gaussian_stds
        
        obs_var_coefs = tf.repeat(obs_var_coefs, nb_tracks, 1)
        hidden_var_coefs = tf.repeat(hidden_var_coefs, nb_tracks, 1)
        biases = tf.repeat(biases, nb_tracks, 1)
        
        current_hidden_var_coefs = hidden_var_coefs[:,:,:,:nb_hidden_vars]
        next_hidden_var_coefs = hidden_var_coefs[:,:,:,nb_hidden_vars:]
        
        reccurent_obs_var_coefs = tf.identity(obs_var_coefs)
        reccurent_hidden_var_coefs = tf.identity(current_hidden_var_coefs)
        reccurent_next_hidden_var_coefs = tf.identity(next_hidden_var_coefs)
        reccurent_biases = tf.identity(biases)
        
        # change of variables to deal with gaussians of variance 1
        initial_hidden_var_coefs = initial_hidden_var_coefs/initial_Gaussian_stds
        initial_obs_var_coefs = initial_obs_var_coefs/initial_Gaussian_stds
        initial_biases = initial_biases/initial_Gaussian_stds
        
        initial_obs_var_coefs = tf.repeat(initial_obs_var_coefs, nb_tracks, 1)
        initial_hidden_var_coefs = tf.repeat(initial_hidden_var_coefs, nb_tracks, 1)
        initial_biases = tf.repeat(initial_biases, nb_tracks, 1)
        
        current_initial_hidden_var_coefs = initial_hidden_var_coefs[:,:,:,:nb_hidden_vars]
        next_initial_hidden_var_coefs = tf.zeros((nb_hidden_vars, nb_tracks, nb_states, nb_hidden_vars), dtype = dtype)  # these coefs must equal 0 as the initial gaussians must only depend on the fist set of hidden states
        
        biases += tf.reduce_sum(obs_var_coefs[:,:,:,:,None] * inputs[0], -2)
        initial_biases += tf.reduce_sum(initial_obs_var_coefs[:,:,:,:,None] * inputs[0], -2)
        
        current_hidden_var_coefs = tf.concat((current_initial_hidden_var_coefs, current_hidden_var_coefs), axis = 0)
        next_hidden_var_coefs =  tf.concat((next_initial_hidden_var_coefs, next_hidden_var_coefs), axis = 0)
        biases = tf.concat((initial_biases, biases), axis = 0)
        
        sequence_length = self.sequence_length
        
        current_hidden_var_coefs = tf.concat([current_hidden_var_coefs]*sequence_length, axis = 2)
        next_hidden_var_coefs = tf.concat([next_hidden_var_coefs]*sequence_length, axis = 2)
        biases = tf.concat([biases]*sequence_length, axis = 2)
        
        sequence_phase_1 = self.initial_sequence_phase_1
        sequence_phase_2 = self.initial_sequence_phase_2
        
        Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                             next_hidden_var_coefs,
                                                             biases,
                                                             sequence_phase_1,
                                                             sequence_phase_2,
                                                             nb_dims,
                                                             dtype = dtype)
        
        current_hidden_var_coefs[:, 0]
        
        transition_hidden_var_coefs = transition_hidden_var_coefs/transition_Gaussian_stds
        transition_biases = transition_biases/transition_Gaussian_stds[:,:,:]
        
        transition_hidden_var_coefs = tf.repeat(transition_hidden_var_coefs, nb_tracks, 1)
        transition_biases = tf.repeat(transition_biases, nb_tracks, 1)
        
        transition_hidden_var_coefs = tf.concat([transition_hidden_var_coefs]*sequence_length*nb_states, 2)
        transition_biases = tf.concat([transition_biases] * nb_states * sequence_length, 2)
        
        initial_Log_factors, Log_factors, transition_Log_factors = self.compute_scaling_factors(param_vars, initial_param_vars)
        
        init_log_fractions = tf.concat([tf.math.log(initial_fractions)]*sequence_length, axis = 1)
        init_log_factors = tf.concat([nb_dims*initial_Log_factors[None]]*sequence_length, axis = 1)
        
        LP = LC + init_log_factors + init_log_fractions + tf.math.log(np.array(1/sequence_length))
        
        Log_factors = nb_dims * Log_factors
        transition_Log_factors = nb_dims * transition_Log_factors
        initial_states = [Next_coefs, Next_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases]
  
        return inputs, initial_states

    def compute_scaling_factors(self, param_vars, initial_param_vars):
        '''
        initial log factors 
        '''
        
        Log_factors = - param_vars[:,0] - param_vars[:,1] - param_vars[:,3]
        
        state_mask = tf.cast(param_vars[:,4]>0.5, dtype = dtype)
        
        initial_anomalous_factor = (- param_vars[:,1] + 0.5*tf.math.log(2*tf.math.sigmoid(param_vars[:,2])))*(1.-state_mask) - param_vars[:,2]*state_mask
        
        initial_Log_factors = Log_factors - initial_param_vars[:,0] + initial_anomalous_factor
        
        transition_Log_factors = Log_factors + initial_anomalous_factor
        
        transition_Log_factors = transition_Log_factors + tf.constant([0]*(transition_Log_factors.shape[0]-1)+[np.log(1.)], dtype = dtype)
        
        return initial_Log_factors, Log_factors, transition_Log_factors

    def duplicate_states(self, param_vars, initial_param_vars, initial_fractions):
        '''
        additional function that can be modified to enable several states to share the same parameters 
        '''
        return param_vars, initial_param_vars, initial_fractions


@tf.function(jit_compile=False)
def RNN_cell(input_i, Prev_coefs, Prev_biases, LP, segment_len, reshaped_Log_factors, reshaped_transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, sequence_phase_1, sequence_phase_2, transition_mask, transition_sequence, transition_mean, transition_var, gamma_dist_mean, gamma_dist_var, states):
    print('LP',LP)
    '''
    First we compute the additional likelihood after integration over the last hidden 
    states for the sequences that transition. Do to so, we make all the previous sequences 
    transition and fuse them into `nb_states` sequences (the fist of the next)
    '''
    
    current_states = states[:,:,-1:]
    
    Prev_coefs[:,0, -1]
    nb_dims = input_i.shape[-1]
    nb_tracks = LP.shape[0]
    nb_hidden_vars = Prev_coefs.shape[3]
    nb_states = reccurent_hidden_var_coefs.shape[2]
    sequence_length = LP.shape[1]//nb_states
    
    Prev_coefs2 = tf.repeat(Prev_coefs, nb_states, axis = 2)
    Prev_biases2 =  tf.repeat(Prev_biases, nb_states, axis = 2)
    LP2 = tf.repeat(LP, nb_states, axis = 1)
    segment_len = tf.repeat(segment_len, nb_states, axis = 1)
    
    alternative_Prev_coefs = tf.concat((Prev_coefs2, tf.identity(transition_hidden_var_coefs)), axis = 0)
    alternative_Prev_biases = tf.concat((Prev_biases2, tf.identity(transition_biases)), axis = 0)
    alternative_Prev_coefs[:,0,-1]
    nb_dims = input_i.shape[-1]
    
    transition_Prev_coefs, transition_Prev_biases, LC = transition_RNN_reccurence_formula(current_hidden_var_coefs = alternative_Prev_coefs, # coefficients of the hidden variables that are updated
                                                                         next_hidden_var_coefs = tf.constant(0, dtype = dtype, shape =  alternative_Prev_coefs.shape),
                                                                         biases = alternative_Prev_biases,
                                                                         transition_sequence = transition_sequence,
                                                                         nb_dims = nb_dims,
                                                                         dtype = dtype)
    transition_Prev_coefs[:, 0, 0]
    LP2 += LC*transition_mask + reshaped_Log_factors
    
    current_shapes = gamma_dist_mean**2/gamma_dist_var
    current_rates = gamma_dist_mean/gamma_dist_var
    
    all_Prev_coefs = transition_Prev_coefs*transition_mask[None,:,:,None] + Prev_coefs2*(1-transition_mask[None,:,:,None])
    all_prev_biases = transition_Prev_biases*transition_mask[None,:,:,None] + Prev_biases2*(1-transition_mask[None,:,:,None])
    # A : transition at time step k, B : no transition at time step k-1 
    # P(A|B) = P(AB)/P(B), if A, B is necessarly verified  
    # Here the probability to consider is the probability to transition given that it did not transition yet 
    # to compute the proba to not transition, we must compute 1 - the probas to transition 
    
    transition_probas = tf.clip_by_value((tf.compat.v1.distributions.Gamma(current_shapes, current_rates).prob(segment_len[:,:]+0.5)+1e-14)/(1-tf.compat.v1.distributions.Gamma(current_shapes, current_rates).cdf(segment_len[:,:]+0.5)+1e-12), clip_value_min=1-20, clip_value_max=1-1e-10) #*segment_len_certainty + transition_rates*(1-segment_len_certainty)
    non_transition_probas = tf.repeat(1-tf.clip_by_value(tf.reduce_sum(tf.reshape(transition_probas*transition_mask, shape = (nb_tracks, nb_states*sequence_length, nb_states)), axis = 2), clip_value_min=1-20, clip_value_max=1-1e-10), nb_states, axis = 1) # this will be useful when we focus on the non-transitioning sequences
    
    transition_probas = transition_probas*transition_mask + non_transition_probas*(1-transition_mask)
    # Then, we update the log probability  
    all_LP = LP2 + tf.math.log(transition_probas) # + reshaped_transition_Log_factors
    '''
    Once we performed transitions on the current states, we want to reshape transition_Prev_coefs,
    transition_Prev_biases and transition_LP to have a shape (nb_tracks, nb_states) instead of
    (nb_tracks, sequence_length*nb_states, nb_states). To do so, we can perform a weighted average
    of the a priori probabilities including LC and integrating over the remaining hidden variables
    to balance.
    Integration: here there is only one Gaussian and one variable so this is easy but when more gaussians
    (and hidden variables) remain, make sure these are triangular and normalize by the determinant (product
    of the diagonal elements).
    '''
    
    '''
    Then we concatenate the transitioned sequences to the previous sequences that will be continued
    '''
    
    current_reccurent_obs_var_coefs = tf.concat([reccurent_obs_var_coefs]*(sequence_length*nb_states), axis = 2)
    current_reccurent_hidden_var_coefs = tf.concat([reccurent_hidden_var_coefs]*(sequence_length*nb_states), axis = 2)
    current_reccurent_next_hidden_var_coefs = tf.concat([reccurent_next_hidden_var_coefs]*(sequence_length*nb_states), axis = 2)
    current_reccurent_biases = tf.concat([reccurent_biases]*(sequence_length*nb_states), axis = 2)
    current_reccurent_hidden_var_coefs[1, 0]
    current_hidden_var_coefs = tf.concat((all_Prev_coefs, tf.identity(current_reccurent_hidden_var_coefs)), axis = 0)
    zero_tensor = tf.constant(0, dtype = dtype, shape = all_Prev_coefs.shape)
    next_hidden_var_coefs = tf.concat((zero_tensor, tf.identity(current_reccurent_next_hidden_var_coefs)), axis = 0)
    current_biases = tf.identity(current_reccurent_biases)
    current_biases += tf.reduce_sum(current_reccurent_obs_var_coefs[:,:,:,:,None] * input_i, (-2))
    biases = tf.concat((all_prev_biases, current_biases), axis =  0)
    
    Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                         next_hidden_var_coefs,
                                                         biases,
                                                         sequence_phase_1,
                                                         sequence_phase_2,
                                                         nb_dims = nb_dims,
                                                         #dependent_variables, # bool array that memorized if coefficients are non-nul 
                                                         #nb_hidden_vars, # number of hidden variables to integrate during this step
                                                         #nb_gaussians, # number of gaussians, must equal hidden_vars_coefs.shape[0]
                                                         dtype = dtype)
    
    all_LP += LC
    
    reshaped_Next_coefs = tf.reshape(Next_coefs, Next_coefs.shape[:2]+[sequence_length*nb_states, nb_states, nb_hidden_vars])
    transition_LPs = tf.reshape(all_LP - 200*(1-transition_mask), (nb_tracks, sequence_length*nb_states, nb_states)) - nb_dims*tf.math.log(tf.math.abs(reshaped_Next_coefs[0, :,:,:,0]*reshaped_Next_coefs[1, :,:,:,1])+1e-20)
    
    max_transition_LPs = tf.reduce_max(transition_LPs, axis = 1, keepdims = True)
    transition_Ps = tf.math.exp(transition_LPs - max_transition_LPs) 
    transition_weights = transition_Ps # We weight by accounting for the determinant of the next coefs (integration over the next coefs)
    transition_weights = transition_weights / tf.reduce_sum(transition_weights, 1, keepdims = True)
    
    # Update the states
    transition_states = tf.reduce_sum(states[:,:,None] * transition_weights[:,:,:,None, None], 1)
    
    transition_Next_coefs = tf.reshape(Next_coefs, Next_coefs.shape[:2]+[sequence_length*nb_states, nb_states, nb_hidden_vars])
    transition_Next_coefs = tf.reduce_sum(transition_Next_coefs*transition_weights[None, :,:,:,None], axis = 2)
    
    transition_Next_biases = tf.reshape(Next_biases, Next_biases.shape[:2]+[sequence_length*nb_states, nb_states, nb_dims])
    transition_Next_biases = tf.reduce_sum(transition_Next_biases*transition_weights[None, :,:,:,None], axis = 2)
    
    transition_LPs = tf.math.log(tf.reduce_sum(transition_Ps, axis = 1)) + max_transition_LPs[:,0] + nb_dims*tf.math.log(tf.math.abs(transition_Next_coefs[0, :,:,0]*transition_Next_coefs[1, :,:,1])+1e-20)
    
    stable_LPs = tf.reshape(all_LP, (nb_tracks, sequence_length* nb_states, nb_states))
    stable_weights = tf.reshape((1-transition_mask), (sequence_length* nb_states, nb_states))[None]
    stable_LPs = tf.reduce_sum(stable_LPs * stable_weights, 2)
    
    stable_states = tf.reduce_sum(states[:,:,None] * stable_weights[:,:,:,None, None], 2)
    
    stable_Next_coefs = tf.reduce_sum(tf.reshape(Next_coefs, Next_coefs.shape[:2]+[sequence_length*nb_states, nb_states, nb_hidden_vars])*stable_weights[None,:,:,:,None], axis = 3)
    stable_Next_biases = tf.reduce_sum(tf.reshape(Next_biases, Next_biases.shape[:2]+[sequence_length*nb_states, nb_states, nb_dims])*stable_weights[None,:,:,:,None], axis = 3)
    stable_segment_len = tf.reduce_sum(tf.reshape(segment_len, (nb_tracks, sequence_length*nb_states, nb_states))*stable_weights, axis = 2)
    
    current_gamma_dist_mean = tf.concat([transition_mean, gamma_dist_mean], axis = 1)
    current_gamma_dist_var = tf.concat([transition_var, gamma_dist_var], axis = 1)
    
    Next_coefs = tf.concat([transition_Next_coefs, stable_Next_coefs], axis = 2)
    Next_biases = tf.concat([transition_Next_biases, stable_Next_biases], axis = 2)
    new_LP = tf.concat([transition_LPs, stable_LPs], axis = 1)
    current_segment_len = tf.concat([tf.ones((nb_tracks, nb_states), dtype = dtype), stable_segment_len+1], axis = 1)
    Next_states = tf.concat([transition_states, stable_states], axis = 1)
    
    '''
    now, the `nb_states` last sequences must be fused with the previous sequences to
    keep the number of sequences to `sequence_length`
    '''
    
    saved_Next_coefs = Next_coefs[:, :, :-nb_states*2]
    saved_Next_biases = Next_biases[:, :, :-nb_states*2]
    saved_LP = new_LP[:, :-nb_states*2]
    saved_segment_len = current_segment_len[:, :-nb_states*2]
    saved_gamma_dist_mean = current_gamma_dist_mean[:, :-nb_states**2*2]
    saved_gamma_dist_var = current_gamma_dist_var[:, :-nb_states**2*2]
    saved_states = Next_states[:, :-nb_states*2]
    
    nb_prev_gaussians = Next_coefs.shape[0]
    last_Next_coefs = tf.reshape(Next_coefs[:, :, -nb_states*2:], (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_hidden_vars))
    last_Next_biases = tf.reshape(Next_biases[:, :, -nb_states*2:], (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_dims))
    last_LP = tf.reshape(new_LP[:, -nb_states*2:], (nb_tracks, 2, nb_states)) - nb_dims*tf.math.log(tf.math.abs(last_Next_coefs[0, :,:,:,0]*last_Next_coefs[1, :,:,:,1])+1e-20)
    last_segment_len = tf.reshape(current_segment_len[:, -nb_states*2:], (nb_tracks, 2, nb_states))
    last_gamma_dist_mean = tf.reshape(current_gamma_dist_mean[:, -nb_states**2*2:], (nb_tracks, 2, nb_states, nb_states))
    last_gamma_dist_var = tf.reshape(current_gamma_dist_var[:, -nb_states**2*2:], (nb_tracks, 2, nb_states, nb_states))
    last_states = tf.reshape(Next_states[:, -nb_states*2:], (nb_tracks, 2, nb_states, sequence_length, nb_states))
    
    last_LP_max = tf.reduce_max(last_LP, axis = 1, keepdims = True)
    last_P = tf.math.exp(last_LP - last_LP_max)
    sum_last_P = tf.reduce_sum(last_P, 1, keepdims = True)
    
    weight_last_LP = last_LP #- nb_dims * tf.math.log(tf.abs(last_Next_coefs[0, :,:,:, 0]*last_Next_coefs[1, :,:,:, 1])+1e-20)
    weight_last_P = tf.math.exp(weight_last_LP - tf.reduce_max(weight_last_LP, axis = 1, keepdims = True))
    last_weights = weight_last_P / tf.reduce_sum(weight_last_P, 1, keepdims = True)
    
    reduced_last_Next_coefs = tf.reduce_sum(last_Next_coefs*last_weights[None,:,:,:,None], axis = 2)
    reduced_last_Next_biases = tf.reduce_sum(last_Next_biases*last_weights[None,:,:,:, None], axis = 2)
    reduced_last_LPs = (tf.math.log(sum_last_P + 1e-100) + last_LP_max)[:,0] + nb_dims*tf.math.log(tf.math.abs(reduced_last_Next_coefs[0,:,:,0]*reduced_last_Next_coefs[1,:,:,1])+1e-20)
    reduced_last_segment_len = tf.reduce_sum(last_segment_len*last_weights, axis = 1)
    reduced_last_gamma_dist_mean = tf.reduce_sum(last_gamma_dist_mean*last_weights[:,:,:,None], axis = 1)
    reduced_last_gamma_dist_var = tf.reduce_sum((last_gamma_dist_var + (last_gamma_dist_mean - reduced_last_gamma_dist_mean[:,None])**2)*last_weights[:,:,:,None], axis = 1)
    reduced_last_gamma_dist_mean = tf.reshape(reduced_last_gamma_dist_mean, (nb_tracks, nb_states**2))
    reduced_last_gamma_dist_var = tf.reshape(reduced_last_gamma_dist_var, (nb_tracks, nb_states**2))
    reduced_last_states = tf.reduce_sum(last_states*last_weights[:,:,:,None, None], axis = 1)
    
    new_Next_coefs = tf.concat((saved_Next_coefs, reduced_last_Next_coefs), axis = 2)
    new_Next_biases = tf.concat((saved_Next_biases, reduced_last_Next_biases), axis = 2)
    new_LPs = tf.concat((saved_LP, reduced_last_LPs), axis = 1)
    new_segment_len = tf.concat((saved_segment_len, reduced_last_segment_len), axis = 1)
    new_gamma_dist_mean = tf.concat((saved_gamma_dist_mean, reduced_last_gamma_dist_mean), axis = 1)
    new_gamma_dist_var = tf.concat((saved_gamma_dist_var, reduced_last_gamma_dist_var), axis = 1)
    new_states = tf.concat((saved_states, reduced_last_states), axis = 1)

    new_states = tf.concat((new_states, current_states), axis = 2)[:,:,1:] # we update the states with the known current states (according to our transition pattern)
    
    return new_Next_coefs, new_Next_biases, new_LPs, new_segment_len, new_gamma_dist_mean, new_gamma_dist_var, new_states


'''
sequence_phase_1 = recurrent_sequence_phase_1
sequence_phase_2 = recurrent_sequence_phase_2
density = 0.001
inputs = sliced_inputs
mask = sliced_mask
Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states
nb_states = 2
inputs[:,:,2]
'''

class Custom_RNN_layer(tf.keras.layers.Layer):
    
    def __init__(self, nb_tracks, transition_shapes, transition_rates, density, nb_states, sequence_phase_1, sequence_phase_2, transition_sequence, transition_param_function, sequence_length = 3, vary_transition_shapes = None, vary_transition_rates = None, **kwargs):
        
        if type(vary_transition_rates) == type(None):
            vary_transition_rates = tf.ones(transition_rates.shape, dtype = dtype)
        
        if type(vary_transition_shapes) == type(None):
            vary_transition_shapes = tf.ones(transition_shapes.shape, dtype = dtype)
        
        self.sequence_phase_1 = sequence_phase_1
        self.sequence_phase_2 = sequence_phase_2
        self.transition_sequence = transition_sequence
        self.nb_states = nb_states + 1
        self.sequence_length = sequence_length
        self.nb_tracks = nb_tracks
        self.initial_transition_params = [transition_shapes, transition_rates]
        self.transition_param_function = transition_param_function
        self.density = density
        self.vary_transition_shapes = vary_transition_shapes
        self.vary_transition_rates = vary_transition_rates
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        nb_states = self.nb_states
        transition_shapes, transition_rates = self.initial_transition_params
        sequence_length = self.sequence_length
        self.transition_rates = tf.Variable(transition_rates, dtype = dtype, name = 'Transition rates', trainable = True)
        self.transition_shapes = tf.Variable(transition_shapes, dtype = dtype, name = 'Transition shape', trainable = True, constraint=lambda w: tf.where(tf.greater_equal(w, 0), w, 0.0001))
        
        indices = tf.stack([tf.repeat(tf.constant(list(np.arange(nb_states))*sequence_length), nb_states), tf.concat([tf.range(nb_states)]*nb_states*sequence_length, 0)], axis = 1)
        transition_mask = tf.cast((indices[:,0] - indices[:,1])!=0, dtype = dtype)[None]
        self.indices = indices
        self.transition_mask = transition_mask

        self.built = True
    
    @tf.function(jit_compile=False)
    def call(self, inputs, mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir): # inputs = current positions, states = outputs of the previous layer, needs to be initialized correctly
        
        nb_tracks = self.nb_tracks
        sequence_phase_1 = self.sequence_phase_1
        sequence_phase_2 = self.sequence_phase_2
        transition_sequence = self.transition_sequence
        transition_rates = self.transition_rates
        transition_shapes = self.transition_shapes
        transition_mask = self.transition_mask
        nb_states = self.nb_states
        indices = self.indices
        sequence_length = self.sequence_length
        density = self.density
        vary_transition_shapes = self.vary_transition_shapes
        vary_transition_rates = self.vary_transition_rates
        
        ds = tf.math.exp(log_ds)
        Fs = tf.math.softmax(softmax_inv_Fractions[0,:-1])
        effective_ds = ds + 2 * tf.math.exp(anomalous_factors) * isdir  # the coefficient 2 is rationalized by the fact that the directed motion is underestimated by a factor 2**0.5 and to which I add another factor 2**0.5 to account for the fact that directed motion is more exploratory than brownian motion
        
        transition_shapes = vary_transition_shapes * transition_shapes + (1 - vary_transition_shapes) * tf.stop_gradient(transition_shapes)
        transition_rates = vary_transition_rates * transition_rates + (1 - vary_transition_rates) * tf.stop_gradient(transition_rates)
        
        transition_shapes, transition_rates = transition_param_function(transition_shapes, transition_rates, density, Fs, effective_ds, dtype)
        
        reshaped_transition_Log_factors = tf.gather(transition_Log_factors, indices = indices[:, 1])[None]
        reshaped_Log_factors = tf.gather(Log_factors, indices = indices[:, 0])[None]
        
        reshaped_Log_factors = reshaped_transition_Log_factors*transition_mask + reshaped_Log_factors*(1-transition_mask)
        
        transition_rates = tf.gather_nd(transition_rates, indices = indices)[None]
        transition_shapes = tf.gather_nd(transition_shapes, indices = indices)[None]
        
        segment_len = tf.ones((nb_tracks, sequence_length*nb_states), dtype = dtype)
        transition_mean = tf.repeat(transition_shapes/transition_rates, nb_tracks, axis=0)[:, :nb_states**2]
        transition_var = tf.repeat(transition_shapes/transition_rates**2, nb_tracks, axis=0)[:, :nb_states**2]
        gamma_dist_mean = tf.repeat(transition_shapes/transition_rates, nb_tracks, axis=0) # We initialize the transition 
        gamma_dist_var = tf.repeat((transition_shapes/transition_rates)**2, nb_tracks, axis=0)
        
        # state vector of dims : nb_tracks, nb_states (+1 for the transition state)*sequence length, sequence length, nb_states
        states = tf.range(0, nb_states*sequence_length, dtype = 'int32')%nb_states
        states = tf.repeat(states[:,None], sequence_length, axis = 1)
        states = tf.repeat(tf.one_hot(states, nb_states, dtype = dtype)[None], nb_tracks, axis=0)
        states.shape
        All_states = tf.zeros((nb_tracks, 0, nb_states), dtype = dtype)
        nb_dims = reccurent_biases.shape[3]
        
        for i in range(inputs.shape[0]):
            
            # Save the states
            # when i == sequence length, we start erasing the information on the previous states. To alleviate this, we save the estimated states
            log_weigths = LP - nb_dims * tf.math.log(tf.math.abs(Prev_coefs[0, :,:, 0]*Prev_coefs[1, :,:, 1]))
            max_log_weigths = tf.reduce_max(log_weigths, 1, keepdims = True)
            weights = tf.math.exp(log_weigths - max_log_weigths)
            weights = weights/tf.reduce_sum(weights, 1, keepdims = True)
            pred_states = tf.reduce_sum(weights[:,:,None]*states[:,:, 0], 1, keepdims = True)
            
            All_states = tf.concat((All_states, pred_states), axis = 1)
            
            input_i = inputs[i]
            mask_i = mask[:,i]
            Next_coefs, Next_biases, Next_LP, Next_segment_len, Next_gamma_dist_mean, Next_gamma_dist_var, Next_states = RNN_cell(input_i, Prev_coefs, Prev_biases, LP, segment_len, reshaped_Log_factors, reshaped_transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, sequence_phase_1, sequence_phase_2, transition_mask, transition_sequence, transition_mean, transition_var, gamma_dist_mean, gamma_dist_var, states)
            Prev_coefs = Next_coefs*mask_i[None, :, None, None] + Prev_coefs*(1-mask_i[None, :, None, None])
            Prev_biases = Next_biases*mask_i[None, :, None, None] + Prev_biases*(1-mask_i[None, :, None, None])
            LP = Next_LP*mask_i[:, None] + LP*(1-mask_i[:, None])
            print(i, LP - nb_dims * tf.math.log(tf.math.abs(Next_coefs[0, :,:, 0]*Next_coefs[1, :,:, 1])))
            segment_len = Next_segment_len*mask_i[:,None] + segment_len*(1-mask_i[:,None])
            print(segment_len)
            gamma_dist_mean = Next_gamma_dist_mean*mask_i[:, None] + gamma_dist_mean*(1-mask_i[:,None])
            gamma_dist_var = Next_gamma_dist_var*mask_i[:,None] + gamma_dist_var*(1-mask_i[:,None])
            states = Next_states * mask_i[:,None,None,None] + states * (1-mask_i[:,None,None,None])
            print(i, LP - nb_dims * tf.math.log(tf.math.abs(Next_coefs[0, :,:, 0]*Next_coefs[1, :,:, 1])))
            
            
        All_states = All_states[:,sequence_length-1:]
        
        return Prev_coefs, Prev_biases, LP, All_states, states  #states

'''
Next_LP[7]
(LP - nb_dims * tf.math.log(tf.math.abs(Next_coefs[0, :,:, 0]*Next_coefs[1, :,:, 1])))[5]
'''
# sequence_phase_1 = final_sequence_phase_1
# sequence_phase_2 = [[], []]

class Final_layer(tf.keras.layers.Layer):
    def __init__(self, sequence_phase_1, nb_dims, sequence_length, **kwargs):
        self.sequence_phase_1 = sequence_phase_1
        self.nb_dims = nb_dims
        self.sequence_length = sequence_length
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
   
    @tf.function(jit_compile=False)
    def call(self, states):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        '''
        nb_dims = self.nb_dims
        Prev_coefs, Prev_biases, LP, All_states, last_states = states
        
        if Prev_coefs.shape[0]>0:
            
            current_hidden_var_coefs = Prev_coefs
            zero_tensor = tf.constant(0, dtype = dtype, shape =  Prev_coefs.shape)
            next_hidden_var_coefs = zero_tensor
            
            biases = Prev_biases
            
            Next_coefs, Next_biases, LC = RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                                                                 next_hidden_var_coefs,
                                                                 biases,
                                                                 self.sequence_phase_1,
                                                                 [[], []],
                                                                 nb_dims = nb_dims,
                                                                 dtype = self.dtype)
            LP += LC
        
        log_weigths = LP
        max_log_weigths = tf.reduce_max(log_weigths, 1, keepdims = True)
        weights = tf.math.exp(log_weigths - max_log_weigths)
        weights = weights/tf.reduce_sum(weights, 1, keepdims = True)
        pred_states = tf.reduce_sum(weights[:,:,None, None]*last_states, 1)
        All_states = tf.concat((All_states, pred_states), axis = 1)
        output = LP
        
        return output, All_states

class transpose_layer(tf.keras.layers.Layer):
    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
   
    def call(self, x, perm):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        '''
        return tf.transpose(x, perm = perm)

def simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2):
    '''
    simplification of RNN_gaussian_product for the function get_sequences
    '''
    
    current_coefs1 = current_hidden_var_coefs_1 / C1
    current_coefs2 = current_hidden_var_coefs_2 / C2
    next_coefs1 = next_hidden_var_coefs_1 / C1
    next_coefs2 = next_hidden_var_coefs_2 / C2
    
    var1 = 1./C1**2
    var2 = 1./C2**2
    
    var3 = var1 + var2
    std3 = var3**0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3
    
    var4 = var1 * var2 / var3
    std4 = var4**0.5
    current_coefs4 = (current_coefs1*var2 + current_coefs2*var1)/(var3*std4)
    next_coefs4 = (next_coefs1*var2 + next_coefs2*var1)/(var3*std4)
    
    return current_coefs3, current_coefs4, next_coefs3, next_coefs4

def get_sequences(params, initial_params, constraint_function, nb_gaussians, nb_hidden_vars, dtype):
    '''
    Function that gets the sequences of integration of the hidden variables: Determined by the indexes to eliminate the coefficents and perform the recursive integration process
    
    The integration process for one time step is composed of 2 phases, phase 1: integration over the current hidden variables, phase 2 rearangement of the matrix of the remaining next hidden variables to minimize the number of gaussians that are dependent on the next hidden variables
    
    In the process, we need to get 2 sequences (for phases 1 and 2) that specify the operations for the initial step, 2 additional sequences (phases 1 and 2) for the recurrence step and 1 final sequence (phase 1) for the last step.
    Each sequence must inform about the coefficient to integrate, the gaussian IDs and the function to use.
    
    The function then needs to compute and return 6 lists : [initial_functions_phase_1, np.array(initial_sequence_phase_1, dtype = 'int32')], [initial_functions_phase_2, np.array(initial_sequence_phase_2, dtype = 'int32')], [recurrent_functions_phase_1, np.array(recurrent_sequence_phase_1, dtype = 'int32')], [recurrent_functions_phase_2, np.array(recurrent_sequence_phase_2, dtype = 'int32')], [final_functions_phase_1, np.array(final_sequence_phase_1, dtype = 'int32')]
    [initial_functions_phase_1, np.array(initial_sequence_phase_1, dtype = 'int32')] : sequence to apply for the phase 1 of the inital step
    [initial_functions_phase_2, np.array(initial_sequence_phase_2, dtype = 'int32')] : sequence to apply for the phase 2 of the inital step
    [recurrent_functions_phase_1, np.array(recurrent_sequence_phase_1, dtype = 'int32')] : sequence to apply for the phase 1 of the recurrence step
    [recurrent_functions_phase_2, np.array(recurrent_sequence_phase_2, dtype = 'int32')] : sequence to apply for the phase 2 of the recurrence step
    [final_functions_phase_1, np.array(final_sequence_phase_1, dtype = 'int32')] : sequence to apply for the phase 1 of the recurrence step, the last step has no phase 2
    '''
    nb_dims = 1
    hidden_var_coefs, _, _, _, initial_hidden_var_coefs, _, _, _,  transition_hidden_var_coefs, _, _, integration_variable_index = constraint_function(params, initial_params, nb_dims, dtype)

    recurrent_current_hidden_var_coefs = np.copy(hidden_var_coefs[:,0,0,:nb_hidden_vars])
    recurrent_next_hidden_var_coefs = np.copy(hidden_var_coefs[:,0,0,nb_hidden_vars:])
    
    current_hidden_var_coefs = hidden_var_coefs[:,0,0,:nb_hidden_vars]
    next_hidden_var_coefs = hidden_var_coefs[:,0,0,nb_hidden_vars:]
            
    current_initial_hidden_var_coefs = initial_hidden_var_coefs[:,0,0,:nb_hidden_vars]
    next_initial_hidden_var_coefs = tf.zeros((nb_hidden_vars, nb_hidden_vars), dtype = dtype) # these coefs must equal 0 as the initial gaussians must only depend on the fist set of hidden states
    
    current_hidden_var_coefs = np.concatenate((current_initial_hidden_var_coefs, current_hidden_var_coefs), axis = 0)
    next_hidden_var_coefs = np.concatenate((next_initial_hidden_var_coefs, next_hidden_var_coefs), axis = 0)
    
    current_nb_gaussians = len(current_hidden_var_coefs)
    
    '''
    Initial step:
    '''
    
    initial_sequence_phase_1 = [] # list of lists containing the sequence of coef_index and gaussian IDs to
    initial_functions_phase_1 = []
    
    for coef_index in np.arange(nb_hidden_vars-1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            initial_sequence_phase_1.append([coef_index, ID_1, ID_2])
            initial_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = current_hidden_var_coefs[ID_1]
            current_hidden_var_coefs_2 = current_hidden_var_coefs[ID_2]
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]
            
            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2)
            
            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4
            
        if len(non_zero_gaussian_IDs)>1:
            initial_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs)==1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            initial_sequence_phase_1.append([coef_index, ID_1, ID_2])
            initial_functions_phase_1.append(no_RNN_function_phase_1)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass
        
        if len(non_zero_gaussian_IDs)>=1:
            current_hidden_var_coefs = np.delete(current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1
    
    initial_sequence_phase_2 = []
    initial_functions_phase_2 = []
    
    saved_Gaussians = np.zeros((nb_hidden_vars, nb_hidden_vars))
    # contrary to the integration step, we cannot remove Gaussians. Instead, we will save them to solve the linear problem
    for coef_index in np.arange(nb_hidden_vars-1, -1, -1):
        
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = next_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs)-1):
            print(i)
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            initial_sequence_phase_2.append([coef_index, ID_1, ID_2])
            initial_functions_phase_2.append(intermediate_RNN_function)
            
            C1 = next_hidden_var_coefs[ID_1, coef_index]
            C2 = next_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]*0
            current_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]*0
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]
            
            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, current_hidden_var_coefs_1, current_hidden_var_coefs_2)
            
            next_hidden_var_coefs[ID_1] = current_coefs3
            next_hidden_var_coefs[ID_2] = current_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            initial_functions_phase_2[-1] = final_RNN_function_phase_2
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            initial_sequence_phase_2.append([coef_index, ID_1, ID_2])
            initial_functions_phase_2.append(no_RNN_function_phase_2)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: 
            saved_Gaussians[coef_index] = next_hidden_var_coefs[ID_2]
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, ID_2, 0)
            current_nb_gaussians += -1
    
    initial_saved_Gaussians = saved_Gaussians
    
    # Recurrence step:
    
    current_hidden_var_coefs = np.concatenate((saved_Gaussians, recurrent_current_hidden_var_coefs), 0)
    next_hidden_var_coefs = np.concatenate((saved_Gaussians*0, recurrent_next_hidden_var_coefs), 0)
    
    current_nb_gaussians = len(current_hidden_var_coefs)
    
    '''
    recurrence step:
    '''
    
    recurrent_sequence_phase_1 = [] # list of lists containing the sequence of coef_index and gaussian IDs to 
    recurrent_functions_phase_1 = []

    #print('LC1',LC)
    for coef_index in np.arange(nb_hidden_vars-1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            recurrent_sequence_phase_1.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_1.append(intermediate_RNN_function)
            
            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = current_hidden_var_coefs[ID_1]
            current_hidden_var_coefs_2 = current_hidden_var_coefs[ID_2]
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]
            
            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2)
            
            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            recurrent_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            recurrent_sequence_phase_1.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_1.append(no_RNN_function_phase_1)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: 
            current_hidden_var_coefs = np.delete(current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1
    
    recurrent_sequence_phase_2 = []
    recurrent_functions_phase_2 = []
    
    saved_Gaussians = np.zeros((nb_hidden_vars, nb_hidden_vars))
    # contrary to the integration step, we cannot remove Gaussians. Instead, we will save them to solve the linear problem
    for coef_index in np.arange(nb_hidden_vars-1, -1, -1):
        
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = next_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            recurrent_sequence_phase_2.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_2.append(intermediate_RNN_function)
            
            C1 = next_hidden_var_coefs[ID_1, coef_index]
            C2 = next_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]*0
            current_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]*0
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]

            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, current_hidden_var_coefs_1, current_hidden_var_coefs_2)

            next_hidden_var_coefs[ID_1] = current_coefs3
            next_hidden_var_coefs[ID_2] = current_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            recurrent_functions_phase_2[-1] = final_RNN_function_phase_2
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            recurrent_sequence_phase_2.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_2.append(no_RNN_function_phase_2)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: # we remove the last gaussian that depend on the coefficient on index coef_index (only valid if at least on gaussian has a non 0 coefficient)
            saved_Gaussians[coef_index] = next_hidden_var_coefs[ID_2]
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, ID_2, 0)
            current_nb_gaussians += -1
            
    print('Checking that the recurrent next Gaussians have the same form than the initial next gaussians:', np.all((initial_saved_Gaussians == 0) == (saved_Gaussians == 0)))
    
    '''
    Transition step:
    '''
    
    current_hidden_var_coefs = saved_Gaussians
    next_hidden_var_coefs = saved_Gaussians*0
    
    current_nb_gaussians = len(current_hidden_var_coefs)
    
    transition_sequence = [] # list of lists containing the sequence of coef_index and gaussian IDs to 
    transition_functions = []
    
    transition_integration_variables = np.arange(integration_variable_index, nb_hidden_vars)[::-1]
    #print('LC1',LC)
    for coef_index in transition_integration_variables:
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            transition_sequence.append([coef_index, ID_1, ID_2])
            transition_functions.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = current_hidden_var_coefs[ID_1]
            current_hidden_var_coefs_2 = current_hidden_var_coefs[ID_2]
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]

            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2)

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4

            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            transition_functions[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            transition_sequence.append([coef_index, ID_1, ID_2])
            transition_functions.append(no_RNN_function_phase_1)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: 
            current_hidden_var_coefs = np.delete(current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1
    
    current_hidden_var_coefs = np.concatenate((current_hidden_var_coefs, transition_hidden_var_coefs[:, 0,0]), 0)
    next_hidden_var_coefs = np.concatenate((next_hidden_var_coefs, transition_hidden_var_coefs[:, 0,0]*0), 0)
    current_nb_gaussians = current_hidden_var_coefs.shape[0]
    
    saved_Gaussians = current_hidden_var_coefs
    
    '''
    Final step
    Contrary to the previous steps, the final step does not introduce new gaussians that depend on the next hidden variables. 
    Therefore, we only need to perform the phase 1 on the gaussians that remain from the previous step
    '''
    
    current_hidden_var_coefs = saved_Gaussians
    current_nb_gaussians = len(current_hidden_var_coefs)

    next_hidden_var_coefs = np.zeros(current_hidden_var_coefs.shape)
    
    final_sequence_phase_1 = [] # list of lists containing the sequence of coef_index and gaussian IDs to 
    final_functions_phase_1 = []
    
    #print('LC1',LC)
    for coef_index in np.arange(nb_hidden_vars-1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID,coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)
        
        for i in range(len(non_zero_gaussian_IDs)-1):
            
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i+1]
            
            final_sequence_phase_1.append([coef_index, ID_1, ID_2])
            final_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            current_hidden_var_coefs_1 = current_hidden_var_coefs[ID_1]
            current_hidden_var_coefs_2 = current_hidden_var_coefs[ID_2]
            next_hidden_var_coefs_1 = next_hidden_var_coefs[ID_1]
            next_hidden_var_coefs_2 = next_hidden_var_coefs[ID_2]
            
            current_coefs3, current_coefs4, next_coefs3, next_coefs4 = simple_RNN_gaussian_product(C1, C2, current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2)

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4

            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4
        
        if len(non_zero_gaussian_IDs)>1:
            recurrent_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1: # if there is already only one gaussian that depend on 
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            
            final_sequence_phase_1.append([coef_index, ID_1, ID_2])
            final_functions_phase_1.append(no_RNN_function_phase_1)
        else: # if next_hidden_var_coefs is independent from the coefficient of index coef_index, nothing happens
            pass 
        
        if len(non_zero_gaussian_IDs) >= 1: 
            current_hidden_var_coefs = np.delete(current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1
    
    return [initial_functions_phase_1, initial_sequence_phase_1], [initial_functions_phase_2, initial_sequence_phase_2], [recurrent_functions_phase_1, recurrent_sequence_phase_1], [recurrent_functions_phase_2, recurrent_sequence_phase_2], [final_functions_phase_1, final_sequence_phase_1], [transition_functions, transition_sequence]

@tf.function
def constraint_function(all_params, all_initial_params, nb_dims, dtype):
    
    '''
    The constraint_function must define the Gaussians and their values based on the model parameters all_params and all_initial_params
    
    This includes the initial Gaussians for each state, the recurrent Gaussians and the transition Gaussians
    When a transition happen, we must integrate over the hidden variables that disappear (e.g. the velocity vector in case of directed motion)
    while conserving other variables that are still relevant for the next step (e.g. the real particle position in case of directed motion).
    
    When considering an anomalous motion model, we have one Gaussian for the localization error, one Gaussian for the diffusion of the particle
    and one Gaussian for the diffusion of the anomalous variable (either the velocity vector or the potential well center). At each time step,
    We have a product of 2 previous Gaussians that represent the hidden variable positions + 3 Gaussians that represent the update of the observed 
    and hidden variables. We then integrate this product over the 2 hidden variables, which results in a product of 3 Gaussians that depend on two 
    hidden variables. One of the Gaussians being independent on the hidden variables, this step outputs two gaussians that will be used as inputs
    for the next step.
    
    When a transition happen, the process is a bit more complex. For instance, if we consider a transition from a directed motion state to a 
    confined motion state, the update of the system updates the potential well. Before this update, we must then initialize the potential well
    position. We then have 2 previous Gaussians that describe the previous knowledge about the real particle position and the velocity vector,
    one Gaussian that initializes the potential well position based on the real particle position and 3 gaussians that update the confined diffusion
    system. In this case, we must 1) integrate the 2 previous Gaussians over the velocity vector, which result in one Gaussian, 2) add the Gaussian
    that initializes the potential well position, 3) integrate the 5 resulting gaussians over the current variables.
    '''
    
    print(all_params)
    nb_states = all_params.shape[0]
    print('nb_states', nb_states)
    
    '''
    First we need to define the variables that need to be integrated. To do so, we define an index that separates the variables 
    that do not need to be integrated from the variables that need to be intergrated. The non-integrated variables must have lower
    indexes than this threshold and the variables whose index is higher or equal to the threshold will be integrated over.
    '''
    integration_variable_index = tf.constant(1)
    nb_hidden_vars = 2
    nb_obs_vars = 1
    nb_transition_gaussians = nb_hidden_vars - integration_variable_index # The number of Gaussians in transition_hidden_vars must be equal to the number of integrations len(transition_integration_variables)
    
    hidden_vars=[]
    obs_vars=[]
    initial_hidden_vars=[]
    transition_hidden_vars = []
    for k in range(nb_states):
        params = all_params[k]
        initial_params = all_initial_params[k]
        d = tf.math.exp(params[1])
        LocErr = tf.math.exp(params[0])
        q = tf.math.exp(params[3])
        initial_position_spread = tf.exp(initial_params[0])
        
        if params[4] < 0.5:
            l = tf.math.sigmoid(params[2])+1e-20
            
            # hidden vars:                               pos_x,           ano_pos_x,        pos_x,    ano_pos_x,
            hidden_vars = hidden_vars + [tf.stack([[[[         1/LocErr,                   0,            0,            0]]],                   # Localization error     
                                         [[[(1-l)/d, l/d, -1/d,            0]]],                # Diffusion + anomalous drift
                                         [[[                   0,          1/q,         0, -1/q]]]])]                    # Diffusion of the anomalous position
            
            obs_vars = obs_vars + [tf.stack([[[[-1/LocErr]]],
                                             [[[        0]]],
                                             [[[        0]]]])]
            
            # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
            # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
            
            well_distance = d/(2*l)**0.5

            #well_distance = tf.math.exp(initial_params[1]) #params[1]/(2*tf.math.abs(params[2])+1e-20)**0.5
            initial_hidden_vars = initial_hidden_vars + [tf.stack([[[[ 1/initial_position_spread,                           0]]],
                                                                   [[[           1/well_distance,            -1/well_distance]]]])]
            
            transition_hidden_vars = transition_hidden_vars + [tf.stack([[[[    1/well_distance,   -1/well_distance]]]])]
        
        else:
            v = tf.math.exp(params[2]) + 1e-20
            
            # hidden vars:                   pos_x,   ano_pos_x,        pos_x,    ano_pos_x,
            hidden_vars = hidden_vars + [tf.stack([[[[1/LocErr,     0,     0,     0]]],               # Localization error    
                                                   [[[     1/d,   1/d,  -1/d,     0]]],               # Diffusion + anomalous drift
                                                   [[[       0,   1/q,     0,  -1/q]]],                    # Diffusion of the anomalous position
                                                                                     ])]
            
            obs_vars = obs_vars + [tf.stack([[[[-1/LocErr]]],
                                             [[[        0]]],
                                             [[[        0]]]])]
            
            # It is important to have the same number of Gaussians and variables, to do so we need to add nb_hidden_vars Gaussians either at the beginning of the recurrence or at the end
            # hidden vars:                     pos_x, pos_y, ano_pos_x, ano_pos_y,  pos_x, pos_y, ano_pos_x,ano_pos_y 
            
            initial_hidden_vars = initial_hidden_vars + [tf.stack([[[[1/initial_position_spread,         0]]], # The initial position and velocity are independent but we need the same sparsity than for confined motion
                                                                   [[[                    1e-15,       1/v]]]])]
            
            transition_hidden_vars = transition_hidden_vars + [tf.stack([[[[   1e-15,   1/v]]]])] # in case of transition to a directed state the new velocity is the same than the initial velocity
    
    Gaussian_stds = tf.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    biases = tf.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states, nb_dims), dtype = dtype)
    initial_obs_vars = tf.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars), dtype = dtype)
    initial_Gaussian_stds = tf.ones((nb_hidden_vars, 1, nb_states, 1), dtype = dtype)
    initial_biases = tf.zeros((nb_transition_gaussians, 1, nb_states, nb_dims), dtype = dtype)
    transition_Gaussian_stds = tf.ones((nb_transition_gaussians, 1, nb_states, 1), dtype = dtype)
    transition_biases = tf.zeros((nb_transition_gaussians, 1, nb_states, nb_dims), dtype = dtype)
    
    hidden_vars = tf.concat(hidden_vars, 2)
    obs_vars = tf.concat(obs_vars, 2)
    initial_hidden_vars = tf.concat(initial_hidden_vars, 2)
    transition_hidden_vars = tf.concat(transition_hidden_vars, 2)
    
    return hidden_vars, obs_vars, Gaussian_stds, biases, initial_hidden_vars, initial_obs_vars, initial_Gaussian_stds, initial_biases, transition_hidden_vars, transition_Gaussian_stds, transition_biases, integration_variable_index

@tf.function
def transition_param_function(transition_shapes, transition_rates, density, Fs, effective_ds, dtype):
    
    '''
    The transition_param_function must define the initial transition parameters and their constraints
    similarly to how constraint_function defines the constraints of the states
    '''
    
    print('transition_shapes', transition_shapes)
    nb_states = transition_shapes.shape[0]
    
    transition_shapes = tf.math.exp(transition_shapes)
    transition_rates = tf.math.softmax(transition_rates, axis = 1)*transition_shapes
    
    new_transition_shapes = tf.concat((transition_shapes, tf.constant([[1]*nb_states], dtype = dtype)), axis = 0)
    new_transition_shapes = tf.concat((new_transition_shapes, tf.constant([[1]]*(nb_states+1), dtype = dtype)), axis = 1)
    
    mislinking_dwell_time = tf.constant([0.9/nb_states]*nb_states, dtype = dtype) # We multiply by (1-tf.reduce_mean(additional_transition_params[:nb_states]) to allow several consecutive mislinkings proportionally to the misslinking probability
    mislinking_dwell_time = tf.concat((mislinking_dwell_time, [0.1]), axis = 0)
        
    #mislinking_rates = tf.constant([0.078,0.146], dtype = dtype)[:, None] # density 1 -> rates 0.052 0.052 
    mislinking_rates = 1-tf.math.exp(-0.5*density *tf.reduce_sum(Fs[None]*(effective_ds[:,None]**2 + effective_ds[None]**2)**0.5, axis = 0)[:, None])# density 0.1 -> rates 0.052 0.052 

    new_transition_rates = tf.concat((transition_rates, mislinking_rates), axis = 1)
    new_transition_rates = tf.concat((new_transition_rates, mislinking_dwell_time[None]), axis = 0)
    
    return new_transition_shapes, new_transition_rates

def get_model_params(model):
    '''
    Function to get the parameters from the model
    '''
    weights = model.weights
    nb_states = weights[-1].shape[0]
    transition_shapes = tf.math.exp(weights[5])
    transition_rates = tf.math.softmax(weights[4], axis = 1)*transition_shapes
    model_types = weights[0][:, -1].numpy().astype(int)
    model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]
    model_parameters = {'Model type': model_types_str,'anomalous factors': list(np.round(tf.sigmoid(weights[0][:, 2])*(1-weights[0][:, 4]) + 2**0.5*tf.exp(weights[0][:, 2])*weights[0][:, 4],3)), 'Localization errors': list(np.round(np.exp(weights[0][:, 0]),3)), 'd': list(np.round(np.exp(weights[0][:, 1]), 3)), 'transition rates': list(np.round(transition_rates, 3).reshape(nb_states**2)), 'transition shapes': list(np.round(transition_shapes, 2).reshape(nb_states**2)), 'Fractions': list(np.round(tf.math.softmax(weights[2][0]), 3))}
 
    return model_parameters


def build_model(track_len, # maximum number of time points in the input tracks 
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions, 
                batch_size, # number of tracks analysed at the same time
                nb_dims = 2, # Number of dimensions of the tracks
                sequence_length = 3, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = 3, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = 0.001, # Estimated density of the sample.
                vary_params = None,
                vary_initial_params = None,
                vary_initial_fractions = None,
                vary_transition_shapes = None,
                vary_transition_rates = None
                ):
    
    # Defining the hyperparameters of the model
    dtype = 'float64'
    
    nb_obs_vars = 1 # number of dependend variables (the x, y, z dimension do not account as dependent variables in our model so keep this to 1)
    nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, 1, nb_independent_vars), dtype = dtype)
    input_mask = tf.keras.Input(batch_shape = (batch_size, track_len), dtype = dtype)
    
    #inputs = tracks
    #input_mask = all_masks
    
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4, 5])
    
    Init_layer = Initial_layer_constraints(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
                                           initial_fractions,
                                           max_linking_distance,
                                           constraint_function,
                                           vary_params = vary_params,
                                           vary_initial_params = vary_initial_params,
                                           vary_initial_fractions = vary_initial_fractions,
                                           sequence_length = sequence_length,
                                           dtype = dtype)
    
    tensor1, initial_states = Init_layer(transposed_inputs)
    
    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds = Init_layer.param_vars[:, 1]
    anomalous_factors = Init_layer.param_vars[:, 2]
    isdir = Init_layer.param_vars[:, 4]
    
    Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states
    
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype = dtype)(input_mask)
    
    layer = Custom_RNN_layer(batch_size, transition_shapes, transition_rates, estimated_density, nb_states, Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, Init_layer.transition_sequence, transition_param_function, sequence_length = sequence_length, vary_transition_shapes = vary_transition_shapes, vary_transition_rates = vary_transition_rates, dtype = dtype)
    states = layer(sliced_inputs, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir)
    
    F_layer = Final_layer(Init_layer.final_sequence_phase_1, nb_dims = nb_independent_vars, sequence_length = sequence_length, dtype = dtype)
    outputs, All_states = F_layer(states)
    
    model = tf.keras.Model(inputs=(inputs, input_mask), outputs=outputs, name="Diffusion_model")
    pred_model = tf.keras.Model(inputs=(inputs, input_mask), outputs=All_states, name="Diffusion_model")
    
    return model, pred_model

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

class get_parameters(tf.keras.callbacks.Callback):
    def __init__(self, layer_name='params'):
        super(get_parameters, self).__init__()
        self.layer_name = layer_name
    
    def on_epoch_end(self, epoch, logs=None):
        # Retrieve the weights of the model
        weights = self.model.weights
        nb_states = weights[-1].shape[0]
        transition_shapes = tf.math.exp(weights[5])
        transition_rates = tf.math.softmax(weights[4], axis = 1)*transition_shapes
        params = {'anomalous factors': list(np.round(tf.sigmoid(weights[0][:, 2])*(1-weights[0][:, 4]) + 2**0.5*tf.exp(weights[0][:, 2])*weights[0][:, 4],3)), 'Localization errors': list(np.round(np.exp(weights[0][:, 0]),3)), 'd': list(np.round(np.exp(weights[0][:, 1]), 3)), 'transition rates': list(np.round(transition_rates, 3).reshape(nb_states**2)), 'transition shapes': list(np.round(transition_shapes, 2).reshape(nb_states**2)), 'Fractions': list(np.round(tf.math.softmax(weights[2][0]), 3))}
        print(params)

def model_to_DataFrame(model, dt):
    weights = model.weights
    nb_states = weights[0].shape[0]
    params = {'anomalous factors': (tf.sigmoid(weights[0][:, 2])*(1-weights[0][:, 4]) + tf.exp(weights[0][:, 2])*weights[0][:, 4]).numpy(), 'Localization errors': np.exp(weights[0][:, 0]), 'd':np.exp(weights[0][:, 1]), 'transition rates': tf.math.softmax(weights[4], axis = 1).numpy(), 'transition shapes': tf.math.exp(weights[5]).numpy(), 'Fractions': (tf.math.softmax(weights[2][0])).numpy()}
    colnames = []
    data = []
    for state in range(nb_states):
        colnames.append('D%s'%state)
        data.append(params['d'][state]**2/(2*dt))
    for state in range(nb_states):
        colnames.append('Fraction %s'%state)
        data.append(params['Fractions'][state])
    for state in range(nb_states):
        colnames.append('Anomalous factor %s'%state)
        data.append(params['anomalous factors'][state])
    for state in range(nb_states):
        colnames.append('Model type state %s'%state)
        data.append(['Confined', 'directed'][int(weights[0][:, 4][state])])
    for state in range(nb_states):
        colnames.append('Localization error %s'%state)
        data.append(params['Localization errors'][state])
    for i in range(nb_states):
        for j in range(nb_states):
            if i != j:
                Tr_shape = tf.math.exp(params['transition shapes'][i, j])
                Tr_rate = tf.math.sigmoid(params['transition rates'], axis = 1)
                colnames.append('Transition rate (per time unit) %s%s'%(i, j))
                data.append(params['transition rates'][i, j])

                colnames.append('Transition shape %s%s'%(i, j))
                data.append(params['transition shapes'][i, j])
    data = pd.DataFrame([data], columns = colnames)
    return data

class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, warmup_steps, peak_lr, decay_rate, decay_start):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.decay_rate = decay_rate
        self.decay_start = decay_start
    
    def __call__(self, step):
        # Linear warmup
        decay_step = tf.reduce_max([step-self.decay_start, 0])
        return self.peak_lr*(1-tf.math.exp(-step/self.warmup_steps))*tf.math.exp(-self.decay_rate*decay_step)

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

def logit(x):
    return -np.log(1/x-1)

def Model_finder(tracks,
                 masks,
                 sequence_length,
                 nb_states, params,
                 initial_params,  
                 initial_fractions, 
                 transition_shapes, 
                 transition_rates, 
                 max_linking_distance, 
                 estimated_density, 
                 epochs, 
                 batch_size,
                 learning_rate = 1/30,
                 decay_threshold = 500,
                 decay_rate = 0.01,
                 device = '/GPU:0', 
                 shuffle = True, 
                 verbose = 1,       
                 vary_params = None,
                 vary_initial_params = None,
                 vary_initial_fractions = None,
                 vary_transition_shapes = None,
                 vary_transition_rates = None):
    '''
    If a state is not found immobile, we test the alternative state hypothesis
    '''
    nb_states = params.shape[0]
    track_len = masks.shape[1]
    nb_dims = tracks.shape[-1]
    initial_anomalous_factors = params[:, 2]
    
    model, pred_model = build_model(track_len, # maximum number of time points in the input tracks 
                    nb_states, # Number of states of their model
                    params, # recurrent parameters of the model
                    initial_params, # initial parameters of the model
                    transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                    transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                    initial_fractions, # initial guess of the fractions (softmax)
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
    
    lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)
    
    with tf.device(device):
        history = model.fit((tracks, masks), tracks, epochs = epochs, batch_size = batch_size, callbacks=[get_parameters()], shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])
    
    All_models = {}
    params, initial_params, initial_fractions, _, transition_rates, transition_shapes = model.get_weights()
    LogLikelihood = - history.history['loss'][-1]
    loss_history = history.history['loss']
    
    All_models['Model 0'] = {'params': params, 'initial_params': initial_params, 'initial_fractions': initial_fractions, 'transition_shapes': transition_shapes, 'transition_rates': transition_rates, 'LogLikelihood': LogLikelihood, 'loss_history': loss_history}
    best_LogLikelihood = LogLikelihood
    best_model = 'Model 0'

    for i in range(nb_states):
        model.weights[0].assign(params)
        model.weights[1].assign(initial_params)
        model.weights[2].assign(initial_fractions)
        model.weights[4].assign(transition_rates)
        model.weights[5].assign(transition_shapes)
        
        model.weights[0][i, 4].assign(1 - model.weights[0][i, 4])
        model.weights[0][i, 2].assign(initial_anomalous_factors[i])

        lr = WarmupLearningRateSchedule(10, 1/50, 0.01, 500) # learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
        model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)
        with tf.device(device):
            history = model.fit((tracks, masks), tracks, epochs = epochs, batch_size = batch_size, callbacks=[get_parameters()], shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])
        model.weights
        
        list(np.round(tf.math.softmax(model.weights[2][0]), 3))
        
        params, initial_params, initial_fractions, transition_shapes, transition_rates = get_model_params(model)
        LogLikelihood = - history.history['loss'][-1]
        loss_history = history.history['loss']
        model_ID = len(All_models)
        All_models['Model %s'%model_ID] = {'params': params.numpy(), 'initial_params': initial_params.numpy(), 'initial_fractions': initial_fractions.numpy(), 'transition_shapes': transition_shapes.numpy(), 'transition_rates': transition_rates.numpy(), 'LogLikelihood': LogLikelihood, 'loss_history': loss_history}
        
        print('Log Likelihood', LogLikelihood)
        print('params', params)
        if LogLikelihood > best_LogLikelihood:
            best_model = 'Model %s'%model_ID
        
        params, initial_params, initial_fractions, transition_shapes, transition_rates = All_models[best_model]['params'], All_models[best_model]['initial_params'], All_models[best_model]['initial_fractions'], All_models[best_model]['transition_shapes'], All_models[best_model]['transition_rates']
        
    model.weights[0].assign(params)
    model.weights[1].assign(initial_params)
    model.weights[2].assign(initial_fractions)
    model.weights[4].assign(transition_rates)
    model.weights[5].assign(transition_shapes)
    return model, pred_model

def build_abrupt_directed_motion_changes_model(track_len, # maximum number of time points in the input tracks 
                nb_states, # Number of states of their model
                all_params, # recurrent parameters of the model
                all_initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions, 
                batch_size, # number of tracks analysed at the same time
                nb_dims = 2, # Number of dimensions of the tracks
                sequence_length = 3, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = 3, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = 0.001, # Estimated density of the sample.
                abrupt_change_state = 2,
                vary_params = None,
                vary_initial_params = None,
                vary_initial_fractions = None,
                vary_transition_shapes = None,
                vary_transition_rates = None):
    
    class Initial_layer_constraints_abrupt_change(Initial_layer_constraints):
        
        def duplicate_states(self, param_vars, initial_param_vars, initial_fractions):
            '''
            initial log factors 
            '''
            param_vars = tf.concat((param_vars[:abrupt_change_state+1], param_vars[abrupt_change_state:]), 0)
            initial_param_vars = tf.concat((initial_param_vars[:abrupt_change_state+1], initial_param_vars[abrupt_change_state:]), 0)
            initial_fractions = tf.concat((initial_fractions[:,:abrupt_change_state],[[1e-10]], initial_fractions[:,abrupt_change_state:]), 1)
            
            return param_vars, initial_param_vars, initial_fractions
        
    @tf.function
    def transition_param_function(transition_shapes, transition_rates, density, Fs, effective_ds, dtype):
        '''
        The transition_param_function must define the initial transition parameters and their constraints
        similarly to how constraint_function defines the constraints of the states
        '''
    
        print('transition_shapes', transition_shapes)
        nb_states = transition_shapes.shape[0]
        
        abrupt_change_state = 0
        
        # We need to assign values to the transitions kinetics directed state 1 <=> directed state 2.
        # To do so, we can use the diagonal values of transition_shapes and transition_rates that are unused
        directed_directed_transition_shape = tf.math.exp(transition_shapes[abrupt_change_state, abrupt_change_state])
        directed_directed_transition_rate = tf.math.sigmoid(transition_rates[abrupt_change_state, abrupt_change_state]-2)
        
        transition_shapes = tf.math.exp(transition_shapes)
        transition_rates = tf.math.softmax(transition_rates, axis = 1)*transition_shapes
        
        new_transition_shapes = tf.concat((transition_shapes, tf.constant([[1]*nb_states], dtype = dtype)), axis = 0)
        new_transition_shapes = tf.concat((new_transition_shapes, tf.constant([[1]]*(nb_states+1), dtype = dtype)), axis = 1)
        
        mislinking_dwell_time = tf.constant([0.9/nb_states]*nb_states, dtype = dtype) # We multiply by (1-tf.reduce_mean(additional_transition_params[:nb_states]) to allow several consecutive mislinkings proportionally to the misslinking probability
        mislinking_dwell_time = tf.concat((mislinking_dwell_time, [0.1]), axis = 0)
        
        #mislinking_rates = tf.constant([0.078,0.146], dtype = dtype)[:, None] # density 1 -> rates 0.052 0.052 
        mislinking_rates = 1-tf.math.exp(-0.5*density *tf.reduce_sum(Fs[None]*(effective_ds[:,None]**2 + effective_ds[None]**2)**0.5, axis = 0)[:, None])# density 0.1 -> rates 0.052 0.052 
    
        new_transition_rates = tf.concat((transition_rates, mislinking_rates), axis = 1)
        new_transition_rates = tf.concat((new_transition_rates, mislinking_dwell_time[None]), axis = 0)
        
        '''
        Once the mislinking state is added we can add the additionnal directed state, constraining
        transitions into a directed state to occur only towards the first directed state duplicated state such that
        transitions from other states can only occur towards the first directed state of state index abrupt_change_state.
        Directed particles can then either transition into the other directed state (duplicate) or into the other
        states. the on rates of the 2nd directed state are 0 except from the 1st directed state and the off rates 
        of the 2 directed states are shared.    
        '''
        abrupt_change_state
        
        second_directed_state_on_rates = tf.stack([1e-10]*abrupt_change_state + [directed_directed_transition_rate] + [1e-10]*(nb_states - abrupt_change_state))
        new_new_transition_rates = tf.concat((new_transition_rates[:,:abrupt_change_state+1], second_directed_state_on_rates[:,None], new_transition_rates[:,abrupt_change_state+1:]), 1)
        second_directed_state_off_rates = tf.concat([new_new_transition_rates[abrupt_change_state, :abrupt_change_state]] + [directed_directed_transition_rate[None]] + [new_new_transition_rates[abrupt_change_state, abrupt_change_state+1:]], axis = 0)
        new_new_transition_rates = tf.concat((new_new_transition_rates[:abrupt_change_state+1], second_directed_state_off_rates[None], new_new_transition_rates[abrupt_change_state+1:]), 0)
        
        second_directed_state_on_shapes = tf.stack([1]*abrupt_change_state + [directed_directed_transition_shape] + [1]*(nb_states - abrupt_change_state))
        new_new_transition_shapes = tf.concat((new_transition_shapes[:,:abrupt_change_state+1], second_directed_state_on_shapes[:,None], new_transition_shapes[:,abrupt_change_state+1:]), 1)
        second_directed_state_off_shapes = tf.concat([new_new_transition_shapes[abrupt_change_state, :abrupt_change_state]] + [directed_directed_transition_shape[None]] + [new_new_transition_shapes[abrupt_change_state, abrupt_change_state+1:]], axis = 0)
        new_new_transition_shapes = tf.concat((new_new_transition_shapes[:abrupt_change_state+1], second_directed_state_off_shapes[None], new_new_transition_shapes[abrupt_change_state+1:]), 0)
            
        return new_new_transition_shapes, new_new_transition_rates
    
    # Defining the hyperparameters of the model
    dtype = 'float64'
    
    nb_obs_vars = 1 # number of dependend variables (the x, y, z dimension do not account as dependent variables in our model so keep this to 1)
    nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars
    nb_states = nb_states + 1
    
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, 1, nb_independent_vars), dtype = dtype)
    input_mask = tf.keras.Input(batch_shape = (batch_size, track_len), dtype = dtype)
    
    #inputs = tracks
    #input_mask = all_masks
    
    transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4, 5])
    
    Init_layer = Initial_layer_constraints_abrupt_change(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           all_params,
                                           all_initial_params,
                                           initial_fractions,
                                           max_linking_distance,
                                           constraint_function,
                                           vary_params = vary_params,
                                           vary_initial_params = vary_initial_params,
                                           vary_initial_fractions = vary_initial_fractions,
                                           sequence_length = sequence_length,
                                           dtype = dtype)
    #inputs = transposed_inputs
    #self = Init_layer
    tensor1, initial_states = Init_layer(transposed_inputs)
    
    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds = Init_layer.param_vars[:, 1]
    anomalous_factors = Init_layer.param_vars[:, 2]
    isdir = Init_layer.param_vars[:, 4]
    
    Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states
    Prev_coefs[:, 0, 7]
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype = dtype)(input_mask)
    
    layer = Custom_RNN_layer(batch_size, transition_shapes, transition_rates, estimated_density, nb_states, Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, Init_layer.transition_sequence, transition_param_function, sequence_length = sequence_length, vary_transition_shapes = vary_transition_shapes, vary_transition_rates = vary_transition_rates, dtype = dtype)
    
    #self = layer
    # inputs = sliced_inputs
    # mask = sliced_mask#
    states = layer(sliced_inputs, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir)
    
    F_layer = Final_layer(Init_layer.final_sequence_phase_1, nb_dims = nb_independent_vars, sequence_length = sequence_length, dtype = dtype)
    outputs, All_states = F_layer(states)
    
    model = tf.keras.Model(inputs=(inputs, input_mask), outputs=outputs, name="Diffusion_model")
    pred_model = tf.keras.Model(inputs=(inputs, input_mask), outputs=All_states, name="Diffusion_model")
    
    return model, pred_model


