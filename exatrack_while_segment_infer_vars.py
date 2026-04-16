# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:48:22 2026

@author: Franc
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN
#from tensorflow.python.keras.layers.recurrent import RNN

dtype = 'float64'
pi = tf.constant(np.pi, dtype = dtype)
minval = np.array(1e-14)

from matplotlib import pyplot as plt
from numba import njit, typed, prange, jit
from scipy.stats import gamma
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import pandas as pd
from glob import glob
from scipy.special import softmax as scipy_softmax

import scipy
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
        raise ValueError('Fs, Ds, velocities, angular_Ds, conf_forces, conf_Ds, conf_dists and transition_matrix must all be arrays of the same length (one element per state)')
    # diff + persistent motion + elastic confinement
    
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
    
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
        while n <= nb_burning_steps*nb_sub_steps:
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
                                         D = D*scipy.stats.chi2.rvs(50, 0, 0.02),
                                         velocity = velocity/nb_sub_steps,
                                         angular_D = angular_D,
                                         conf_force = conf_force/nb_sub_steps,
                                         conf_D = conf_D,
                                         conf_dist = conf_dist,
                                         LocErr_std = LocErr_std,
                                         dt = dt/nb_sub_steps,
                                         nb_sub_steps = 1,
                                         initial_positions = initial_positions)
                
                #np.mean(scipy.stats.chi2.rvs(50, 0, 0.02, 100000))
                #np.std(scipy.stats.chi2.rvs(50, 0, 0.02, 100000))

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
        all_tracks[k,track_len:] = track[-1]

        all_states[k,:track_len] = states
        all_states[k,track_len:] = states[-1]

        all_masks[k,:track_len-1] = 1
        all_masks[k,-1] = 1
        
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
        # Directed motion update:
        angle = angles[i]
        pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
        positions[i+1] = positions[i] + pesistent_disp + disps[i]
        # Confinement update:
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
            
            movie = emit_photons(pixel_pos, nb_photons, movie, time, emission_std, pixel_dims)
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

def padding(track_list, frame_list = None, batch_size = None):
    '''
    If tracks have multiple lengths, we need to homogenize the shape to the longest track lengths using padding and mask the state updates of the padding
    
    This function takes a list of tracks as imput and returns a padded array of tracks and the corresponding padding mask.
    
    fitting_type: can be either 'All', 'Directed', 'Confined' or 'Brownian' depending on the type of motion you want to analyse.
    'All': works for all types of motion but requires tracks of at least 5 time steps
    'Directed': works for both directed and Brownian types of motion.
    'Confined': works for both confined and Brownian types of motion.
    'Brownian': only works for Brownian motion.
    If your tracks are all 5 time points or more you can ignore the `fitting_type` argument.
    
    Indicate the batch size to complete the last batch with dummy tracks when needed
    '''
    start_len = 1
    max_len = 0
    for track in track_list:
        if track.shape[0] > max_len:
            max_len = track.shape[0]
    nb_tracks = len(track_list)
    
    if batch_size is not None:
        nb_tracks = int(np.ceil(nb_tracks/batch_size))*batch_size
    padded_tracks = np.zeros((nb_tracks, max_len, track_list[0].shape[1]), dtype = track_list[0].dtype)
    if type(frame_list)!= type(None):
        padded_frames =  np.zeros((nb_tracks, max_len), dtype = frame_list[0].dtype)
    else:
        padded_frames = None
    mask = np.zeros((nb_tracks, max_len), dtype = track_list[0].dtype)
    
    for i, track in enumerate(track_list):
        if track.shape[0]>=start_len:
            cur_len = track.shape[0]
            padded_tracks[i, :cur_len] = track
            padded_tracks[i, cur_len:] = track[-1] # padding that replicates the edges in case we want to use ExaTrack on the reversed tracks

            mask[i, :cur_len] = 1
            if type(frame_list)!= type(None):
                frames = frame_list[i]
                padded_frames[i, :cur_len] = frames
                padded_frames[i, cur_len:] = frames[-1]
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
               colnames = ['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],  # if multiple columns are required to identify a track, the string used to identify the track ID can be replaced by a list of strings represening the column names e.g. ['TRACK_ID', 'Movie_ID']
               opt_colnames = [], # list of additional metrics to collect e.g. ['QUALITY', 'ID']
               remove_no_disp = True):
    
    if type(paths) == str or type(paths) == np.str_:
        paths = [paths]
        
    colnames = list(colnames)
    
    tracks = []
    frames = []
    track_IDs = []
    opt_metrics = {}
    for m in opt_colnames:
        opt_metrics[m] = []
    
    for path in paths:
        if fmt == 'csv':
            data = pd.read_csv(path, sep=',', low_memory=False)
            #data = data.dropna()
        elif fmt == 'pkl':
            data = pd.read_pickle(path)
        else:
            data = pd.read_csv(path, sep = fmt, low_memory=False)
        
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
        data = data.dropna()
        
        try:
            for ID, track in data.groupby(colnames[3]):
                
                track = track.sort_values(colnames[2], axis = 0)
                track_mat = track.values[:,:4].astype('float64')
                dists2 = (track_mat[1:, :2] - track_mat[:-1, :2])**2
                if remove_no_disp:
                    if len(dists2) > 0 and np.mean(dists2==0)>0.05:
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
                            for k in range(len(track_mat)//np.max(lengths)):
                                l = np.max(lengths)
                                tracks.append(track_mat[l*k:l*(k+1), 0:2])
                                frames.append(track_mat[l*k:l*(k+1), 2])
                                for m in opt_colnames:
                                    opt_metrics[m].append(track[m].values[l*k:l*(k+1)]) 
        
        except Exception as e:
            import logging
            # log the error and continue, if appropriate
            logging.error(f"An unexpected error occurred: {e}")
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
        carryover = False, # do we want a segmented model that carries over the hidden states of the model to the next batches 
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
        self.carryover = carryover
    
    def build(self, input_shape):
        dtype = self.dtype
        '''
        param_vars = tf.Variable(params,  dtype = dtype, name = 'recurrence_variables', constraint=lambda w: tf.where(tf.greater_equal(w, -1), w, 0.0000001))
        initial_param_vars = tf.Variable(initial_params,  dtype = dtype, name = 'initial_variables', constraint=lambda w: tf.where(tf.greater_equal(w, 0), w, 0.0000001))
        initial_fractions = (np.random.rand(1, nb_states+1)*0+1)
        initial_fractions[0,-1] = -1
        '''
        self.param_vars = tf.Variable(self.params,  dtype = dtype, name = 'recurrence_variables', constraint=lambda w: tf.where(tf.greater_equal(w, tf.math.log(minval)), w, tf.math.log(minval)))
        self.initial_param_vars = tf.Variable(self.initial_params,  dtype = dtype, name = 'initial_variables', trainable = True, constraint=lambda w: tf.where(tf.greater_equal(w, tf.math.log(minval)), w, tf.math.log(minval)))
        self.max_linking_distance_param = tf.Variable(self.max_linking_distance, dtype = dtype, name = 'max linking distance', trainable = False)
        initial_fractions = self.initial_fractions
        self.initial_fractions = tf.Variable(initial_fractions, dtype = dtype, name = 'Fractions', trainable = True)
        
        nb_sequences = self.sequence_length * (self.nb_states + 1)
        if self.carryover:
            self.carryout_coefs = tf.Variable(np.zeros((self.nb_hidden_vars, input_shape[2], nb_sequences, input_shape[5])), dtype = dtype, trainable = False)
            self.carryout_biases = tf.Variable(np.zeros(self.carryout_coefs.shape), dtype = dtype, trainable = False)
            self.carryout_LP = tf.Variable(np.zeros((input_shape[2], nb_sequences)), dtype = dtype, trainable = False)
    
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
Prev_coefs.shape
Prev_coefs[:,0,0]
Prev_biases[:,0,0]
nb_states = 2
inputs[:,:,2]
'''

class Custom_RNN_layer(tf.keras.layers.Layer):
    
    def __init__(self, nb_tracks, transition_shapes, transition_rates, density, nb_states, sequence_phase_1, sequence_phase_2, transition_sequence, transition_param_function, sequence_length = 3, vary_transition_shapes = None, vary_transition_rates = None, carryover = False, **kwargs):
        
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
        self.carryover = carryover
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        nb_states = self.nb_states
        transition_shapes, transition_rates = self.initial_transition_params
        sequence_length = self.sequence_length
        nb_tracks = self.nb_tracks
        self.transition_rates = tf.Variable(transition_rates, dtype = dtype, name = 'Transition rates', trainable = True, constraint=lambda w: tf.where(tf.greater_equal(w, tf.math.log(minval)), w, tf.math.log(minval)))
        self.transition_shapes = tf.Variable(transition_shapes, dtype = dtype, name = 'Transition shape', trainable = True)
        
        indices = tf.stack([tf.repeat(tf.constant(list(np.arange(nb_states))*sequence_length), nb_states), tf.concat([tf.range(nb_states)]*nb_states*sequence_length, 0)], axis = 1)
        transition_mask = tf.cast((indices[:,0] - indices[:,1])!=0, dtype = dtype)[None]
        self.indices = indices
        self.transition_mask = transition_mask
        
        if self.carryover:
            self.carryout_segment_len = tf.Variable(np.zeros((nb_tracks, sequence_length * nb_states)), dtype=dtype, name = 'carryover_segment_length', trainable = False)
            self.carryout_gamma_dist_mean = tf.Variable(np.zeros((nb_tracks, sequence_length * nb_states**2)), dtype=dtype, name = 'carryover_gamma_dist_mean', trainable = False)
            self.carryout_gamma_dist_var = tf.Variable(np.zeros((nb_tracks, sequence_length * nb_states**2)), dtype=dtype, name = 'carryover_gamma_dist_var', trainable = False)

        self.built = True
        
    @tf.function(jit_compile=False)
    def call(self, inputs, mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors,
             reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs,
             reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds,
             softmax_inv_Fractions, anomalous_factors, isdir, isfirst = None):
        
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
        Fs = tf.math.softmax(softmax_inv_Fractions[0, :-1])
        effective_ds = ds + 2 * tf.math.exp(anomalous_factors) * isdir
    
        transition_shapes = vary_transition_shapes * transition_shapes + (1 - vary_transition_shapes) * tf.stop_gradient(transition_shapes)
        transition_rates = vary_transition_rates * transition_rates + (1 - vary_transition_rates) * tf.stop_gradient(transition_rates)
    
        transition_shapes, transition_rates = self.transition_param_function(
            transition_shapes, transition_rates, density, Fs, effective_ds, dtype)
        '''
        reshaped_transition_Log_factors = tf.gather(transition_Log_factors, indices=indices[:, 1])[None]
        reshaped_Log_factors = tf.gather(Log_factors, indices=indices[:, 0])[None]
        reshaped_Log_factors = reshaped_transition_Log_factors * transition_mask + reshaped_Log_factors * (1 - transition_mask)

        transition_rates = tf.gather_nd(transition_rates, indices=indices)[None]
        transition_shapes = tf.gather_nd(transition_shapes, indices=indices)[None]
        '''
        
        # One-hot encodings of the row/column indices (pre-cast once)
        oh_row = tf.cast(tf.one_hot(indices[:, 0], nb_states), dtype)   # (n_pairs, nb_total)
        oh_col = tf.cast(tf.one_hot(indices[:, 1], nb_states), dtype)   # (n_pairs, nb_total)
        oh_src = tf.cast(tf.one_hot(indices[:, 1], nb_states), dtype)  # for Log_factors
        
        # Dense gather replacements — no sparse gradients
        reshaped_transition_Log_factors = (oh_src @ transition_Log_factors[:, None])[:, 0][None]
        reshaped_Log_factors            = (oh_row @ Log_factors[:, None])[:, 0][None]
        transition_rates_flat  = tf.einsum('ki,ij,kj->k', oh_row, transition_rates,  oh_col)
        transition_shapes_flat = tf.einsum('ki,ij,kj->k', oh_row, transition_shapes, oh_col)
        transition_rates  = transition_rates_flat[None]
        transition_shapes = transition_shapes_flat[None]
        reshaped_Log_factors = reshaped_transition_Log_factors * transition_mask + reshaped_Log_factors * (1 - transition_mask)
        
        segment_len = tf.ones((nb_tracks, sequence_length * nb_states), dtype=dtype)
        transition_mean = tf.repeat(transition_shapes / transition_rates, nb_tracks, axis=0)[:, :nb_states**2]
        transition_var = tf.repeat(transition_shapes / transition_rates**2, nb_tracks, axis=0)[:, :nb_states**2]
        
        gamma_dist_mean = tf.repeat(transition_shapes / transition_rates, nb_tracks, axis=0)
        gamma_dist_var = tf.repeat((transition_shapes / transition_rates)**2, nb_tracks, axis=0)
        
        if self.carryover:
            segment_len = isfirst[:, None]*segment_len + (1 - isfirst[:, None])*self.carryout_segment_len
            gamma_dist_mean = isfirst[:, None]*gamma_dist_mean + (1 - isfirst[:, None])*self.carryout_gamma_dist_mean
            gamma_dist_var = isfirst[:, None]*gamma_dist_var + (1 - isfirst[:, None])*self.carryout_gamma_dist_var
        
        # State tracking tensor: (nb_tracks, sequence_length*nb_states, sequence_length, nb_states)
        states_indices = tf.range(0, nb_states * sequence_length, dtype='int32') % nb_states
        states_indices = tf.repeat(states_indices[:, None], sequence_length, axis=1)
        states = tf.repeat(tf.one_hot(states_indices, nb_states, dtype=dtype)[None], nb_tracks, axis=0)
        
        nb_dims = reccurent_biases.shape[3]
        num_steps = tf.shape(inputs)[0]
        
        # TensorArray to accumulate predicted states at each step
        All_states_ta = tf.TensorArray(
            dtype=dtype,
            size=num_steps,
            dynamic_size=False,
            element_shape=(nb_tracks, 1, nb_states))
        
        # TensorArrays to accumulate the coefficients, biases and log probabilities
        All_coefs_ta = tf.TensorArray(dtype=dtype, size=num_steps, dynamic_size=False,
                                      element_shape=Prev_coefs.shape)   # (nb_hv, nb_tracks, nb_seq, nb_hv)
        
        All_biases_ta = tf.TensorArray(dtype=dtype, size=num_steps, dynamic_size=False,
                                       element_shape=Prev_biases.shape)  # (nb_hv, nb_tracks, nb_seq, nb_dims)
        
        All_LP_ta = tf.TensorArray(dtype=dtype, size=num_steps, dynamic_size=False,
                                       element_shape=LP.shape)  # (nb_hv, nb_tracks, nb_seq, nb_dims)
       
        # ---- loop body ----
        def body(i, Prev_coefs, Prev_biases, LP, segment_len,
                 gamma_dist_mean, gamma_dist_var, states, All_states_ta,
                 All_coefs_ta, All_biases_ta, All_LP_ta):
            
            # Compute predicted states before the RNN cell update
            log_weights = LP - nb_dims * tf.math.log(
                tf.math.abs(Prev_coefs[0, :, :, 0] * Prev_coefs[1, :, :, 1]) + 1e-20)
            
            max_log_weights = tf.reduce_max(log_weights, 1, keepdims=True)
            weights = tf.math.exp(log_weights - max_log_weights)
            weights = weights / tf.reduce_sum(weights, 1, keepdims=True)
            pred_states = tf.reduce_sum(weights[:, :, None] * states[:, :, 0], 1, keepdims=True)
            
            All_states_ta = All_states_ta.write(i, pred_states)
            All_coefs_ta = All_coefs_ta.write(i, Prev_coefs)
            All_biases_ta = All_biases_ta.write(i, Prev_biases)
            All_LP_ta = All_LP_ta.write(i, LP)
            # Gather current input and mask
            input_i = inputs[i]
            mask_i = mask[:, i]
    
            # Core recurrence
            Next_coefs, Next_biases, Next_LP, Next_segment_len, \
                Next_gamma_dist_mean, Next_gamma_dist_var, Next_states = RNN_cell(
                    input_i, Prev_coefs, Prev_biases, LP, segment_len,
                    reshaped_Log_factors, reshaped_transition_Log_factors,
                    reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
                    reccurent_next_hidden_var_coefs, reccurent_biases,
                    transition_hidden_var_coefs, transition_biases,
                    sequence_phase_1, sequence_phase_2,
                    transition_mask, transition_sequence,
                    transition_mean, transition_var,
                    gamma_dist_mean, gamma_dist_var, states)
            
            # Masked update: only advance tracks that are still alive at step i
            mask_coef = mask_i[None, :, None, None]
            mask_scalar = mask_i[:, None]
    
            Prev_coefs = Next_coefs * mask_coef + Prev_coefs * (1 - mask_coef)
            Prev_biases = Next_biases * mask_coef + Prev_biases * (1 - mask_coef)
            LP = Next_LP * mask_scalar + LP * (1 - mask_scalar)
            segment_len = Next_segment_len * mask_scalar + segment_len * (1 - mask_scalar)
            gamma_dist_mean = Next_gamma_dist_mean * mask_scalar + gamma_dist_mean * (1 - mask_scalar)
            gamma_dist_var = Next_gamma_dist_var * mask_scalar + gamma_dist_var * (1 - mask_scalar)
            
            mask_state = mask_i[:, None, None, None]
            states = Next_states * mask_state + states * (1 - mask_state)
            Next_coefs[:, 0]
            # compute the biases
            #weights = tf.math.exp(LP - nb_dims*tf.math.log(tf.math.abs(Prev_coefs[0, :,:,:,0]*Prev_coefs[1, :,:,:,1])+1e-20))
            #Prev_biases = 
            
            return i + 1, Prev_coefs, Prev_biases, LP, segment_len, \
                   gamma_dist_mean, gamma_dist_var, states, All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta
            
        cond = lambda i, *_: i < num_steps
        
        # ---- run the loop ----
        i, Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var, states, All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta = tf.while_loop(
            cond,
            body,
            loop_vars=[
                tf.constant(0),
                Prev_coefs,
                Prev_biases,
                LP,
                segment_len,
                gamma_dist_mean,
                gamma_dist_var,
                states,
                All_states_ta,
                All_coefs_ta,
                All_biases_ta,
                All_LP_ta
            ],
            parallel_iterations=1,        # sequential dependency, no benefit from >1
            swap_memory=True,              # offload to CPU RAM for long tracks
        )
        
        # Stack the TensorArray → (num_steps, nb_tracks, 1, nb_states) then transpose
        All_states = tf.transpose(All_states_ta.stack(), perm=[1, 0, 2, 3])  # (nb_tracks, num_steps, 1, nb_states)
        All_states = All_states[:, :, 0, :]  # (nb_tracks, num_steps, nb_states)
        # change dims (nb steps, nb gaussians, nb tracks, nb sequences, nb hv) to (nb tracks, nb steps, nb sequences, nb gaussians, nb hv)
        All_coefs = tf.transpose(All_coefs_ta.stack(), perm=[2, 0, 3, 1, 4])
        
        All_biases = tf.transpose(All_biases_ta.stack(), perm=[2, 0, 3, 1, 4])
        All_LPs =  tf.transpose(All_LP_ta.stack(), perm=[1, 0, 2])
        # Trim the burn-in prefix (first sequence_length - 1 steps carry incomplete history)
        All_states = All_states[:, sequence_length - 1:]
        
        return Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var, All_states, All_coefs, All_biases, All_LPs, states


def marginalise_variable(All_coefs, All_biases, integrate_index):
    """
    coefs = All_coefs[..., ::-1, :]
    biases = All_biases[..., ::-1, :]
    Given nb_gaussians=2 Gaussians characterised by coefficients and biases,
    integrate (marginalise) over the hidden variable at `integrate_index`
    using the Gaussian-product substitution.
    
    Parameters
    ----------
    coefs : ndarray, shape (..., 2, 2)
        Coefficient matrix.  coefs[..., g, h] is the coefficient of
        hidden variable h in Gaussian g.
    biases : ndarray, shape (..., 2, D)
        Bias vectors.  biases[..., g, d] is the bias of Gaussian g in
        spatial dimension d.
    integrate_index : int  (0 or 1)
        Index of the hidden variable to integrate out.
    
    Returns
    -------
    remaining_coef : ndarray, shape (...)
        Coefficient of the *remaining* hidden variable in the marginal
        Gaussian.
    remaining_bias : ndarray, shape (..., D)
        Bias of the marginal Gaussian.
    
    The marginal distribution of the remaining variable x_k is:
        P(x_k) ∝ exp(−½ (remaining_coef · x_k + remaining_bias)²)
    so that  x_k_MAP = −remaining_bias / remaining_coef
    and      Var(x_k) = 1 / remaining_coef²
    """
    keep_index = 1 - integrate_index
    nb_dims = All_biases.shape[-1]
    
    C1 = All_coefs[..., 0:1, integrate_index]   # coef of var-to-integrate in G0
    C2 = All_coefs[..., 1:, integrate_index]   # coef of var-to-integrate in G1
    
    # Normalise each Gaussian by its coefficient at integrate_index
    # (same operation as in RNN_gaussian_product)
    coefs1 = All_coefs[..., 0, :] / (C1 + 1e-30)
    coefs2 = All_coefs[..., 1, :] / (C2 + 1e-30)
    biases1 = All_biases[..., 0, :] / (C1 + 1e-30)
    biases2 = All_biases[..., 1, :] / (C2 + 1e-30)
    
    var1 = 1.0 / (C1**2 + 1e-30)
    var2 = 1.0 / (C2**2 + 1e-30)
    var3 = var1 + var2
    std3 = np.sqrt(var3)
    C1.shape
    coefs2.shape
    All_coefs.shape
    coefs3 = (coefs1 - coefs2) / std3
    biases3 = (biases1 - biases2)/std3
    
    var4 = var1 * var2 / var3
    std4 = var4**0.5
    coefs4 = (coefs1*var2 + coefs2*var1)/(var3*std4)
    biases4 = (biases1*var2 + biases2*var1)/(var3*std4)
    
    LogConstant = -nb_dims*np.log(np.abs(C1*C2*std4*std3))[:,:,:,0]

    return coefs3, biases3, coefs4, biases4, LogConstant

def extract_hidden_variables(All_coefs, All_biases, All_LPs, nb_dims, sequence_length):
    '''
    Algorithm to estimate the distribution of the hidden variables at a given time step
    given the prior positions (online estimate, equivalent to filtering in the context 
    of Kalman filters). 
    '''
    All_coefs = np.array(All_coefs)
    All_biases = np.array(All_biases)
    All_LPs = np.array(All_LPs)
    nb_tracks, track_len, nb_sequences = All_LPs.shape
    nb_states = nb_sequences//sequence_length
    
    integrate_index = 0
    coefs3, biases3, coefs4, biases4, LogConstant = marginalise_variable(All_coefs, All_biases, integrate_index)
    
    All_LPs_ano = All_LPs + LogConstant -nb_dims * np.log(np.abs(coefs4[..., integrate_index]) + 1e-30)
    All_LPs_ano = All_LPs_ano.reshape((nb_tracks, track_len, sequence_length, nb_states)) 
    ano_MAP = -biases3/coefs3[:, :, :, 1:]
    ano_MAP = ano_MAP.reshape((nb_tracks, track_len, sequence_length, nb_states, nb_dims))
    '''
    ano_MAP.shape
    ano_MAP[3, 50, 0]
    reshaped_coefs3 = coefs3[:, :, :, 1:].reshape((nb_tracks, track_len, nb_states, sequence_length))
    reshaped_coefs3.shape
    reshaped_coefs3[3, :, 0]
    ano_MAP[3, 50, :,:,0]
    coefs3[3, 50]
    coefs3.shape
    [3, :, -6, 1:]
    reshaped_coefs3[3, 50, 0]
    All_LPs_ano[3, 50]
    np.arange(30).reshape((10, 3))[:, 0]
    '''
    ano_var = 1/(coefs3[..., 1:]**2 + 1e-50)
    ano_var = ano_var.reshape((nb_tracks, track_len, sequence_length, nb_states, 1))
    w_ano = scipy_softmax(All_LPs_ano, axis=2)[..., None]  # (..., 1)
    weighted_ano_MAP = ano_MAP * w_ano
    anomalous_mean = np.sum(weighted_ano_MAP, axis=2)  # (tracks, time, nb states, nb dims)
    weighted_ano_var_term1 = (ano_var + ano_MAP**2) * w_ano
    weighted_ano_var_term2 = ano_MAP * w_ano
    anomalous_var = np.sum(weighted_ano_var_term1, axis = 2) - np.sum(weighted_ano_var_term2, axis=2)**2
    anomalous_std = anomalous_var**0.5

    integrate_index = 1
    #coefs3, biases3, coefs4, biases4, LogConstant = marginalise_variable(All_coefs, All_biases, integrate_index)
    All_LPs_pos = All_LPs - nb_dims * np.log(np.abs(All_coefs[..., integrate_index, integrate_index]) + 1e-30)
    #All_LPs_pos = All_LPs_pos.reshape((nb_tracks, track_len, sequence_length, nb_states)) 
    pos_MAP = - All_biases[..., 0, :]/All_coefs[..., 0, :1]
    #pos_MAP = pos_MAP.reshape((nb_tracks, track_len, sequence_length, nb_states, nb_dims))
    pos_var = 1/(All_coefs[..., 0, :1]**2 + 1e-50)
    #pos_var = pos_var.reshape((nb_tracks, track_len, sequence_length, nb_states, 1))
    
    w_pos = scipy_softmax(All_LPs_pos, axis=2)[..., None]  # (..., 1)
    weighted_pos_MAP = pos_MAP * w_pos
    pos_mean = np.sum(weighted_pos_MAP, axis=2)  # (tracks, T, D)
    weighted_pos_var_term1 = (pos_var + pos_MAP**2) * w_pos
    weighted_pos_var_term2 = pos_MAP * w_pos
    position_var = np.sum(weighted_pos_var_term1, axis = 2) - np.sum(weighted_pos_var_term2, axis=2)**2
    position_std = position_var**0.5
    
    # Estimate the mixture variance = E[Var] + Var[E]
    return pos_mean, anomalous_mean, position_std, anomalous_std

def extract_smooth_hidden_variables(tracks, masks, pred_model, batch_size, sequence_length, motion_types):
    '''
    Sloppy estimate of the hidden variables given all the known positions. While
    the proper way to do it is to integrate over all the hidden variables (very
    feasible but a little complex), we do that by running pred_model.predict on
    both the track and the inverse track and averaging the estimates resulting
    from extract_hidden_variables
    the filtering 
    smoothing by running the filterings on both ends
    
    motion_types: list of booleans with nb_states elements, the element i is 1
    if the state i is directed or 0 if the state i is confined.
    '''
    motion_types = np.array(motion_types)
    nb_dims = tracks.shape[-1]
    preds_1, All_coefs_1, All_biases_1, All_LPs_1 = pred_model.predict((tracks, masks), batch_size = batch_size)
    
    # transpose the shape and rate matrices for the hidden state inference on the reversed tracks
    pred_model.weights[4].assign(tf.transpose(pred_model.weights[4]))
    pred_model.weights[5].assign(tf.transpose(pred_model.weights[5]))
    
    preds_2, All_coefs_2, All_biases_2, All_LPs_2 = pred_model.predict((tracks[:,:, ::-1], masks[:,::-1]), batch_size = batch_size)
    pos_mean_1, anomalous_mean_1, position_std_1, anomalous_std_1 = extract_hidden_variables(All_coefs_1, All_biases_1, All_LPs_1, nb_dims, sequence_length)
    pos_mean_2, anomalous_mean_2, position_std_2, anomalous_std_2 = extract_hidden_variables(All_coefs_2, All_biases_2, All_LPs_2, nb_dims, sequence_length)
    
    pos_mean_1 = pos_mean_1[:, 1:]
    position_std_1 = position_std_1[:, 1:]
    pos_mean_2 = pos_mean_2[:, :0:-1]
    position_std_2 = position_std_2[:, :0:-1]
    
    # In case the motion type is directed, we need to inverse the sign of anomalous parameter 
    # of the reversed track as it represents the drift
    motion_type_sign = -1*(motion_types==1) + 1*(motion_types==0)
    motion_type_sign = motion_type_sign[None,None,:,None]
    
    anomalous_mean_1 = anomalous_mean_1[:, 1:]
    anomalous_std_1 = anomalous_std_1[:, 1:]
    anomalous_mean_2 = motion_type_sign * anomalous_mean_2[:, :0:-1]
    anomalous_std_2 = anomalous_std_2[:, :0:-1]
    
    def optimal_estimator(x1, x2, var1, var2):
        w1 = 1 / var1
        w2 = 1 / var2
        return (w1 * x1 + w2 * x2) / (w1 + w2)
    
    pos_mean = optimal_estimator(pos_mean_1, pos_mean_2, position_std_1**2, position_std_2**2)
    pos_std = (1/((1 / position_std_1**2 + 1 / position_std_2**2)))**0.5
    
    anomalous_mean = optimal_estimator(anomalous_mean_1, anomalous_mean_2, anomalous_std_1**2, anomalous_std_2**2)
    anomalous_std = (1/((1 / anomalous_std_1**2 + 1 / anomalous_std_2**2)))**0.5
    
    mean_preds = (preds_1 + preds_2[:,::-1])/2
    
    # We can average the position along the state axis as the state has a small influence on the estimated position
    w = mean_preds[:,1:-1,:, None]
    position_mean = np.sum(w * pos_mean, axis = 2)
    position_var = np.sum((pos_std**2 + pos_mean**2) * w, axis = 2) - np.sum(pos_mean * w, axis=2)**2
    position_std = position_var**0.5
    
    # The anomalous variable cannot be averaged along the state axis to avoid mixing velocity vector and potential well position which have different orders of magnitude
    return position_mean, position_std, anomalous_mean, anomalous_std, mean_preds

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

def get_model_raw_params(model, track_segmentation = False, return_dict = False):
    '''
    Function to get the raw (log-space) parameters from the model
    '''
    weights = model.get_weights()
    params = weights[0].copy()
    initial_params = weights[1].copy()
    initial_fractions = weights[2].copy()
    if track_segmentation:
        transition_shapes = weights[8].copy()
        transition_rates = weights[7].copy()
    else:
        transition_shapes = weights[5].copy()
        transition_rates = weights[4].copy()
        
    if return_dict:
        return {'params': params, 'initial_params':initial_params, 'initial_fractions': initial_fractions, 'transition_shapes':transition_shapes, 'transition_rates': transition_rates}
    else:
        return params, initial_params, initial_fractions, transition_shapes, transition_rates

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
                vary_transition_rates = None):
    
    # Defining the hyperparameters of the model
    dtype = 'float64'
    
    nb_obs_vars = 1 # number of dependend variables (the x, y, z dimension do not account as dependent variables in our model so keep this to 1)
    nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, 1, nb_independent_vars), dtype = dtype)
    input_mask = tf.keras.Input(batch_shape = (batch_size, track_len), dtype = dtype)
    
    #inputs = tracks
    #input_mask = masks
    
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
    
    self = Init_layer
    tensor1, initial_states = Init_layer(transposed_inputs)
    
    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds = Init_layer.param_vars[:, 1]
    anomalous_factors = Init_layer.param_vars[:, 2]
    isdir = Init_layer.param_vars[:, 4]
    
    Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states
    
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype = dtype)(input_mask)
    
    layer = Custom_RNN_layer(batch_size, transition_shapes, transition_rates, estimated_density, nb_states, Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, Init_layer.transition_sequence, transition_param_function, sequence_length = sequence_length, vary_transition_shapes = vary_transition_shapes, vary_transition_rates = vary_transition_rates, dtype = dtype)
    self = layer

    Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var, All_states, All_coefs, All_biases, All_LPs, states = layer(sliced_inputs, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir)
    
    F_layer = Final_layer(Init_layer.final_sequence_phase_1, nb_dims = nb_independent_vars, sequence_length = sequence_length, dtype = dtype)
    outputs, All_states = F_layer([Prev_coefs, Prev_biases, LP, All_states, states])
    
    model = tf.keras.Model(inputs=(inputs, input_mask), outputs=outputs, name="Diffusion_model")
    pred_model = tf.keras.Model(inputs=(inputs, input_mask), outputs=(All_states, All_coefs, All_biases, All_LPs), name="Diffusion_model")
    
    return model, pred_model

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

class get_parameters(tf.keras.callbacks.Callback):
    def __init__(self, track_segmentation = False, layer_name='params'):
        super(get_parameters, self).__init__()
        self.layer_name = layer_name
        self.track_segmentation = track_segmentation
    
    def on_epoch_end(self, epoch, logs=None):
        # Retrieve the weights of the model
        weights = self.model.weights
        nb_states = weights[0].shape[0]
        if self.track_segmentation:
            shape_idx = 8
            rate_idx = 7
        else:
            shape_idx = 5
            rate_idx = 4
        transition_shapes = tf.math.exp(weights[shape_idx])
        transition_rates = tf.math.softmax(weights[rate_idx], axis = 1)*transition_shapes
        
        transition_rates = np.round(transition_rates, 3)
        transition_shapes = np.round(transition_shapes, 3)
        
        transition_rates = [list(rates) for rates in transition_rates]
        transition_shapes = [list(shapes) for shapes in transition_shapes]

        model_types = weights[0][:, -1].numpy().astype(int)
        model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]
        params = {'Model types': model_types_str, 'anomalous factors': list(np.round(tf.sigmoid(weights[0][:, 2])*(1-weights[0][:, 4]) + 2**0.5*tf.exp(weights[0][:, 2])*weights[0][:, 4], 4)), 'Localization errors': list(np.round(np.exp(weights[0][:, 0]),3)), 'd': list(np.round(np.exp(weights[0][:, 1]), 3)), 'anomalous variation': list(np.round(np.exp(weights[0][:, 3]), 5)), 'transition rates': transition_rates, 'transition shapes': transition_shapes, 'Fractions': list(np.round(tf.math.softmax(weights[2][0]), 3))}
        print(params)
        #if 'loss' in model.history.history.keys() and model.history.history['loss'][-1] < -150:
        #    dfgg

def get_model_params(model, track_segmentation = False):
    weights = model.weights
    nb_states = weights[-1].shape[0]
    if track_segmentation:
        shape_IDs = 8
        rates_IDs = 7
    else:
        shape_IDs = 5
        rates_IDs = 4    
    transition_shapes = tf.math.exp(weights[shape_IDs]).numpy()
    transition_rates = tf.math.softmax(weights[rates_IDs], axis = 1)*transition_shapes
    transition_rates = transition_rates.numpy()
    model_types = weights[0][:, -1].numpy().astype(int)
    model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]
    anomalous_factors = tf.sigmoid(weights[0][:, 2])*(1-weights[0][:, 4]) + 2**0.5*tf.exp(weights[0][:, 2])*weights[0][:, 4]
    anomalous_factors = anomalous_factors.numpy()
    param_dict = {'Model types': model_types_str, 'anomalous factors': anomalous_factors, 'Localization errors': np.exp(weights[0][:, 0]), 'd': np.exp(weights[0][:, 1]), 'q': np.exp(weights[0][:, 3]), 'transition rates': transition_rates, 'transition shapes': transition_shapes, 'Fractions': tf.math.softmax(weights[2][0])}
    return param_dict

def equilibrium_distribution(P):
    """
    Compute the stationary (equilibrium) distribution of a Markov chain
    given its transition matrix P.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Row-stochastic transition matrix where P[i, j] is the probability
        of transitioning from state i to state j.

    Returns
    -------
    pi : ndarray, shape (n,)
        The equilibrium distribution vector (sums to 1).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    # We need π such that πP = π, i.e. Pᵀπ = π
    # Rearranged: (Pᵀ - I)π = 0, plus the constraint Σπ_i = 1.
    # Replace one equation with the normalization constraint.
    A = P.T - np.eye(n)
    A[-1, :] = 1.0          # replace last row with sum-to-one constraint

    b = np.zeros(n)
    b[-1] = 1.0             # right-hand side for the constraint row

    pi = np.linalg.solve(A, b)
    return pi

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

def logit(x):
    return -np.log(1/x-1)


def segment_tracks(track_list, batch_size, segment_length=20, min_segment_length=4, cutoff_batch_treshhold = 0.5, shuffle = False):
    """
    Split long tracks into shorter segments and output a batch of tracks, mask, and 
    a 1D array (batch_size) that indicates wether a segment is the first segment of
    a track or not.
    
    Parameters
    ----------
    track_list : list of np.ndarray
        List of tracks, each of shape (track_len, nb_dims).
    batch_size: integer
        Batch size
    segment_length : int
        Maximum length of each segment.
    min_segment_length : int
        Minimum length of a segment (shorter trailing segments are discarded).
    cutoff_batch_treshhold: float between 0 and 1
        Remove the last batches depending on the threshold value. If 1
    Returns
    -------
    track_batches : np.ndarray
        batches of tracks of the shape (number of batch, batch_size, seg_len, nb_dims).
    mask_batches : np.ndarray
        batches of masks of the shape (number of batch, batch_size, seg_len).
    isfirst_batches : np.ndarray
        batches of binaries indicating if each segment is the first segment of a track or not
    """
    # clip the cutoff so it selects the right number of batches even in case of incorrect cutoff entries
    if cutoff_batch_treshhold>1:
        cutoff_batch_treshhold = 1
    elif cutoff_batch_treshhold<1/batch_size:
        cutoff_batch_treshhold = 0.5/batch_size
        
    max_track_length = np.max([len(track) for track in track_list])
    max_nb_batches = (len(track_list) // batch_size + 1) * (max_track_length // segment_length + 1)*2
    track_batches = np.zeros((max_nb_batches, batch_size, segment_length, track_list[0].shape[1]))
    mask_batches = np.zeros((max_nb_batches, batch_size, segment_length))
    isfirst_batches = np.zeros((max_nb_batches, batch_size))
    
    if shuffle:
        track_list = [track_list[i] for i in np.random.permutation(len(track_list))]
    
    for track in track_list:
        
        nb_segments = len(track)//segment_length
        if len(track)%segment_length > min_segment_length:
            nb_segments += 1
        
        # Then we find where to add the segments of this track
        batch_IDs, index_IDs = np.where(mask_batches[:,:, 0]==0)
        batch_ID, index_ID = (batch_IDs[0], index_IDs[0])
        
        for i in range(nb_segments):
            segment = track[i*(segment_length-1):(i+1)*segment_length-i]
            track_batches[batch_ID + i, index_ID, :len(segment)] = segment
            mask_batches[batch_ID + i, index_ID, :len(segment)] = 1
            if i == 0:
                isfirst_batches[batch_ID, index_ID] = 1 
    
    nb_batches = np.argmin(np.mean(mask_batches[:,:,0], 1) >= cutoff_batch_treshhold)
    track_batches = track_batches[:nb_batches]
    mask_batches = mask_batches[:nb_batches]
    isfirst_batches = isfirst_batches[:nb_batches]
    return track_batches, mask_batches, isfirst_batches

'''
def data_generator(track_list, batch_size, segment_length=20,
                   min_segment_length=4, cutoff_batch_treshhold=0.5):
    """
    Non-lazy generator that yields pre-computed batches of
    (tracks, masks, isfirsts) for a Keras model.
    """

    while True:
        track_batches, mask_batches, isfirst_batches = segment_tracks(
            track_list, batch_size, segment_length,
            min_segment_length, cutoff_batch_treshhold)
        
        nb_batches = len(track_batches)
        for i in range(nb_batches):
            yield ({"tracks": track_batches[i],
                    "masks": mask_batches[i],
                    "isfirsts": isfirst_batches[i]},
                   track_batches[i],  # replace with your actual targets
            )
'''

class TrackSegmentSequence(tf.keras.utils.Sequence):
    """Keras Sequence that pre-computes all batches from segment_tracks."""

    def __init__(self, track_list, batch_size, segment_length=20,
                 min_segment_length=4, cutoff_batch_treshhold=0.5, shuffle = False):
        """
        Parameters
        ----------
        track_list, batch_size, segment_length, min_segment_length,
        cutoff_batch_treshhold : forwarded to segment_tracks.
        dummy_label_shape : tuple or None
            Per-sample label shape *after* the batch_size dimension.
            If None, defaults to a scalar zero per sample, i.e. shape (batch_size,).
        """
        self.track_list = track_list
        self.segment_length = segment_length
        self.min_segment_length = min_segment_length
        self.cutoff_batch_treshhold = cutoff_batch_treshhold
        self.shuffle = shuffle
        
        self.tracks, self.masks, self.isfirsts = segment_tracks(
            track_list, batch_size, segment_length,
            min_segment_length, cutoff_batch_treshhold)
        self.batch_size = batch_size
        
        # Pre-build dummy labels once (same array every batch)
        self.dummy_labels = np.zeros((self.batch_size,), dtype=dtype)
    
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        #inputs = {"tracks":   self.tracks[idx],    # (batch_size, seg_len, nb_dims)
        #          "masks":    self.masks[idx],      # (batch_size, seg_len)
        #          "isfirsts": self.isfirsts[idx]}   # (batch_size,)
        inputs = (self.tracks[idx],    # (batch_size, seg_len, nb_dims)
                  self.masks[idx],      # (batch_size, seg_len)
                  self.isfirsts[idx])   # (batch_size,)
        return inputs, self.dummy_labels
    
    def on_epoch_end(self):
        """Shuffle batch order between epochs."""
        if self.shuffle:
            self.tracks, self.masks, self.isfirsts = segment_tracks(
                self.track_list, self.batch_size, self.segment_length,
                self.min_segment_length, self.cutoff_batch_treshhold, self.shuffle)

class IsfirstMaskLayer(tf.keras.layers.Layer):
    """Element-wise   init_val * isfirst + prev_val * (1 - isfirst)
    with correct broadcasting for tensors of various ranks.
    """
    def __init__(self, **kwargs):
        """rank: rank of the value tensors (init_val / prev_val)."""
        super().__init__(**kwargs)

    def call(self, init_val, prev_val, isfirst):
        return init_val * isfirst + prev_val * (1 - isfirst)

class CarryoverAssignLayer(tf.keras.layers.Layer):
    def __init__(self, carryout_variables, **kwargs):
        """
        Args:
            carryout_variables: list of tf.Variables (trainable=False) 
                                to assign new values to.
        """
        super().__init__(**kwargs)
        self.carryout_variables = carryout_variables

    def call(self, output, new_states):
        """
        Args:
            output: the tensor to pass through unchanged (e.g. final output).
            new_states: list of tensors, same order as carryout_variables,
                        whose values will be assigned to the variables.
        """
        assign_ops = []
        for var, state in zip(self.carryout_variables, new_states):
            assign_ops.append(var.assign(tf.stop_gradient(state)))

        with tf.control_dependencies(assign_ops):
            return tf.identity(output)

def build_segment_model(track_len, # maximum number of time points in the input tracks 
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
                vary_transition_rates = None):
    
    # Defining the hyperparameters of the model    
    nb_obs_vars = 1 # number of dependend variables (the x, y, z dimension do not account as dependent variables in our model so keep this to 1)
    nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars
    
    inputs = tf.keras.Input(batch_shape=(batch_size, track_len, nb_independent_vars), name = 'tracks', dtype = dtype)
    input_mask = tf.keras.Input(batch_shape = (batch_size, track_len), name = 'masks', dtype = dtype)
    input_isfirst = tf.keras.Input(batch_shape = (batch_size,), name = 'isfirsts', dtype = dtype)
    '''
    seq = TrackSegmentSequence(
        track_list,
        batch_size=50,
        segment_length=20,
        min_segment_length=4,
        cutoff_batch_treshhold=0.5,
    )
    all_inputs, outputs = seq[0]
    inputs = all_inputs[0]
    input_mask = tf.constant(all_inputs[1], dtype = dtype)
    input_isfirst = tf.constant(all_inputs[2], dtype = dtype)
    
    inputs = tracks
    input_mask = masks
    input_isfirst = masks[:, 0]
    '''
    reshaped_inputs = tf.keras.layers.Lambda(lambda x: x[:, None, :, None, None, :], dtype = dtype)(inputs)
    transposed_inputs = transpose_layer(dtype = dtype)(reshaped_inputs, perm = [2, 1, 0, 3, 4, 5])
    
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
                                           carryover = True,
                                           dtype = dtype)
    
    #self = Init_layer
    #Init_layer = model.layers[5]
    tensor1, initial_states = Init_layer(transposed_inputs)
    
    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds = Init_layer.param_vars[:, 1]
    anomalous_factors = Init_layer.param_vars[:, 2]
    isdir = Init_layer.param_vars[:, 4]
    
    Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states
    
    first_mask_layer = IsfirstMaskLayer(dtype = dtype)
    Prev_coefs = first_mask_layer(Prev_coefs, Init_layer.carryout_coefs, input_isfirst[None, :, None, None])
    Prev_biases = first_mask_layer(Prev_biases, Init_layer.carryout_biases, input_isfirst[None, :, None, None])
    LP = first_mask_layer(LP, Init_layer.carryout_LP, input_isfirst[:, None])
    
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype = dtype)(input_mask)
    
    layer = Custom_RNN_layer(batch_size, transition_shapes, transition_rates, estimated_density, nb_states, Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, Init_layer.transition_sequence, transition_param_function, sequence_length = sequence_length, vary_transition_shapes = vary_transition_shapes, vary_transition_rates = vary_transition_rates, carryover = True, dtype = dtype)
    #self=layer
    Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var, All_motion_states, All_coefs, All_biases, All_LPs, motion_states = layer(sliced_inputs, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir, isfirst = input_isfirst)
    states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]
    carryover_layer = CarryoverAssignLayer(carryout_variables=[Init_layer.carryout_coefs,
                                                               Init_layer.carryout_biases,
                                                               Init_layer.carryout_LP,
                                                               layer.carryout_segment_len,
                                                               layer.carryout_gamma_dist_mean,
                                                               layer.carryout_gamma_dist_var],
                                           dtype=dtype)
    
    F_layer = Final_layer(Init_layer.final_sequence_phase_1, nb_dims = nb_independent_vars, sequence_length = sequence_length, dtype = dtype)
    outputs, All_states = F_layer(states)
    
    outputs = carryover_layer(outputs, [Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var])
    model = tf.keras.Model(inputs=(inputs, input_mask, input_isfirst),
                           outputs=outputs, name="Diffusion_model")
    
    pred_model = tf.keras.Model(inputs=(inputs, input_mask, input_isfirst),
                                outputs=(All_states, All_coefs, All_biases, All_LPs), name="Diffusion_model")
    
    '''
    model.compile(optimizer="adam", loss=MLE_loss)
    seq = TrackSegmentSequence(
        track_list,
        batch_size=50,
        segment_length=20,
        min_segment_length=4,
        cutoff_batch_treshhold=0.5)
    model.fit(seq, epochs = 5)
    '''
    
    return model, pred_model

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
                 vary_transition_rates = None,
                 track_segmentation = False):
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
    callbacks = [get_parameters(track_segmentation = track_segmentation)]
    with tf.device(device):
        history = model.fit((tracks, masks), tracks, epochs = epochs, batch_size = batch_size, callbacks = callbacks, shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])
    
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
            history = model.fit((tracks, masks), tracks, epochs = epochs, batch_size = batch_size, callbacks=callbacks, shuffle=shuffle, verbose = verbose) #, callbacks  = [l_callback])
        model.weights
        
        list(np.round(tf.math.softmax(model.weights[2][0]), 3))
        
        params, initial_params, initial_fractions, transition_shapes, transition_rates = get_model_raw_params(model)
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
    # mask = sliced_mask
    states = layer(sliced_inputs, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir)
    
    F_layer = Final_layer(Init_layer.final_sequence_phase_1, nb_dims = nb_independent_vars, sequence_length = sequence_length, dtype = dtype)
    outputs, All_states = F_layer(states)
    
    model = tf.keras.Model(inputs=(inputs, input_mask), outputs=outputs, name="Diffusion_model")
    pred_model = tf.keras.Model(inputs=(inputs, input_mask), outputs=All_states, name="Diffusion_model")
    
    return model, pred_model

def get_number_of_states(tracks,
                         params,
                         initial_params,
                         transition_shapes,
                         transition_rates,
                         initial_fractions,
                         nb_dims,
                         sequence_length,
                         max_linking_distance,
                         estimated_density,
                         epochs = 100,
                         epoch_decay = 70,
                         learning_rate = 0.02,
                         decay_rate = 0.005,
                         batch_size = 100,
                         track_masks = None,
                         vary_params = None,
                         vary_initial_params = None,
                         vary_initial_fractions = None,
                         vary_transition_shapes = False,
                         vary_transition_rates = None,
                         device = '/GPU:0',
                         track_segmentation = True,
                         segment_length = 10):
    
    if track_segmentation:
        model_builder = build_segment_model
        shuffle = False
        seq = TrackSegmentSequence(tracks,
                                   batch_size=batch_size,
                                   segment_length=segment_length,
                                   min_segment_length=4,
                                   cutoff_batch_treshhold=0.5)
        
        nb_batches = len(seq)
        tracks = np.concatenate([seq[i][0][0] for i in range(nb_batches)], axis = 0)
        track_masks = np.concatenate([seq[i][0][1] for i in range(nb_batches)], axis = 0)
        isfirst =  np.concatenate([seq[i][0][2] for i in range(nb_batches)], axis = 0)
        inputs = (tracks, track_masks, isfirst)
        decay_step = epoch_decay * nb_batches
        
    else:
        model_builder = build_model
        shuffle = True
        if type(tracks)==list:
            tracks, _, track_masks = padding(tracks)
        else:
            if type(track_masks) == type(None):
                track_masks = np.ones(tracks.shape[:2])
        if len(tracks.shape)==3:
            tracks = tracks[:, None, :, None, None, :]
        inputs = (tracks, track_masks)
        decay_step = epoch_decay * track_masks.shape[0] // batch_size
    
    track_len = track_masks.shape[1]
    nb_states = params.shape[0]
    nb_tracks = track_masks.shape[0]
    callbacks = [get_parameters(track_segmentation = track_segmentation)]

    
    # Store results for all models
    model_results = {}
    current_nb_states = nb_states
    
    # Initial parameters for the full model
    current_params = params.copy()
    current_initial_params = initial_params.copy()
    current_transition_shapes = transition_shapes.copy()
    current_transition_rates = transition_rates.copy()
    current_initial_fractions = initial_fractions.copy()
    
    while current_nb_states >= 1:
        print(f"\n{'='*60}")
        print(f"Training model with {current_nb_states} states")
        print(f"{'='*60}")
        
        # Build and train model with current number of states
        model, pred_model = model_builder(
            track_len,
            current_nb_states,
            current_params,
            current_initial_params,
            current_transition_rates,
            current_transition_shapes,
            current_initial_fractions,
            batch_size,
            nb_dims=nb_dims,
            sequence_length=sequence_length,
            max_linking_distance=max_linking_distance,
            estimated_density=estimated_density,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            vary_transition_shapes=vary_transition_shapes,
            vary_transition_rates=vary_transition_rates)
        
        preds = model.predict(inputs, batch_size=batch_size)
        print('initial predictions:', MLE_loss(preds, preds))
        
        # Compile and train
        if nb_states == current_nb_states:
            cur_epochs = 2*epochs
            lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_step*2)
        else:
            lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_step)
            cur_epochs = epochs

        adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0)
        model.compile(loss=MLE_loss, optimizer=adam, jit_compile=False)
        
        with tf.device(device):
            history = model.fit(
                inputs, 
                tracks,
                epochs= cur_epochs,
                batch_size=batch_size,
                callbacks=callbacks, 
                shuffle=shuffle, 
                verbose=1)
        
        # Calculate metrics for model selection
        with tf.device(device):
            final_preds = model.predict(inputs, batch_size = batch_size) # history.history['loss'][-1]
        log_likelihood = -MLE_loss(final_preds, final_preds)*nb_tracks  # Total log likelihood
        
        # Count parameters (excluding the mislinking state)
        num_params = (current_nb_states * 5 +  # params: LocErr, D, anomalous, q, model_type
                      current_nb_states * 1 +  # initial_params
                      current_nb_states +      # initial_fractions
                      current_nb_states ** 2 * 2 ) # transition rates and shapes
        
        # Calculate information criteria
        aic = 2 * num_params - 2 * log_likelihood
        bic = np.log(track_masks.shape[0]) * num_params - 2 * log_likelihood
        
        # Get fitted parameters
        # Get fitted parameters
        fitted_weights = model.get_weights()
        fitted_params = fitted_weights[0].copy()
        fitted_initial_params = fitted_weights[1].copy()
        fitted_initial_fractions = fitted_weights[2].copy()
        if track_segmentation:
            fitted_transition_rates = fitted_weights[7].copy()
            fitted_transition_shapes = fitted_weights[8].copy()
        else:
            fitted_transition_rates = fitted_weights[4].copy()
            fitted_transition_shapes = fitted_weights[5].copy()
        
        (fitted_params, fitted_initial_params, fitted_initial_fractions, 
        fitted_transition_shapes, fitted_transition_rates) = get_model_raw_params(
                                                    model, 
                                                    track_segmentation = track_segmentation,
                                                    return_dict = False)
        parameters = get_model_params(model, track_segmentation)
        raw_parameters = get_model_raw_params(model, track_segmentation = track_segmentation,
                                              return_dict = True)
        
        # Store results
        model_results[current_nb_states] = {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'num_params': num_params,
            'loss_history': history.history['loss'],
            'parameters': parameters,
            'raw_parameters': raw_parameters,
            'model': model,
            'pred_model': pred_model}
        
        print(f"Log-likelihood: {log_likelihood:.2f}")
        print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        # If we have more than 1 state, determine which state to remove
        if current_nb_states > 1:
            state_influences = []
            
            for state_to_remove in range(current_nb_states):
                print(f"\nTesting removal of state {state_to_remove}")
                
                # Create reduced parameter sets (removing one state)
                keep_states = [i for i in range(current_nb_states) if i != state_to_remove]
                
                reduced_params = fitted_params[keep_states]
                reduced_initial_params = fitted_initial_params[keep_states]
                
                # Adjust fractions (renormalize after removing one state)
                fractions_softmax = tf.math.softmax(fitted_initial_fractions[0])
                reduced_fractions_values = fractions_softmax.numpy()[keep_states+[current_nb_states]]
                reduced_fractions_values = reduced_fractions_values / reduced_fractions_values.sum()
                reduced_initial_fractions = np.log(reduced_fractions_values / (1 - reduced_fractions_values)).reshape(1, -1)
                
                # Reduce transition matrices
                reduced_transition_rates = fitted_transition_rates[np.ix_(keep_states, keep_states)]
                reduced_transition_shapes = fitted_transition_shapes[np.ix_(keep_states, keep_states)]
                
                # Build reduced model
                test_model, _ = model_builder(
                    track_len,
                    current_nb_states - 1,
                    reduced_params,
                    reduced_initial_params,
                    reduced_transition_rates,
                    reduced_transition_shapes,
                    reduced_initial_fractions,
                    batch_size,
                    nb_dims=nb_dims,
                    sequence_length=sequence_length,
                    max_linking_distance=max_linking_distance,
                    estimated_density=estimated_density)
                
                with tf.device(device):
                    test_preds = test_model.predict(inputs, batch_size = batch_size) #-test_history.history['loss'][-1] * track_masks.shape[0]
                test_likelihood = MLE_loss(test_preds, test_preds)
                state_influences.append((state_to_remove, test_likelihood))
                print(f"  Likelihood: {test_likelihood:.2f}")
            
            # Remove state with smallest influence (higher likelihood)
            state_influences.sort(key=lambda x: x[1])
            state_to_remove = state_influences[0][0]
            
            print(f"\n→ Removing state {state_to_remove} (least influence: {state_influences[0][1]:.2f})")
            
            # Prepare parameters for next iteration
            keep_states = [i for i in range(current_nb_states) if i != state_to_remove]
            current_params = fitted_params[keep_states]
            current_initial_params = fitted_initial_params[keep_states]
            
            # Renormalize fractions
            fractions_softmax = tf.math.softmax(fitted_initial_fractions[0])
            reduced_fractions_values = fractions_softmax.numpy()[keep_states + [current_nb_states]]
            reduced_fractions_values = reduced_fractions_values / reduced_fractions_values.sum()
            current_initial_fractions = np.log(reduced_fractions_values / (1 - reduced_fractions_values)).reshape(1, -1)
            
            current_transition_rates = fitted_transition_rates[np.ix_(keep_states, keep_states)]
            current_transition_shapes = fitted_transition_shapes[np.ix_(keep_states, keep_states)]
            
            current_nb_states -= 1
        else:
            break
    
    print(f"\n{'='*60}")
    print(f"Model Selection Results:")
    print(f"{'='*60}")
    for n_states in sorted(model_results.keys(), reverse=True):
        result = model_results[n_states]
        print(f"{n_states} states: LL={result['log_likelihood']:.1f}, "
              f"AIC={result['aic']:.1f}, BIC={result['bic']:.1f}")
    
    return model_results

'''
Estimating the parameter standard deviations or confidence interval by bootstrapping
'''

def sample_tracks_with_replacement(tracks, masks):
    nb_tracks = tracks.shape[0]
    sampling_indices = np.random.randint(0, nb_tracks, size = nb_tracks)
    sampled_tracks = tracks[sampling_indices]
    sampled_masks = masks[sampling_indices]
    return sampled_tracks, sampled_masks

def bootstrapping(model, 
                  tracks,
                  masks,
                  bootstrap_number=100,
                  epochs=100, 
                  batch_size=65,
                  learning_rate = 1/100,
                  decay_threshold = None,
                  decay_rate = None,
                  device = '/GPU:0', 
                  verbose = 1,
                  track_segmentation = False):
    '''
    If a state is not found immobile, we test the alternative state hypothesis
    
    track_segmentation to finish
    '''
    nb_tracks = tracks.shape[0]
    nb_batchs = nb_tracks // batch_size
    
    if type(decay_threshold)==type(None):
        decay_threshold = int(epochs*nb_batchs * 0.75)
    if type(decay_rate)==type(None):
        #0.001 = np.exp(-decay_rate * epochs*nb_batchs * 0.25)
        decay_rate = - np.log(0.001)/(0.25*epochs*nb_batchs) # rate so that the learning rate decreases by a 0.001 factor in the last 25% steps
    
    lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)
    callbacks = [get_parameters(track_segmentation = track_segmentation)]
    
    original_weights = [w.numpy() for w in model.weights]
    
    all_model_parameters = []
    all_likelihoods = []
    for i in range(bootstrap_number):
        # Reset model weights
        for w, ow in zip(model.weights, original_weights):
            w.assign(ow)
    
        # Reset optimizer state (momentum, variance, step counter)
        for var in model.optimizer.variables():
            var.assign(tf.zeros_like(var))
        
        sampled_tracks, sampled_masks = sample_tracks_with_replacement(tracks, masks)
        with tf.device(device):
            history = model.fit((sampled_tracks, sampled_masks), sampled_tracks, epochs = epochs, batch_size = batch_size, callbacks=callbacks, verbose = verbose) #, callbacks  = [l_callback])
                
        parameter_dict = get_model_params(model, track_segmentation)
        all_model_parameters.append(parameter_dict)
        all_likelihoods.append(history.history['loss'][-1])
    
    reshaped_params = {}
    for p in all_model_parameters[0]:
        reshaped_params[p] = np.array([all_model_parameters[k][p] for k in range(len(all_model_parameters))])

    return all_model_parameters, all_likelihoods



'''
HMC sampler for Bayesian inference
'''
from copy import deepcopy
dtype = 'float64'

# ---------------------------------------------------------------------------
# Utility: flatten / unflatten model parameters
# ---------------------------------------------------------------------------

def get_trainable_param_indices(model):
    """
    Identify which model weights are trainable and return their indices.
    In the ExaTrack model the weight order is typically:
        0 – params  (recurrence variables)
        1 – initial_params (initial variables)
        2 – initial_fractions (Fractions)
        3 – max_linking_distance  (not trainable)
        4 – transition_rates
        5 – transition_shapes
    """
    indices = []
    for i, w in enumerate(model.weights):
        if w.trainable:
            indices.append(i)
    return indices


def flatten_params(model, indices):
    """Flatten selected model weights into a single 1-D tensor."""
    parts = []
    for i in indices:
        parts.append(tf.reshape(tf.cast(model.weights[i], dtype), [-1]))
    return tf.concat(parts, axis=0)


def unflatten_params(flat, model, indices):
    """Write a flat parameter vector back into the model weights."""
    offset = 0
    for i in indices:
        w = model.weights[i]
        size = tf.reduce_prod(w.shape)
        chunk = tf.reshape(flat[offset:offset + size], w.shape)
        w.assign(tf.cast(chunk, w.dtype))
        offset += size


def shapes_and_sizes(model, indices):
    """Return the shapes and sizes of selected model weights."""
    shapes, sizes = [], []
    for i in indices:
        shapes.append(model.weights[i].shape)
        sizes.append(int(tf.reduce_prod(model.weights[i].shape)))
    return shapes, sizes


# ---------------------------------------------------------------------------
# Log-likelihood wrapper
# ---------------------------------------------------------------------------

def compute_log_likelihood(model, tracks, masks, batch_size):
    """
    Compute the *total* log-likelihood of the data under the current model
    parameters.  This mirrors ``MLE_loss`` but returns a positive scalar
    (the total LL rather than the negative mean).

    Parameters
    ----------
    model : tf.keras.Model
        The compiled ExaTrack likelihood model (not the pred_model).
    tracks : ndarray, shape (N, 1, T, 1, 1, D)
        Padded track tensor.
    masks : ndarray, shape (N, T)
        Padding masks.
    batch_size : int
        Batch size used for prediction.

    Returns
    -------
    log_lik : tf.Tensor, scalar float64
        Total log-likelihood summed over all tracks.
    """
    nb_tracks = masks.shape[0]
    y_pred = model.predict((tracks, masks), batch_size=batch_size, verbose=0)
    y_pred = tf.cast(y_pred, dtype)
    # log-sum-exp over the state/sequence dimension (axis 1) per track
    max_lp = tf.reduce_max(y_pred, axis=1, keepdims=True)
    per_track_ll = tf.math.log(
        tf.reduce_sum(tf.exp(y_pred - max_lp), axis=1, keepdims=True)
    ) + max_lp
    return tf.reduce_sum(per_track_ll)


@tf.function
def _log_likelihood_from_flat(flat_params, model_weights_trainable,
                              shapes, sizes, model_fn,
                              tracks_tensor, masks_tensor):
    """
    Pure-TF computation of log-likelihood given a flat parameter vector.
    ``model_fn`` is a callable that runs the forward pass and returns the
    raw log-probability tensor (shape [N, S]).

    This variant is used inside the gradient tape so that TF can
    differentiate through it.
    """
    # Assign flat params into weight variables
    offset = 0
    for w, sh, sz in zip(model_weights_trainable, shapes, sizes):
        w.assign(tf.reshape(flat_params[offset:offset + sz], sh))
        offset += sz

    y_pred = model_fn(tracks_tensor, masks_tensor)
    y_pred = tf.cast(y_pred, dtype)
    max_lp = tf.reduce_max(y_pred, axis=1, keepdims=True)
    per_track_ll = tf.math.log(
        tf.reduce_sum(tf.exp(y_pred - max_lp), axis=1, keepdims=True)
    ) + max_lp
    return tf.reduce_sum(per_track_ll)


# ---------------------------------------------------------------------------
# Log-prior (weakly informative defaults – customise as needed)
# ---------------------------------------------------------------------------

def default_log_prior(flat_params):
    """
    Weakly informative Gaussian prior centred at 0 with std = 10 for every
    parameter.  Replace or extend this with domain-specific priors.
    """
    return -0.5 * tf.reduce_sum(flat_params ** 2) / (10.0 ** 2)


# ---------------------------------------------------------------------------
# Leapfrog integrator
# ---------------------------------------------------------------------------

def leapfrog(q, p, grad_log_prob_fn, step_size, num_leapfrog_steps,
             mass_inv=None):
    """
    Leapfrog (Störmer-Verlet) integrator for Hamiltonian dynamics.

    Parameters
    ----------
    q : tf.Tensor  – position (parameters), shape [D]
    p : tf.Tensor  – momentum, shape [D]
    grad_log_prob_fn : callable(q) -> (log_prob, grad)
        Returns log-probability and its gradient w.r.t. q.
    step_size : float or tf.Tensor
        Integration step size (epsilon).
    num_leapfrog_steps : int
        Number of leapfrog steps (L).
    mass_inv : tf.Tensor or None
        Inverse mass (diagonal).  If None, identity is used.

    Returns
    -------
    q_new, p_new, log_prob_new : updated position, momentum, and log-prob.
    """
    if mass_inv is None:
        mass_inv = tf.ones_like(q)

    q = tf.identity(q)
    p = tf.identity(p)

    # Half-step for momentum
    log_prob, grad = grad_log_prob_fn(q)
    p = p + 0.5 * step_size * grad
    
    for i in range(num_leapfrog_steps - 1):
        # Full step for position
        q = q + step_size * mass_inv * p
        # Full step for momentum
        log_prob, grad = grad_log_prob_fn(q)
        p = p + step_size * grad

    # Final full step for position
    q = q + step_size * mass_inv * p
    # Final half-step for momentum
    log_prob, grad = grad_log_prob_fn(q)
    p = p + 0.5 * step_size * grad

    # Negate momentum for reversibility (not strictly necessary for MH)
    p = -p
    
    return q, p, log_prob


# ---------------------------------------------------------------------------
# Hamiltonian Monte Carlo sampler
# ---------------------------------------------------------------------------
class HMCSampler:
    """
    Hamiltonian Monte Carlo sampler for ExaTrack model parameters.
    
    Parameters
    ----------
    model : tf.keras.Model
        The ExaTrack *likelihood* model (the one whose output is the
        log-probability tensor of shape [N, S]).
    tracks : ndarray
        Track data, shape (N, 1, T, 1, 1, D).
    masks : ndarray
        Padding masks, shape (N, T).
    batch_size : int
        Batch size for likelihood evaluation.
    step_size : float
        Initial leapfrog step size (epsilon).
    num_leapfrog_steps : int
        Number of leapfrog steps per HMC iteration.
    log_prior_fn : callable or None
        Function  flat_params -> scalar log-prior.
        Defaults to a weak Gaussian prior.
    param_indices : list[int] or None
        Indices of model.weights to sample.  If None, all trainable
        weights are sampled.
    mass_diag : ndarray or None
        Diagonal of the mass matrix.  If None, identity is used.
        A good heuristic is to set this to the inverse variance of each
        parameter estimated from a short optimisation run.
    target_accept_rate : float
        Target Metropolis acceptance rate for dual-averaging step-size
        adaptation (used during warmup).
    fix_model_type : bool
        If True, the model-type flags (column 4 of weight 0, i.e.
        ``params[:, 4]``) are held fixed at their initial values
        throughout sampling.  This prevents HMC from drifting the
        confined/directed flag away from its initial integer value.
    """

    def __init__(self,
                 model,
                 tracks,
                 masks,
                 batch_size,
                 step_size=1e-3,
                 num_leapfrog_steps=10,
                 log_prior_fn=None,
                 param_indices=None,
                 mass_diag=None,
                 target_accept_rate=0.65,
                 fix_model_type=True):
        
        self.model = model
        self.tracks = tf.constant(tracks, dtype=dtype)
        self.masks = tf.constant(masks, dtype=dtype)
        self.batch_size = batch_size

        # Determine which weights to sample
        if param_indices is None:
            param_indices = get_trainable_param_indices(model)
        self.param_indices = param_indices
        self._shapes, self._sizes = shapes_and_sizes(model, param_indices)
        self._trainable_weights = [model.weights[i] for i in param_indices]
        self._ndim = sum(self._sizes)

        # HMC tuning knobs
        self.step_size = tf.Variable(step_size, dtype=dtype)
        self.num_leapfrog_steps = num_leapfrog_steps
        self.log_prior_fn = log_prior_fn or default_log_prior

        # Mass matrix (diagonal)
        if mass_diag is not None:
            self.mass_inv = tf.constant(1.0 / mass_diag, dtype=dtype)
        else:
            self.mass_inv = tf.ones(self._ndim, dtype=dtype)/5
        
        # ----------------------------------------------------------
        # Fixed model-type handling
        # ----------------------------------------------------------
        self.fix_model_type = fix_model_type
        
        if fix_model_type:
            # Build a boolean mask over the flat parameter vector:
            # True  = parameter is FREE  (sampled normally)
            # False = parameter is FIXED (gradient zeroed, value restored)
            free_mask = np.ones(self._ndim, dtype=bool)
            
            # Locate weight-0 (params) inside the flat vector
            flat_offset = 0
            weight_0_offset = None
            weight_0_shape = None
            for idx, sh, sz in zip(self.param_indices,
                                   self._shapes, self._sizes):
                if idx == 0:
                    weight_0_offset = flat_offset
                    weight_0_shape = sh
                    break
                flat_offset += sz
            
            if weight_0_offset is not None and weight_0_shape is not None:
                nb_states_model = weight_0_shape[0]
                nb_cols = weight_0_shape[1]  # should be 5
                for s in range(nb_states_model):
                    flat_idx = weight_0_offset + s * nb_cols + 4
                    free_mask[flat_idx] = False
            
            self._free_mask = tf.constant(free_mask, dtype=tf.bool)
            self._free_mask_float = tf.cast(self._free_mask, dtype=dtype)
            
            # Snapshot the initial fixed values so we can restore them
            q_init = flatten_params(self.model, self.param_indices)
            self._fixed_values = tf.where(self._free_mask,
                                          tf.zeros_like(q_init),
                                          q_init)
        else:
            self._free_mask = None
            self._free_mask_float = None
            self._fixed_values = None
        
        # Dual averaging for step-size adaptation
        self.target_accept_rate = target_accept_rate
        self._mu = tf.cast(tf.math.log(10.0 * step_size), dtype=dtype)
        self._log_step_size_bar = tf.Variable(0.0, dtype=dtype)
        self._h_bar = tf.Variable(0.0, dtype=dtype)
        self._gamma = 0.05
        self._t0 = 10.0
        self._kappa = 0.75
        
        # Book-keeping
        self.samples = []
        self.log_probs = []
        self.accept_count = 0
        self.total_count = 0

    # ------------------------------------------------------------------
    # helper: restore fixed parameters in a flat vector
    # ------------------------------------------------------------------
    def _enforce_fixed(self, q):
        """Replace fixed entries in *q* with their frozen initial values."""
        if self._fixed_values is not None:
            return tf.where(self._free_mask, q, self._fixed_values)
        return q

    # ------------------------------------------------------------------
    # gradient of log posterior
    # ------------------------------------------------------------------
    def _grad_log_posterior(self, q):
        """
        Returns (log_posterior, gradient) evaluated at parameter vector q.
        Fixed parameters (when fix_model_type is True) have their
        gradients zeroed out so that HMC never moves them.
        """
        q = tf.cast(q, dtype)
        # Ensure fixed values are in place before the forward pass
        q = self._enforce_fixed(q)
        
        with tf.GradientTape() as tape:
            tape.watch(q)
            # Write q into model weights
            offset = 0
            for w, sh, sz in zip(self._trainable_weights,
                                  self._shapes, self._sizes):
                w.assign(tf.reshape(q[offset:offset + sz], sh))
                offset += sz

            # Forward pass  ── uses model.__call__ so TF traces the graph
            y_pred = self.model((self.tracks, self.masks), training=False)
            y_pred = tf.cast(y_pred, dtype)

            max_lp = tf.reduce_max(y_pred, axis=1, keepdims=True)
            per_track_ll = (
                tf.math.log(
                    tf.reduce_sum(tf.exp(y_pred - max_lp), axis=1,
                                  keepdims=True)
                ) + max_lp
            )
            log_lik = tf.reduce_sum(per_track_ll)
            log_prior = self.log_prior_fn(q)
            log_post = log_lik + log_prior

        grad = tape.gradient(log_post, q)
        # Replace NaN / Inf gradients with 0 (numerical safety net)
        grad = tf.where(tf.math.is_finite(grad), grad,
                        tf.zeros_like(grad))
        
        # Zero out gradients for fixed parameters
        if self._free_mask_float is not None:
            grad = grad * self._free_mask_float
        
        return log_post, grad

    # ------------------------------------------------------------------
    # single HMC step
    # ------------------------------------------------------------------
    def _hmc_step(self, q_current, log_prob_current):
        """
        One iteration of HMC: sample momentum, leapfrog, MH accept/reject.
        Fixed parameters have zero momentum so they never move.
        """
        # Sample momentum from N(0, M)
        p_current = tf.random.normal([self._ndim], dtype=dtype
                                     ) / tf.sqrt(self.mass_inv)
        
        # Zero out momentum for fixed parameters
        if self._free_mask_float is not None:
            p_current = p_current * self._free_mask_float
        
        # Current Hamiltonian
        kinetic_current = 0.5 * tf.reduce_sum(self.mass_inv * p_current ** 2)
        H_current = -log_prob_current + kinetic_current
        
        # Leapfrog integration
        q_proposed, p_proposed, log_prob_proposed = leapfrog(
            q_current, p_current,
            self._grad_log_posterior,
            self.step_size,
            self.num_leapfrog_steps,
            self.mass_inv)
        
        # Safety: enforce fixed values after leapfrog (belt-and-suspenders)
        q_proposed = self._enforce_fixed(q_proposed)
        
        # Proposed Hamiltonian
        kinetic_proposed = 0.5 * tf.reduce_sum(
            self.mass_inv * p_proposed ** 2)
        H_proposed = -log_prob_proposed + kinetic_proposed
        
        # Metropolis-Hastings acceptance
        log_accept_ratio = H_current - H_proposed
        accept_prob = tf.minimum(1.0, tf.exp(
            tf.minimum(log_accept_ratio, tf.constant(20.0, dtype=dtype))
        ))

        u = tf.random.uniform([], dtype=dtype)
        accepted = u < accept_prob

        if accepted:
            return q_proposed, log_prob_proposed, accept_prob, True
        else:
            # Restore current params into model
            offset = 0
            for w, sh, sz in zip(self._trainable_weights,
                                  self._shapes, self._sizes):
                w.assign(tf.reshape(q_current[offset:offset + sz], sh))
                offset += sz
            return q_current, log_prob_current, accept_prob, False
    
    # ------------------------------------------------------------------
    # Dual-averaging step-size adaptation (Hoffman & Gelman, 2014)
    # ------------------------------------------------------------------
    def _adapt_step_size(self, iteration, accept_prob):
        """Dual averaging to tune epsilon during warmup."""
        m = iteration + 1.0
        w = 1.0 / (m + self._t0)
        self._h_bar.assign((1.0 - w) * self._h_bar +
            w * (self.target_accept_rate - accept_prob))
        log_eps = (self._mu -
                   tf.sqrt(m) / self._gamma * self._h_bar)
        self.step_size.assign(tf.exp(log_eps))
        m_kappa = m ** (-self._kappa)
        self._log_step_size_bar.assign(m_kappa * log_eps +
            (1.0 - m_kappa) * self._log_step_size_bar)
    
    # ------------------------------------------------------------------
    # Mass matrix adaptation from warmup samples
    # ------------------------------------------------------------------
    def _adapt_mass_matrix(self, warmup_samples):
        """
        Set the diagonal mass matrix to the empirical variance of the
        warmup samples (Welford online algorithm could be used for very
        long warmups).
        """
        if len(warmup_samples) < 20:
            return
        stacked = tf.stack(warmup_samples)
        var = tf.math.reduce_variance(stacked[::-1][:200], axis=0)
        # Regularise: don't let any variance be too small
        var = tf.maximum(var, tf.constant(1e-8, dtype=dtype))
        self.mass_inv = 1.0 / var
    
    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    def sample(self,
               num_samples=500,
               num_warmup=200,
               thin=1,
               adapt_step_size=True,
               adapt_mass_matrix=True,
               verbose=True):
        """
        Run the HMC sampler.

        Parameters
        ----------
        num_samples : int
            Number of post-warmup samples to collect.
        num_warmup : int
            Number of warmup (burn-in) iterations with adaptation.
        thin : int
            Keep every ``thin``-th sample.
        adapt_step_size : bool
            Whether to adapt the step size during warmup via dual averaging.
        adapt_mass_matrix : bool
            Whether to adapt the diagonal mass matrix from warmup samples.
        verbose : bool
            Print progress information.

        Returns
        -------
        samples : np.ndarray, shape (num_samples // thin, D)
            Posterior samples (flat parameter vectors).
        log_probs : np.ndarray, shape (num_samples // thin,)
            Log-posterior at each kept sample.
        accept_rate : float
            Overall Metropolis acceptance rate.
        """
        # Initialise from current model weights
        q = flatten_params(self.model, self.param_indices)
        q = self._enforce_fixed(q)  # ensure consistency at start
        log_prob, _ = self._grad_log_posterior(q)
        
        warmup_samples = []
        
        total_iterations = num_warmup + num_samples
        self.samples = []
        self.log_probs = []
        self.accept_count = 0
        self.total_count = 0
        
        for i in range(total_iterations):
            print(i)
            is_warmup = i < num_warmup
            
            q, log_prob, accept_prob, accepted = self._hmc_step(q, log_prob)
            
            self.total_count += 1
            if accepted:
                self.accept_count += 1
            
            # ---- Adaptation during warmup ----
            if is_warmup:
                warmup_samples.append(q.numpy().copy())
                
                if adapt_step_size:
                    self._adapt_step_size(tf.cast(i, dtype = dtype), accept_prob)
                
                # Adapt mass matrix at the midpoint of warmup
                if (adapt_mass_matrix and i % 50 == 0):
                    self._adapt_mass_matrix([tf.constant(s, dtype=dtype)
                         for s in warmup_samples])
                    if verbose:
                        print(f"  [warmup {i}] mass matrix adapted")
                
                # At the end of warmup, fix the step size
                if i == num_warmup - 1:
                    if adapt_step_size:
                        self.step_size.assign(
                            tf.exp(self._log_step_size_bar))
                    if verbose:
                        print(f"  Warmup complete.  "
                              f"step_size = {self.step_size.numpy():.6g}, "
                              f"accept rate = "
                              f"{self.accept_count / self.total_count:.2%}")
                        self.accept_count = 0
                        self.total_count = 0

            # ---- Collect samples after warmup ----
            else:
                sample_idx = i - num_warmup
                if sample_idx % thin == 0:
                    self.samples.append(q.numpy().copy())
                    self.log_probs.append(float(log_prob.numpy()))

            # ---- Progress reporting ----
            if verbose and (i + 1) % max(1, 5) == 0:
                phase = "warmup" if is_warmup else "sampling"
                rate = (self.accept_count /
                        max(1, self.total_count))
                print(f"  [{phase}  iter {i + 1}/{total_iterations}]  "
                      f"log_post = {float(log_prob.numpy()):.2f}  "
                      f"accept = {rate:.2%}  "
                      f"params = {get_model_params(self.model)}"
                      f"eps = {float(self.step_size.numpy()):.4g}")

        accept_rate = (self.accept_count /
                       max(1, self.total_count))
        if verbose:
            print(f"\nSampling done.  "
                  f"Collected {len(self.samples)} samples, "
                  f"accept rate = {accept_rate:.2%}")

        return (np.array(self.samples),
                np.array(self.log_probs),
                accept_rate)

    # ------------------------------------------------------------------
    # Convenience: unflatten samples into named parameter dicts
    # ------------------------------------------------------------------
    def unflatten_samples(self, flat_samples):
        """
        Convert an array of flat samples (N, D) into a list of
        dictionaries keyed by weight index.
        
        Returns
        -------
        list[dict[int, np.ndarray]]
        """
        results = []
        for s in flat_samples:
            d = {}
            offset = 0
            for idx, sh, sz in zip(self.param_indices,
                                    self._shapes, self._sizes):
                d[idx] = np.reshape(s[offset:offset + sz], sh)
                offset += sz
            results.append(d)
        return results
    
    def get_param_samples(self, flat_samples, weight_index):
        """
        Extract the samples for a single model weight (by its position in
        ``model.weights``) from the flat sample array.

        Parameters
        ----------
        flat_samples : ndarray, shape (N, D)
        weight_index : int
            Index into ``model.weights``.

        Returns
        -------
        ndarray, shape (N, *weight_shape)
        """
        if weight_index not in self.param_indices:
            raise ValueError(
                f"Weight {weight_index} is not among the sampled indices "
                f"{self.param_indices}")
        offset = 0
        for idx, sh, sz in zip(self.param_indices,
                                self._shapes, self._sizes):
            if idx == weight_index:
                return flat_samples[:, offset:offset + sz].reshape(
                    (-1,) + tuple(sh))
            offset += sz


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------
def run_hmc(model, tracks, masks, batch_size,
            num_samples=500,
            num_warmup=200,
            step_size=1e-3,
            num_leapfrog_steps=10,
            thin=1,
            log_prior_fn=None,
            param_indices=None,
            target_accept_rate=0.65,
            fix_model_type=True,
            verbose=True):
    
    sampler = HMCSampler(
        model=model,
        tracks=tracks,
        masks=masks,
        batch_size=batch_size,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        log_prior_fn=log_prior_fn,
        param_indices=param_indices,
        target_accept_rate=target_accept_rate,
        fix_model_type=fix_model_type)

    samples, log_probs, accept_rate = sampler.sample(
        num_samples=num_samples,
        num_warmup=num_warmup,
        thin=thin,
        verbose=verbose)

    return sampler, samples, log_probs, accept_rate

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def effective_sample_size(samples):
    """
    Estimate ESS for each parameter dimension using the initial positive
    sequence estimator (Geyer 1992).
    
    Parameters
    ----------
    samples : ndarray, shape (N, D)
    
    Returns
    -------
    ess : ndarray, shape (D,)
    """
    n, d = samples.shape
    ess = np.zeros(d)
    for j in range(d):
        x = samples[:, j]
        x = x - x.mean()
        # Auto-correlation via FFT
        fft_x = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
        acf /= acf[0]
        # Sum consecutive pairs; stop when the sum turns negative
        sum_rho = 0.0
        for t in range(0, n - 1, 2):
            rho_pair = acf[t] + (acf[t + 1] if t + 1 < n else 0.0)
            if rho_pair < 0:
                break
            sum_rho += rho_pair
        tau = -1.0 + 2.0 * sum_rho
        ess[j] = n / max(tau, 1.0)
    return ess

def r_hat(chains):
    """
    Compute the Gelman-Rubin R-hat statistic for multiple chains.

    Parameters
    ----------
    chains : list of ndarray, each shape (N, D)
        Multiple chains (at least 2).

    Returns
    -------
    rhat : ndarray, shape (D,)
    """
    m = len(chains)
    n = chains[0].shape[0]
    
    chain_means = np.array([c.mean(axis=0) for c in chains])  # (m, d)
    grand_mean = chain_means.mean(axis=0)                       # (d,)
    
    B = n / (m - 1.0) * np.sum(
        (chain_means - grand_mean[None, :]) ** 2, axis=0)
    
    W = np.mean([c.var(axis=0, ddof=1) for c in chains], axis=0)
    
    var_hat = (n - 1.0) / n * W + B / n
    return np.sqrt(var_hat / (W + 1e-30))

def transform_hmc_samples(flat_samples, sampler):
    """
    Transform raw HMC flat parameter samples into physical parameters.
    
    Parameters
    ----------
    flat_samples : ndarray, shape (N, D)
        Raw flat samples from HMCSampler.sample().
    sampler : HMCSampler
        The sampler instance (needed to unflatten columns into weight shapes).
    
    Returns
    -------
    dict of ndarray
        Each key maps to an array of shape (N, ...) with the transformed
        parameter values across all samples.
    """
    unflattened = sampler.unflatten_samples(flat_samples)
    N = len(unflattened)
    
    # Peek at first sample to determine sizes
    s0 = unflattened[0]
    nb_states = s0[0].shape[0]           # weight 0: params, shape (nb_states, 5)
    nb_fractions = s0[2].shape[1]        # weight 2: initial_fractions, shape (1, nb_states+1)
    
    # Pre-allocate output arrays
    model_types        = np.zeros((N, nb_states), dtype=int)
    anomalous_factors  = np.zeros((N, nb_states))
    localization_errors = np.zeros((N, nb_states))
    d_values           = np.zeros((N, nb_states))
    fractions          = np.zeros((N, nb_fractions))
    tr_shapes          = np.zeros((N, nb_states, nb_states))
    tr_rates           = np.zeros((N, nb_states, nb_states))
    
    for i, sample_dict in enumerate(unflattened):
        params              = sample_dict[0]   # (nb_states, 5): [LocErr, d, anomalous, q, model_type]
        # initial_params    = sample_dict[1]   # (nb_states, 1) — not needed for physical params
        initial_fractions   = sample_dict[2]   # (1, nb_states+1)
        transition_rates_raw  = sample_dict[4] # (nb_states, nb_states)
        transition_shapes_raw = sample_dict[5] # (nb_states, nb_states)
        
        is_dir = params[:, 4]                  # ~0 for confined, ~1 for directed
        
        model_types[i]         = (is_dir > 0.5).astype(int)
        localization_errors[i] = np.exp(params[:, 0])
        d_values[i]            = np.exp(params[:, 1])
        
        # Anomalous factor: sigmoid(a)*(1-isdir) + sqrt(2)*exp(a)*isdir
        a = params[:, 2]
        anomalous_factors[i] = (scipy.special.expit(a) * (1.0 - is_dir)
                                + np.sqrt(2) * np.exp(a) * is_dir)
        
        # Fractions via softmax
        fractions[i] = scipy.special.softmax(initial_fractions[0])
        
        # Transition kinetics: shapes are exp, rates are softmax * shapes
        tr_shapes[i] = np.exp(transition_shapes_raw)
        tr_rates[i]  = scipy.special.softmax(transition_rates_raw, axis=1) * tr_shapes[i]
    
    return {'Model types':          model_types,
            'anomalous factors':    anomalous_factors,
            'Localization errors':  localization_errors,
            'd':                    d_values,
            'Fractions':            fractions,
            'transition shapes':    tr_shapes,
            'transition rates':     tr_rates}


