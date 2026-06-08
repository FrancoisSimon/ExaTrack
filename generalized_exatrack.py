# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:14:51 2026

@author: Franc
"""

import numpy as np
import tensorflow as tf

dtype = 'float64'
pi = tf.constant(np.pi, dtype = dtype)
minval = np.array(1e-14)

from matplotlib import pyplot as plt
from numba import njit, typed, prange, jit
from scipy.stats import gamma
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import gc

import pandas as pd
from glob import glob
from scipy.special import softmax as scipy_softmax

import scipy
from scipy.spatial.transform import Rotation as R
jit_compile = False

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

def padding(track_list, LocErr_list = None, dt_list = None, batch_size = None):
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
    if type(dt_list)!= type(None):
        padded_dts =  np.zeros((nb_tracks, max_len+1), dtype = dt_list[0].dtype) # we need to have a frame array with one additional time step to compute the time step ratios in constraint_function
    else:
        padded_dts = None
    if type(LocErr_list)!= type(None):
        if len(LocErr_list[0].shape)==2:
            padded_LocErrs =  np.zeros((nb_tracks, max_len, LocErr_list[0].shape[1]), dtype = LocErr_list[0].dtype)
        else:
            padded_LocErrs =  np.zeros((nb_tracks, max_len), dtype = LocErr_list[0].dtype)
    else:
        padded_LocErrs = None
    
    mask = np.zeros((nb_tracks, max_len), dtype = track_list[0].dtype)
    
    for i, track in enumerate(track_list):
        if track.shape[0]>=start_len:
            cur_len = track.shape[0]
            padded_tracks[i, :cur_len] = track
            padded_tracks[i, cur_len:] = track[-1] # padding that replicates the edges in case we want to use ExaTrack on the reversed tracks

            mask[i, :cur_len] = 1
            if type(dt_list)!= type(None):
                dts = dt_list[i]
                padded_dts[i, :cur_len] = dts
                padded_dts[i, cur_len:] = dts[-1]
                
            if type(LocErr_list)!= type(None):
                LocErrs = LocErr_list[i]
                padded_LocErrs[i, :cur_len] = LocErrs
                padded_LocErrs[i, cur_len:] = LocErrs[-1]
        else:
            raise Warning('The minimal track length supported is 2 time points. Tracks of 1 time point were discarded.')
    
    return padded_tracks, padded_LocErrs, padded_dts, mask


def _sample_transitions(state, current_sub_idx, cum_sub_times,
                        shape_matrix, transition_matrix, nb_states, dt_mean):
    """
    Sample, for each candidate target state, the *lifetime* of the current
    state segment in continuous time, then convert it to a sub-step count
    through `cum_sub_times`.

    Because each sub-step can have a different physical duration, the same
    sampled lifetime can map to different sub-step counts depending on where
    the segment starts. Under fixed dt this reduces exactly to the original
    CPD-based sampling, since:
        gamma(shape, dt/rate)        -- in time units
        =  (dt/nb_sub_steps) * gamma(shape, nb_sub_steps/rate)  -- in sub-steps
    """
    transitions = np.full(nb_states, len(cum_sub_times), dtype=np.int64)
    current_time = cum_sub_times[current_sub_idx]
    for target in range(nb_states):
        if target != state and transition_matrix[state, target] > 0:
            lifetime = np.random.gamma(
                shape_matrix[state, target],
                dt_mean / transition_matrix[state, target],
            )
            end_idx = np.searchsorted(cum_sub_times, current_time + lifetime)
            # at least 1 sub-step so the simulation always makes progress
            transitions[target] = max(1, end_idx - current_sub_idx)
    return transitions

def anomalous_diff_transition(max_track_len=100,                          # Maximal track length
                              nb_tracks=100,                              # Number of tracks
                              LocErr=0.02,                                # Localization error
                              Fs=np.array([0., 1]),                       # Initial fractions of each state
                              Ds=np.array([0.0, 0.25]),                   # Diffusion coefficients
                              nb_dims=2,                                  # number of dimensions (x, y, ..)
                              velocities=np.array([0.03, 0.0]),           # Speed of the directed motion
                              angular_Ds=np.array([0.0, 0.0]),            # Angular diffusion coefficient of the directed velocity vector 
                              conf_forces=np.array([0.0, 0.2]),           # Force of the attraction towards the potential well (in between 0 for no confinement and 1 for purely confined particles)
                              conf_Ds=np.array([0.0, 0.0]),               # Diffusion coefficient of the potential well (for confined motion)
                              conf_dists=np.array([0.0, 0.0]),            # Standard deviation of the potential well when confinement appears 
                              transition_matrix=np.array([[0.00, 0.1],    # Matrix of the rates for the transition model
                                                          [0.1, 0.00]]),  
                              shape_matrix=np.array([[0, 1],              # Matrix of the shapes for the transition model
                                                     [1, 0]]),             
                              bleaching_rate=1e-10,                       # Bleaching rate per time step
                              LocErr_std=0.002,                           # Standard deviation of the localization error, can be set to 0 for constant localization error
                              dt=0.02,                                    # Time step
                              dt_std=0.002,                               # Standard deviation of the time steps, can be set to 0 for constant time steps
                              field_of_view=np.array([10, 10]),           # Size of the field of view
                              nb_burning_steps=100,                       # Number of steps used before actually simulating the tracks to better simulate lifetimes and equilibrium fractions
                              nb_sub_steps=10,
                              return_list = False):                           # Number of sub-steps that compose each step (to simulate continuous transitions)
    
    nb_states = len(velocities)
    if not np.all(np.array([len(Fs), len(Ds), len(velocities), len(angular_Ds),
                            len(conf_forces), len(conf_Ds), len(conf_dists),
                            len(transition_matrix)]) == nb_states):
        raise ValueError('Fs, Ds, velocities, angular_Ds, conf_forces, conf_Ds, '
                         'conf_dists and transition_matrix must all be arrays of '
                         'the same length (one element per state)')
    
    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for s in range(1, nb_states):
        cum_Fs[s] = cum_Fs[s - 1] + Fs[s]
    
    # NOTE: the precomputed CPD_transition_mat is no longer needed. With variable
    # time steps, sub-step durations are not constant, so transition lifetimes
    # are now sampled in continuous time and converted to sub-step indices via
    # the cumulative sub-step time array `cum_sub_times` (built per track below).
    
    all_tracks   = np.zeros((nb_tracks, max_track_len, nb_dims))
    all_states   = np.zeros((nb_tracks, max_track_len))
    all_masks    = np.zeros((nb_tracks, max_track_len))
    all_LocErrs  = np.zeros((nb_tracks, max_track_len, nb_dims))
    all_dts      = np.zeros((nb_tracks, max_track_len))
    
    LocErr = np.array([LocErr])
    
    for k in range(nb_tracks):
        # ---- Track length (bleaching) ----
        if bleaching_rate / nb_sub_steps > 1e-8:
            track_len = min(max_track_len, np.random.geometric(p=bleaching_rate))
        else:
            track_len = max_track_len

        # ---- Variable real time steps (one per actual time step) ----
        if dt_std > 0:
            dt_scale = dt_std ** 2 / dt
            dt_shape = dt / dt_scale
            dts = np.random.gamma(dt_shape, dt_scale, track_len)
            #dts[:] = dts[0]
            dts[dts<0.05*dt] = 0.05*dt
            dts[dts>3*dt] = 3*dt
        else:
            dts = np.full(track_len, dt)
        
        # ---- Cumulative sub-step times (the new state-clock) ----
        # Each sub-step within real time-step `i` has duration dts[i]/nb_sub_steps.
        # The burn-in does not simulate positions, so we use the mean dt there;
        # only the *timing* of state transitions matters during burn-in.
        burn_in_subs_total = nb_burning_steps * nb_sub_steps
        burn_sub_dts = np.full(burn_in_subs_total, dt / nb_sub_steps)
        main_sub_dts = np.repeat(dts / nb_sub_steps, nb_sub_steps)
        all_sub_dts  = np.concatenate([burn_sub_dts, main_sub_dts])
        cum_sub_times = np.concatenate([[0.0], np.cumsum(all_sub_dts)])
        # cum_sub_times[i] = time at the *start* of sub-step i.
        # Length = burn_in_subs_total + track_len * nb_sub_steps + 1.
        
        # ---- Initial position and state ----
        initial_positions = np.random.rand(nb_dims) * field_of_view
        track = []
        states = []
        next_state = np.argmin(np.random.rand() > cum_Fs)
        
        # ---- Burn-in: randomise the initial state ----
        n = 0
        while n <= burn_in_subs_total:
            state = next_state
            transitions = _sample_transitions(
                state, n, cum_sub_times, shape_matrix,
                transition_matrix, nb_states, dt)
            next_state = np.argmin(transitions)
            n += int(np.min(transitions))
        # Override the chosen transition's count with the residual that spills
        # over into the main simulation. Because that residual is measured in
        # sub-steps of `cum_sub_times`, it already accounts for variable dt.
        transitions[next_state] = n - burn_in_subs_total

        # If the segment ended exactly at the burn-in cutoff, draw a fresh one.
        if transitions[next_state] == 0:
            state = next_state
            transitions = _sample_transitions(
                state, burn_in_subs_total, cum_sub_times, shape_matrix,
                transition_matrix, nb_states, dt,
            )
        
        # ---- Main simulation loop ----
        track_len_subs = track_len * nb_sub_steps
        while len(track) < track_len_subs:
            if len(track) > 0:
                transitions = _sample_transitions(
                    state, burn_in_subs_total + len(track), cum_sub_times,
                    shape_matrix, transition_matrix, nb_states, dt)
        
            l = min(int(np.min(transitions)), track_len_subs - len(track))
        
            D, velocity, angular_D, conf_force, conf_D, conf_dist = (
                Ds[state], velocities[state], angular_Ds[state],
                conf_forces[state], conf_Ds[state], conf_dists[state])
            
            # Per-sub-step vectors covering the entire state segment of length `l`.
            # The state is constant across the segment, so velocity/conf_force are
            # uniform; dt varies because consecutive sub-steps may belong to
            # different real time steps with different sampled dts[ts_idx].
            seg_start_sub = len(track)
            sub_indices = np.arange(seg_start_sub, seg_start_sub + l)
            ts_indices  = sub_indices // nb_sub_steps
            sub_dts_seg     = dts[ts_indices] / nb_sub_steps
            velocity_seg    = np.full(l, velocity   / nb_sub_steps)
            conf_force_seg  = np.full(l, conf_force / nb_sub_steps)
            
            if nb_dims < 3:
                segment = anomalous_diff_2D(track_len=l + 1,
                                            LocErr=0,
                                            D=D,
                                            velocity=velocity_seg,
                                            angular_D=angular_D,
                                            conf_force=conf_force_seg,
                                            conf_D=conf_D,
                                            conf_dist=conf_dist,
                                            dt=sub_dts_seg,
                                            nb_sub_steps=1,
                                            initial_positions=initial_positions)
                segment = segment[:, :nb_dims]
            elif nb_dims == 3:
                segment = anomalous_diff_3D(track_len=l + 1,
                                            LocErr=0,
                                            D=D,
                                            velocity=velocity_seg,
                                            angular_D=angular_D,
                                            conf_force=conf_force_seg,
                                            conf_D=conf_D,
                                            conf_dist=conf_dist,
                                            dt=sub_dts_seg,
                                            nb_sub_steps=1,
                                            initial_positions=initial_positions)
            else:
                raise ValueError('The number of dimensions must be 1, 2 or 3')
                
            track  += list(segment[:-1])
            states += [state] * l
            initial_positions = segment[-1]
        
            state = np.argmin(transitions)
            
        # ---- Localisation error (unchanged) ----
        if LocErr_std > 0:
            scale = LocErr_std ** 2 / LocErr
            shape = LocErr / scale
            LocErrs = np.random.gamma(shape, scale, (track_len, nb_dims))
        else:
            LocErrs = LocErr

        track  = np.array(track)[::nb_sub_steps] + np.random.normal(0, LocErrs, (track_len, nb_dims))
        states = np.array(states)[::nb_sub_steps]

        all_tracks[k, :track_len]  = track
        all_tracks[k, track_len:]  = track[-1]
        all_LocErrs[k, :track_len] = LocErrs
        all_LocErrs[k, track_len:] = LocErr[-1]
        all_dts[k, :track_len]     = dts
        all_dts[k, track_len:]     = dts[-1]
        all_states[k, :track_len]  = states
        all_states[k, track_len:]  = states[-1]
        all_masks[k, :track_len - 1] = 1
        all_masks[k, -1]             = 1

    return all_tracks, all_LocErrs, all_dts, all_states, all_masks

@njit
def anomalous_diff_2D(track_len=20,
                      LocErr=0.02,
                      D=0.05,
                      velocity=None,     # array, length track_len*nb_sub_steps - 1
                      angular_D=0.0,
                      conf_force=None,   # array, length track_len*nb_sub_steps - 1
                      conf_D=0.0,
                      conf_dist=0.0,
                      dt=None,           # array, length track_len*nb_sub_steps - 1
                      nb_sub_steps=10,
                      initial_positions=np.array([0., 0.])):
    nb_dims = 2
    n_disps = track_len * nb_sub_steps - 1

    positions = np.zeros((track_len * nb_sub_steps, nb_dims))
    positions[0] = initial_positions

    # Thermal diffusion: per-sub-step std = sqrt(2 * D * dt[i])
    disps = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * D * dt[i])
        for d in range(nb_dims):
            disps[i, d] = np.random.normal(0.0, s)

    # Confining anchor positions: random walk with per-step std sqrt(2 * conf_D * dt[i])
    anchor_positions = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * conf_D * dt[i])
        for d in range(nb_dims):
            anchor_positions[i, d] = np.random.normal(0.0, s)
    for d in range(nb_dims):
        anchor_positions[0, d] = positions[0, d] + np.random.normal(0.0, conf_dist)
    for i in range(1, n_disps):
        for d in range(nb_dims):
            anchor_positions[i, d] += anchor_positions[i - 1, d]

    # Angular diffusion: cumulative angle, per-step std = sqrt(2 * angular_D * dt[i-1])
    angles = np.zeros(n_disps)
    angles[0] = np.random.rand() * 2.0 * np.pi
    for i in range(1, n_disps):
        d_angle = np.random.normal(0.0, np.sqrt(2.0 * angular_D * dt[i - 1]))
        angles[i] = angles[i - 1] + d_angle

    # Per-sub-step update
    for i in range(n_disps):
        cos_a = np.cos(angles[i])
        sin_a = np.sin(angles[i])
        # velocity[i] is the directed-displacement magnitude for this sub-step
        positions[i + 1, 0] = positions[i, 0] + cos_a * velocity[i] + disps[i, 0]
        positions[i + 1, 1] = positions[i, 1] + sin_a * velocity[i] + disps[i, 1]
        cf = conf_force[i]
        positions[i + 1, 0] = (1.0 - cf) * positions[i + 1, 0] + cf * anchor_positions[i, 0]
        positions[i + 1, 1] = (1.0 - cf) * positions[i + 1, 1] + cf * anchor_positions[i, 1]

    final_track = np.zeros((track_len, nb_dims))
    for i in range(track_len):
        final_track[i] = positions[i * nb_sub_steps]

    if LocErr > 0:
        final_track += np.random.normal(0.0, LocErr, (track_len, nb_dims))
    return final_track


def anomalous_diff_3D(track_len=20,
                      LocErr=0.02,
                      D=0.05,
                      velocity=None,     # array, length track_len*nb_sub_steps - 1
                      angular_D=0.0,
                      conf_force=None,   # array, length track_len*nb_sub_steps - 1
                      conf_D=0.0,
                      conf_dist=0.0,
                      dt=None,           # array, length track_len*nb_sub_steps - 1
                      nb_sub_steps=10,
                      initial_positions=np.array([0., 0., 0.])):
    nb_dims = 3
    n_disps = track_len * nb_sub_steps - 1

    positions = np.zeros((track_len * nb_sub_steps, nb_dims))
    positions[0] = initial_positions

    # Thermal diffusion (per-sub-step std)
    disps = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * D * dt[i])
        disps[i] = np.random.normal(0.0, s, nb_dims)

    # Confining anchors
    anchor_positions = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * conf_D * dt[i])
        anchor_positions[i] = np.random.normal(0.0, s, nb_dims)
    anchor_positions[0] = positions[0] + np.random.normal(0.0, conf_dist, nb_dims)
    for i in range(1, n_disps):
        anchor_positions[i] += anchor_positions[i - 1]

    # Persistent (directed) displacements with rotational diffusion — now vector-aware
    pesistent_displacements = simulate_3D_rotational_diffusion(
        n_disps, velocity, angular_D, dt
    )

    for i in range(n_disps):
        positions[i + 1] = positions[i] + pesistent_displacements[i] + disps[i]
        cf = conf_force[i]
        positions[i + 1] = (1.0 - cf) * positions[i + 1] + cf * anchor_positions[i]

    final_track = np.zeros((track_len, nb_dims))
    for i in range(track_len):
        final_track[i] = positions[i * nb_sub_steps]

    if LocErr > 0:
        final_track += np.random.normal(0.0, LocErr, (track_len, nb_dims))
    return final_track


def simulate_3D_rotational_diffusion(nb_steps, velocities, D_r, dts):
    """
    velocities : array of length nb_steps -- per-step directed-displacement magnitude
    dts        : array of length nb_steps -- per-step time
    Returns an (nb_steps, 3) array of directed displacements.
    """
    theta = 2.0 * np.pi * np.random.rand()
    phi = np.arccos(2.0 * np.random.rand() - 1.0)
    v = np.array([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)])
    v = v / np.linalg.norm(v)

    vs = np.zeros((nb_steps, 3))
    vs[0] = v
    for i in range(1, nb_steps):
        sigma_theta = np.sqrt(2.0 * D_r * dts[i - 1])
        dtheta = np.random.normal(0.0, sigma_theta, size=3)
        v = R.from_rotvec(dtheta).apply(v)
        v = v / np.linalg.norm(v)
        vs[i] = v

    # Magnitude scaling per step
    return vs * velocities[:, None]

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

@tf.function(jit_compile=jit_compile)
def log_gaussian(top, variance=tf.constant(1, dtype = dtype)):
    return - 0.5*tf.math.log(2*pi*variance) - top**2/(2*variance)


@tf.function(jit_compile=jit_compile)
def norm_log_gaussian(top):
    return - 0.5*(tf.math.log(2*pi) + top**2)

def RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2, next_hidden_var_coefs_1, next_hidden_var_coefs_2, biases_1, biases_2, coef_index, nb_dims = 1):
    '''
    Basic function of the method to simplify a product of two Gaussians that both depend on
    a hidden variable of index `coef_index` into one gaussian that depends on this variable
    and another Gaussian that is independent of this variable. 
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

#%% Functions that need to be executed during the different steps of the automated integration process

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
#%% Functions that need to be executed during the different steps of the automated integration process


@tf.function(jit_compile=jit_compile)
def RNN_reccurence_formula(current_hidden_var_coefs, # coefficients of the hidden variables that are updated
                           next_hidden_var_coefs,
                           biases,
                           sequence_phase_1,
                           sequence_phase_2,
                           nb_dims,
                           dtype = 'float64'): # False by default, set to true when aiming to compute the scaling factor
    '''
    RNN_reccurence_formula is the function that organizes and executes the different
    steps of the automated integration process. 
    
    We first integrate over the current hidden variables. During the phase 1, we 
    perform Gaussian swaps to express the joint probability so a single Gaussian
    depends on the variable to be integrated over. This last Gaussian is then eliminated
    as the integrale of a single gaussian equals 1. This is repeated for all the 
    hidden variables of the current step to retreive the Gaussians that reprensent
    the posterior on the hidden variables (next_hidden_var_coefs, biases, LP).
    
    The phase 2 is then applied to reorder the remaining Gaussians (posterior) so the coefficients
    have the same patterns than the prior gaussians.
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
    Addaptation of RNN_reccurence_formula in case of transitions between states i to j with i!=j
    This additional step is required to integrate over the previous hidden variable that 
    disappears during transitions
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

# self =tf.keras.layers.Layer(dtype = dtype)
class Initial_layer_constraints(tf.keras.layers.Layer):
    '''
    First layer of the model that initializes the parameters and variables

    Responsibilities
    ----------------
    1. Owns and creates all trainable parameters of the dynamical model:
         - `param_vars`           : per-state recurrence parameters
                                    (log_LocErr, log_d, ano, log_q, is_directed_flag)
         - `initial_param_vars`   : per-state initial-spread parameters
         - `initial_fractions`    : softmax-parametrised initial state mix
         - `max_linking_distance` : (non-trainable) mislinking radius
       plus optional non-trainable carry-over buffers used when the model is
       run in segmented mode (`carryover=True`).

    2. On `call`, builds the full set of Gaussian coefficients/biases for
       every time step by invoking `constraint_function`, appends an extra
       "mislinking" state, performs the first recurrence step (t = 0) by
       calling `RNN_reccurence_formula`, and returns the loop-carried tuple
       consumed by `Custom_RNN_layer`.

    Constructor arguments
    ---------------------
    nb_states              : int, number of physical (non-mislinking) states.
    nb_gaussians           : int, number of Gaussian factors per recurrence step.
    nb_obs_vars            : int, number of observed variables (typically 1: position).
    nb_hidden_vars         : int, number of hidden variables per time step
                             (e.g. 2 = position + anomalous variable).
    params                 : array (nb_states, 5), initial recurrence parameters.
    initial_params         : array (nb_states, >=1), initial-spread parameters.
    initial_fractions      : array (1, nb_states+1), pre-softmax initial mixture.
    max_linking_distance   : scalar, used to define the mislinking state.
    constraint_function    : callable, see `constraint_function` below.
    reference_dt           : scalar, reference frame duration the parameters
                             are expressed in. Per-step rescaling to actual
                             `dts` is done inside `constraint_function`.
    vary_params,
    vary_initial_params,
    vary_initial_fractions : optional float masks (same shape as the
                             corresponding parameter arrays). Entries equal
                             to 1 train normally; entries equal to 0 are
                             frozen via `tf.stop_gradient`. Defaults to all-ones.
    sequence_length        : int, number of parallel state-history sequences
                             tracked by the segmented RNN (default 3).
    carryover              : bool, if True allocate buffers that persist
                             between successive batches of the same tracks.
    LocErr_type            : string,
                            'Identity': Use the localization error inputs as is.
                            'Linear'  : Multiplies the localization error inputs 
                                        by the localization error parameter.
                                        Useful to avoid a biased estimates 
                                        (adding an ofset would be good too)
                            'Photon'  : In case the localization error represents
                                        a metric proportional to the number of photons
                                        (e.g. the quality), we estimate the localization
                                        error as input_LocErrs**0.5 * LocErr parameter. 
                            'Constant': Do not use input_LocErrs but only the 
                                        localization error parameter.

    `call` arguments
    ----------------
    inputs        : tensor of shape (track_len, nb_gaussians, nb_tracks,
                                     nb_states, nb_obs_vars, nb_dims).
                    Already-transposed observations.
    input_LocErrs : per-step localisation error, shape compatible with
                    `constraint_function` (see its docstring).
    input_dts     : per-step frame durations, shape (nb_tracks, track_len+1).
    
    `call` outputs
    --------------
    inputs         : passed through unchanged for downstream layers.
    initial_states : list of tensors fed to `Custom_RNN_layer`:
                     [Next_coefs, Next_biases, LP,
                      Log_factors, transition_Log_factors,
                      reccurent_obs_var_coefs,
                      reccurent_hidden_var_coefs,
                      reccurent_next_hidden_var_coefs,
                      reccurent_biases,
                      transition_hidden_var_coefs,
                      transition_biases].
    '''
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
        reference_dt,
        vary_params = None,
        vary_initial_params = None,
        vary_initial_fractions = None,
        sequence_length = 3,
        carryover = True, # do we want a segmented model that carries over the hidden states of the model to the next batches 
        LocErr_type = 'Linear', # Choose between Identity to use
        **kwargs):
        '''
        Stores configuration on `self` and pre-computes the integration
        schedules used by `RNN_reccurence_formula`:
            initial_sequence_phase_1 and _2,
            recurrent_sequence_phase_1 and _2,
            final_sequence_phase_1,
            transition_sequence
        via `get_sequences(...)`.
        
        Vary-masks default to all-ones (i.e. fully trainable) when None.
        '''
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
        self.reference_dt = reference_dt
        
        initial_sequence_phase_1, initial_sequence_phase_2, recurrent_sequence_phase_1, recurrent_sequence_phase_2, final_sequence_phase_1, transition_sequence = get_sequences(params, initial_params, constraint_function, nb_gaussians, nb_hidden_vars, dtype)
        
        self.initial_sequence_phase_1 = initial_sequence_phase_1
        self.initial_sequence_phase_2 = initial_sequence_phase_2
        self.recurrent_sequence_phase_1 = recurrent_sequence_phase_1
        self.recurrent_sequence_phase_2 = recurrent_sequence_phase_2
        self.transition_sequence = transition_sequence
        self.final_sequence_phase_1 = final_sequence_phase_1
        self.carryover = carryover
        self.LocErr_type = LocErr_type
    
    def build(self, input_shape):
        '''
        Allocates the trainable model parameters as tf.Variables:
            - param_vars              : log-domain recurrence parameters,
                                        constrained from below by log(minval).
            - initial_param_vars      : log-domain initial-spread parameters,
                                        constrained from below by log(minval).
            - initial_fractions       : pre-softmax mixture weights.
            - max_linking_distance_param : non-trainable scalar.
        
        If `carryover=True`, also allocates non-trainable buffers
        (carryout_coefs, carryout_biases, carryout_LP) sized so the
        hidden-state representation can be carried from one batch to the
        next without re-initialising. `nb_sequences = sequence_length *
        (nb_states + 1)` accounts for the extra mislinking state.
        
        inputs = transposed_inputs
        input_shape = inputs.shape
        '''
        
        dtype = self.dtype
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
        
        if self.LocErr_type == 'Identity':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErrs
        elif self.LocErr_type == 'Linear':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErrs*LocErr_param
        elif self.LocErr_type == 'Photon':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErr_param/LocErrs**0.5
        elif self.LocErr_type == 'Constant':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErrs*0 + LocErr_param
        else:
            raise ValueError("Wrong LocErr_type, can be 'Identity', 'Linear', 'Photon' or 'Constant'.")
        self.LocErr_function = LocErr_function
    
    def call(self, inputs, input_LocErrs, input_dts):
        '''
        Performs the first time step (t = 0) of the RNN.

        Inputs
        ------
        inputs        : (track_len, nb_gaussians, nb_tracks, nb_states,
                         nb_obs_vars, nb_dims) — observed tracks.
        input_LocErrs : per-step localisation errors for each track.
        input_dts     : per-step frame durations for each track.

        Pipeline
        --------
        1. Apply `vary_*` stop-gradient masks to the trainable parameters that should not be varied.
        2. Optional `duplicate_states` hook (for parameter sharing in personalized versions of the model).
        3. Append the mislinking state to `param_vars` and `initial_param_vars`; bump `nb_states` by one.
        4. Call `constraint_function` to obtain the Gaussian coefficients,
           biases, std-rescaling factors and per-step Log_factors for aLL
           time steps.
        5. Normalise coefficients/biases by Gaussian_stds (unnecessary but potentially useful).
        6. Slice out the t = 0 coefficients, fold the observation `inputs[0]`
           into the biases, broadcast to all `sequence_length` parallel
           histories, and run one pass of `RNN_reccurence_formula`.
        7. Initialise `LP` with the log-fractions, log-factors and a
           uniform 1/sequence_length term (normalize the probabilities so they sum to 1).

        Outputs
        -------
        inputs         : unchanged.
        initial_states : the loop-carried tuple consumed by Custom_RNN_layer
                         (see class docstring).
        
        inputs = transposed_inputs # in build_model
        '''
        
        nb_tracks = inputs.shape[2]
        nb_hidden_vars = self.nb_hidden_vars
        dtype = self.dtype
        constraint_function = self.constraint_function
        reference_dt = self.reference_dt

        param_vars = self.param_vars
        initial_param_vars = self.initial_param_vars
        nb_states = self.nb_states
        max_linking_distance = self.max_linking_distance_param
        vary_params = self.vary_params
        vary_initial_params = self.vary_initial_params
        initial_fractions = tf.math.softmax(self.initial_fractions)+1e-8
        vary_initial_fractions = self.vary_initial_fractions
        LocErr_function = self.LocErr_function
        
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
        
        hidden_var_coefs, obs_var_coefs, Gaussian_stds, biases, initial_hidden_var_coefs, initial_obs_var_coefs, initial_Gaussian_stds, initial_biases, transition_hidden_var_coefs, transition_Gaussian_stds, transition_biases, integration_variable_index, Log_factors, initial_Log_factors, transition_Log_factors = constraint_function(param_vars, initial_param_vars, input_LocErrs, input_dts, nb_dims, reference_dt, LocErr_function, dtype)
        
        hidden_var_coefs = hidden_var_coefs/Gaussian_stds
        obs_var_coefs = obs_var_coefs/Gaussian_stds
        biases = biases/Gaussian_stds
        
        current_hidden_var_coefs = hidden_var_coefs[...,:nb_hidden_vars]
        next_hidden_var_coefs = hidden_var_coefs[...,nb_hidden_vars:]
        
        # parameters of the gaussians for all the recurrence steps
        reccurent_obs_var_coefs = tf.identity(obs_var_coefs)
        reccurent_hidden_var_coefs = tf.identity(current_hidden_var_coefs)
        reccurent_next_hidden_var_coefs = tf.identity(next_hidden_var_coefs)
        reccurent_biases = tf.identity(biases)
        
        # change of variables to deal with gaussians of variance 1
        initial_hidden_var_coefs = initial_hidden_var_coefs/initial_Gaussian_stds
        initial_obs_var_coefs = initial_obs_var_coefs/initial_Gaussian_stds
        initial_biases = initial_biases/initial_Gaussian_stds
        
        current_initial_hidden_var_coefs = initial_hidden_var_coefs[...,:nb_hidden_vars]
        next_initial_hidden_var_coefs = tf.zeros((nb_hidden_vars, nb_tracks, nb_states, nb_hidden_vars), dtype = dtype)  # these coefs must equal 0 as the initial gaussians must only depend on the fist set of hidden states

        # Gaussians added at state transitions
        transition_hidden_var_coefs = transition_hidden_var_coefs/transition_Gaussian_stds
        transition_biases = transition_biases/transition_Gaussian_stds

        sequence_length = self.sequence_length
        transition_hidden_var_coefs = tf.concat([transition_hidden_var_coefs]*sequence_length*nb_states, 3)
        transition_biases = tf.concat([transition_biases] * nb_states * sequence_length, 3)
        
        # now that we have the coefficients and biases to carry to the next layers, 
        # we compute the first iteration 
        
        biases = reccurent_biases[0]
        obs_var_coefs = reccurent_obs_var_coefs[0]
        current_hidden_var_coefs = reccurent_hidden_var_coefs[0]
        
        next_hidden_var_coefs = reccurent_next_hidden_var_coefs[0]
        biases += tf.reduce_sum(obs_var_coefs[...,None] * inputs[0], -2)
        biases.shape
        initial_biases += tf.reduce_sum(initial_obs_var_coefs[...,None] * inputs[0], -2)
        
        current_hidden_var_coefs = tf.concat((current_initial_hidden_var_coefs, current_hidden_var_coefs), axis = 0)
        next_hidden_var_coefs =  tf.concat((next_initial_hidden_var_coefs, next_hidden_var_coefs), axis = 0)
        biases = tf.concat((initial_biases, biases), axis = 0)
        
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
        
        #initial_Log_factors, Log_factors, transition_Log_factors = self.compute_scaling_factors(param_vars, initial_param_vars)
        
        init_log_fractions = tf.concat([tf.math.log(initial_fractions)]*sequence_length, axis = 1)
        init_log_factors = tf.concat([nb_dims*initial_Log_factors]*sequence_length, axis = 1)
        
        LP = LC + init_log_factors + init_log_fractions + tf.math.log(np.array(1/sequence_length))
        
        Log_factors = nb_dims * Log_factors
        transition_Log_factors = nb_dims * transition_Log_factors
        initial_states = [Next_coefs, Next_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases]
        
        return inputs, initial_states

    def duplicate_states(self, param_vars, initial_param_vars, initial_fractions):
        '''
        Hook intended to be overridden by subclasses when several physical
        states should share the same parameter row (e.g. tying a directed
        and a confined state's diffusion coefficient). Default implementation
        is the identity map.

        Inputs / outputs
        ----------------
        Same three tensors, possibly expanded along the state axis.
        '''
        return param_vars, initial_param_vars, initial_fractions


@tf.function
def constraint_function(all_params, all_initial_params, LocErrs, dts,
                        nb_dims, reference_dt, LocErr_function, dtype):
    '''
    
    
    Vectorised, time-varying constraint function that makes the link between
    the model variables and the characteristic parameters of the Gaussians.

    Builds the per-step Gaussian coefficients, biases, std-rescaling factors
    and log-normalising factors describing the joint distribution
        p(observation_t, hidden_t, hidden_{t+1} | state_t)
    for all (time, track, state) triples in one pass. Designed to be used in the
    call of the layer `Initial_layer_constraints`.

    Parameters
    ----------
    all_params         : (nb_states, 5) — columns
                         [log_LocErr_unused, log_d, ano, log_q, is_directed_flag].
                         `log_d`  : log diffusion length per reference_dt.
                         `ano`    : in directed regime acts as log drift speed,
                                    in confined regime acts as a logistic well
                                    confinement (l = sigmoid(ano)).
                         `log_q`  : log std of the anomalous-variable noise.
                         `is_directed_flag` : 1 = directed motion, 0 = confined.
    all_initial_params : (nb_states, >=1), column 0 is log(initial spread).
    LocErr             : per-step localisation error. Accepted shapes
                         (nb_tracks, track_len),
                         (nb_tracks, track_len, 1) or
                         (nb_tracks, track_len, nb_dims). A trailing dim axis
                         is averaged out.
    dts                : per-step frame durations, shape (nb_tracks, track_len+1)
                         or (nb_tracks, track_len+1, 1). Must have one extra
                         time step relative to the track length to support
                         the directed-mode `dt_ratio_next` rescaling at
                         segment carryovers.
    nb_dims            : int, spatial dimensionality. Typically set to 2.
    reference_dt       : scalar, reference frame duration the parameters
                         are expressed in.
    dtype              : tensorflow dtype string (e.g. 'float64').
    
    Per-step continuous-time rescaling
    ----------------------------------
    Given the parameters at `reference_dt`, this routine rescales them to
    the actual `dts[t]`:
        d  = d_ref * sqrt(dt_ratio)
        q  = q_ref * sqrt(dt_ratio)
        v  = v_ref * dt_ratio                 (directed regime)
        l  = 1 - exp(-l_ref_c * dt_ratio)     (confined regime, with l_ref_c = -log(1 - sigmoid(ano)))
    For directed states the ano-coefficient of the recurrent g2 Gaussian is
    further scaled by `dts[t+1]/dts[t]` to keep
        E[ano_{t+1} | ano_t] = (dts[t+1]/dts[t]) * ano_t.

    Returns
    -------
    A 15-tuple (in this order):
        hidden_vars              : (track_len, 3, nb_tracks, nb_states, 4)
                                   recurrent hidden-variable coefficients,
                                   last axis = [pos_t, ano_t, pos_{t+1}, ano_{t+1}].
        obs_vars                 : (track_len, 3, nb_tracks, nb_states, 1)
                                   recurrent observation coefficients.
        Gaussian_stds            : ones, shape compatible with hidden_vars.
        biases                   : zeros, shape (track_len, 3, nb_tracks,
                                   nb_states, nb_dims).
        initial_hidden_vars      : (2, nb_tracks, nb_states, 2)
                                   initial-step hidden-variable coefficients.
        initial_obs_vars         : zeros, (nb_hidden_vars, nb_tracks,
                                   nb_states, nb_obs_vars).
        initial_Gaussian_stds    : ones, (nb_hidden_vars, nb_tracks,
                                   nb_states, 1).
        initial_biases           : zeros, (nb_transition_gaussians,
                                   nb_tracks, nb_states, nb_dims).
        transition_hidden_vars   : (track_len, 1, nb_tracks, nb_states, 2)
                                   coefficients of the extra Gaussian inserted
                                   at state transitions.
        transition_Gaussian_stds : ones, matching shape.
        transition_biases        : zeros, matching shape.
        integration_variable_index : tf.constant(1) — index of the variable
                                   integrated out at transitions.
        Log_factors              : (track_len, nb_tracks, nb_states)
                                   per-step log-normalising factor for the
                                   recurrent Gaussians.
        initial_Log_factors      : (nb_tracks, nb_states) factor for t = 0.
        transition_Log_factors   : (track_len, nb_tracks, nb_states) factor
                                   to apply at state transitions.

    All coefficient tensors carry a leading time axis so callers slice
    `tensor[t]` at time step t.
    
    all_params = param_vars
    all_initial_params = initial_param_vars
    LocErrs = input_LocErrs
    dts = input_dts
    '''
    
    # ------------------------------------------------------------------
    # Bookkeeping constants
    # ------------------------------------------------------------------
    nb_states                  = all_params.shape[0]
    integration_variable_index = tf.constant(1)
    nb_hidden_vars             = 2
    nb_obs_vars                = 1
    nb_transition_gaussians    = 1          # = nb_hidden_vars - integration_variable_index
    
    # ------------------------------------------------------------------
    # Normalise LocErrs and dts to shape (track_len, nb_tracks, 1)
    # ------------------------------------------------------------------
    LocErrs = tf.cast(LocErrs, dtype)
    if LocErrs.shape.rank == 2:
        LocErrs = LocErrs[..., None]                            # (nb_tracks, track_len, 1)
    LocErrs = tf.reduce_mean(LocErrs, axis=-1, keepdims=True)   # (nb_tracks, track_len, 1)
    LocErrs = tf.transpose(LocErrs, [1, 0, 2])                  # (track_len, nb_tracks, 1)
    
    dts = tf.cast(dts, dtype)
    if dts.shape.rank == 2:
        dts = dts[..., None]                                  # (nb_tracks, track_len, 1)
    dts = tf.reduce_mean(dts, axis=-1, keepdims=True)         # (nb_tracks, track_len, 1)
    dts = tf.transpose(dts, [1, 0, 2])                        # (track_len, nb_tracks, 1)
    
    reference_dt = tf.cast(reference_dt, dtype)
    
    # Dynamic shape helpers
    track_len = tf.shape(LocErrs)[0]
    nb_tracks = tf.shape(LocErrs)[1]
    
    # ------------------------------------------------------------------
    # Per-state parameters, broadcast-ready on (1, 1, nb_states)
    # ------------------------------------------------------------------
    LocErr_param    = tf.math.exp(all_params[:, 0][None, None, :])
    LocErrs = LocErr_function(LocErrs, LocErr_param)
    
    log_d           = all_params[:, 1][None, None, :]
    ano             = all_params[:, 2][None, None, :]
    log_q           = all_params[:, 3][None, None, :]
    is_dir          = all_params[:, 4][None, None, :]
    log_init_spread = all_initial_params[:, 0][None, :]

    # State-selection masks, shape (1, 1, nb_states)
    isdir_mask  = tf.cast(is_dir >= 0.5, dtype)
    isconf_mask = 1.0 - isdir_mask
    
    # ------------------------------------------------------------------
    # Step-size scaling from reference_dt to the actual dts.
    # All scaled tensors have shape (track_len, nb_tracks, nb_states).
    # ------------------------------------------------------------------
    dt_ratio      = dts / reference_dt   + 0.9e-20                    # (track_len, nb_tracks, 1)
    dt_sqrt_ratio = tf.sqrt(dt_ratio)                                 # (track_len, nb_tracks, 1)
    dt_ratio.shape
    # Values at reference_dt (per-state only)
    d_ref = tf.exp(log_d)            # (1, 1, nb_states)
    q_ref = tf.exp(log_q)
    l_ref = tf.math.sigmoid(ano)
    v_ref = tf.exp(ano)
    
    # Continuous-discrete conversion for l:
    #   ld = 1 - exp(-lc)   <=>   lc = -log(1 - ld)
    # so l scales correctly with dt_ratio in the continuous domain.
    d       = d_ref * dt_sqrt_ratio[:track_len] + 1e-20                  # (track_len, nb_tracks, nb_states)
    q       = q_ref * dt_sqrt_ratio[:track_len] + 1e-20
    l_ref_c = -tf.math.log(1.0 - l_ref)
    l_c     = l_ref_c * dt_ratio[:track_len]
    l       =  - tf.math.expm1(-l_c) + 1e-20#   = 1.0 - tf.math.exp(-l_c) + 1e-20 without underflow
    one_minus_l = tf.math.exp(-l_c) + 1e-20
    v       = v_ref * dt_ratio[:track_len] + 1e-20
    
    # ------------------------------------------------------------------
    # NEW: per-step rescaling of the ano_t coefficient in recurrent g2.
    # ano_t = v * dts[t] in the directed regime, so the deterministic
    # part of the ano dynamics is
    #     E[ano_{t+1} | ano_t] = (dts[t+1]/dts[t]) * ano_t.
    # For confined motion ano_t is the well anchor, dt-independent.
    # ------------------------------------------------------------------
    dt_ratio_next         = dt_ratio[1:]
    ano_step_ratio        = dt_ratio_next / dt_ratio[:-1]                    # (track_len, nb_tracks, 1)
    ano_rescale_per_state = ano_step_ratio * isdir_mask + (1.0 - isdir_mask)
    # shape: (track_len, nb_tracks, nb_states)
    
    # Characteristic well distance for confined motion
    #well_distance = d / tf.sqrt(1-tf.math.exp(-2*l_c))    # (nb_tracks, nb_states) independent of time or localization error
    well_distance = d / tf.sqrt(2*(1-tf.math.exp(-2*l_c)))    # (nb_tracks, nb_states) independent of time or localization error
    
    # Initial position spread, broadcast to full (track_len, nb_tracks, nb_states)
    initial_position_spread = tf.broadcast_to(tf.exp(log_init_spread),
                                              tf.shape(d[0]))
    
    # LocErrs broadcast across states
    LocErr_b = tf.broadcast_to(LocErrs, (track_len, nb_tracks, nb_states)) + 1e-20  # (track_len, nb_tracks, nb_states)
    
    zeros = tf.zeros_like(LocErr_b)
    tiny  = tf.fill((track_len, nb_tracks, nb_states), tf.constant(1e-15, dtype=dtype))
    
    # ==================================================================
    # Recurrent hidden-variable coefficients
    # Per-gaussian tensor shape: (track_len, nb_tracks, nb_states, 4)
    # Stacking at axis=1 keeps time at axis 0, puts gaussians at axis 1.
    # Final shape: (track_len, 3, nb_tracks, nb_states, 4)
    # Last axis ordering: [pos_t, ano_t, pos_{t+1}, ano_{t+1}]
    # ==================================================================

    # Gaussian 0 -- localisation error:   [1/LocErrs, 0, 0, 0]
    g0 = tf.stack([1.0 / LocErr_b, zeros, zeros, zeros], axis=-1)

    # Gaussian 1 -- diffusion + anomalous drift
    #   confined :  [(1-l)/d, l/d, -1/d, 0]
    #   directed :  [   1/d, 1/d, -1/d, 0]
    
    #g1_std = d * isdir_mask + d*((1-tf.math.exp(-2*l))/l)**0.5 * isconf_mask
    
    #Var = D dt/lambda dt * (1-exp(-2 lambda dt))
    #Var = d**2/2 lambda dt * (1-exp(-2 lambda dt))

    g1_std = d * isdir_mask + d/(2*l_c)**0.5*(1-tf.math.exp(-2*l_c))**0.5 * isconf_mask
    
    inv_d = 1.0 / g1_std
    g1_c0 = (one_minus_l * isconf_mask + isdir_mask) * inv_d 
    g1_c1 = (l * isconf_mask + isdir_mask) * inv_d + 1.1e-20
    g1    = tf.stack([g1_c0, g1_c1, -inv_d, zeros], axis=-1)
    
    # Gaussian 2 -- anomalous-variable evolution:   [0, 1/q, 0, -1/q]
    #inv_q = 1.0 / q
    #g2    = tf.stack([zeros, inv_q, zeros, -inv_q], axis=-1)
    
    # g2 -- ano evolution (MODIFIED: ano_t coefficient scaled by dt ratio
    # for directed states; confined states unchanged)
    inv_q = 1.0 / q
    g2_c1 = ano_rescale_per_state * inv_q   # was simply inv_q
    g2    = tf.stack([zeros, g2_c1, zeros, -inv_q], axis=-1)
    
    hidden_vars = tf.stack([g0, g1, g2], axis=1)             # (track_len, 3, nb_tracks, nb_states, 4)
    
    # ==================================================================
    # Recurrent observation coefficients
    # Final shape: (track_len, 3, nb_tracks, nb_states, 1)
    # Only Gaussian 0 depends on the observation: [-1/LocErrs, 0, 0]
    # ==================================================================
    obs_g0   = (-1.0 / LocErr_b)[..., None]                  # (track_len, nb_tracks, nb_states, 1)
    obs_zero = zeros[..., None]
    obs_vars = tf.stack([obs_g0, obs_zero, obs_zero], axis=1)
    # ==================================================================
    # Initial hidden-variable coefficients
    # Final shape: (2, nb_tracks, nb_states, 2)
    # ==================================================================
    init_g0 = tf.stack([1.0 / initial_position_spread, zeros[0]], axis=-1) # does not need the time axis to participate to the initialization
    
    # init_g1 needs the time axis to participate to the transition gaussians
    init_g1_c0 = (1.0  / well_distance) * isconf_mask + tiny * isdir_mask
    init_g1_c1 = (-1.0 / well_distance) * isconf_mask + (1.0 / v) * isdir_mask
    init_g1    = tf.stack([init_g1_c0, init_g1_c1], axis=-1)
    
    initial_hidden_vars = tf.stack([init_g0, init_g1[0]], axis=0)   # (track_len, 2, nb_tracks, nb_states, 2)
    
    # ==================================================================
    # Transition hidden-variable coefficients
    # Final shape: (track_len, 1, nb_tracks, nb_states, 2)
    # ==================================================================
    transition_hidden_vars = init_g1[:, None]                # insert gaussian axis at position 1
    # ==================================================================
    # Unit-std / zero-bias scaffolding tensors, all carrying the leading
    # time axis so they can be sliced [t] uniformly with the coefficient
    # tensors above.
    # ==================================================================
    Gaussian_stds = tf.ones((track_len, nb_obs_vars + nb_hidden_vars,
         nb_tracks, nb_states, 1), dtype=dtype)
    biases = tf.zeros((track_len, nb_obs_vars + nb_hidden_vars,
         nb_tracks, nb_states, nb_dims), dtype=dtype)
    initial_obs_vars         = tf.zeros((nb_hidden_vars,
                                         nb_tracks, nb_states, nb_obs_vars), dtype=dtype)
    initial_Gaussian_stds    = tf.ones((nb_hidden_vars,
                                        nb_tracks, nb_states, 1), dtype=dtype)
    initial_biases           = tf.zeros((nb_transition_gaussians,
                                         nb_tracks, nb_states, nb_dims), dtype=dtype)
    transition_Gaussian_stds = tf.ones((track_len, nb_transition_gaussians,
         nb_tracks, nb_states, 1), dtype=dtype)
    transition_biases = tf.zeros((track_len, nb_transition_gaussians,
         nb_tracks, nb_states, nb_dims), dtype=dtype)
    
    # Then, we compute the time and localization error varying scaling factors    
    Log_factors = - tf.math.log(LocErrs + 1e-20) - tf.math.log(g1_std) - tf.math.log(q)
        
    #initial_anomalous_factor = (- param_vars[:,1] + 0.5*tf.math.log(2*tf.math.sigmoid(param_vars[:,2])))*(1.-state_mask) - param_vars[:,2]*state_mask
    initial_anomalous_factor = (- tf.math.log(d) + 0.5*tf.math.log(2*(1-tf.math.exp(-2*l_c))+1e-20))*isconf_mask - tf.math.log(v)*isdir_mask
    initial_Log_factors = Log_factors[0] - log_init_spread + initial_anomalous_factor[0]
    
    transition_Log_factors = Log_factors + initial_anomalous_factor
    transition_Log_factors = transition_Log_factors
    
    return (hidden_vars, obs_vars, Gaussian_stds, biases,
            initial_hidden_vars, initial_obs_vars,
            initial_Gaussian_stds, initial_biases,
            transition_hidden_vars, transition_Gaussian_stds,
            transition_biases, integration_variable_index, 
            Log_factors, initial_Log_factors, transition_Log_factors)

@tf.function(jit_compile=False)
def RNN_cell(input_i, Prev_coefs, Prev_biases, LP, segment_len, reshaped_Log_factors, reshaped_transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, sequence_phase_1, sequence_phase_2, transition_mask, transition_sequence, transition_mean, transition_var, gamma_dist_mean, gamma_dist_var, states, dt_ratios):
    print('LP',LP)
    '''
    One recurrent step of the model.

    Conceptual flow
    ---------------
    Each track maintains `sequence_length * nb_states` parallel hypotheses
    ("sequences") about its hidden-state history. At every time step:
    
      1. For every existing sequence, generate `nb_states` candidate
         continuations: one for each possible next state. Sequences whose
         next state differs from their current state ("transition" branches,
         selected by `transition_mask`) get an extra integration over the
         hidden variable that changes at the transition (via
         `transition_RNN_reccurence_formula` and `transition_sequence`).
      2. Weight each branch by the Gamma-distributed dwell-time hazard
            P(transition at age k | survived to k)
              = pdf(k) / (1 - cdf(k))
         using `gamma_dist_mean`/`gamma_dist_var` as the per-pair Gamma
         moments. Non-transition branches get the complementary survival
         probability.
      3. Fold the new observation `input_i` into the recurrent biases and
         run one pass of `RNN_reccurence_formula` to update the analytic
         hidden-variable coefficients.
      4. For each (sequence, next_state), reduce the duplicated branches
         back to a single sequence by exp-log-sum-weighting them by their
         likelihood (including the determinant of the next-step Gaussian
         coefficients), updating `states`, segment lengths and Gamma
         moments accordingly.
      5. The sequence buffer would now contain `sequence_length+1` entries
         per state; the oldest two-state slab is merged back into the rest
         by another weighted reduction so the buffer stays at
         `sequence_length * nb_states`.

    Inputs
    ------
    input_i                          : (1, nb_tracks, 1, 1, nb_dims) — current obs.
    Prev_coefs, Prev_biases          : analytic hidden-variable coefficients
                                       and biases carried from previous step.
    LP                               : (nb_tracks, sequence_length*nb_states)
                                       running log-likelihood per sequence.
    segment_len                      : (nb_tracks, sequence_length*nb_states)
                                       age (in steps assuming 1 step = reference_dt) 
                                       of the current segment.
    reshaped_Log_factors,
    reshaped_transition_Log_factors  : per-pair log normalisers for this step.
    reccurent_*  /  transition_*     : per-step coefficient slices coming
                                       from `constraint_function`.
    sequence_phase_1/2               : integration schedules for the
                                       recurrent step.
    transition_mask                  : (1, sequence_length*nb_states**2)
                                       1.0 where the candidate is a true
                                       state change, 0.0 otherwise.
    transition_sequence              : integration schedule for the
                                       transition branches.
    transition_mean, transition_var  : (nb_tracks, nb_states**2) per-pair
                                       Gamma moments at the current step.
    gamma_dist_mean, gamma_dist_var  : (nb_tracks, sequence_length*nb_states**2)
                                       Gamma moments inherited by each
                                       sequence (set when the segment was born).
    states                           : (nb_tracks, sequence_length*nb_states,
                                       sequence_length, nb_states) one-hot
                                       state-history per sequence.
    dt_ratios                        : (nb_tracks,) dt_i / reference_dt for
                                       this step, used to advance segment
                                       lengths.

    Returns
    -------
    new_Next_coefs       : updated hidden-variable coefficients.
    new_Next_biases      : updated biases.
    new_LPs              : updated log-likelihoods per sequence.
    new_segment_len      : updated segment ages.
    new_gamma_dist_mean,
    new_gamma_dist_var   : updated per-sequence Gamma moments.
    new_states           : updated state-history tensor (length
                           sequence_length, oldest entry dropped).
    
    reshaped_Log_factors, reshaped_transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases,  transition_mean, transition_var = (log_factors_i, trans_log_factors_i,  rec_obs_i, rec_hid_i, rec_next_hid_i, rec_bias_i, trans_hid_i, trans_bias_i, trans_mean_i, trans_var_i)
    '''
    
    current_states = states[:,:,-1:]
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
    
    transition_Prev_coefs, transition_Prev_biases, LC = transition_RNN_reccurence_formula(current_hidden_var_coefs = alternative_Prev_coefs, # coefficients of the hidden variables that are updated
                                                                         next_hidden_var_coefs = tf.constant(0, dtype = dtype, shape =  alternative_Prev_coefs.shape),
                                                                         biases = alternative_Prev_biases,
                                                                         transition_sequence = transition_sequence,
                                                                         nb_dims = nb_dims,
                                                                         dtype = dtype)
    
    LP2 += LC*transition_mask + reshaped_Log_factors
    
    current_shapes = gamma_dist_mean**2/gamma_dist_var
    current_rates = gamma_dist_mean/gamma_dist_var
    
    all_Prev_coefs = transition_Prev_coefs*transition_mask[None,:,:,None] + Prev_coefs2*(1-transition_mask[None,:,:,None])
    all_prev_biases = transition_Prev_biases*transition_mask[None,:,:,None] + Prev_biases2*(1-transition_mask[None,:,:,None])
    # A : transition at time step k, B : no transition at time step k-1 
    # P(A|B) = P(AB)/P(B), if A, B is necessarly verified  
    # Here the probability to consider is the probability to transition given that it did not transition yet 
    # to compute the proba to not transition, we must compute 1 - the probas to transition 
    
    #transition_probas = tf.clip_by_value((tf.compat.v1.distributions.Gamma(current_shapes, current_rates).prob(segment_len[:,:]+0.5*dt_ratios[:, None])+1e-14)/(1-tf.compat.v1.distributions.Gamma(current_shapes, current_rates).cdf(segment_len[:,:]+0.5*dt_ratios[:, None])+1e-12), clip_value_min=1-20, clip_value_max=1-1e-10) #*segment_len_certainty + transition_rates*(1-segment_len_certainty)
    gamma = tf.compat.v1.distributions.Gamma(current_shapes, current_rates)
    S_old = 1 - gamma.cdf(segment_len+1e-12) + 1e-12
    S_new = 1 - gamma.cdf(segment_len+1e-12 + dt_ratios[:, None]) + 1e-12
    transition_probas = tf.clip_by_value(1 - S_new / S_old,
    clip_value_min=1e-20, clip_value_max=1 - 1e-10)
    
    '''dt_ratios[:,None]*
    transition_probas = tf.clip_by_value(dt_ratios[:,None]*(tf.compat.v1.distributions.Gamma(current_shapes, current_rates).prob(segment_len[:,:]+1/6*dt_ratios[:, None])/3 + 
                                                            tf.compat.v1.distributions.Gamma(current_shapes, current_rates).prob(segment_len[:,:]+3/6*dt_ratios[:, None])/3 +
                                                            tf.compat.v1.distributions.Gamma(current_shapes, current_rates).prob(segment_len[:,:]+5/6*dt_ratios[:, None])/3 + 1e-14)/(1-tf.compat.v1.distributions.Gamma(current_shapes, current_rates).cdf(segment_len[:,:]+0*dt_ratios[:, None])+1e-12), clip_value_min=1-20, clip_value_max=1-1e-10) #*segment_len_certainty + transition_rates*(1-segment_len_certainty)
    '''
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
    transition_LPs = tf.reshape(all_LP - 200*(1-transition_mask), (nb_tracks, sequence_length*nb_states, nb_states)) - nb_dims*tf.math.log(tf.math.abs(tf.reduce_prod(tf.linalg.diag_part(tf.transpose(reshaped_Next_coefs, [1, 2, 3, 0, 4])), axis=-1))+1e-20)
    
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
    
    transition_LPs = tf.math.log(tf.reduce_sum(transition_Ps, axis = 1)) + max_transition_LPs[:,0] + nb_dims*tf.math.log(tf.math.abs(tf.reduce_prod(tf.linalg.diag_part(tf.transpose(transition_Next_coefs, [1, 2, 0, 3])), axis=-1)) + 1e-20)
    
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
    additional_stable_segment_len = dt_ratios[:, None]
    #transition_segment_len = tf.broadcast_to(additional_stable_segment_len, (nb_tracks, nb_states))
    
    #current_segment_len = tf.concat([transition_segment_len, stable_segment_len + additional_stable_segment_len], axis = 1)
    current_segment_len = tf.concat([tf.zeros((nb_tracks, nb_states), dtype = dtype), stable_segment_len], axis = 1)
    current_segment_len = current_segment_len + additional_stable_segment_len
    Next_states = tf.concat([transition_states, stable_states], axis = 1)
    
    '''
    Now, the `nb_states` last sequences must be fused with the previous sequences to
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
    last_LP = tf.reshape(new_LP[:, -nb_states*2:], (nb_tracks, 2, nb_states)) - nb_dims*tf.math.log(tf.math.abs(tf.reduce_prod(tf.linalg.diag_part(tf.transpose(last_Next_coefs, [1, 2, 3, 0, 4])), axis=-1)) + 1e-20)
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
    reduced_last_LPs = (tf.math.log(sum_last_P + 1e-100) + last_LP_max)[:,0] + nb_dims*tf.math.log(tf.math.abs(tf.reduce_prod(tf.linalg.diag_part(tf.transpose(reduced_last_Next_coefs, [1, 2, 0, 3])), axis=-1))+1e-20)
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
isfirst = input_isfirst
Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states
Prev_coefs.shape
Prev_coefs[:,-1,0]
Prev_biases[:,0,0]
nb_states = 2
inputs[:,:,2]
'''


class Custom_RNN_layer(tf.keras.layers.Layer):
    '''
    Time-varying recurrent layer that drives `RNN_cell` across the track.

    Responsibilities
    ----------------
    1. Owns the trainable Gamma-distribution parameters of the dwell-time
       model (`transition_shapes`, `transition_rates`).
    2. Builds the (1, sequence_length*nb_states**2) `transition_mask` and
       state-pair `indices`.
    3. Optionally allocates carry-over buffers
       (carryout_segment_len, carryout_gamma_dist_mean,
       carryout_gamma_dist_var) when `carryover=True`.
    4. On `call`, computes the per-step Gamma moments via
       `transition_param_function`, slices the time-varying tensors so that
       index 0 lines up with `inputs[0]` (= actual time step 1, since step 0
       was consumed by `Initial_layer_constraints`), and runs a
       `tf.while_loop` whose body invokes `RNN_cell`. Tracks marked dead by
       `mask` are frozen on a per-step basis. This mask is important to account
       for tracks of various lengths. Per-step diagnostics are collected in
       TensorArrays.
    
    Constructor arguments
    ---------------------
    nb_tracks                : int.
    transition_shapes        : (nb_states, nb_states) initial Gamma shape.
    transition_rates         : (nb_states, nb_states) initial Gamma rate.
    density                  : float, used to derive the mislinking rate
                               from the effective diffusion.
    nb_states                : int, number of physical states (NOT including
                               the mislinking state — the layer adds one,
                               so `self.nb_states = nb_states + 1`).
    sequence_phase_1/2,
    transition_sequence      : integration schedules for `RNN_cell`.
    transition_param_function: callable
                               (transition_shapes, transition_rates,
                                density, Fs, effective_ds, dts_TN,
                                reference_dt, dtype)
                               -> (transition_shapes_full (number of states, number of states),
                                   transition_rates_full  (number of time steps, number of tracks, number of states, number of states)).
    sequence_length          : int (default 3).
    vary_transition_shapes,
    vary_transition_rates    : optional float stop-gradient masks.
    carryover                : bool. used to indicate if tracks are splitted into 
                               segments across batches

    `call` arguments
    ----------------
    inputs                          : (number of time steps-1, 1, number of tracks, 1, 1, nb dims) already sliced.
    input_dts                       : (number of tracks, number of time steps) per-step frame durations
                                      (NOT yet sliced).
    reference_dt                    : scalar reference frame duration.
    mask                            : (number of tracks, number of time steps-1) per-step alive mask (already
                                      sliced).
    Prev_coefs, Prev_biases, LP     : initial loop-carried tensors from
                                      `Initial_layer_constraints`.
    Log_factors,
    transition_Log_factors          : (number of time steps, number of tracks, number of states) per-step normalisers.
    reccurent_obs_var_coefs,
    reccurent_hidden_var_coefs,
    reccurent_next_hidden_var_coefs,
    reccurent_biases,
    transition_hidden_var_coefs,
    transition_biases               : per-step coefficient stacks (full,
                                      not yet sliced).
    log_ds                          : log diffusion of the physical states,
                                      used to compute effective_ds.
    softmax_inv_Fractions           : pre-softmax mixture weights, used to
                                      compute Fs (state occupancies).
    anomalous_factors               : ano values per state.
    isdir                           : 0/1 mask, directed vs confined per state.
    isfirst                         : (N,) 0/1 flag per track. When 1 the
                                      carryover buffers are ignored and the
                                      sequence is started fresh; when 0 the
                                      track resumes from the buffers
                                      (only meaningful if `carryover=True`).

    Returns
    -------
    Prev_coefs, Prev_biases, LP        : final loop-carried recurrence state.
    segment_len, gamma_dist_mean,
    gamma_dist_var                     : final dwell-time bookkeeping.
    All_states                         : (number of tracks, number of time steps-sequence_length, nb_states)
                                          per-step predicted state distributions
                                          (cropped to discard the warm-up).
    All_coefs, All_biases, All_LPs     : per-step diagnostics, transposed to
                                          (track, time, ...) layout.
    states                             : final per-sequence state-history
                                          tensor (shape as in `RNN_cell`).
    '''
    def __init__(self, nb_tracks, transition_shapes, transition_rates, density, nb_states, sequence_phase_1, sequence_phase_2, transition_sequence, transition_param_function, sequence_length = 3, vary_transition_shapes = None, vary_transition_rates = None, carryover = False, **kwargs):
        '''
        Stores configuration and `vary_*` masks (defaulting to all-ones).
        Note `self.nb_states = nb_states + 1`: the constructor argument
        counts only the physical states, the layer internally accounts for
        the appended mislinking state. No tf.Variables are created here.
        '''
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
        '''
        Allocates trainable Gamma-distribution parameters and pair-indexing
        helpers used by `call`:
            transition_rates   : tf.Variable, log-floored at log(minval).
            transition_shapes  : tf.Variable, unconstrained.
            indices            : (sequence_length*nb_states**2, 2) int tensor
                                 enumerating (current_state, next_state) pairs.
            transition_mask    : (1, sequence_length*nb_states**2) float, 1.0 where current != next.
        If `carryover=True`, also allocates non-trainable buffers
        carryout_segment_len, carryout_gamma_dist_mean,
        carryout_gamma_dist_var so dwell-time bookkeeping survives across
        successive batches of the same tracks.
        '''
        nb_states = self.nb_states
        transition_shapes, transition_rates = self.initial_transition_params
        sequence_length = self.sequence_length
        nb_tracks = self.nb_tracks
        self.transition_rates = tf.Variable(transition_rates, dtype = dtype, name = 'Transition rates', trainable = True, constraint=lambda w: tf.clip_by_value(w, clip_value_min=-10, clip_value_max=4))
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
    def call(self, inputs, input_dts, reference_dt, mask,
         Prev_coefs, Prev_biases, LP,
         Log_factors, transition_Log_factors,
         reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
         reccurent_next_hidden_var_coefs, reccurent_biases,
         transition_hidden_var_coefs, transition_biases,
         log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
         isfirst=None):
        '''
        Drives `RNN_cell` across the track in a tf.while_loop.

        Pipeline
        --------
        with T the number of time steps
        N the nuumber of tracks
        P the number of sequences considering the transitions (nb_states**2*sequence_length)
        
        1. Apply stop-gradient `vary_*` masks to the trainable Gamma params.
        2. Compute effective diffusion per state
               effective_ds = exp(log_ds) + 2 * exp(anomalous_factors) * isdir
           and softmax fractions Fs, then call `transition_param_function` to 
           obtain the time-varying transition rates (T, N, S, S).
        3. Flatten the (S, S) state-pair axis into the unrolled
           P = sequence_length * S**2 axis used by `RNN_cell` via einsum
           contractions with one-hot row/column matrices, producing per-step
              flat_Log_full        : (T, N, P) merged Log_factors /
                                     transition_Log_factors selected by
                                     transition_mask
              transition_mean_full : (T, N, P)  Gamma mean
              transition_var_full  : (T, N, P)  Gamma variance
        4. Slice every time-varying tensor [1:] so that index 0 aligns with
           inputs[0] (= actual time step 1; step 0 was consumed by
           Initial_layer_constraints).
        5. Initialise loop carriers:
               segment_len      = ones,
               gamma_dist_mean  = transition_mean_full[0],
               gamma_dist_var   = transition_var_full[0],
               states           = identity-like one-hot history,
           overwriting them with the carryover buffers where isfirst == 0
           (only when self.carryover is True).
        6. Loop body: for each step i, log diagnostics into TensorArrays,
           call `RNN_cell`, then apply the per-step alive `mask` to either
           accept or reject the update on a track-by-track basis.
        7. After the loop: stack the TensorArrays and transpose them to
           (track, time, ...) layout. Drop the first `sequence_length - 1`
           predicted-state entries (warm-up where the buffer is not yet full).
        '''
        
        nb_tracks              = self.nb_tracks
        sequence_phase_1       = self.sequence_phase_1
        sequence_phase_2       = self.sequence_phase_2
        transition_sequence    = self.transition_sequence
        transition_mask        = self.transition_mask
        nb_states              = self.nb_states
        indices                = self.indices
        sequence_length        = self.sequence_length
        density                = self.density
        vary_transition_shapes = self.vary_transition_shapes
        vary_transition_rates  = self.vary_transition_rates
        
        transition_rates  = self.transition_rates
        transition_shapes = self.transition_shapes
        
        # ---- stop-gradient on fixed parameters ---------------------------
        transition_shapes = (vary_transition_shapes * transition_shapes
                             + (1 - vary_transition_shapes)
                                * tf.stop_gradient(transition_shapes))
        transition_rates  = (vary_transition_rates  * transition_rates
                             + (1 - vary_transition_rates)
                                * tf.stop_gradient(transition_rates))
        
        # ---- effective diffusion (used only for the mislinking rate) -----
        ds           = tf.math.exp(log_ds)
        Fs           = tf.math.softmax(softmax_inv_Fractions[0, :-1])
        effective_ds = ds + 2 * tf.math.exp(anomalous_factors) * isdir
        
        # ------------------------------------------------------------------
        # Time-varying transition kinetics.
        #
        # input_dts          : (N, T)
        # transition_param_function expects dts as (T, N), so we transpose
        # once here.  It returns
        #   transition_shapes_full : (S, S)             constant
        #   transition_rates_full  : (T, N, S, S)       time-varying
        # where S already includes the appended mislinking row/column.
        # ------------------------------------------------------------------
        dts_TN = tf.transpose(input_dts, [1, 0])                  # (T, N)
        # dts = dts_TN
        transition_shapes_full, transition_rates_full = self.transition_param_function(
                transition_shapes, transition_rates, density,
                Fs, effective_ds, dts_TN, reference_dt, dtype)
        transition_rates_full[0,0]
        # ------------------------------------------------------------------
        # Dense one-hot operators that "flatten" the (S, S) state-pair axis
        # into the unrolled n_pairs = sequence_length * S**2 axis used by
        # RNN_cell.
        # ------------------------------------------------------------------
        oh_row = tf.cast(tf.one_hot(indices[:, 0], nb_states), dtype)   # (P, S)
        oh_col = tf.cast(tf.one_hot(indices[:, 1], nb_states), dtype)   # (P, S)
        oh_src = oh_col   # next-state index for transition_Log_factors
        
        # ---- per-step Log_factors over the n_pairs axis ------------------
        # Log_factors / transition_Log_factors : (T, N, S)  ->  (T, N, P)
        flat_Log_full       = tf.einsum('tns,ps->tnp',
                                        Log_factors,            oh_row)
        flat_trans_Log_full = tf.einsum('tns,ps->tnp',
                                        transition_Log_factors, oh_src)
        flat_Log_full = (flat_trans_Log_full * transition_mask
                         + flat_Log_full     * (1 - transition_mask))
        
        # ---- per-step gamma-distribution moments over the n_pairs axis ---
        # transition_rates_full : (T, N, S, S)  ->  (T, N, P)
        transition_rates_flat_full = tf.einsum(
            'tnij,pi,pj->tnp', transition_rates_full,  oh_row, oh_col)
        transition_shapes_flat = tf.einsum(
            'ij,pi,pj->p',     transition_shapes_full, oh_row, oh_col)
        
        transition_mean_full = (transition_shapes_flat[None, None]
                                / transition_rates_flat_full)         # (T, N, P)
        transition_var_full  = (transition_shapes_flat[None, None]
                                / (transition_rates_flat_full ** 2))  # (T, N, P)
        
        # ------------------------------------------------------------------
        # Slice the time axis: index 0 of "_seq" tensors lines up with
        # inputs[0], i.e. actual time index 1 (time index 0 was consumed
        # by the initial layer).
        # ------------------------------------------------------------------
        rec_obs_var_coefs_seq           = reccurent_obs_var_coefs[1:]
        rec_hidden_var_coefs_seq        = reccurent_hidden_var_coefs[1:]
        rec_next_hidden_var_coefs_seq   = reccurent_next_hidden_var_coefs[1:]
        rec_biases_seq                  = reccurent_biases[1:]
        transition_hidden_var_coefs_seq = transition_hidden_var_coefs[1:]
        transition_biases_seq           = transition_biases[1:]
                
        flat_Log_seq        = flat_Log_full[1:]
        flat_trans_Log_seq  = flat_trans_Log_full[1:]
        transition_mean_seq = transition_mean_full[1:, :, :nb_states**2]
        transition_var_seq  = transition_var_full[1:,  :, :nb_states**2]
        
        # ------------------------------------------------------------------
        # Initial loop-carried state.  The first segments started at t = 0,
        # so we initialise gamma_dist_mean / _var from the rates at t = 0.
        # Inside the loop, RNN_cell prepends the *fresh* moments at t = i+1
        # and rolls the buffer, which gives every segment the gamma
        # distribution corresponding to the dt at which it was born.
        # ------------------------------------------------------------------
        segment_len     = tf.zeros((nb_tracks, sequence_length * nb_states),
                                  dtype=dtype)
        gamma_dist_mean = transition_mean_full[0]                   # (N, P)
        gamma_dist_var  = transition_var_full[0]                    # (N, P)
        
        if self.carryover:
            br_isfirst_1 = tf.broadcast_to(isfirst[:, None], segment_len.shape)
            segment_len     = (br_isfirst_1 * segment_len
                               + (1 - br_isfirst_1) * self.carryout_segment_len)
            br_isfirst_2 = tf.broadcast_to(isfirst[:, None], gamma_dist_mean.shape)
            gamma_dist_mean = (br_isfirst_2 * gamma_dist_mean
                               + (1 - br_isfirst_2) * self.carryout_gamma_dist_mean)
            gamma_dist_var  = (br_isfirst_2 * gamma_dist_var
                               + (1 - br_isfirst_2) * self.carryout_gamma_dist_var)
        
        # ------------------------------------------------------------------
        # State-tracking tensor for online state predictions.
        # ------------------------------------------------------------------
        states_indices = tf.range(0, nb_states * sequence_length,
                                  dtype='int32') % nb_states
        states_indices = tf.repeat(states_indices[:, None],
                                   sequence_length, axis=1)
        states = tf.repeat(tf.one_hot(states_indices, nb_states,
                                      dtype=dtype)[None],
                           nb_tracks, axis=0)
        
        nb_dims   = reccurent_biases.shape[4]
        num_steps = tf.shape(inputs)[0]
        
        # ---- per-step diagnostics ----------------------------------------
        All_states_ta = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=(nb_tracks, 1, nb_states))
        All_coefs_ta  = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=Prev_coefs.shape)
        All_biases_ta = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=Prev_biases.shape)
        All_LP_ta     = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=LP.shape)
        
        # ------------------------------------------------------------------
        # Recurrent loop body.
        # ------------------------------------------------------------------
        def body(i, Prev_coefs, Prev_biases, LP, segment_len,
                 gamma_dist_mean, gamma_dist_var, states,
                 All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta):
            
            # predicted state distribution BEFORE the cell update
            log_w = LP - nb_dims * tf.math.log(
                tf.math.abs(tf.reduce_prod(tf.linalg.diag_part(tf.transpose(Prev_coefs, [1, 2, 0, 3])), axis=-1))
                + 1e-20)
            
            max_log_w = tf.reduce_max(log_w, 1, keepdims=True)
            w = tf.math.exp(log_w - max_log_w)
            w = w / tf.reduce_sum(w, 1, keepdims=True)
            pred_states = tf.reduce_sum(w[:, :, None] * states[:, :, 0],
                                        1, keepdims=True)
            
            All_states_ta = All_states_ta.write(i, pred_states)
            All_coefs_ta  = All_coefs_ta.write(i,  Prev_coefs)
            All_biases_ta = All_biases_ta.write(i, Prev_biases)
            All_LP_ta     = All_LP_ta.write(i,     LP)
            
            # gather data + per-step coefficients for step i
            input_i = inputs[i]
            mask_i  = mask[:, i]
            
            rec_obs_i      = rec_obs_var_coefs_seq[i]
            rec_hid_i      = rec_hidden_var_coefs_seq[i]
            rec_next_hid_i = rec_next_hidden_var_coefs_seq[i]
            rec_bias_i     = rec_biases_seq[i]
            trans_hid_i    = transition_hidden_var_coefs_seq[i]
            trans_bias_i   = transition_biases_seq[i]
            
            log_factors_i        = flat_Log_seq[i]
            trans_log_factors_i  = flat_trans_Log_seq[i]
            trans_mean_i         = transition_mean_seq[i]
            trans_var_i          = transition_var_seq[i]
            dt_ratios = input_dts[:, i+1]/reference_dt
            
            # core single-step recurrence (RNN_cell unchanged)
            (Next_coefs, Next_biases, Next_LP, Next_segment_len,
             Next_gamma_mean, Next_gamma_var, Next_states) = RNN_cell(
                input_i, Prev_coefs, Prev_biases, LP, segment_len,
                log_factors_i, trans_log_factors_i,
                rec_obs_i, rec_hid_i, rec_next_hid_i, rec_bias_i,
                trans_hid_i, trans_bias_i,
                sequence_phase_1, sequence_phase_2,
                transition_mask, transition_sequence,
                trans_mean_i, trans_var_i,
                gamma_dist_mean, gamma_dist_var, states, dt_ratios)
            
            #(Next_LP - nb_dims * tf.math.log(
            #    tf.math.abs(tf.reduce_prod(tf.linalg.diag_part(tf.transpose(Next_coefs, [1, 2, 0, 3])), axis=-1)) + 1e-20))[10:20]
            
            # masked update: only advance still-alive tracks
            mask_coef   = mask_i[None, :, None, None]
            mask_scalar = mask_i[:, None]
            mask_state  = mask_i[:, None, None, None]
            
            Prev_coefs      = Next_coefs       * mask_coef   + Prev_coefs      * (1 - mask_coef)
            Prev_biases     = Next_biases      * mask_coef   + Prev_biases     * (1 - mask_coef)
            LP              = Next_LP          * mask_scalar + LP              * (1 - mask_scalar)
            segment_len     = Next_segment_len * mask_scalar + segment_len     * (1 - mask_scalar)
            gamma_dist_mean = Next_gamma_mean  * mask_scalar + gamma_dist_mean * (1 - mask_scalar)
            gamma_dist_var  = Next_gamma_var   * mask_scalar + gamma_dist_var  * (1 - mask_scalar)
            states          = Next_states      * mask_state  + states          * (1 - mask_state)
            
            return (i + 1, Prev_coefs, Prev_biases, LP, segment_len,
                    gamma_dist_mean, gamma_dist_var, states,
                    All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta)
        
        cond = lambda i, *_: i < num_steps
        
        (i, Prev_coefs, Prev_biases, LP, segment_len,
         gamma_dist_mean, gamma_dist_var, states,
         All_states_ta, All_coefs_ta,
         All_biases_ta, All_LP_ta) = tf.while_loop(cond, body,
            loop_vars=[tf.constant(0), Prev_coefs, Prev_biases, 
                       LP, segment_len, gamma_dist_mean, 
                       gamma_dist_var, states, All_states_ta, 
                       All_coefs_ta, All_biases_ta, All_LP_ta],
            parallel_iterations=1, swap_memory=True)
        
        # ---- stack and rearrange to the layer's output convention --------
        All_states = tf.transpose(All_states_ta.stack(),
                                  perm=[1, 0, 2, 3])[:, :, 0, :]
        All_coefs  = tf.transpose(All_coefs_ta.stack(),
                                  perm=[2, 0, 3, 1, 4])
        All_biases = tf.transpose(All_biases_ta.stack(),
                                  perm=[2, 0, 3, 1, 4])
        All_LPs    = tf.transpose(All_LP_ta.stack(),
                                  perm=[1, 0, 2])
        All_states = All_states[:, sequence_length - 1:]
        
        return (Prev_coefs, Prev_biases, LP, segment_len,
                gamma_dist_mean, gamma_dist_var,
                All_states, All_coefs, All_biases, All_LPs, states)
    
    
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
"""
def extract_smooth_hidden_variables(tracks, LocErrs, dts, masks, pred_model, batch_size, sequence_length, motion_types):
    '''
    tracks = tf_tracks
    dts = time_steps
    
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
    is_first = np.ones(tracks.shape[0])
    
    nb_dims = tracks.shape[-1]
    LPs, preds_1, All_coefs_1, All_biases_1, All_LPs_1 = pred_model.predict((tracks, LocErrs, dts, masks, is_first), batch_size = batch_size)
    
    # transpose the shape and rate matrices for the hidden state inference on the reversed tracks
    pred_model.weights[7].assign(tf.transpose(pred_model.weights[7]))
    pred_model.weights[8].assign(tf.transpose(pred_model.weights[8]))
    
    inverse_dts = np.concatenate((dts[:, -1:], dts[:, :-1]), axis = 1)[:, ::-1] # the last time step of dts is a dummy value that must not be used
    
    LPs, preds_2, All_coefs_2, All_biases_2, All_LPs_2 = pred_model.predict((tracks[:, ::-1], LocErrs[:, ::-1], inverse_dts, masks[:,::-1], is_first), batch_size = batch_size)
    
    # restore the model so subsequent calls behave normally
    pred_model.weights[7].assign(tf.transpose(pred_model.weights[7]))
    pred_model.weights[8].assign(tf.transpose(pred_model.weights[8]))

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
    
    position_mean = optimal_estimator(pos_mean_1, pos_mean_2, position_std_1**2, position_std_2**2)
    position_std = (1/((1 / position_std_1**2 + 1 / position_std_2**2)))**0.5
    
    anomalous_mean = optimal_estimator(anomalous_mean_1, anomalous_mean_2, anomalous_std_1**2, anomalous_std_2**2)
    anomalous_std = (1/((1 / anomalous_std_1**2 + 1 / anomalous_std_2**2)))**0.5
    
    mean_preds = (preds_1 + preds_2[:,::-1])/2
    
    # The anomalous variable cannot be averaged along the state axis to avoid mixing velocity vector and potential well position which have different orders of magnitude
    return position_mean, position_std, anomalous_mean, anomalous_std, mean_preds
"""
def extract_smooth_hidden_variables(tracks, LocErrs, dts, masks, pred_model,
                                    batch_size, sequence_length, motion_types, reference_dt):
    '''
    Variable-dt-aware version: returns per-step VELOCITY (dt-independent)
    for directed states instead of the step-wise displacement returned by
    the fixed-dt version.

    Why a rescaling is needed
    -------------------------
    Inside `constraint_function` the anomalous variable is encoded as
        directed :  ano_t  ~  velocity_t  *  dts[t]            (depends on dt)
        confined :  ano_t  ~  well anchor                       (dt-independent)
    so the raw `ano` returned by the forward pass is a step displacement
    for directed states.  Dividing by the dt the model actually used at
    that step recovers the velocity, which is what we want when dts vary
    between steps.

    Parameters
    ----------
    motion_types : iterable of length nb_states (the *physical* states only)
        1 for a directed state, 0 for a confined state.  The mislinking
        state appended internally by the model has is_directed_flag = 0
        and is treated as confined here.

    Other arguments and return signature are unchanged from the fixed-dt
    version.  `anomalous_mean` / `anomalous_std` are now velocities (for
    directed states) and well-anchor positions (for confined states) — both
    dt-independent quantities.
    dts=time_steps
    '''
    motion_types = np.array(motion_types)
    nb_dims = tracks.shape[-1]
    is_first = np.ones(tracks.shape[0])
    
    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    LPs, preds_1, All_coefs_1, All_biases_1, All_LPs_1 = pred_model.predict(
        (tracks, LocErrs, dts, masks, is_first), batch_size=batch_size)
    
    # ------------------------------------------------------------------
    # Reverse pass — transpose the Gamma shape/rate matrices so that
    # transitions s -> s' in the original direction become s' -> s.
    # ------------------------------------------------------------------
    pred_model.weights[7].assign(tf.transpose(pred_model.weights[7]))
    pred_model.weights[8].assign(tf.transpose(pred_model.weights[8]))
    
    # The last column of dts is a dummy (carryover slot) that must not be
    # used as a real step.  Putting it at position 0 of the reversed array
    # keeps every "real" dt aligned with the reversed observations.
    inverse_dts = np.concatenate((dts[:, -1:], dts[:, :-1]), axis=1)[:, ::-1]
    
    LPs, preds_2, All_coefs_2, All_biases_2, All_LPs_2 = pred_model.predict(
        (tracks[:, ::-1], LocErrs[:, ::-1], inverse_dts, masks[:, ::-1], is_first),
        batch_size=batch_size)
    
    # restore the model so subsequent calls behave normally
    pred_model.weights[7].assign(tf.transpose(pred_model.weights[7]))
    pred_model.weights[8].assign(tf.transpose(pred_model.weights[8]))
    
    # ------------------------------------------------------------------
    # Per-pass marginals of the hidden variables
    # ------------------------------------------------------------------
    pos_mean_1, anomalous_mean_1, position_std_1, anomalous_std_1 = \
        extract_hidden_variables(All_coefs_1, All_biases_1, All_LPs_1,
                                 nb_dims, sequence_length)
    
    pos_mean_2, anomalous_mean_2, position_std_2, anomalous_std_2 = \
        extract_hidden_variables(All_coefs_2, All_biases_2, All_LPs_2,
                                 nb_dims, sequence_length)
    
    # ==================================================================
    # NEW: convert directed-state ano (= velocity * dt) to velocity.
    #
    # All_coefs_*[i] encodes p(pos_{i+1}, ano_{i+1} | obs_{0..i}), so the
    # i-th entry of anomalous_mean_1 used dts[:, i+1] and the i-th entry
    # of anomalous_mean_2 used inverse_dts[:, i+1].  Confined states are
    # left unchanged because their ano is the dt-independent well anchor.
    # The mislinking state (appended last) has is_directed_flag = 0, so
    # we extend motion_types with a 0.
    # ==================================================================
    isdir_state  = motion_types[None, None, :, None]   # broadcasts to (1, 1, nb_states+1, 1)
    isconf_state = 1.0 - isdir_state
    
    nb_time = anomalous_mean_1.shape[1]                      # = T - 1
    
    fwd_dt = dts[:,         1:1+nb_time][:, :, None, None] # (N, T-1, 1, 1)
    rev_dt = inverse_dts[:, 1:1+nb_time][:, :, None, None] # (N, T-1, 1, 1)
    
    # 1/dt for directed states, 1.0 for confined states
    fwd_scale = isdir_state / fwd_dt * reference_dt + isconf_state
    rev_scale = isdir_state / rev_dt * reference_dt + isconf_state
    
    anomalous_mean_1 = anomalous_mean_1 * fwd_scale
    anomalous_std_1  = anomalous_std_1  * fwd_scale
    anomalous_mean_2 = anomalous_mean_2 * rev_scale
    anomalous_std_2  = anomalous_std_2  * rev_scale
    # ==================================================================
    
    # ------------------------------------------------------------------
    # Align forward and reverse passes to the same time axis
    # ------------------------------------------------------------------
    pos_mean_1     = pos_mean_1[:, 1:]
    position_std_1 = position_std_1[:, 1:]
    pos_mean_2     = pos_mean_2[:, :0:-1]
    position_std_2 = position_std_2[:, :0:-1]
    
    # Reversing time flips the sign of a velocity (drift) but not of a
    # well anchor.  This is exactly the same logic as before — it acts on
    # the *velocity* now rather than on `velocity * dt`, which is fine
    # because dt is positive.
    motion_type_sign = (-1 * (motion_types == 1)
                        + 1 * (motion_types == 0))
    motion_type_sign = motion_type_sign[None, None, :, None]
    
    anomalous_mean_1 = anomalous_mean_1[:, 1:]
    anomalous_std_1  = anomalous_std_1[:, 1:]
    anomalous_mean_2 = motion_type_sign * anomalous_mean_2[:, :0:-1]
    anomalous_std_2  = anomalous_std_2[:, :0:-1]
    
    '''
    anomalous_mean_1[1, :, 0]
    anomalous_mean_2[1, :, 0]
    anomalous_std_1[0, 60]
    anomalous_std_2[0, 60]
    '''
    # ------------------------------------------------------------------
    # Precision-weighted fusion (smoothing)
    # ------------------------------------------------------------------
    def optimal_estimator(x1, x2, var1, var2):
        w1 = 1 / var1
        w2 = 1 / var2
        return (w1 * x1 + w2 * x2) / (w1 + w2)
    
    position_mean = optimal_estimator(pos_mean_1, pos_mean_2, position_std_1**2, position_std_2**2)
    position_std  = (1 / (1 / position_std_1**2 + 1 / position_std_2**2))**0.5
    
    anomalous_mean = optimal_estimator(anomalous_mean_1, anomalous_mean_2, anomalous_std_1**2, anomalous_std_2**2)
    anomalous_std  = (1 / (1 / anomalous_std_1**2 + 1 / anomalous_std_2**2))**0.5
        
    mean_preds = (preds_1 + preds_2[:, ::-1]) / 2
    
    # `anomalous_mean` is now:
    #   - velocity (dt-independent) for directed states
    #   - well-anchor position       for confined states
    return position_mean, position_std, anomalous_mean, anomalous_std, mean_preds


class Final_layer(tf.keras.layers.Layer):
    def __init__(self, sequence_phase_1, nb_dims, sequence_length, **kwargs):
        self.sequence_phase_1 = sequence_phase_1
        self.nb_dims = nb_dims
        self.sequence_length = sequence_length
        super().__init__(**kwargs)
    '''
    Final layer of the model that integrates over the remaining hidden variables
    '''
    
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
    LocErrs = np.ones((1,1))
    def LocErr_function(LocErrs, LocErr_param):
        return LocErrs
    dts = np.ones((1,2))
    hidden_var_coefs, _, _, _, initial_hidden_var_coefs, _, _, _,  transition_hidden_var_coefs, _, _, integration_variable_index, _, _, _ = constraint_function(params, initial_params, LocErrs, dts, nb_dims, 1., LocErr_function, dtype)
    hidden_var_coefs = hidden_var_coefs[0]
    transition_hidden_var_coefs = transition_hidden_var_coefs[0]
    
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

'''
#dts=dts_TN
@tf.function
def transition_param_function(transition_shapes, transition_rates, density, Fs,effective_ds,  dts, reference_dt, dtype):
    
    
    print('transition_shapes', transition_shapes)
    nb_states = transition_shapes.shape[0]
    nb_time_points, nb_tracks = dts.shape
    
    transition_shapes = tf.math.exp(transition_shapes)
    transition_rates = tf.math.softmax(transition_rates, axis = 1)*transition_shapes/reference_dt
    transition_rates = transition_rates[None,None]*dts[..., None, None]+1e-20
    
    new_transition_shapes = tf.concat((transition_shapes, tf.constant([[1]*nb_states], dtype = dtype)), axis = 0)
    new_transition_shapes = tf.concat((new_transition_shapes, tf.constant([[1]]*(nb_states+1), dtype = dtype)), axis = 1)
    
    mislinking_dwell_time = tf.constant([0.9/nb_states]*nb_states, dtype = dtype) # We multiply by (1-tf.reduce_mean(additional_transition_params[:nb_states]) to allow several consecutive mislinkings proportionally to the misslinking probability
    mislinking_dwell_time = tf.concat((mislinking_dwell_time, [0.1]), axis = 0)
    mislinking_dwell_time = tf.broadcast_to(mislinking_dwell_time[None,None,None], (nb_time_points, nb_tracks, 1, nb_states + 1))
    mislinking_dwell_time.shape
    #mislinking_rates = tf.constant([0.078,0.146], dtype = dtype)[:, None] # density 1 -> rates 0.052 0.052 
    mislinking_rates = 1-tf.math.exp(-0.5*density *tf.reduce_sum(Fs[None]*(effective_ds[:,None]**2 + effective_ds[None]**2)**0.5, axis = 0)[:, None])# density 0.1 -> rates 0.052 0.052 
    mislinking_rates = tf.broadcast_to(mislinking_rates[None,None],
                                              (nb_time_points, nb_tracks, nb_states, 1))
    new_transition_rates = tf.concat((transition_rates, mislinking_rates), axis = 3)
    new_transition_rates = tf.concat((new_transition_rates, mislinking_dwell_time), axis = 2)
    
    return new_transition_shapes, new_transition_rates
'''

min_lifetime = 3

@tf.function
def transition_param_function(transition_shapes, transition_rates, density, Fs,effective_ds,  dts, reference_dt, dtype):
    '''
    The transition_param_function must define the initial transition parameters and their constraints
    similarly to how constraint_function defines the constraints of the states
    '''
    
    print('transition_shapes', transition_shapes)
    nb_states = transition_shapes.shape[0]
    nb_time_points, nb_tracks = dts.shape
    
    max_off_rate  = 1/min_lifetime
    
    transition_shapes = tf.math.exp(transition_shapes)
    transition_rates = (tf.eye(nb_states, dtype = dtype) * (1 - max_off_rate + max_off_rate*tf.math.softmax(transition_rates, axis = 1)) 
                       + (1-tf.eye(nb_states, dtype = dtype)) * ( max_off_rate*tf.math.softmax(transition_rates, axis = 1)) + 1e-7)
    
    transition_rates = transition_rates[None,None]*transition_shapes[None,None]*(1 + 0*dts[..., None, None]/reference_dt)+1e-20
    
    new_transition_shapes = tf.concat((transition_shapes, tf.constant([[1]*nb_states], dtype = dtype)), axis = 0)
    new_transition_shapes = tf.concat((new_transition_shapes, tf.constant([[1]]*(nb_states+1), dtype = dtype)), axis = 1)
    
    mislinking_dwell_time = tf.constant([0.9/nb_states]*nb_states, dtype = dtype) # We multiply by (1-tf.reduce_mean(additional_transition_params[:nb_states]) to allow several consecutive mislinkings proportionally to the misslinking probability
    mislinking_dwell_time = tf.concat((mislinking_dwell_time, [0.1]), axis = 0)
    mislinking_dwell_time = tf.broadcast_to(mislinking_dwell_time[None,None,None], (nb_time_points, nb_tracks, 1, nb_states + 1))
    mislinking_dwell_time.shape
    #mislinking_rates = tf.constant([0.078,0.146], dtype = dtype)[:, None] # density 1 -> rates 0.052 0.052 
    mislinking_rates = 1-tf.math.exp(-0.5*density *tf.reduce_sum(Fs[None]*(effective_ds[:,None]**2 + effective_ds[None]**2)**0.5, axis = 0)[:, None])# density 0.1 -> rates 0.052 0.052 
    mislinking_rates = tf.broadcast_to(mislinking_rates[None,None],
                                              (nb_time_points, nb_tracks, nb_states, 1))
    new_transition_rates = tf.concat((transition_rates, mislinking_rates), axis = 3)
    new_transition_rates = tf.concat((new_transition_rates, mislinking_dwell_time), axis = 2)
    
    return new_transition_shapes, new_transition_rates




def get_model_raw_params(model, track_segmentation = True, return_dict = False):
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
                reference_dt,
                nb_dims = 2, # Number of dimensions of the tracks
                sequence_length = 3, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = 3, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = 0.001, # Estimated density of the sample.
                vary_params = True,
                vary_initial_params = True,
                vary_initial_fractions = True,
                vary_transition_shapes = False,
                vary_transition_rates = True, 
                nb_LocErr_dims = 0,
                LocErr_type = 'Linear'):
    
    # Defining the hyperparameters of the model    
    nb_obs_vars = 1 # number of dependend variables (the x, y, z dimension do not account as dependent variables in our model so keep this to 1)
    nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars
    
    inputs = tf.keras.Input(batch_shape=(batch_size, track_len, nb_independent_vars), name = 'tracks', dtype = dtype)
    if nb_LocErr_dims>0:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len, nb_LocErr_dims), name = 'Localization errors', dtype = dtype)
    else:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len), name = 'Localization errors', dtype = dtype)
    input_dts = tf.keras.Input(batch_shape=(batch_size, track_len+1), name = 'frame durations', dtype = dtype)
    input_mask = tf.keras.Input(batch_shape = (batch_size, track_len), name = 'masks', dtype = dtype)
    input_isfirst = tf.keras.Input(batch_shape = (batch_size,), name = 'isfirsts', dtype = dtype)
    '''
    seq = TrackSegmentSequence(
        track_list,
        batch_size=50,
        segment_length=20,
        min_segment_length=4,
        cutoff_batch_treshhold=0.5)
    
    all_inputs, outputs = seq[0]
    inputs = all_inputs[0]
    input_LocErrs = all_inputs[1]
    input_dts = all_inputs[2]
    input_mask = tf.constant(all_inputs[3], dtype = dtype)
    input_isfirst = tf.constant(all_inputs[4], dtype = dtype)
    
    inputs = tf.constant(inputs, dtype=dtype)
    input_LocErrs = tf.constant(input_LocErrs, dtype=dtype)
    input_dts = tf.constant(input_dts, dtype=dtype)
    input_mask = tf.constant(input_mask, dtype=dtype)
    input_isfirst = tf.constant(input_isfirst, dtype=dtype)

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
                                           reference_dt = reference_dt,
                                           vary_params = vary_params,
                                           vary_initial_params = vary_initial_params,
                                           vary_initial_fractions = vary_initial_fractions,
                                           sequence_length = sequence_length,
                                           carryover = True,
                                           LocErr_type = LocErr_type,
                                           dtype = dtype)
    
    #self = Init_layer
    #Init_layer = model.layers[7]
    tensor1, initial_states = Init_layer(transposed_inputs, input_LocErrs, input_dts)
    
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
    # self = model.layers[14]
    
    Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var, All_motion_states, All_coefs, All_biases, All_LPs, motion_states = layer(sliced_inputs, input_dts, reference_dt, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir, isfirst = input_isfirst)
    
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
    
    model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
                           outputs=outputs, name="Diffusion_model")
    
    #carried_Prev_coefs, carried_Prev_biases, carried_LP, carried_segment_len, carried_gamma_dist_mean, carried_gamma_dist_var = All_states
    pred_model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
                                outputs=(outputs, All_states, All_coefs, All_biases, All_LPs), name="Diffusion_model")
    
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

def build_model(track_len, # maximum number of time points in the input tracks 
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions, 
                batch_size, # number of tracks analysed at the same time
                reference_dt,
                nb_dims = 2, # Number of dimensions of the tracks
                sequence_length = 3, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = 3, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = 0.001, # Estimated density of the sample.
                vary_params = None,
                vary_initial_params = None,
                vary_initial_fractions = None,
                vary_transition_shapes = None,
                vary_transition_rates = None,
                nb_LocErr_dims = 1,
                LocErr_type = 'Linear'):
    
    # Defining the hyperparameters of the model
    dtype = 'float64'
    
    nb_obs_vars = 1 # number of dependend variables (the x, y, z dimension do not account as dependent variables in our model so keep this to 1)
    nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars
    
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, 1, nb_independent_vars), dtype = dtype)
    if nb_LocErr_dims>0:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len, nb_LocErr_dims), name = 'Localization errors', dtype = dtype)
    else:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len), name = 'Localization errors', dtype = dtype)
    input_dts = tf.keras.Input(batch_shape=(batch_size, track_len+1), name = 'frame durations', dtype = dtype)
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
                                           reference_dt = reference_dt,
                                           vary_params = vary_params,
                                           vary_initial_params = vary_initial_params,
                                           vary_initial_fractions = vary_initial_fractions,
                                           sequence_length = sequence_length,
                                           LocErr_type = LocErr_type,
                                           dtype = dtype)
    
    tensor1, initial_states = Init_layer(transposed_inputs, input_LocErrs, input_dts)
    
    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds = Init_layer.param_vars[:, 1]
    anomalous_factors = Init_layer.param_vars[:, 2]
    isdir = Init_layer.param_vars[:, 4]
    
    Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states
    
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype = dtype)(input_mask)
    
    layer = Custom_RNN_layer(batch_size, transition_shapes, transition_rates, estimated_density, nb_states, Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, Init_layer.transition_sequence, transition_param_function, sequence_length = sequence_length, vary_transition_shapes = vary_transition_shapes, vary_transition_rates = vary_transition_rates, dtype = dtype)
    
    isfirst = tf.ones((batch_size), dtype = dtype)
    Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var, All_states       , All_coefs, All_biases, All_LPs,        states = layer(sliced_inputs, input_dts, reference_dt, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir, isfirst = isfirst)

    F_layer = Final_layer(Init_layer.final_sequence_phase_1, nb_dims = nb_independent_vars, sequence_length = sequence_length, dtype = dtype)
    outputs, All_states = F_layer([Prev_coefs, Prev_biases, LP, All_states, states])
    
    model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask), outputs=outputs, name="Diffusion_model")
    pred_model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask), outputs=(All_states, All_coefs, All_biases, All_LPs), name="Diffusion_model")
    
    return model, pred_model

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

class get_parameters(tf.keras.callbacks.Callback):
    def __init__(self, track_segmentation = True, layer_name='params'):
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
        #transition_rates = tf.math.softmax(weights[rate_idx], axis = 1)*transition_shapes
        max_off_rate = 1/min_lifetime
        transition_rates = weights[rate_idx]
        transition_rates = (tf.eye(nb_states, dtype = dtype) * (1 - max_off_rate + max_off_rate*tf.math.softmax(transition_rates, axis = 1)) 
                           + (1-tf.eye(nb_states, dtype = dtype)) * ( max_off_rate*tf.math.softmax(transition_rates, axis = 1)) + 1e-7)
        transition_rates = transition_rates * transition_shapes
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
    nb_states = weights[0].shape[0]
    if track_segmentation:
        shape_IDs = 8
        rates_IDs = 7
    else:
        shape_IDs = 5
        rates_IDs = 4    
    transition_shapes = tf.math.exp(weights[shape_IDs]).numpy()
    max_off_rate = 1/min_lifetime
    transition_rates = weights[rates_IDs]
    transition_rates = (tf.eye(nb_states, dtype = dtype) * (1 - max_off_rate + max_off_rate*tf.math.softmax(transition_rates, axis = 1)) 
                       + (1-tf.eye(nb_states, dtype = dtype)) * ( max_off_rate*tf.math.softmax(transition_rates, axis = 1)) + 1e-7)
    transition_rates = transition_rates * transition_shapes
    transition_rates = transition_rates.numpy()
    model_types = weights[0][:, -1].numpy().astype(int)
    model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]
    anomalous_factors = tf.sigmoid(weights[0][:, 2])*(1-weights[0][:, 4]) + 2**0.5*tf.exp(weights[0][:, 2])*weights[0][:, 4]
    anomalous_factors = anomalous_factors.numpy()
    param_dict = {'Model types': model_types_str, 'anomalous factors': anomalous_factors, 'Localization errors': np.exp(weights[0][:, 0]), 'd': np.exp(weights[0][:, 1]), 'q': np.exp(weights[0][:, 3]), 'transition rates': transition_rates, 'transition shapes': transition_shapes, 'Fractions': tf.math.softmax(weights[2][0]).numpy()}
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


def segment_tracks(track_list, LocErr_list, dt_list, batch_size, segment_length=20, min_segment_length=4, cutoff_batch_treshhold = 0.5, shuffle = False):
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
    dt_batches : np.ndarray
        batches of times between steps of the shape (number of batch, batch_size, seg_len+1).
        the number of time steps must be one step longer than seg_len to accomodate the changes of velocity accross time steps in the constraint_function
    LocErr_batches : np.ndarray
        batches of estimated localization errors of the shape (number of batch, batch_size, seg_len+1).
        the number of time steps must be one step longer than seg_len to accomodate the changes of velocity accross time steps in the constraint_function
    
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
    if len(LocErr_list[0].shape)==2:
        LocErr_batches = np.zeros((max_nb_batches, batch_size, segment_length, LocErr_list[0].shape[1]))
    elif len(LocErr_list[0].shape)==1:
        LocErr_batches = np.zeros((max_nb_batches, batch_size, segment_length))

    mean_dt = np.mean(np.concatenate(dt_list))
    dt_batches = np.zeros((max_nb_batches, batch_size, segment_length+1)) + mean_dt
    mask_batches = np.zeros((max_nb_batches, batch_size, segment_length))
    isfirst_batches = np.ones((max_nb_batches, batch_size))
    
    if shuffle:
        shuffling_IDs = np.random.permutation(len(track_list))
        track_list = [track_list[i] for i in shuffling_IDs]
        LocErr_list = [LocErr_list[i] for i in shuffling_IDs]
        dt_list = [dt_list[i] for i in shuffling_IDs]
    
    for track, LocErrs, dts in zip(track_list, LocErr_list, dt_list):
        
        #track, LocErrs, dts = track_list[1], LocErr_list[1], dt_list[1]
        nb_segments = len(track)//segment_length
        if len(track)%segment_length > min_segment_length:
            nb_segments += 1
        
        # Then we find where to add the segments of this track
        batch_IDs, index_IDs = np.where(mask_batches[:,:, 0]==0)
        batch_ID, index_ID = (batch_IDs[0], index_IDs[0])
        
        for i in range(nb_segments):
            
            segment = track[i*(segment_length-1):(i+1)*segment_length-i]
            LocErr_segment = LocErrs[i*(segment_length-1):(i+1)*segment_length-i]
            dt_segment = dts[i*(segment_length-1):(i+1)*segment_length-i+1]
            track_batches[batch_ID + i, index_ID, :len(segment)] = segment
            track_batches[batch_ID + i, index_ID, len(segment):] = segment[-1]
            LocErr_batches[batch_ID + i, index_ID, :len(LocErr_segment)] = LocErr_segment
            LocErr_batches[batch_ID + i, index_ID, len(LocErr_segment):] = LocErr_segment[-1]
            dt_batches[batch_ID + i, index_ID, :len(dt_segment)] = dt_segment
            dt_batches[batch_ID + i, index_ID, len(dt_segment):] = dt_segment[-1]
            
            mask_batches[batch_ID + i, index_ID, :len(segment)] = 1
            if i != 0:
                isfirst_batches[batch_ID + i, index_ID] = 0
        
    nb_batches = np.argmin(np.mean(mask_batches[:,:,0], 1) >= cutoff_batch_treshhold)
    
    track_batches = track_batches[:nb_batches]
    LocErr_batches = LocErr_batches[:nb_batches]
    dt_batches = dt_batches[:nb_batches]    
    mask_batches = mask_batches[:nb_batches]
    isfirst_batches = isfirst_batches[:nb_batches]
    return track_batches, LocErr_batches, dt_batches, mask_batches, isfirst_batches

"""
Numba-accelerated version of the above segment_tracks function.

Drop-in replacement for the original function: same signature, same outputs
(verified against the original on random data).

Key changes vs. the original:
  * Variable-length lists are concatenated into flat float64 arrays + offsets
    before entering the jitted core (Numba is much happier with flat arrays
    than with reflected lists of ndarrays).
  * The per-track `np.where(mask_batches[:, :, 0] == 0)[0]` scan is replaced
    by an O(batch_size) "leftmost column with minimum next-empty-row" lookup
    that produces the exact same placement order as the original C-order
    np.where call.
  * The hot inner loop (copy segment, pad with last value, write mask) is
    compiled with @njit(cache=True).
"""

import numpy as np
from numba import njit

@njit(cache=True)
def _segment_tracks_core(
    tracks_flat,        # (total_len, nb_dims)        float64
    locerrs_flat,       # (total_len, locerr_dims)    float64
    dts_flat,           # (total_dt_len,)             float64
    track_offsets,      # (nb_tracks + 1,)            int64  -- cumsum of track lengths
    dt_offsets,         # (nb_tracks + 1,)            int64  -- cumsum of dt lengths
    batch_size,
    segment_length,
    min_segment_length,
    max_nb_batches,
    mean_dt):
    
    nb_dims = tracks_flat.shape[1]
    locerr_dims = locerrs_flat.shape[1]

    track_batches   = np.zeros((max_nb_batches, batch_size, segment_length, nb_dims))
    locerr_batches  = np.zeros((max_nb_batches, batch_size, segment_length, locerr_dims))
    dt_batches      = np.full((max_nb_batches, batch_size, segment_length + 1), mean_dt)
    mask_batches    = np.zeros((max_nb_batches, batch_size, segment_length))
    isfirst_batches = np.ones((max_nb_batches, batch_size))

    # next_row[c] = next empty row in column c.
    # Picking the leftmost column with the minimum next_row reproduces the
    # original "first empty cell in C-order" placement.
    next_row = np.zeros(batch_size, dtype=np.int64)

    nb_tracks = len(track_offsets) - 1

    for t in range(nb_tracks):
        track_start = track_offsets[t]
        track_end   = track_offsets[t + 1]
        track_len   = track_end - track_start

        dt_start      = dt_offsets[t]
        dt_track_len  = dt_offsets[t + 1] - dt_start

        nb_segments = track_len // segment_length
        if track_len % segment_length > min_segment_length:
            nb_segments += 1
        if nb_segments == 0:
            continue

        # Find leftmost column with min next_row (equivalent to original np.where).
        min_row = next_row[0]
        min_col = 0
        for c in range(1, batch_size):
            if next_row[c] < min_row:
                min_row = next_row[c]
                min_col = c

        batch_ID = min_row
        index_ID = min_col

        for i in range(nb_segments):
            # Slicing arithmetic copied verbatim from the original.
            seg_start         = i * (segment_length - 1)
            seg_end_unclipped = (i + 1) * segment_length - i

            seg_end = seg_end_unclipped
            if seg_end > track_len:
                seg_end = track_len
            seg_len = seg_end - seg_start
            last_track_idx = track_start + seg_end - 1

            row = batch_ID + i

            # --- tracks: bulk copy then pad with last value ---
            track_batches[row, index_ID, :seg_len] = \
                tracks_flat[track_start + seg_start : track_start + seg_end]
            for k in range(nb_dims):
                pad_val = tracks_flat[last_track_idx, k]
                for j in range(seg_len, segment_length):
                    track_batches[row, index_ID, j, k] = pad_val

            # --- localization errors: same pattern ---
            locerr_batches[row, index_ID, :seg_len] = \
                locerrs_flat[track_start + seg_start : track_start + seg_end]
            for k in range(locerr_dims):
                pad_val = locerrs_flat[last_track_idx, k]
                for j in range(seg_len, segment_length):
                    locerr_batches[row, index_ID, j, k] = pad_val

            # --- dt: one extra time step, uses the *unclipped* end + 1 ---
            dt_end = seg_end_unclipped + 1
            if dt_end > dt_track_len:
                dt_end = dt_track_len
            dt_seg_len   = dt_end - seg_start
            last_dt_idx  = dt_start + dt_end - 1

            dt_batches[row, index_ID, :dt_seg_len] = \
                dts_flat[dt_start + seg_start : dt_start + dt_end]
            pad_val_dt = dts_flat[last_dt_idx]
            for j in range(dt_seg_len, segment_length + 1):
                dt_batches[row, index_ID, j] = pad_val_dt

            # --- mask ---
            for j in range(seg_len):
                mask_batches[row, index_ID, j] = 1.0

            if i != 0:
                isfirst_batches[row, index_ID] = 0.0

        next_row[index_ID] = batch_ID + nb_segments

    return track_batches, locerr_batches, dt_batches, mask_batches, isfirst_batches

def segment_tracks(
    track_list,
    LocErr_list,
    dt_list,
    batch_size,
    segment_length=20,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5,
    shuffle=False):
    """
    Numba-accelerated version of segment_tracks.
     
    Drop-in replacement for the original function: same signature, same outputs
    (verified against the original on random data).
     
    Key changes vs. the original:
      * Variable-length lists are concatenated into flat float64 arrays + offsets
        before entering the jitted core (Numba is much happier with flat arrays
        than with reflected lists of ndarrays).
      * The per-track `np.where(mask_batches[:, :, 0] == 0)[0]` scan is replaced
        by an O(batch_size) "leftmost column with minimum next-empty-row" lookup
        that produces the exact same placement order as the original C-order
        np.where call.
      * The hot inner loop (copy segment, pad with last value, write mask) is
        compiled with @njit(cache=True).
    """

    # ---- clip cutoff (preserve original behavior) ----
    if cutoff_batch_treshhold > 1:
        cutoff_batch_treshhold = 1
    elif cutoff_batch_treshhold < 1 / batch_size:
        cutoff_batch_treshhold = 0.5 / batch_size
    
    if LocErr_list is None:
        LocErr_list = [np.ones((len(track))) for track in track_list]
    
    nb_tracks = len(track_list)
    nb_dims   = track_list[0].shape[1]
    locerr_is_1d = (LocErr_list[0].ndim == 1)
    locerr_dims  = 1 if locerr_is_1d else LocErr_list[0].shape[1]

    # ---- optional shuffle (kept outside the jit to use numpy's RNG) ----
    if shuffle:
        perm = np.random.permutation(nb_tracks)
        track_list  = [track_list[i]  for i in perm]
        LocErr_list = [LocErr_list[i] for i in perm]
        dt_list     = [dt_list[i]     for i in perm]
    
    # ---- flatten variable-length lists into contiguous arrays + offsets ----
    track_lens = np.fromiter((len(t) for t in track_list), dtype=np.int64, count=nb_tracks)
    dt_lens    = np.fromiter((len(d) for d in dt_list),    dtype=np.int64, count=nb_tracks)
    
    track_offsets = np.empty(nb_tracks + 1, dtype=np.int64)
    track_offsets[0] = 0
    np.cumsum(track_lens, out=track_offsets[1:])

    dt_offsets = np.empty(nb_tracks + 1, dtype=np.int64)
    dt_offsets[0] = 0
    np.cumsum(dt_lens, out=dt_offsets[1:])

    total_track_len = int(track_offsets[-1])
    total_dt_len    = int(dt_offsets[-1])

    tracks_flat  = np.zeros((total_track_len, nb_dims),     dtype=np.float64) - 1
    locerrs_flat = np.zeros((total_track_len, locerr_dims), dtype=np.float64)
    dts_flat     = np.zeros(total_dt_len,                   dtype=np.float64)
    
    for i in range(nb_tracks):
        s, e = int(track_offsets[i]), int(track_offsets[i + 1])
        tracks_flat[s:e] = track_list[i]
        if locerr_is_1d:
            locerrs_flat[s:e, 0] = LocErr_list[i]
        else:
            locerrs_flat[s:e] = LocErr_list[i]
        ds, de = int(dt_offsets[i]), int(dt_offsets[i + 1])
        dts_flat[ds:de] = dt_list[i]
    
    mean_dt = float(dts_flat.mean()) if total_dt_len > 0 else 0.0
    max_track_length = int(track_lens.max())
    max_nb_batches = (nb_tracks // batch_size + 1) * (max_track_length // segment_length + 1) * 2
    
    # ---- jitted core ----
    track_batches, locerr_batches, dt_batches, mask_batches, isfirst_batches = \
        _segment_tracks_core(
            tracks_flat, locerrs_flat, dts_flat,
            track_offsets, dt_offsets,
            int(batch_size), int(segment_length), int(min_segment_length),
            int(max_nb_batches), mean_dt)
    
    # ---- trim trailing under-filled batches (same rule as original) ----
    nb_batches = int(np.argmin(mask_batches[:, :, 0].mean(axis=1) >= cutoff_batch_treshhold))
    
    track_batches   = track_batches[:nb_batches]
    locerr_batches  = locerr_batches[:nb_batches]
    dt_batches      = dt_batches[:nb_batches]
    mask_batches    = mask_batches[:nb_batches]
    isfirst_batches = isfirst_batches[:nb_batches]
    
    # ---- restore 1D LocErr shape if input was 1D per track ----
    if locerr_is_1d:
        locerr_batches = locerr_batches[..., 0]
    
    return track_batches, locerr_batches, dt_batches, mask_batches, isfirst_batches

'''
lines to visually test if the segments are built well. 
track_batches.shape

for i in range(len(track_batches)):
    track_batches[i, :, 0] = track_batches[i, :, 0] * (1-isfirst_batches[i, :, None])

img = track_batches.transpose([0,2,1,3])
img = np.concatenate(img)[:,:,1]

plt.figure()
plt.imshow(img[-200:, :])
'''

self = tf.keras.utils.Sequence()
class TrackSegmentSequence(tf.keras.utils.Sequence):
    """Keras Sequence that pre-computes all batches from segment_tracks."""
    
    def __init__(self, track_list, LocErr_list, dt_list, batch_size, segment_length=20,
                 min_segment_length=4, cutoff_batch_treshhold=0.01, shuffle = False):
        """
        Parameters
        ----------
        track_list: List of numpy arrays of shape (number of time points, number of dimensions)
                    The number of time points can vary but not the number od dimensions.
        LocErr_list: List of localization errors corresponding to the localizations in track_list.
                     Can also be set to None to assume a single localization error (per state) 
                     shared across tracks. Each element must have the shape (number of time points)
        dt_list:     List of durations between frames corresponding to the localizations in track_list.
                     For a given track, dt_list[i] must equal time[i+1] - time[i]. A last element with
                     value reference_dt must be provided.
                     Can also be set to None to assume a dt shared across tracks and time points.
                     Each element of dt_list must have the shape (number of time points).
                     
        batch_size: (int) Batch size of the model.
        segment_length: (int) length of the track segments (number of time-steps per batch of the model).
        min_segment_length: (int) minimal segment length that we consider.
        cutoff_batch_treshhold : Cutoff to know which fraction of the non-full batches we keep.
                                 if 0 we remove all the non-full batches (which might be problematic),
                                 and if 1 we keep all the non-full batches.
        dummy_label_shape : tuple or None
            Per-sample label shape *after* the batch_size dimension.
            If None, defaults to a scalar zero per sample, i.e. shape (batch_size,).
        
        track_list = cur_track_list
        dt_list = cur_dt_list
        """
        self.track_list = track_list
        if LocErr_list is None:
            LocErr_list = [np.ones(len(track)) for track in track_list]
        if dt_list is None:
            dt_list = [np.ones(len(track)) for track in track_list]
                    
        self.LocErr_list = LocErr_list
        self.dt_list = dt_list
        self.segment_length = segment_length
        self.min_segment_length = min_segment_length
        self.cutoff_batch_treshhold = cutoff_batch_treshhold
        self.shuffle = shuffle

        self.tracks, self.LocErrs, self.dts, self.masks, self.isfirsts = segment_tracks(
            track_list, LocErr_list, dt_list, batch_size, segment_length,
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
                  self.LocErrs[idx],
                  self.dts[idx],
                  self.masks[idx],      # (batch_size, seg_len)
                  self.isfirsts[idx])   # (batch_size,)
        return inputs, self.dummy_labels
    
    def on_epoch_end(self):
        """Shuffle batch order between epochs."""
        if self.shuffle:
            self.tracks, self.LocErrs, self.dts, self.masks, self.isfirsts = segment_tracks(
                self.track_list, self.batch_size, self.segment_length,
                self.min_segment_length, self.cutoff_batch_treshhold, self.shuffle)

def Model_finder(track_list,
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
                 LocErr_list = None,
                 dt_list = None,
                 segment_length = 10,
                 learning_rate = 1/50,
                 decay_fraction = 0.2,
                 decay_rate = 0.002,
                 device = '/GPU:0', 
                 verbose = 1,       
                 shuffle = False,
                 vary_params = None,
                 vary_initial_params = None,
                 vary_initial_fractions = None,
                 vary_transition_shapes = None,
                 vary_transition_rates = None,
                 LocErr_type = 'Linear'):
    '''
    If a state is not found immobile, we test the alternative state hypothesis
    '''
    
    nb_states = params.shape[0]
    if LocErr_list is None:
        LocErr_list = [np.ones(len(track)) for track in track_list]
        nb_LocErr_dims = 0
    elif LocErr_list[0].ndim==2:
        nb_LocErr_dims = LocErr_list[0].shape[1]
    elif LocErr_list[0].ndim==1:
        nb_LocErr_dims = 0

    if dt_list is None:
        dt_list = [np.ones(len(track)) for track in dt_list] 
    
    seq = TrackSegmentSequence(track_list, 
                               LocErr_list,
                               dt_list,
                               batch_size=batch_size,
                               segment_length=segment_length,
                               min_segment_length=4,
                               cutoff_batch_treshhold=0.5,
                               shuffle = shuffle)
    
    nb_batches = len(seq)
    nb_dims = track_list[0].shape[-1]
    initial_anomalous_factors = params[:, 2]
    
    model, pred_model = build_segment_model(segment_length, # maximum number of time points in the input tracks
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
                    nb_LocErr_dims = nb_LocErr_dims,
                    LocErr_type = LocErr_type)
    
    beta_1 = max(1 - 5/nb_batches, 0.8)
    beta_2 = 1 - 0.1/nb_batches
    
    decay_threshold = int(epochs * nb_batches * decay_fraction)
    
    lr = WarmupLearningRateSchedule(20, learning_rate, decay_rate, decay_threshold) # learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, clipvalue=0.01) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
    model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)
    callbacks = [get_parameters(track_segmentation = True)]
    #preds = model.predict(seq)
    #MLE_loss(preds, preds)    
    
    with tf.device(device):
        history = model.fit(seq, epochs = epochs, callbacks = callbacks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
    
    All_models = {}
    params, initial_params, initial_fractions, transition_shapes, transition_rates = get_model_raw_params(model, track_segmentation = True)

    LogLikelihood = - history.history['loss'][-1]
    loss_history = history.history['loss']
    
    All_models['Model 0'] = {'params': params, 'initial_params': initial_params, 'initial_fractions': initial_fractions, 'transition_shapes': transition_shapes, 'transition_rates': transition_rates, 'LogLikelihood': LogLikelihood, 'loss_history': loss_history}
    best_LogLikelihood = LogLikelihood
    best_model = 'Model 0'
    
    for i in range(nb_states):
        model.weights[0].assign(params)
        model.weights[1].assign(initial_params)
        model.weights[2].assign(initial_fractions)
        model.weights[7].assign(transition_rates)
        model.weights[8].assign(transition_shapes)
        
        model.weights[0][i, 4].assign(1 - model.weights[0][i, 4])
        model.weights[0][i, 2].assign(initial_anomalous_factors[i])
        
        lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, clipvalue=0.01) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
        model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)
        with tf.device(device):
            history = model.fit(seq, epochs = epochs, callbacks=callbacks, shuffle=False, verbose = verbose) #, callbacks  = [l_callback])
        
        params, initial_params, initial_fractions, transition_shapes, transition_rates = get_model_raw_params(model, track_segmentation = True)
        
        LogLikelihood = - history.history['loss'][-1]
        loss_history = history.history['loss']
        model_ID = len(All_models)
        All_models['Model %s'%model_ID] = {'params': params, 'initial_params': initial_params, 'initial_fractions': initial_fractions, 'transition_shapes': transition_shapes, 'transition_rates': transition_rates, 'LogLikelihood': LogLikelihood, 'loss_history': loss_history}
        
        print('Log Likelihood', LogLikelihood)
        print('params', params)
        if LogLikelihood > best_LogLikelihood:
            best_model = 'Model %s'%model_ID
            best_LogLikelihood = LogLikelihood
        
        params, initial_params, initial_fractions, transition_shapes, transition_rates = All_models[best_model]['params'], All_models[best_model]['initial_params'], All_models[best_model]['initial_fractions'], All_models[best_model]['transition_shapes'], All_models[best_model]['transition_rates']
        
    model.weights[0].assign(params)
    model.weights[1].assign(initial_params)
    model.weights[2].assign(initial_fractions)
    model.weights[7].assign(transition_rates)
    model.weights[8].assign(transition_shapes)
    return All_models, model, pred_model



def build_abrupt_directed_motion_changes_model(segment_length, # maximum number of time points in the input tracks 
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions, 
                batch_size, # number of tracks analysed at the same time
                reference_dt,
                nb_dims = 2, # Number of dimensions of the tracks
                sequence_length = 3, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = 3, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = 0.001, # Estimated density of the sample.
                abrupt_change_state = 1,
                vary_params = None,
                vary_initial_params = None,
                vary_initial_fractions = None,
                vary_transition_shapes = None,
                vary_transition_rates = None,
                LocErr_type = 'Linear'):
    
    class Initial_layer_constraints_abrupt_change(Initial_layer_constraints):
        
        def duplicate_states(self, param_vars, initial_param_vars, initial_fractions):
            '''
            initial log factors 
            '''
            param_vars = tf.concat((param_vars[:abrupt_change_state+1], param_vars[abrupt_change_state:]), 0)
            initial_param_vars = tf.concat((initial_param_vars[:abrupt_change_state+1], initial_param_vars[abrupt_change_state:]), 0)
            initial_fractions = tf.concat((initial_fractions[:,:abrupt_change_state+1],[[1e-10]], initial_fractions[:,abrupt_change_state+1:]), 1)
            
            return param_vars, initial_param_vars, initial_fractions
    """
    @tf.function
    def transition_param_function(transition_shapes, transition_rates, density, Fs, effective_ds, dts, reference_dt, dtype):
        '''
        The transition_param_function must define the initial transition parameters and their constraints
        similarly to how constraint_function defines the constraints of the states
        
        dts = dts_TN
        '''
    
        print('transition_shapes', transition_shapes)
        nb_states = transition_shapes.shape[0]
        nb_time_points, nb_tracks = dts.shape
        
        # We need to assign values to the transitions kinetics directed state 1 <=> directed state 2.
        # To do so, we can use the diagonal values of transition_shapes and transition_rates that are unused
        directed_directed_transition_shape = tf.math.exp(transition_shapes[abrupt_change_state, abrupt_change_state])
        directed_directed_transition_rate = tf.math.sigmoid(transition_rates[abrupt_change_state, abrupt_change_state]-2)
                
        transition_shapes = tf.math.exp(transition_shapes)
        transition_shapes = np.arange(1,10).reshape((3,3))
        transition_rates = tf.math.softmax(transition_rates, axis = 1)*transition_shapes/reference_dt
        transition_rates = transition_rates[None,None]*dts[..., None, None]+1e-20
        
        new_transition_shapes = tf.concat((transition_shapes, tf.constant([[1]*nb_states], dtype = dtype)), axis = 0)
        new_transition_shapes = tf.concat((new_transition_shapes, tf.constant([[1]]*(nb_states+1), dtype = dtype)), axis = 1)
        
        mislinking_dwell_time = tf.constant([0.9/nb_states]*nb_states, dtype = dtype) # We multiply by (1-tf.reduce_mean(additional_transition_params[:nb_states]) to allow several consecutive mislinkings proportionally to the misslinking probability
        mislinking_dwell_time = tf.concat((mislinking_dwell_time, [0.1]), axis = 0)
        mislinking_dwell_time = tf.broadcast_to(mislinking_dwell_time[None,None,None], (nb_time_points, nb_tracks, 1, nb_states + 1))
        
        #mislinking_rates = tf.constant([0.078,0.146], dtype = dtype)[:, None] # density 1 -> rates 0.052 0.052 
        mislinking_rates = 1-tf.math.exp(-0.5*density *tf.reduce_sum(Fs[None]*(effective_ds[:,None]**2 + effective_ds[None]**2)**0.5, axis = 0)[:, None])# density 0.1 -> rates 0.052 0.052 
        mislinking_rates = tf.broadcast_to(mislinking_rates[None,None], (nb_time_points, nb_tracks, nb_states, 1))
        
        new_transition_rates = tf.concat((transition_rates, mislinking_rates), axis = 3)
        new_transition_rates = tf.concat((new_transition_rates, mislinking_dwell_time), axis = 2)

        '''
        Once the mislinking state is added we can add the additionnal directed state, constraining
        transitions into a directed state to occur only towards the first directed state duplicated state such that
        transitions from other states can only occur towards the first directed state of state index abrupt_change_state.
        Directed particles can then either transition into the other directed state (duplicate) or into the other
        states. the on rates of the 2nd directed state are 0 except from the 1st directed state and the off rates 
        of the 2 directed states are shared.    
        '''
        
        second_directed_state_on_rates = tf.stack([1e-10]*abrupt_change_state + [directed_directed_transition_rate] + [1e-10]*(nb_states - abrupt_change_state))
        second_directed_state_on_rates = tf.broadcast_to(second_directed_state_on_rates[None,None, :, None], (nb_time_points, nb_tracks, nb_states + 1, 1))
        second_directed_state_on_rates = second_directed_state_on_rates * dts[:,:,None,None] / reference_dt
        new_new_transition_rates = tf.concat((new_transition_rates[:,:,:,:abrupt_change_state+1], second_directed_state_on_rates, new_transition_rates[:,:,:,abrupt_change_state+1:]), 3)
        
        #new_new_transition_rates[:,:,abrupt_change_state, :abrupt_change_state].shape
        br_directed_directed_transition_rate = tf.broadcast_to(directed_directed_transition_rate[None,None,None], (nb_time_points, nb_tracks, 1))
        #new_new_transition_rates[:,:,abrupt_change_state, abrupt_change_state+1:].shape
                
        second_directed_state_off_rates = tf.concat([new_new_transition_rates[:,:,abrupt_change_state, :abrupt_change_state], br_directed_directed_transition_rate, new_new_transition_rates[:,:,abrupt_change_state, abrupt_change_state+1:]], axis =2)
        new_new_transition_rates = tf.concat((new_new_transition_rates[:,:,:abrupt_change_state+1], second_directed_state_off_rates[:,:,None], new_new_transition_rates[:,:,abrupt_change_state+1:]), 2)
        
        second_directed_state_on_shapes = tf.stack([1]*abrupt_change_state + [directed_directed_transition_shape] + [1]*(nb_states - abrupt_change_state))
        #second_directed_state_on_shapes = tf.broadcast_to(second_directed_state_on_shapes[None,None, :, None], (nb_time_points, nb_tracks, nb_states + 1, 1))
            
        
        
        #new_new_transition_shapes = tf.concat((new_transition_shapes[:,:,:,:abrupt_change_state+1], second_directed_state_on_shapes, new_transition_shapes[:,:,:,abrupt_change_state+1:]), 3)
        new_new_transition_shapes = tf.concat((new_transition_shapes[:,:abrupt_change_state+1], second_directed_state_on_shapes[:, None], new_transition_shapes[:,abrupt_change_state+1:]), 1)
        second_directed_state_off_shapes = tf.concat([new_new_transition_shapes[abrupt_change_state, :abrupt_change_state], directed_directed_transition_shape[None], new_new_transition_shapes[abrupt_change_state, abrupt_change_state+1:]], axis = 0)
        new_new_transition_shapes = tf.concat((new_new_transition_shapes[:abrupt_change_state+1], second_directed_state_off_shapes[None], new_new_transition_shapes[abrupt_change_state+1:]), 0)
            
        return new_new_transition_shapes, new_new_transition_rates
    """
    
    #@tf.function
    def transition_param_function(transition_shapes, transition_rates, density, Fs, effective_ds, dts, reference_dt, dtype):
        '''
        The transition_param_function must define the initial transition parameters and their constraints
        similarly to how constraint_function defines the constraints of the states
        
        dts = dts_TN
        '''
        
        print('transition_shapes', transition_shapes)
        nb_states = transition_shapes.shape[0]
        nb_time_points, nb_tracks = dts.shape
        
        # We need to assign values to the transitions kinetics directed state 1 <=> directed state 2.
        # To do so, we can use the diagonal values of transition_shapes and transition_rates that are unused
        directed_directed_transition_shape = tf.math.exp(transition_shapes[abrupt_change_state, abrupt_change_state])
        directed_directed_transition_rate = transition_rates[abrupt_change_state, abrupt_change_state]-4
        
        transition_shapes = tf.math.exp(transition_shapes)
        #transition_shapes = tf.math.exp(transition_shapes) + np.arange(1,10).reshape((3,3))
        #transition_rates = tf.math.softmax(transition_rates, axis = 1)*transition_shapes/reference_dt
        transition_rates = transition_rates #*dts[..., None, None]+1e-20
        
        new_transition_shapes = tf.concat((transition_shapes, tf.constant([[1]*nb_states], dtype = dtype)), axis = 0)
        new_transition_shapes = tf.concat((new_transition_shapes, tf.constant([[1]]*(nb_states+1), dtype = dtype)), axis = 1)
        
        mislinking_dwell_time = tf.constant([0.9/nb_states]*nb_states, dtype = dtype) # We multiply by (1-tf.reduce_mean(additional_transition_params[:nb_states]) to allow several consecutive mislinkings proportionally to the misslinking probability
        mislinking_dwell_time = tf.concat((mislinking_dwell_time, [0.1]), axis = 0)[None]
        #mislinking_dwell_time = tf.broadcast_to(mislinking_dwell_time[None,None,None], (nb_time_points, nb_tracks, 1, nb_states + 1))
        
        #mislinking_rates = tf.constant([0.078,0.146], dtype = dtype)[:, None] # density 1 -> rates 0.052 0.052 
        mislinking_rates = tf.math.log(1-tf.math.exp(-0.5*density *tf.reduce_sum(Fs[None]*(effective_ds[:,None]**2 + effective_ds[None]**2)**0.5, axis = 0)[:, None]))# density 0.1 -> rates 0.052 0.052 
        #mislinking_rates = tf.broadcast_to(mislinking_rates[None,None], (nb_time_points, nb_tracks, nb_states, 1))
        
        new_transition_rates = tf.concat((transition_rates, mislinking_rates), axis = 1)
        new_transition_rates = tf.concat((new_transition_rates, mislinking_dwell_time), axis = 0)

        '''
        Once the mislinking state is added we can add the additionnal directed state, constraining
        transitions into a directed state to occur only towards the first directed state duplicated state such that
        transitions from other states can only occur towards the first directed state of state index abrupt_change_state.
        Directed particles can then either transition into the other directed state (duplicate) or into the other
        states. the on rates of the 2nd directed state are 0 except from the 1st directed state and the off rates 
        of the 2 directed states are shared.    
        '''
        
        second_directed_state_on_rates = tf.stack([-10]*abrupt_change_state + [directed_directed_transition_rate] + [-10]*(nb_states - abrupt_change_state))[:, None]
        #second_directed_state_on_rates = tf.broadcast_to(second_directed_state_on_rates[None,None, :, None], (nb_time_points, nb_tracks, nb_states + 1, 1))
        #second_directed_state_on_rates = second_directed_state_on_rates * dts[:,:,None,None] / reference_dt
        new_new_transition_rates = tf.concat((new_transition_rates[:,:abrupt_change_state+1], second_directed_state_on_rates, new_transition_rates[:,abrupt_change_state+1:]), 1)
        
        #new_new_transition_rates[:,:,abrupt_change_state, :abrupt_change_state].shape
        #br_directed_directed_transition_rate = tf.broadcast_to(directed_directed_transition_rate[None,None,None], (nb_time_points, nb_tracks, 1))
        #new_new_transition_rates[:,:,abrupt_change_state, abrupt_change_state+1:].shape
                
        second_directed_state_off_rates = tf.concat([new_new_transition_rates[abrupt_change_state, :abrupt_change_state], directed_directed_transition_rate[None], new_new_transition_rates[abrupt_change_state, abrupt_change_state:abrupt_change_state+1], new_new_transition_rates[abrupt_change_state, abrupt_change_state+2:]], axis =0)
        new_new_transition_rates = tf.concat((new_new_transition_rates[:abrupt_change_state+1], second_directed_state_off_rates[None], new_new_transition_rates[abrupt_change_state+1:]), 0)
        

        
        second_directed_state_on_shapes = tf.stack([1]*abrupt_change_state + [directed_directed_transition_shape] + [1]*(nb_states - abrupt_change_state))
        #second_directed_state_on_shapes = tf.broadcast_to(second_directed_state_on_shapes[None,None, :, None], (nb_time_points, nb_tracks, nb_states + 1, 1))
        #new_new_transition_shapes = tf.concat((new_transition_shapes[:,:,:,:abrupt_change_state+1], second_directed_state_on_shapes, new_transition_shapes[:,:,:,abrupt_change_state+1:]), 3)
        new_new_transition_shapes = tf.concat((new_transition_shapes[:,:abrupt_change_state+1], second_directed_state_on_shapes[:, None], new_transition_shapes[:,abrupt_change_state+1:]), 1)
        second_directed_state_off_shapes = tf.concat([new_new_transition_shapes[abrupt_change_state, :abrupt_change_state], directed_directed_transition_shape[None], new_new_transition_shapes[abrupt_change_state, abrupt_change_state+1:]], axis = 0)
        new_new_transition_shapes = tf.concat((new_new_transition_shapes[:abrupt_change_state+1], second_directed_state_off_shapes[None], new_new_transition_shapes[abrupt_change_state+1:]), 0)
        
        new_new_transition_rates = tf.math.softmax(new_new_transition_rates, axis = 1)*new_new_transition_shapes/reference_dt
        new_new_transition_rates = tf.broadcast_to(new_new_transition_rates[None,None], (nb_time_points, nb_tracks) + new_new_transition_rates.shape)
        new_new_transition_rates = new_new_transition_rates*dts[..., None, None]

        return new_new_transition_shapes, new_new_transition_rates
    
    # Defining the hyperparameters of the model
    dtype = 'float64'
    
    nb_obs_vars = 1 # number of dependend variables (the x, y, z dimension do not account as dependent variables in our model so keep this to 1)
    nb_independent_vars = nb_dims # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars
    nb_states = nb_states + 1
    
    inputs = tf.keras.Input(batch_shape=(batch_size, segment_length, nb_independent_vars), dtype = dtype)
    input_LocErrs = tf.keras.Input(batch_shape=(batch_size, segment_length, nb_independent_vars), name = 'Localization errors', dtype = dtype)
    input_dts = tf.keras.Input(batch_shape=(batch_size, segment_length+1), name = 'frame durations', dtype = dtype)
    input_mask = tf.keras.Input(batch_shape = (batch_size, segment_length), dtype = dtype)
    input_isfirst = tf.keras.Input(batch_shape = (batch_size,), name = 'isfirsts', dtype = dtype)
    '''
    seq = TrackSegmentSequence(
        track_list,
        batch_size=50,
        segment_length=20,
        min_segment_length=4,
        cutoff_batch_treshhold=0.5)
    
    all_inputs, outputs = seq[1]
    inputs = all_inputs[0]
    input_LocErrs = all_inputs[1]
    input_dts = all_inputs[2]
    input_mask = tf.constant(all_inputs[3], dtype = dtype)
    input_isfirst = tf.constant(all_inputs[4], dtype = dtype)
    
    inputs = tf.constant(inputs, dtype=dtype)
    input_LocErrs = tf.constant(input_LocErrs, dtype=dtype)
    input_dts = tf.constant(input_dts, dtype=dtype)
    input_mask = tf.constant(input_mask, dtype=dtype)
    input_isfirst = tf.constant(input_isfirst, dtype=dtype)
    '''
    #inputs = tracks
    #input_mask = all_masks
    
    reshaped_inputs = tf.keras.layers.Lambda(lambda x: x[:, None, :, None, None, :], dtype = dtype)(inputs)
    transposed_inputs = transpose_layer(dtype = dtype)(reshaped_inputs, perm = [2, 1, 0, 3, 4, 5])
    
    Init_layer = Initial_layer_constraints_abrupt_change(nb_states,
                                           nb_gaussians,
                                           nb_obs_vars,
                                           nb_hidden_vars,
                                           params,
                                           initial_params,
                                           initial_fractions,
                                           max_linking_distance,
                                           constraint_function,
                                           reference_dt,
                                           vary_params = vary_params,
                                           vary_initial_params = vary_initial_params,
                                           vary_initial_fractions = vary_initial_fractions,
                                           sequence_length = sequence_length,
                                           LocErr_type = LocErr_type,
                                           dtype = dtype)
    
    #inputs = transposed_inputs
    #self = Init_layer
    tensor1, initial_states = Init_layer(transposed_inputs, input_LocErrs, input_dts)
    
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
    # self = layer
    Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var, All_motion_states, All_coefs, All_biases, All_LPs, motion_states = layer(sliced_inputs, input_dts, reference_dt, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors, isdir, isfirst = input_isfirst)
    
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
    
    model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
                           outputs=outputs, name="Diffusion_model")
    
    #carried_Prev_coefs, carried_Prev_biases, carried_LP, carried_segment_len, carried_gamma_dist_mean, carried_gamma_dist_var = All_states
    pred_model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
                                outputs=(outputs, All_states, All_coefs, All_biases, All_LPs), name="Diffusion_model")
        
    return model, pred_model



    tensor1, initial_states = Init_layer(transposed_inputs, input_LocErrs, input_dts)
    
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
    

def get_number_of_states(track_list,
                         params,
                         initial_params,
                         transition_shapes,
                         transition_rates,
                         initial_fractions,
                         reference_dt,
                         dt_list = None,
                         LocErr_list = None,
                         nb_dims = 2,
                         sequence_length = 10,
                         max_linking_distance = 0.4,
                         estimated_density = 0.001,
                         epochs = 50,
                         epoch_decay = 40,
                         learning_rate = 0.02,
                         decay_rate = 0.005,
                         batch_size = 100,
                         vary_params = True,
                         vary_initial_params = True,
                         vary_initial_fractions = True,
                         vary_transition_shapes = False,
                         vary_transition_rates = True,
                         device = '/GPU:0',
                         track_segmentation = True,
                         segment_length = 10,
                         LocErr_type = 'Linear'):
    
    if not track_segmentation:
        segment_length = np.max([len(track) for track in track_list])
    
    nb_tracks = len(track_list)
    shuffle = False
    seq = TrackSegmentSequence(track_list,
                               LocErr_list,
                               dt_list,
                               batch_size=batch_size,
                               segment_length=segment_length,
                               min_segment_length=4,
                               cutoff_batch_treshhold=0.5)
    
    nb_states = params.shape[0]
    
    if vary_params is True:
        vary_params = np.ones((nb_states, 5))
    elif vary_params is False:
        vary_params = np.zeros((nb_states, 5))

    if vary_initial_params is True:
        vary_initial_params = np.ones((nb_states, 5))
    elif vary_initial_params is False:
        vary_initial_params = np.zeros((nb_states, 1))
    
    if vary_initial_fractions is True:
        vary_initial_fractions = np.ones((1, nb_states+1))
    elif vary_initial_fractions is False:
        vary_initial_fractions = np.zeros((1, nb_states+1))
     
    if vary_transition_shapes is True:
        vary_transition_shapes = np.ones((nb_states, nb_states))
    elif vary_transition_shapes is False:
        vary_transition_shapes = np.zeros((nb_states, nb_states))
     
    if vary_transition_rates is True:
        vary_transition_rates = np.ones((nb_states, nb_states))
    elif vary_transition_rates is False:
        vary_transition_rates = np.zeros((nb_states, nb_states))
     
    nb_batches = len(seq)
    #tracks = np.concatenate([seq[i][0][0] for i in range(nb_batches)], axis = 0)
    mask_array = np.concatenate([seq[i][0][1] for i in range(nb_batches)], axis = 0)
    nb_data_points = np.sum(mask_array[:, 1:])
    #isfirst =  np.concatenate([seq[i][0][2] for i in range(nb_batches)], axis = 0)
    #inputs = (tracks, track_masks, isfirst)
    decay_step = epoch_decay * nb_batches
    
    #track_len = track_masks.shape[1]
    #nb_tracks = track_masks.shape[0]
    callbacks = [get_parameters(track_segmentation = True)] 
    
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
        model, pred_model = build_segment_model(
            segment_length,
            current_nb_states,
            current_params,
            current_initial_params,
            current_transition_rates,
            current_transition_shapes,
            current_initial_fractions,
            batch_size,
            reference_dt,
            nb_dims=nb_dims,
            sequence_length=sequence_length,
            max_linking_distance=max_linking_distance,
            estimated_density=estimated_density,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            vary_transition_shapes=vary_transition_shapes,
            vary_transition_rates=vary_transition_rates,
            LocErr_type = LocErr_type)
        
        preds = model.predict(seq)
        
        print('initial predictions:', MLE_loss(preds, preds))
        
        # Compile and train
        if nb_states == current_nb_states:
            cur_epochs = 2*epochs
            lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_step*2)
        else:
            lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_step)
            cur_epochs = epochs
        
        beta_1 = max(1 - 5/nb_batches, 0.8)
        beta_2 = 1 - 0.2/nb_batches
        adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, clipvalue=1.0)
        model.compile(loss=MLE_loss, optimizer=adam, jit_compile=False)
        
        with tf.device(device):
            history = model.fit(seq, 
                epochs= cur_epochs,
                callbacks=callbacks, 
                shuffle=shuffle, 
                verbose=1)
        
        # Calculate metrics for model selection
        with tf.device(device):
            final_preds = model.predict(seq) # history.history['loss'][-1]
        log_likelihood = -MLE_loss(final_preds, final_preds)*nb_tracks  # Total log likelihood
        
        # Count parameters (excluding the mislinking state)
        num_params = (current_nb_states * 5 +  # params: LocErr, D, anomalous, q, model_type
                      current_nb_states * 1 +  # initial_params
                      current_nb_states +      # initial_fractions
                      current_nb_states ** 2 * 2 ) # transition rates and shapes
        
        # Calculate information criteria
        aic = 2 * num_params - 2 * log_likelihood
        bic = np.log(nb_data_points) * num_params - 2 * log_likelihood
        
        # Get fitted parameters
        # Get fitted parameters
        fitted_weights = model.get_weights()
        fitted_params = fitted_weights[0].copy()
        fitted_initial_params = fitted_weights[1].copy()
        fitted_initial_fractions = fitted_weights[2].copy()
        fitted_transition_rates = fitted_weights[7].copy()
        fitted_transition_shapes = fitted_weights[8].copy()
      
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
            'raw_parameters': raw_parameters}
       
        del model, pred_model
        tf.keras.backend.clear_session()
        gc.collect()
        
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
                try:
                    del test_model
                except NameError:
                    pass
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Build reduced model
                test_model, _ = build_segment_model(
                    segment_length,
                    current_nb_states - 1,
                    reduced_params,
                    reduced_initial_params,
                    reduced_transition_rates,
                    reduced_transition_shapes,
                    reduced_initial_fractions,
                    batch_size,
                    reference_dt,
                    nb_dims=nb_dims,
                    sequence_length=sequence_length,
                    max_linking_distance=max_linking_distance,
                    estimated_density=estimated_density,
                    LocErr_type = LocErr_type)
                
                with tf.device(device):
                    test_preds = test_model.predict(seq) #-test_history.history['loss'][-1] * track_masks.shape[0]
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
            
            vary_params = vary_params[keep_states]
            vary_initial_params = vary_initial_params[keep_states]
            vary_initial_fractions = vary_initial_fractions[:, keep_states + [current_nb_states]]
            vary_transition_shapes = vary_transition_shapes[np.ix_(keep_states, keep_states)]
            vary_transition_rates = vary_transition_rates[np.ix_(keep_states, keep_states)]
            
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


