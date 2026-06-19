# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:13:30 2026

@author: Franc
"""

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
    
    #LC = tf.constant(0, shape = current_hidden_var_coefs_cp[0].shape[:2], dtype = dtype)
    LC = tf.zeros(tf.shape(current_hidden_var_coefs_cp[0])[:2], dtype = dtype)

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
    
    #LC = tf.constant(0, shape = current_hidden_var_coefs_cp[0].shape[:2], dtype = dtype)
    LC = tf.zeros(tf.shape(current_hidden_var_coefs_cp[0])[:2], dtype = dtype)
    
    for f, s in zip(transition_sequence[0], transition_sequence[1]):
        print('1...')
        coef_index, ID_1, ID_2 = s
        current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp = f(current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp, coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians, kept_next_hidden_var_coefs_cp, kept_biases_cp, nb_dims)
    
    Next_coefs = current_hidden_var_coefs_cp
    Next_biases = biases_cp
    
    return Next_coefs, Next_biases, LC

class constraint_function_arbitrary_KF:
    """
    Generic constraint function for a recurrent conditional-Gaussian process
    with an arbitrary number of hidden (H = nb_hidden_vars) and observed
    (O = nb_obs_vars) variables. This implementation is aimed to be strictly
    equivalent to a state space model with triangular coef matrices and dense Q and R.
    
    It produces, for every (time, track, state) triple, the coefficients,
    biases, STANDARD DEVIATIONS and log-normalisers of the univariate Gaussian
    factors
    
        f_i( sum_j C[i, j] * var_j  +  bias_i )           (std_i learnable)
 
    that, taken together, encode the joint distribution
        p(observation_t, hidden_t, hidden_{t+1} | state_t).
    
    Arbitrary-Kalman-filter parameterisation
    ----------------------------------------
    This version is in bijection with a GENERAL linear-Gaussian state-space
    model (full transition matrix, full process- and measurement-noise
    covariances, full prior), so -- transitions between processes aside -- it
    is equivalent to an arbitrary Kalman filter.  The equivalence is achieved
    with three blocks whose diagonal is fixed and whose scale lives in explicit
    learnable standard deviations:
 
        * Acur     -> DENSE H x H        (general transition, current-var block)
        * dyn_next -> lower-tri, unit(+1) diagonal, learnable strictly-lower
                      -> full process-noise covariance Q
        * meas_obs -> lower-tri, unit(+1) diagonal, learnable strictly-lower
                      -> full measurement-noise covariance R
        * prior L  -> lower-tri, unit(+1) diagonal, learnable strictly-lower
                      -> full prior covariance Sigma_0  (with init_bias -> mu_0)
 
    A block being unit-lower-triangular is exactly an LDL^T factor of a
    precision matrix, which is why the conditional noise covariances become
    full rather than diagonal.
 
    Mapping to Kalman matrices (per state s, derived below)
    ------------------------------------------------------
    Let N = dyn_next_coefs (unit-lower-tri), P = meas_obs (unit-lower-tri),
    L~ = prior L (unit-lower-tri), and sigma_* the corresponding stds.
 
        dynamics  :  N x_{t+1} + Acur x_t + b_dyn  ~ N(0, diag(sigma_dyn^2))
            A_kf = -N^{-1} Acur            (any H x H, since Acur is free dense)
            b_kf = -N^{-1} b_dyn
            Q    =  N^{-1} diag(sigma_dyn^2) N^{-T}       (any PD H x H)
 
        measurement: P y_t + meas_hidden x_t + b_meas ~ N(0, diag(sigma_meas^2))
            M_kf = -P^{-1} meas_hidden     (any O x H, since meas_hidden is free)
            d_kf = -P^{-1} b_meas
            R    =  P^{-1} diag(sigma_meas^2) P^{-T}      (any PD O x O)
 
        prior     :  L~ x_0 + init_bias ~ N(0, diag(init_std^2))
            mu_0    = -L~^{-1} init_bias
            Sigma_0 =  L~^{-1} diag(init_std^2) L~^{-T}   (any PD H x H)
 
    The inverse map (Kalman -> params) uses the unpivoted LDL^T of each
    covariance:  Q = G diag(sigma_dyn^2) G^T with G unit-lower-tri gives
    N = G^{-1}, dyn_next strictly-lower = strict-lower(N), Acur = -N A_kf,
    b_dyn = -N b_kf; analogously for (P, R, meas_hidden) and (L~, Sigma_0).
 
    Parameter packing
    -----------------
    all_initial_params[s] : length  H*(H-1)/2 + 2*H, in order
        [ L_lower   : H*(H-1)/2   strictly-lower prior off-diagonals (diag = 1)
          init_std  : H           log-std of each initial / prior gaussian
          init_bias : H           bias  of each initial / prior gaussian ]
 
    all_params[s] : length  H*H + H*(H-1)/2 + O*H + O*(O-1)/2 + 2*(H+O), order
        [ Acur_dense   : H*H          dense transition current-var coefs
          dyn_next_off : H*(H-1)/2    strictly-lower next-var coefs (diag = 1)
          meas_hidden  : O*H          measurement coefs over current hidden vars
          meas_obs_off : O*(O-1)/2    strictly-lower observed-var coefs (diag=1)
          stds         : H+O          log-std of [dynamics_0..H-1, meas_0..O-1]
          biases       : H+O          bias    of [dynamics_0..H-1, meas_0..O-1] ]
 
    This length equals exactly the degrees of freedom of a general KF:
        A_kf(H*H) + Q(H(H+1)/2) + M_kf(O*H) + R(O(O+1)/2) + b(H) + d(O)
    so the parameterisation is minimal (no redundancy).
 
    Standard deviations everywhere use sigma = exp(param) + eps.
 
    initial_hidden_vars
    <tf.Tensor: shape=(2, 50, 2, 2)
 
    hidden_vars
    <tf.Tensor: shape=(20, 3, 50, 2, 4)
    
    initial_params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + nb_hidden_vars)*4 - 2
    params = np.random.rand(nb_states, nb_hidden_vars**2 + (nb_hidden_vars + 1) * nb_hidden_vars //2 + (nb_obs_vars + 1) * nb_obs_vars //2 +  (nb_hidden_vars + 1) * (nb_obs_vars + 1))*4-2

    """
 
    def __init__(self, nb_hidden_vars=2, nb_obs_vars=1, integration_variable_index=1):
        self.nb_hidden_vars = nb_hidden_vars
        self.nb_obs_vars = nb_obs_vars
        self.integration_variable_index = integration_variable_index
 
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
 
    # ------------------------------------------------------------------
    # static index maps (depend only on H and O, built once at trace time)
    # ------------------------------------------------------------------
    @staticmethod
    def _strictly_lower(n):
        """index/mask for the strictly-lower triangle of an n x n matrix
        (row k touches columns 0..k-1), filled row by row."""
        idx = np.zeros((n, n), dtype=np.int32)
        mask = np.zeros((n, n), dtype=np.float64)
        p = 0
        for k in range(n):
            for m in range(k):
                idx[k, m] = p
                mask[k, m] = 1.0
                p += 1
        return idx, mask, p                      # p = n*(n-1)/2
 
    def _build_index_maps(self):
        H = self.nb_hidden_vars
        O = self.nb_obs_vars
        # strictly-lower map for H x H (used by prior L AND dynamics next-block)
        SLH_idx, SLH_mask, n_sl_H = self._strictly_lower(H)
        # strictly-lower map for O x O (used by measurement obs-block)
        SLO_idx, SLO_mask, n_sl_O = self._strictly_lower(O)
        return SLH_idx, SLH_mask, n_sl_H, SLO_idx, SLO_mask, n_sl_O
    
    #@tf.function(jit_compile=False)
    def call(self, all_params, all_initial_params, LocErrs, dts,
             nb_dims, reference_dt, LocErr_function, dtype):
 
        H = self.nb_hidden_vars
        O = self.nb_obs_vars
        idx = self.integration_variable_index
        n_g = O + H                                        # recurrent gaussians
        n_trans = H - idx                                  # transition gaussians
        eps = tf.constant(1e-20, dtype=dtype)
 
        # unit (+1) pivot on every diagonal -> |pivot| = 1 -> normaliser = -log(std)
        prior_diag = tf.constant(1.0, dtype=dtype)
 
        all_params = tf.cast(all_params, dtype)
        all_initial_params = tf.cast(all_initial_params, dtype)
        S = all_params.shape[0]                            # nb_states (static)
 
        (SL_idx, SL_mask, n_sl_H,
         SLO_idx, SLO_mask, n_sl_O) = self._build_index_maps()
        SL_mask = tf.constant(SL_mask, dtype=dtype)
        SLO_mask = tf.constant(SLO_mask, dtype=dtype)
 
        # --------------------------------------------------------------
        # dynamic track_len / nb_tracks from the (only) shape-carrying input
        # --------------------------------------------------------------
        LocErrs = tf.cast(LocErrs, dtype)
        if LocErrs.shape.rank == 2:
            LocErrs = LocErrs[..., None]
        LocErrs = tf.transpose(LocErrs, [1, 0, 2])         # (track_len, nb_tracks, *)
        track_len = tf.shape(LocErrs)[0]
        nb_tracks = tf.shape(LocErrs)[1]
 
        # --------------------------------------------------------------
        # slice the recurrent parameter vector
        # --------------------------------------------------------------
        o0 = 0
        A_flat       = all_params[:, o0:o0 + H * H];       o0 += H * H
        dyn_next_off = all_params[:, o0:o0 + n_sl_H];      o0 += n_sl_H
        meas_hidden  = all_params[:, o0:o0 + O * H];       o0 += O * H
        meas_obs_off = all_params[:, o0:o0 + n_sl_O];      o0 += n_sl_O
        stds         = all_params[:, o0:o0 + H + O];       o0 += H + O
        biases       = all_params[:, o0:o0 + H + O];       o0 += H + O
        
        #nb_hidden_vars**2 + (nb_hidden_vars + 1) * nb_hidden_vars //2 + (nb_obs_vars + 1) * nb_obs_vars //2 +  (nb_hidden_vars + 1) * (nb_obs_vars + 1) 
        
        Acur = tf.reshape(A_flat, (S, H, H))               # dense transition (current block)
        meas_hidden = tf.reshape(meas_hidden, (S, O, H))
        stds = tf.math.exp(stds) + eps                     # (S, H + O)
 
        # --------------------------------------------------------------
        # slice the initial / prior parameter vector
        # --------------------------------------------------------------
        i0 = 0
        L_lower   = all_initial_params[:, i0:i0 + n_sl_H]; i0 += n_sl_H
        init_std  = all_initial_params[:, i0:i0 + H];      i0 += H
        init_bias = all_initial_params[:, i0:i0 + H];      i0 += H
        init_std = tf.math.exp(init_std) + eps             # (S, H)
 
        # --------------------------------------------------------------
        # helper: unit-lower-triangular matrix from strictly-lower params
        # --------------------------------------------------------------
        def unit_lower(flat, n, idx_map, mask):
            if n * (n - 1) // 2 > 0:
                off = tf.reshape(tf.gather(flat, idx_map.reshape(-1), axis=1),
                                 (S, n, n)) * mask
            else:
                off = tf.zeros((S, n, n), dtype=dtype)
            return off + tf.eye(n, batch_shape=[S], dtype=dtype)
 
        # next-variable block (unit-lower-tri -> full Q); obs block (-> full R);
        # prior L (-> full Sigma_0). prior_diag scales the prior pivot only.
        dyn_next_coefs = unit_lower(dyn_next_off, H, SL_idx, SL_mask)        # (S,H,H)
        meas_obs       = unit_lower(meas_obs_off, O, SLO_idx, SLO_mask)      # (S,O,O)
        L = (unit_lower(L_lower, H, SL_idx, SL_mask)
             - tf.eye(H, batch_shape=[S], dtype=dtype)
             + prior_diag * tf.eye(H, batch_shape=[S], dtype=dtype))        # diag = prior_diag
 
        # --------------------------------------------------------------
        # recurrent hidden-variable coefficients, last axis = 2H
        #   [ current_0..H-1 , next_0..H-1 ]
        # --------------------------------------------------------------
        dyn_rows = tf.concat([Acur, dyn_next_coefs], axis=-1)       # (S, H, 2H)
        meas_next = tf.zeros((S, O, H), dtype=dtype)
        meas_rows = tf.concat([meas_hidden, meas_next], axis=-1)    # (S, O, 2H)
 
        C = tf.concat([dyn_rows, meas_rows], axis=1)        # (S, n_g, 2H) dynamics first
        C = tf.transpose(C, [1, 0, 2])                      # (n_g, S, 2H)
        hidden_vars = tf.broadcast_to(
            C[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, 2 * H]))
 
        # recurrent observation coefficients, last axis = O  (obs block lower-tri)
        obs_dyn = tf.zeros((H, S, O), dtype=dtype)
        obs_meas = tf.transpose(meas_obs, [1, 0, 2])        # (O, S, O)
        OBS = tf.concat([obs_dyn, obs_meas], axis=0)        # (n_g, S, O)
        obs_vars = tf.broadcast_to(
            OBS[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, O]))
 
        # recurrent biases (one per gaussian), broadcast over nb_dims
        rec_bias = tf.broadcast_to(biases[:, :, None], (S, H + O, nb_dims))
        rec_bias = tf.transpose(rec_bias, [1, 0, 2])        # (n_g, S, nb_dims)
        biases = tf.broadcast_to(rec_bias[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, nb_dims]))
 
        # recurrent standard deviations, last axis = 1
        rec_std = tf.transpose(stds, [1, 0])                # (n_g, S)
        Gaussian_stds = tf.broadcast_to(
            rec_std[None, :, None, :, None],
            tf.stack([track_len, n_g, nb_tracks, S, 1]))
 
        # --------------------------------------------------------------
        # initial gaussians (no time axis), last axis = H
        # --------------------------------------------------------------
        IL = tf.transpose(L, [1, 0, 2])                     # (H, S, H)
        initial_hidden_vars = tf.broadcast_to(
            IL[:, None, :, :], tf.stack([H, nb_tracks, S, H]))
        initial_obs_vars = tf.zeros(tf.stack([H, nb_tracks, S, O]), dtype=dtype)
 
        init_std_g = tf.transpose(init_std, [1, 0])         # (H, S)
        initial_Gaussian_stds = tf.broadcast_to(
            init_std_g[:, None, :, None], tf.stack([H, nb_tracks, S, 1]))
 
        init_bias_g = tf.transpose(init_bias, [1, 0])       # (H, S)
        initial_biases = tf.broadcast_to(
            init_bias_g[:, None, :, None], tf.stack([H, nb_tracks, S, nb_dims]))
 
        # --------------------------------------------------------------
        # transition gaussians (fresh prior on vars idx..H-1 at a state change)
        # --------------------------------------------------------------
        TL = tf.transpose(L[:, idx:H, :], [1, 0, 2])        # (n_trans, S, H)
        transition_hidden_vars = tf.broadcast_to(
            TL[None, :, None, :, :], tf.stack([track_len, n_trans, nb_tracks, S, H]))
 
        trans_std = tf.transpose(init_std[:, idx:H], [1, 0])    # (n_trans, S)
        transition_Gaussian_stds = tf.broadcast_to(
            trans_std[None, :, None, :, None],
            tf.stack([track_len, n_trans, nb_tracks, S, 1]))
 
        trans_bias = tf.transpose(init_bias[:, idx:H], [1, 0])  # (n_trans, S)
        transition_biases = tf.broadcast_to(
            trans_bias[None, :, None, :, None],
            tf.stack([track_len, n_trans, nb_tracks, S, nb_dims]))
 
        integration_variable_index = tf.constant(idx)
 
        # --------------------------------------------------------------
        # log-normalising factors  (unit pivots everywhere -> -log(std))
        # --------------------------------------------------------------
        rec_logs = -tf.reduce_sum(tf.math.log(rec_std), axis=0)         # (S,)
        Log_factors = tf.broadcast_to(rec_logs[None, None, :],
                                      tf.stack([track_len, nb_tracks, S]))
 
        log_init = tf.math.log(init_std)                                # (S, H)
        init_norm_all = -tf.reduce_sum(log_init, axis=-1)               # (S,)
        init_norm_reset = -tf.reduce_sum(log_init[:, idx:H], axis=-1)   # (S,)
 
        initial_Log_factors = Log_factors[0] + init_norm_all[None, :]                # (N, S)
        transition_Log_factors = Log_factors + init_norm_reset[None, None, :]        # (T, N, S)
 
        return (hidden_vars, obs_vars, Gaussian_stds, biases,
                initial_hidden_vars, initial_obs_vars,
                initial_Gaussian_stds, initial_biases,
                transition_hidden_vars, transition_Gaussian_stds,
                transition_biases, integration_variable_index,
                Log_factors, initial_Log_factors, transition_Log_factors)

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
    
    constraint_function = current_constraint_function
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

min_lifetime = 3

@tf.function
def transition_param_function(transition_shapes, transition_rates,  dts, reference_dt, dtype):
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
    
    return transition_shapes, transition_rates


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


# -*- coding: utf-8 -*-
"""
LSTM-driven dynamics for the segmented state-space / abrupt-transition model.

This module replaces the analytic `constraint_function` (e.g.
`constraint_function_arbitrary_KF`) by a *deep LSTM* that, at every recurrence
step, ingests the running latent description of each track

    (Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var)

and emits the parameters of the next Gaussians (current/next hidden-variable
coefficients, observation coefficients, biases, standard deviations and
log-normalisers), in *exactly the same packing and structural pattern* that the
constraint function produces.

Why this works without touching the integrator
----------------------------------------------
`RNN_reccurence_formula`, `transition_RNN_reccurence_formula` and the schedules
built by `get_sequences` depend ONLY on the zero / unit-diagonal *pattern* of the
coefficient matrices, never on their numeric values.  We therefore:

  * keep a *template* `constraint_function_arbitrary_KF` instance whose sole job
    is to let `get_sequences` discover the integration schedules
    (it is never trained, never evaluated at run time), and
  * let the LSTM produce the numeric parameters, which we then assemble with the
    very same structural code (dense `Acur`, unit-lower-triangular
    `dyn_next` / `meas_obs` / prior `L`).  Identical pattern  =>  identical, valid
    schedules.

What is replaced vs. kept
-------------------------
Replaced (now LSTM-produced, per (time, track, next-state)):
    reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
    reccurent_next_hidden_var_coefs, reccurent_biases,
    transition_hidden_var_coefs, transition_biases,
    Log_factors, transition_Log_factors,
    the initial (prior) Gaussians at t = 0.

Kept untouched:
    the Gamma dwell-time hazard machinery and `transition_param_function`
    (these were never constraint-function outputs), the analytic integration,
    the carry-over / is_first segmentation logic.

The module assumes the following symbols are importable from your original file
(same names, same behaviour):

    dtype, minval, pi,
    RNN_reccurence_formula, transition_RNN_reccurence_formula,
    get_sequences, constraint_function_arbitrary_KF,
    transition_param_function,
    IsfirstMaskLayer, CarryoverAssignLayer, transpose_layer, Final_layer,
    log_gaussian, norm_log_gaussian

If you keep everything in one file, just paste the classes below under the
originals and call `build_segment_model_LSTM(...)` instead of
`build_segment_model(...)`.
"""

# ============================================================================
#  The deep-LSTM "constraint function"
# ============================================================================
class LSTM_constraint_function(tf.keras.layers.Layer):
    """
    Deep LSTM that produces, per recurrence step, the same structured Gaussian
    parameters as `constraint_function_arbitrary_KF`, but conditioned on the
    running latent state of each track.

    Public attributes mirrored from the constraint function (so downstream code
    and `get_sequences` keep working):
        nb_hidden_vars, nb_obs_vars, integration_variable_index.

    Parameter packing produced by the head (per track, per state)
    -------------------------------------------------------------
    recurrent block  (length L_rec):
        [ Acur        : H*H          dense transition current-var coefs
          dyn_next_off: H*(H-1)/2    strictly-lower next-var coefs (diag = 1)
          meas_hidden : O*H          measurement coefs over current hidden vars
          meas_obs_off: O*(O-1)/2    strictly-lower observed-var coefs (diag=1)
          log_stds    : H+O          log std of [dynamics_0..H-1, meas_0..O-1]
          biases      : H+O ]
    prior block      (length L_init):
        [ L_lower     : H*(H-1)/2    strictly-lower prior off-diagonals (diag=1)
          log_init_std: H
          init_bias   : H ]

    The diagonals of dyn_next / meas_obs / prior L are fixed to +1; all scale
    lives in the (exp) standard deviations, exactly as in the KF version.
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 nb_hidden_vars=2,
                 nb_obs_vars=1,
                 integration_variable_index=1,
                 nb_states=2,                 # physical states
                 nb_dims=2,
                 batch_size=50,
                 sequence_length=3,
                 lstm_units=128,
                 nb_lstm_layers=2,
                 feature_hidden=128,
                 carryover=True,
                 stop_gradient_features=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.nb_hidden_vars = nb_hidden_vars
        self.nb_obs_vars = nb_obs_vars
        self.integration_variable_index = integration_variable_index
        self.nb_states = nb_states
        self.nb_dims = nb_dims
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.nb_lstm_layers = nb_lstm_layers
        self.feature_hidden = feature_hidden
        self.carryover = carryover
        self.stop_gradient_features = stop_gradient_features

        H = nb_hidden_vars
        O = nb_obs_vars
        self.n_sl_H = H * (H - 1) // 2
        self.n_sl_O = O * (O - 1) // 2
        self.L_rec = H * H + self.n_sl_H + O * H + self.n_sl_O + 2 * (H + O)
        self.L_init = self.n_sl_H + 2 * H
        self.n_g = O + H                                  # recurrent gaussians
        self.n_trans = H - integration_variable_index    # transition gaussians

        # strictly-lower index maps (depend only on H, O)
        self.SL_idx, self.SL_mask_np, _ = self._strictly_lower(H)
        self.SLO_idx, self.SLO_mask_np, _ = self._strictly_lower(O)
        self.M = nb_states * sequence_length          # number of state-sequences (hypotheses)
        
    # ------------------------------------------------------------------ #
    @staticmethod
    def _strictly_lower(n):
        idx = np.zeros((n, n), dtype=np.int32)
        mask = np.zeros((n, n), dtype=np.float64)
        p = 0
        for k in range(n):
            for m in range(k):
                idx[k, m] = p
                mask[k, m] = 1.0
                p += 1
        return idx, mask, p
    
    # ------------------------------------------------------------------ #
    def _feature_dim(self):
        H = self.nb_hidden_vars
        S = self.nb_states
        seq = self.sequence_length
        D = self.nb_dims
        pc = H * (seq * S) * H            # Prev_coefs  (H, N, seq*S, H)
        pb = H * (seq * S) * D            # Prev_biases (H, N, seq*S, D)
        lp = seq * S                      # LP
        sl = seq * S                      # segment_len
        gm = seq * S * S                  # gamma_dist_mean
        gv = seq * S * S                  # gamma_dist_var
        return pc + pb + lp + sl + gm + gv
        #return pc + pb + sl + gm + gv
    
    # ------------------------------------------------------------------ #
    def build(self, input_shape):
        d = self.dtype

        self.SL_mask = tf.constant(self.SL_mask_np, dtype=d)
        self.SLO_mask = tf.constant(self.SLO_mask_np, dtype=d)

        # --- sub-layers (built explicitly so no variable is created inside the
        #     while_loop) -------------------------------------------------------
        self.input_proj = tf.keras.layers.Dense(
            self.feature_hidden, activation='tanh', dtype=d, name='lstm_in_proj')
        self.cells = [tf.keras.layers.LSTMCell(self.lstm_units, dtype=d,
                                               name='lstm_cell_%d' % i)
                      for i in range(self.nb_lstm_layers)]
        self.head = tf.keras.layers.Dense(
            self.nb_states * (self.L_rec + self.L_init), dtype=d, name='param_head')

        feat_dim = self._feature_dim()
        self.input_proj.build((None, feat_dim))
        prev = self.feature_hidden
        for cell in self.cells:
            cell.build((None, prev))
            prev = self.lstm_units
        self.head.build((None, self.lstm_units))

        # learned "start token" used as LSTM input at t = 0 (no prev state yet)
        self.init_token = self.add_weight(
            name='init_token', shape=(1, self.feature_hidden),
            initializer='zeros', dtype=d, trainable=True)

        # cross-batch carry-over of the LSTM (h, c) for every layer
        if self.carryover:
            self.carry_h = [
                tf.Variable(np.zeros((self.batch_size, self.lstm_units)),
                            dtype=d, trainable=False, name='carry_h_%d' % i)
                for i in range(self.nb_lstm_layers)]
            self.carry_c = [
                tf.Variable(np.zeros((self.batch_size, self.lstm_units)),
                            dtype=d, trainable=False, name='carry_c_%d' % i)
                for i in range(self.nb_lstm_layers)]

        self.built = True

    # ------------------------------------------------------------------ #
    def zero_state(self):
        d = self.dtype
        return [[tf.zeros((self.batch_size, self.lstm_units), dtype=d),
                 tf.zeros((self.batch_size, self.lstm_units), dtype=d)]
                for _ in range(self.nb_lstm_layers)]

    def carry_state(self):
        return [[self.carry_h[i], self.carry_c[i]]
                for i in range(self.nb_lstm_layers)]
    
    # ------------------------------------------------------------------ #
    def featurize(self, Prev_coefs, Prev_biases, LP,
                  segment_len, gamma_dist_mean, gamma_dist_var):
        """Flatten the per-track latent description into an LSTM input vector."""
        N = self.batch_size
        if self.stop_gradient_features:
            sg = tf.stop_gradient
        else:
            sg = tf.identity
        pc = tf.reshape(tf.transpose(sg(Prev_coefs),  [1, 0, 2, 3]), (N, -1))
        pb = tf.reshape(tf.transpose(sg(Prev_biases), [1, 0, 2, 3]), (N, -1))
        lp = LP - tf.reduce_max(LP, axis=1, keepdims=True)          # (N, seq*S)
        feat = tf.concat([pc, pb, lp, segment_len,
                          gamma_dist_mean, gamma_dist_var], axis=1)
        return feat
    '''
    # ------------------------------------------------------------------ #
    def featurize(self, Prev_coefs, Prev_biases,
                  segment_len, gamma_dist_mean, gamma_dist_var):
        """Flatten the per-track latent description into an LSTM input vector."""
        N = self.batch_size
        if self.stop_gradient_features:
            sg = tf.stop_gradient
        else:
            sg = tf.identity
        pc = tf.reshape(tf.transpose(sg(Prev_coefs),  [1, 0, 2, 3]), (N, -1))
        pb = tf.reshape(tf.transpose(sg(Prev_biases), [1, 0, 2, 3]), (N, -1))
        feat = tf.concat([pc, pb, segment_len,
                          gamma_dist_mean, gamma_dist_var], axis=1)
        return feat
    '''
    # ------------------------------------------------------------------ #
    def _run_cells(self, x, states):
        new_states = []
        for i, cell in enumerate(self.cells):
            x, [h, c] = cell(x, states[i])
            new_states.append([h, c])
        return x, new_states

    def step(self, features, states):
        x = self.input_proj(features)
        out, new_states = self._run_cells(x, states)
        raw = self.head(out)
        raw = tf.reshape(raw, (self.batch_size, self.nb_states,
                               self.L_rec + self.L_init))
        return raw, new_states

    def step_initial(self, states):
        x = tf.broadcast_to(self.init_token, (self.batch_size, self.feature_hidden))
        out, new_states = self._run_cells(x, states)
        raw = self.head(out)
        raw = tf.reshape(raw, (self.batch_size, self.nb_states,
                               self.L_rec + self.L_init))
        return raw, new_states

    # ------------------------------------------------------------------ #
    def _unit_lower(self, flat, n, idx_map, mask):
        """(N, S, n*(n-1)/2) strictly-lower params  ->  (N, S, n, n) unit-lower."""
        d = self.dtype
        N = self.batch_size
        S = self.nb_states
        if n * (n - 1) // 2 > 0:
            off = tf.reshape(tf.gather(flat, idx_map.reshape(-1), axis=-1),
                             (N, S, n, n)) * mask
        else:
            off = tf.zeros((N, S, n, n), dtype=d)
        eye = tf.eye(n, batch_shape=[N, S], dtype=d)
        return off + eye

    # ------------------------------------------------------------------ #
    def assemble(self, raw, nb_dims):
        """
        Turn the raw (N, S, L_rec+L_init) LSTM output into the structured,
        std-normalised, *one time step* Gaussian tensors expected downstream.

        Returns a dict with recurrent / initial / transition coefficients,
        biases and (nb_dims-scaled) log-normalisers.  Shapes:
            rec_hidden_cur   (n_g, N, S, H)
            rec_hidden_next  (n_g, N, S, H)
            rec_obs          (n_g, N, S, O)
            rec_biases       (n_g, N, S, nb_dims)
            Log_factors      (N, S)
            init_hidden      (H,  N, S, H)
            init_obs         (H,  N, S, O)
            init_biases      (H,  N, S, nb_dims)
            initial_Log_factors (N, S)
            trans_hidden     (n_trans, N, S, H)
            trans_biases     (n_trans, N, S, nb_dims)
            transition_Log_factors (N, S)
            
        self = lstm
        raw = raw0
        """
        d = self.dtype
        N = self.batch_size
        S = self.nb_states
        H = self.nb_hidden_vars
        O = self.nb_obs_vars
        idx = self.integration_variable_index
        n_sl_H = self.n_sl_H
        n_sl_O = self.n_sl_O
        n_g = self.n_g

        rec = raw[..., :self.L_rec]
        ini = raw[..., self.L_rec:]
        
        # ---- slice recurrent block ----
        o = 0
        A_flat       = rec[..., o:o + H * H];     o += H * H
        dyn_next_off = rec[..., o:o + n_sl_H];    o += n_sl_H
        meas_hidden  = rec[..., o:o + O * H];     o += O * H
        meas_obs_off = rec[..., o:o + n_sl_O];    o += n_sl_O
        log_stds     = rec[..., o:o + H + O];     o += H + O
        biases_p     = rec[..., o:o + H + O];     o += H + O
        
        #c = 5
        #A_flat = c * tf.tanh(A_flat)
        #dyn_next_off = c * tf.tanh(dyn_next_off)
        #meas_obs_off = c * tf.tanh(meas_obs_off)
        
        Acur = tf.reshape(A_flat, (N, S, H, H))
        meas_hidden = tf.reshape(meas_hidden, (N, S, O, H))
        #stds = tf.math.exp(log_stds) + eps                  # (N, S, H+O)
        LOG_STD_MIN = tf.math.log(tf.constant(1e-10, dtype=d))   # was implicitly 1e-20 via eps
        LOG_STD_MAX = tf.math.log(tf.constant(1e10,  dtype=d))
        log_stds = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * tf.math.sigmoid(log_stds)
        stds = tf.math.exp(log_stds)

        # ---- slice prior block ----
        i0 = 0
        L_lower   = ini[..., i0:i0 + n_sl_H]; i0 += n_sl_H
        log_init  = ini[..., i0:i0 + H];      i0 += H
        init_bias = ini[..., i0:i0 + H];      i0 += H
        #init_std = tf.math.exp(log_init) + eps              # (N, S, H)
        log_init = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * tf.math.sigmoid(log_init)
        init_std = tf.math.exp(log_init)
        
        #tf.print('max std', tf.reduce_max(stds), tf.reduce_max(init_std))

        # ---- structural matrices (unit diagonals) ----
        dyn_next_coefs = self._unit_lower(dyn_next_off, H, self.SL_idx, self.SL_mask)
        meas_obs       = self._unit_lower(meas_obs_off, O, self.SLO_idx, self.SLO_mask)
        L              = self._unit_lower(L_lower, H, self.SL_idx, self.SL_mask)

        # ---- recurrent hidden coefs  [current(H) | next(H)] ----
        dyn_rows = tf.concat([Acur, dyn_next_coefs], axis=-1)      # (N,S,H,2H)
        meas_next = tf.zeros((N, S, O, H), dtype=d)
        meas_rows = tf.concat([meas_hidden, meas_next], axis=-1)   # (N,S,O,2H)
        C = tf.concat([dyn_rows, meas_rows], axis=2)              # (N,S,n_g,2H)
        hidden = tf.transpose(C, [2, 0, 1, 3])                    # (n_g,N,S,2H)

        obs_dyn = tf.zeros((H, N, S, O), dtype=d)
        obs_meas = tf.transpose(meas_obs, [2, 0, 1, 3])          # (O,N,S,O)
        obs = tf.concat([obs_dyn, obs_meas], axis=0)             # (n_g,N,S,O)

        gbias = tf.transpose(biases_p, [2, 0, 1])                # (n_g,N,S)
        rbias = tf.broadcast_to(gbias[..., None], (n_g, N, S, nb_dims))
        gstd = tf.transpose(stds, [2, 0, 1])                     # (n_g,N,S)
        gstd = gstd[..., None]                             # (n_g,N,S,1) ->broadcast
        
        #tf.print('hidden', hidden[:, 0])
        #tf.print( 'obs', obs[:, 0])
        #tf.print( 'gstd', gstd[:, 0])
        
        # std-normalise so every gaussian has unit variance
        hidden = hidden / gstd
        obs = obs / gstd
        rbias = rbias / gstd

        rec_hidden_cur = hidden[..., :H]
        rec_hidden_next = hidden[..., H:]

        Log_factors = -tf.reduce_sum(tf.math.log(stds), axis=-1)  # (N,S)

        # ---- initial / prior gaussians ----
        init_hidden = tf.transpose(L, [2, 0, 1, 3])              # (H,N,S,H)
        init_obs = tf.zeros((H, N, S, O), dtype=d)
        gistd = tf.transpose(init_std, [2, 0, 1])[..., None]   # (H,N,S,1)
        gibias = tf.transpose(init_bias, [2, 0, 1])
        init_biases = tf.broadcast_to(gibias[..., None], (H, N, S, nb_dims))
        gistd.shape
        init_hidden = init_hidden / gistd
        init_obs = init_obs / gistd
        init_biases = init_biases / gistd
        initial_Log_factors = -tf.reduce_sum(tf.math.log(init_std), axis=-1)  # (N,S)
        
        # ---- transition gaussians (fresh prior on vars idx..H-1) ----
        TL = L[:, :, idx:H, :]                                   # (N,S,n_trans,H)
        trans_hidden = tf.transpose(TL, [2, 0, 1, 3])            # (n_trans,N,S,H)
        t_std = init_std[:, :, idx:H]
        gtstd = tf.transpose(t_std, [2, 0, 1])[..., None]  # (n_trans,N,S,1)
        t_bias = init_bias[:, :, idx:H]
        gtbias = tf.transpose(t_bias, [2, 0, 1])
        trans_biases = tf.broadcast_to(gtbias[..., None],
                                       (self.n_trans, N, S, nb_dims))
        trans_hidden = trans_hidden / gtstd
        trans_biases = trans_biases / gtstd
        transition_Log_factors = -tf.reduce_sum(
            tf.math.log(init_std[:, :, idx:H]), axis=-1)         # (N,S)

        return {
            'rec_hidden_cur': rec_hidden_cur,
            'rec_hidden_next': rec_hidden_next,
            'rec_obs': obs,
            'rec_biases': rbias,
            'Log_factors': nb_dims * Log_factors,
            'init_hidden': init_hidden,
            'init_obs': init_obs,
            'init_biases': init_biases,
            'initial_Log_factors': nb_dims * (Log_factors + initial_Log_factors),
            'trans_hidden': trans_hidden,
            'trans_biases': trans_biases,
            'transition_Log_factors': nb_dims * (Log_factors + transition_Log_factors),
        }
    
# ============================================================================
#  LSTM-driven single recurrence step (front-end swapped, integrator unchanged)
# ============================================================================
@tf.function(jit_compile=False)
def RNN_cell_LSTM(input_i, Prev_coefs, Prev_biases, LP, segment_len,
                  lstm_module, lstm_states,
                  oh_row, oh_col, transition_mask,
                  sequence_phase_1, sequence_phase_2, transition_sequence,
                  transition_mean, transition_var,
                  gamma_dist_mean, gamma_dist_var, states, dt_ratios):
    """
    Identical bookkeeping to the original `RNN_cell`, except the per-step
    Gaussian factors are produced by `lstm_module` from the current latent state
    instead of being passed in pre-computed.  Returns the updated carriers plus
    the new LSTM (h, c) state list.
    """
    nb_dims = input_i.shape[-1]
    nb_tracks = LP.shape[0]
    nb_hidden_vars = Prev_coefs.shape[3]
    nb_states = lstm_module.nb_states
    sequence_length = LP.shape[1] // nb_states

    # -------------------- LSTM forward (replaces constraint function) --------
    features = lstm_module.featurize(Prev_coefs, Prev_biases, LP,
                                     segment_len, gamma_dist_mean, gamma_dist_var)
    #features = lstm_module.featurize(Prev_coefs, Prev_biases,
    #                                 segment_len, gamma_dist_mean, gamma_dist_var)
    raw, new_lstm_states = lstm_module.step(features, lstm_states)
    g = lstm_module.assemble(raw, nb_dims)

    reccurent_obs_var_coefs         = g['rec_obs']           # (n_g,N,S,O)
    reccurent_hidden_var_coefs      = g['rec_hidden_cur']    # (n_g,N,S,H)
    reccurent_next_hidden_var_coefs = g['rec_hidden_next']   # (n_g,N,S,H)
    reccurent_biases                = g['rec_biases']        # (n_g,N,S,nb_dims)
    transition_hidden_var_coefs     = g['trans_hidden']      # (n_trans,N,S,H)
    transition_biases               = g['trans_biases']      # (n_trans,N,S,nb_dims)
    Log_factors_NS                  = g['Log_factors']       # (N,S)
    transition_Log_factors_NS       = g['transition_Log_factors']  # (N,S)

    # tile transition coefs across the candidate axis (seq*S), next-state fastest
    transition_hidden_var_coefs = tf.concat(
        [transition_hidden_var_coefs] * (sequence_length * nb_states), axis=2)
    transition_biases = tf.concat(
        [transition_biases] * (sequence_length * nb_states), axis=2)

    # per-candidate log-normaliser (current state if no transition, next if transition)
    flat_Log = tf.einsum('ns,ps->np', Log_factors_NS, oh_row)
    flat_trans = tf.einsum('ns,ps->np', transition_Log_factors_NS, oh_col)
    reshaped_Log_factors = flat_trans * transition_mask + flat_Log * (1 - transition_mask)

    # ============================== original cell body ======================
    current_states = states[:, :, -1:]

    Prev_coefs2 = tf.repeat(Prev_coefs, nb_states, axis=2)
    Prev_biases2 = tf.repeat(Prev_biases, nb_states, axis=2)
    LP2 = tf.repeat(LP, nb_states, axis=1)
    segment_len = tf.repeat(segment_len, nb_states, axis=1)

    alternative_Prev_coefs = tf.concat(
        (Prev_coefs2, tf.identity(transition_hidden_var_coefs)), axis=0)
    alternative_Prev_biases = tf.concat(
        (Prev_biases2, tf.identity(transition_biases)), axis=0)

    transition_Prev_coefs, transition_Prev_biases, LC = transition_RNN_reccurence_formula(
        current_hidden_var_coefs=alternative_Prev_coefs,
        next_hidden_var_coefs=tf.constant(0, dtype=dtype,
                                          shape=alternative_Prev_coefs.shape),
        biases=alternative_Prev_biases,
        transition_sequence=transition_sequence,
        nb_dims=nb_dims,
        dtype=dtype)

    LP2 += LC * transition_mask + reshaped_Log_factors

    current_shapes = gamma_dist_mean ** 2 / gamma_dist_var
    current_rates = gamma_dist_mean / gamma_dist_var

    all_Prev_coefs = (transition_Prev_coefs * transition_mask[None, :, :, None]
                      + Prev_coefs2 * (1 - transition_mask[None, :, :, None]))
    all_prev_biases = (transition_Prev_biases * transition_mask[None, :, :, None]
                       + Prev_biases2 * (1 - transition_mask[None, :, :, None]))

    gamma = tf.compat.v1.distributions.Gamma(current_shapes, current_rates)
    S_old = 1 - gamma.cdf(segment_len + 1e-12) + 1e-12
    S_new = 1 - gamma.cdf(segment_len + 1e-12 + dt_ratios[:, None]) + 1e-12
    transition_probas = tf.clip_by_value(1 - S_new / S_old,
                                         clip_value_min=1e-20, clip_value_max=1 - 1e-10)

    non_transition_probas = tf.repeat(
        1 - tf.clip_by_value(
            tf.reduce_sum(tf.reshape(transition_probas * transition_mask,
                                     shape=(nb_tracks, nb_states * sequence_length, nb_states)),
                          axis=2),
            clip_value_min=1 - 20, clip_value_max=1 - 1e-10),
        nb_states, axis=1)

    transition_probas = (transition_probas * transition_mask
                         + non_transition_probas * (1 - transition_mask))
    all_LP = LP2 + tf.math.log(transition_probas)

    current_reccurent_obs_var_coefs = tf.concat(
        [reccurent_obs_var_coefs] * (sequence_length * nb_states), axis=2)
    current_reccurent_hidden_var_coefs = tf.concat(
        [reccurent_hidden_var_coefs] * (sequence_length * nb_states), axis=2)
    current_reccurent_next_hidden_var_coefs = tf.concat(
        [reccurent_next_hidden_var_coefs] * (sequence_length * nb_states), axis=2)
    current_reccurent_biases = tf.concat(
        [reccurent_biases] * (sequence_length * nb_states), axis=2)

    current_hidden_var_coefs = tf.concat(
        (all_Prev_coefs, tf.identity(current_reccurent_hidden_var_coefs)), axis=0)
    zero_tensor = tf.constant(0, dtype=dtype, shape=all_Prev_coefs.shape)
    next_hidden_var_coefs = tf.concat(
        (zero_tensor, tf.identity(current_reccurent_next_hidden_var_coefs)), axis=0)
    current_biases = tf.identity(current_reccurent_biases)
    current_biases += tf.reduce_sum(
        current_reccurent_obs_var_coefs[:, :, :, :, None] * input_i, (-2))
    biases = tf.concat((all_prev_biases, current_biases), axis=0)

    Next_coefs, Next_biases, LC = RNN_reccurence_formula(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        sequence_phase_1, sequence_phase_2, nb_dims=nb_dims, dtype=dtype)
    
    #tf.print('min|pivot|',
    #         tf.reduce_min(tf.math.abs(tf.linalg.diag_part(
    #             tf.transpose(Next_coefs, [1, 2, 0, 3])))),
    #         'max|LP|', tf.reduce_max(tf.math.abs(all_LP)))

    all_LP += LC

    reshaped_Next_coefs = tf.reshape(
        Next_coefs, Next_coefs.shape[:2] + [sequence_length * nb_states, nb_states, nb_hidden_vars])
    transition_LPs = tf.reshape(all_LP - 200 * (1 - transition_mask),
                                (nb_tracks, sequence_length * nb_states, nb_states)) \
        - nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(reshaped_Next_coefs, [1, 2, 3, 0, 4])),
            axis=-1)) + 1e-20)

    max_transition_LPs = tf.reduce_max(transition_LPs, axis=1, keepdims=True)
    transition_Ps = tf.math.exp(transition_LPs - max_transition_LPs)
    transition_weights = transition_Ps / tf.reduce_sum(transition_Ps, 1, keepdims=True)

    transition_states = tf.reduce_sum(
        states[:, :, None] * transition_weights[:, :, :, None, None], 1)

    transition_Next_coefs = tf.reshape(
        Next_coefs, Next_coefs.shape[:2] + [sequence_length * nb_states, nb_states, nb_hidden_vars])
    transition_Next_coefs = tf.reduce_sum(
        transition_Next_coefs * transition_weights[None, :, :, :, None], axis=2)

    transition_Next_biases = tf.reshape(
        Next_biases, Next_biases.shape[:2] + [sequence_length * nb_states, nb_states, nb_dims])
    transition_Next_biases = tf.reduce_sum(
        transition_Next_biases * transition_weights[None, :, :, :, None], axis=2)

    transition_LPs = tf.math.log(tf.reduce_sum(transition_Ps, axis=1)) \
        + max_transition_LPs[:, 0] \
        + nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(transition_Next_coefs, [1, 2, 0, 3])),
            axis=-1)) + 1e-20)

    stable_LPs = tf.reshape(all_LP, (nb_tracks, sequence_length * nb_states, nb_states))
    stable_weights = tf.reshape((1 - transition_mask),
                                (sequence_length * nb_states, nb_states))[None]
    stable_LPs = tf.reduce_sum(stable_LPs * stable_weights, 2)
    stable_states = tf.reduce_sum(states[:, :, None] * stable_weights[:, :, :, None, None], 2)

    stable_Next_coefs = tf.reduce_sum(
        tf.reshape(Next_coefs, Next_coefs.shape[:2] +
                   [sequence_length * nb_states, nb_states, nb_hidden_vars])
        * stable_weights[None, :, :, :, None], axis=3)
    stable_Next_biases = tf.reduce_sum(
        tf.reshape(Next_biases, Next_biases.shape[:2] +
                   [sequence_length * nb_states, nb_states, nb_dims])
        * stable_weights[None, :, :, :, None], axis=3)
    stable_segment_len = tf.reduce_sum(
        tf.reshape(segment_len, (nb_tracks, sequence_length * nb_states, nb_states))
        * stable_weights, axis=2)

    current_gamma_dist_mean = tf.concat([transition_mean, gamma_dist_mean], axis=1)
    current_gamma_dist_var = tf.concat([transition_var, gamma_dist_var], axis=1)

    Next_coefs = tf.concat([transition_Next_coefs, stable_Next_coefs], axis=2)
    Next_biases = tf.concat([transition_Next_biases, stable_Next_biases], axis=2)
    new_LP = tf.concat([transition_LPs, stable_LPs], axis=1)
    additional_stable_segment_len = dt_ratios[:, None]
    current_segment_len = tf.concat(
        [tf.zeros((nb_tracks, nb_states), dtype=dtype), stable_segment_len], axis=1)
    current_segment_len = current_segment_len + additional_stable_segment_len
    Next_states = tf.concat([transition_states, stable_states], axis=1)

    # ---- fuse the oldest slab back so the buffer stays sequence_length long --
    saved_Next_coefs = Next_coefs[:, :, :-nb_states * 2]
    saved_Next_biases = Next_biases[:, :, :-nb_states * 2]
    saved_LP = new_LP[:, :-nb_states * 2]
    saved_segment_len = current_segment_len[:, :-nb_states * 2]
    saved_gamma_dist_mean = current_gamma_dist_mean[:, :-nb_states ** 2 * 2]
    saved_gamma_dist_var = current_gamma_dist_var[:, :-nb_states ** 2 * 2]
    saved_states = Next_states[:, :-nb_states * 2]

    nb_prev_gaussians = Next_coefs.shape[0]
    last_Next_coefs = tf.reshape(Next_coefs[:, :, -nb_states * 2:],
                                 (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_hidden_vars))
    last_Next_biases = tf.reshape(Next_biases[:, :, -nb_states * 2:],
                                  (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_dims))
    last_LP = tf.reshape(new_LP[:, -nb_states * 2:], (nb_tracks, 2, nb_states)) \
        - nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(last_Next_coefs, [1, 2, 3, 0, 4])),
            axis=-1)) + 1e-20)
    last_segment_len = tf.reshape(current_segment_len[:, -nb_states * 2:],
                                  (nb_tracks, 2, nb_states))
    last_gamma_dist_mean = tf.reshape(current_gamma_dist_mean[:, -nb_states ** 2 * 2:],
                                      (nb_tracks, 2, nb_states, nb_states))
    last_gamma_dist_var = tf.reshape(current_gamma_dist_var[:, -nb_states ** 2 * 2:],
                                     (nb_tracks, 2, nb_states, nb_states))
    last_states = tf.reshape(Next_states[:, -nb_states * 2:],
                             (nb_tracks, 2, nb_states, sequence_length, nb_states))

    last_LP_max = tf.reduce_max(last_LP, axis=1, keepdims=True)
    last_P = tf.math.exp(last_LP - last_LP_max)
    sum_last_P = tf.reduce_sum(last_P, 1, keepdims=True)

    weight_last_P = tf.math.exp(last_LP - tf.reduce_max(last_LP, axis=1, keepdims=True))
    last_weights = weight_last_P / tf.reduce_sum(weight_last_P, 1, keepdims=True)

    reduced_last_Next_coefs = tf.reduce_sum(
        last_Next_coefs * last_weights[None, :, :, :, None], axis=2)
    reduced_last_Next_biases = tf.reduce_sum(
        last_Next_biases * last_weights[None, :, :, :, None], axis=2)
    reduced_last_LPs = (tf.math.log(sum_last_P + 1e-100) + last_LP_max)[:, 0] \
        + nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(reduced_last_Next_coefs, [1, 2, 0, 3])),
            axis=-1)) + 1e-20)
    reduced_last_segment_len = tf.reduce_sum(last_segment_len * last_weights, axis=1)
    reduced_last_gamma_dist_mean = tf.reduce_sum(
        last_gamma_dist_mean * last_weights[:, :, :, None], axis=1)
    reduced_last_gamma_dist_var = tf.reduce_sum(
        (last_gamma_dist_var + (last_gamma_dist_mean - reduced_last_gamma_dist_mean[:, None]) ** 2)
        * last_weights[:, :, :, None], axis=1)
    reduced_last_gamma_dist_mean = tf.reshape(reduced_last_gamma_dist_mean,
                                              (nb_tracks, nb_states ** 2))
    reduced_last_gamma_dist_var = tf.reshape(reduced_last_gamma_dist_var,
                                             (nb_tracks, nb_states ** 2))
    reduced_last_states = tf.reduce_sum(last_states * last_weights[:, :, :, None, None], axis=1)

    new_Next_coefs = tf.concat((saved_Next_coefs, reduced_last_Next_coefs), axis=2)
    new_Next_biases = tf.concat((saved_Next_biases, reduced_last_Next_biases), axis=2)
    new_LPs = tf.concat((saved_LP, reduced_last_LPs), axis=1)
    new_segment_len = tf.concat((saved_segment_len, reduced_last_segment_len), axis=1)
    new_gamma_dist_mean = tf.concat((saved_gamma_dist_mean, reduced_last_gamma_dist_mean), axis=1)
    new_gamma_dist_var = tf.concat((saved_gamma_dist_var, reduced_last_gamma_dist_var), axis=1)
    new_states = tf.concat((saved_states, reduced_last_states), axis=1)
    new_states = tf.concat((new_states, current_states), axis=2)[:, :, 1:]

    return (new_Next_coefs, new_Next_biases, new_LPs, new_segment_len,
            new_gamma_dist_mean, new_gamma_dist_var, new_states, new_lstm_states)


# ============================================================================
#  Initial layer (t = 0)  — LSTM driven
# ============================================================================
class Initial_LSTM_layer(tf.keras.layers.Layer):
    """
    First time step.  Reads the LSTM carry-over state (masked by is_first),
    runs LSTM step 0, builds the prior + first-observation joint, performs one
    `RNN_reccurence_formula` pass and returns the loop-carried tuple plus the
    LSTM state to be threaded into `Custom_RNN_LSTM_layer`.
    """

    def __init__(self, lstm_module, initial_fractions,
                 initial_sequence_phase_1, initial_sequence_phase_2,
                 nb_states, sequence_length, reference_dt,
                 vary_initial_fractions=None, carryover=True, **kwargs):
        super().__init__(**kwargs)
        self.lstm_module = lstm_module
        self.initial_fractions = initial_fractions
        self.initial_sequence_phase_1 = initial_sequence_phase_1
        self.initial_sequence_phase_2 = initial_sequence_phase_2
        self.nb_states = nb_states
        self.sequence_length = sequence_length
        self.reference_dt = reference_dt
        self.vary_initial_fractions = vary_initial_fractions
        self.carryover = carryover

    def build(self, input_shape):
        d = self.dtype
        if self.vary_initial_fractions is None:
            self.vary_initial_fractions = np.ones(self.initial_fractions.shape, dtype=d)
        self.initial_fractions = tf.Variable(self.initial_fractions, dtype=d,
                                             name='Fractions', trainable=True)
        nb_sequences = self.sequence_length * self.nb_states
        if self.carryover:
            self.carryout_coefs = tf.Variable(
                np.zeros((self.lstm_module.nb_hidden_vars, input_shape[2],
                          nb_sequences, self.lstm_module.nb_hidden_vars)),
                dtype=d, trainable=False, name='carryout_coefs')
            self.carryout_biases = tf.Variable(
                np.zeros((self.lstm_module.nb_hidden_vars, input_shape[2],
                          nb_sequences, input_shape[5])),
                dtype=d, trainable=False, name='carryout_biases')
            self.carryout_LP = tf.Variable(
                np.zeros((input_shape[2], nb_sequences)),
                dtype=d, trainable=False, name='carryout_LP')
        self.built = True

    def call(self, inputs, input_dts, input_isfirst):
        '''
        inputs, input_dts, input_isfirst = (transposed_inputs, input_dts, input_isfirst)
        self = Init_layer
        '''
        
        
        d = self.dtype
        lstm = self.lstm_module
        nb_tracks = inputs.shape[2]
        nb_dims = inputs.shape[-1]
        H = lstm.nb_hidden_vars
        S = self.nb_states
        seq = self.sequence_length

        initial_fractions = tf.math.softmax(self.initial_fractions) + 1e-8
        vif = self.vary_initial_fractions
        initial_fractions = vif * initial_fractions + (1 - vif) * tf.stop_gradient(initial_fractions)

        # ---- LSTM initial state: resume from carry if not is_first ----------
        if lstm.carryover:
            base = lstm.carry_state()
        else:
            base = lstm.zero_state()
        zero = lstm.zero_state()
        isf = input_isfirst[:, None]
        lstm_states0 = []
        for i in range(lstm.nb_lstm_layers):
            h = isf * zero[i][0] + (1 - isf) * base[i][0]
            c = isf * zero[i][1] + (1 - isf) * base[i][1]
            lstm_states0.append([h, c])

        raw0, lstm_states1 = lstm.step_initial(lstm_states0)
        g = lstm.assemble(raw0, nb_dims)

        # recurrent step-0 gaussians
        rec_obs = g['rec_obs']
        rec_cur = g['rec_hidden_cur']
        rec_next = g['rec_hidden_next']
        rec_bias = g['rec_biases']
        # initial (prior) gaussians
        init_hidden = g['init_hidden']
        init_obs = g['init_obs']
        init_bias = g['init_biases']
        initial_Log_factors = g['initial_Log_factors']        # (N,S)
        
        # fold first observation into both blocks
        biases = rec_bias + tf.reduce_sum(rec_obs[..., None] * inputs[0], -2)
        init_bias = init_bias + tf.reduce_sum(init_obs[..., None] * inputs[0], -2)

        current_initial = init_hidden                          # (H,N,S,H)
        next_initial = tf.zeros((H, nb_tracks, S, H), dtype=d)

        current_hidden = tf.concat((current_initial, rec_cur), axis=0)
        next_hidden = tf.concat((next_initial, rec_next), axis=0)
        biases = tf.concat((init_bias, biases), axis=0)

        # tile sequence_length copies (no transition branching yet at t=0)
        current_hidden = tf.concat([current_hidden] * seq, axis=2)
        next_hidden = tf.concat([next_hidden] * seq, axis=2)
        biases = tf.concat([biases] * seq, axis=2)
        
        Next_coefs, Next_biases, LC = RNN_reccurence_formula(
            current_hidden, next_hidden, biases,
            self.initial_sequence_phase_1, self.initial_sequence_phase_2,
            nb_dims, dtype=d)

        init_log_fractions = tf.concat([tf.math.log(initial_fractions)] * seq, axis=1)
        init_log_factors = tf.concat([initial_Log_factors] * seq, axis=1)
        LP = LC + init_log_factors + init_log_fractions + tf.math.log(np.array(1 / seq))

        return inputs, [Next_coefs, Next_biases, LP], lstm_states1


# ============================================================================
#  Recurrent layer  — LSTM driven
# ============================================================================
class Custom_RNN_LSTM_layer(tf.keras.layers.Layer):
    """
    Drives `RNN_cell_LSTM` across the track in a tf.while_loop.  The Gamma
    dwell-time machinery (transition_param_function, hazard weighting) is the
    same as in the original layer; only the source of the Gaussian factors
    changed.  The LSTM (h, c) state is a loop variable and is carried across
    batches via the LSTM module's carry-over variables.
    """

    def __init__(self, nb_tracks, lstm_module,
                 transition_shapes, transition_rates, nb_states,
                 sequence_phase_1, sequence_phase_2, transition_sequence,
                 transition_param_function, sequence_length=3,
                 vary_transition_shapes=None, vary_transition_rates=None,
                 carryover=False, **kwargs):
        if vary_transition_rates is None:
            vary_transition_rates = tf.ones(transition_rates.shape, dtype=dtype)
        if vary_transition_shapes is None:
            vary_transition_shapes = tf.ones(transition_shapes.shape, dtype=dtype)
        self.lstm_module = lstm_module
        self.sequence_phase_1 = sequence_phase_1
        self.sequence_phase_2 = sequence_phase_2
        self.transition_sequence = transition_sequence
        self.nb_states = nb_states
        self.sequence_length = sequence_length
        self.nb_tracks = nb_tracks
        self.initial_transition_params = [transition_shapes, transition_rates]
        self.transition_param_function = transition_param_function
        self.vary_transition_shapes = vary_transition_shapes
        self.vary_transition_rates = vary_transition_rates
        self.carryover = carryover
        super().__init__(**kwargs)

    def build(self, input_shape):
        nb_states = self.nb_states
        transition_shapes, transition_rates = self.initial_transition_params
        sequence_length = self.sequence_length
        nb_tracks = self.nb_tracks
        self.transition_rates = tf.Variable(
            transition_rates, dtype=dtype, name='Transition rates', trainable=True,
            constraint=lambda w: tf.clip_by_value(w, clip_value_min=-10, clip_value_max=4))
        self.transition_shapes = tf.Variable(
            transition_shapes, dtype=dtype, name='Transition shape', trainable=True)

        indices = tf.stack(
            [tf.repeat(tf.constant(list(np.arange(nb_states)) * sequence_length), nb_states),
             tf.concat([tf.range(nb_states)] * nb_states * sequence_length, 0)], axis=1)
        transition_mask = tf.cast((indices[:, 0] - indices[:, 1]) != 0, dtype=dtype)[None]
        self.indices = indices
        self.transition_mask = transition_mask
        self.oh_row = tf.cast(tf.one_hot(indices[:, 0], nb_states), dtype)   # (P,S) current
        self.oh_col = tf.cast(tf.one_hot(indices[:, 1], nb_states), dtype)   # (P,S) next

        if self.carryover:
            self.carryout_segment_len = tf.Variable(
                np.zeros((nb_tracks, sequence_length * nb_states)), dtype=dtype,
                name='carryover_segment_length', trainable=False)
            self.carryout_gamma_dist_mean = tf.Variable(
                np.zeros((nb_tracks, sequence_length * nb_states ** 2)), dtype=dtype,
                name='carryover_gamma_dist_mean', trainable=False)
            self.carryout_gamma_dist_var = tf.Variable(
                np.zeros((nb_tracks, sequence_length * nb_states ** 2)), dtype=dtype,
                name='carryover_gamma_dist_var', trainable=False)
        self.built = True
    
    @tf.function(jit_compile=False)
    def call(self, inputs, input_dts, reference_dt, mask,
             Prev_coefs, Prev_biases, LP, lstm_states, isfirst=None):
        nb_tracks = self.nb_tracks
        sequence_phase_1 = self.sequence_phase_1
        sequence_phase_2 = self.sequence_phase_2
        transition_sequence = self.transition_sequence
        transition_mask = self.transition_mask
        oh_row = self.oh_row
        oh_col = self.oh_col
        nb_states = self.nb_states
        sequence_length = self.sequence_length
        lstm = self.lstm_module
        
        # ---- stop-gradient on fixed dwell-time params ----
        ts = (self.vary_transition_shapes * self.transition_shapes
              + (1 - self.vary_transition_shapes) * tf.stop_gradient(self.transition_shapes))
        tr = (self.vary_transition_rates * self.transition_rates
              + (1 - self.vary_transition_rates) * tf.stop_gradient(self.transition_rates))

        dts_TN = tf.transpose(input_dts, [1, 0])
        transition_shapes_full, transition_rates_full = self.transition_param_function(
            ts, tr, dts_TN, reference_dt, dtype)

        # Gamma moments flattened onto the candidate (P) axis (unchanged logic)
        transition_rates_flat_full = tf.einsum(
            'tnij,pi,pj->tnp', transition_rates_full, oh_row, oh_col)
        transition_shapes_flat = tf.einsum(
            'ij,pi,pj->p', transition_shapes_full, oh_row, oh_col)
        transition_mean_full = transition_shapes_flat[None, None] / transition_rates_flat_full
        transition_var_full = transition_shapes_flat[None, None] / (transition_rates_flat_full ** 2)

        # slice time: index 0 of "_seq" lines up with inputs[0] (step 1)
        transition_mean_seq = transition_mean_full[1:, :, :nb_states ** 2]
        transition_var_seq = transition_var_full[1:, :, :nb_states ** 2]

        # ---- loop carriers ----
        segment_len = tf.zeros((nb_tracks, sequence_length * nb_states), dtype=dtype)
        gamma_dist_mean = transition_mean_full[0]
        gamma_dist_var = transition_var_full[0]

        if self.carryover:
            br1 = tf.broadcast_to(isfirst[:, None], segment_len.shape)
            segment_len = br1 * segment_len + (1 - br1) * self.carryout_segment_len
            br2 = tf.broadcast_to(isfirst[:, None], gamma_dist_mean.shape)
            gamma_dist_mean = br2 * gamma_dist_mean + (1 - br2) * self.carryout_gamma_dist_mean
            gamma_dist_var = br2 * gamma_dist_var + (1 - br2) * self.carryout_gamma_dist_var

        states_indices = tf.range(0, nb_states * sequence_length, dtype='int32') % nb_states
        states_indices = tf.repeat(states_indices[:, None], sequence_length, axis=1)
        states = tf.repeat(tf.one_hot(states_indices, nb_states, dtype=dtype)[None],
                           nb_tracks, axis=0)

        nb_dims = Prev_biases.shape[3]
        num_steps = tf.shape(inputs)[0]

        All_states_ta = tf.TensorArray(dtype=dtype, size=num_steps, dynamic_size=False,
                                       element_shape=(nb_tracks, 1, nb_states))
        All_coefs_ta = tf.TensorArray(dtype=dtype, size=num_steps, dynamic_size=False,
                                      element_shape=Prev_coefs.shape)
        All_biases_ta = tf.TensorArray(dtype=dtype, size=num_steps, dynamic_size=False,
                                       element_shape=Prev_biases.shape)
        All_LP_ta = tf.TensorArray(dtype=dtype, size=num_steps, dynamic_size=False,
                                   element_shape=LP.shape)

        # flatten lstm states into loop vars (h0,c0,h1,c1,...)
        lstm_flat = []
        for s in lstm_states:
            lstm_flat += [s[0], s[1]]
        n_lstm = len(lstm_flat)

        def body(i, Prev_coefs, Prev_biases, LP, segment_len,
                 gamma_dist_mean, gamma_dist_var, states,
                 All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta, *lstm_flat):
            # predicted-state distribution BEFORE the update
            log_w = LP - nb_dims * tf.math.log(
                tf.math.abs(tf.reduce_prod(tf.linalg.diag_part(
                    tf.transpose(Prev_coefs, [1, 2, 0, 3])), axis=-1)) + 1e-20)
            max_log_w = tf.reduce_max(log_w, 1, keepdims=True)
            w = tf.math.exp(log_w - max_log_w)
            w = w / tf.reduce_sum(w, 1, keepdims=True)
            pred_states = tf.reduce_sum(w[:, :, None] * states[:, :, 0], 1, keepdims=True)

            All_states_ta = All_states_ta.write(i, pred_states)
            All_coefs_ta = All_coefs_ta.write(i, Prev_coefs)
            All_biases_ta = All_biases_ta.write(i, Prev_biases)
            All_LP_ta = All_LP_ta.write(i, LP)

            input_i = inputs[i]
            mask_i = mask[:, i]
            trans_mean_i = transition_mean_seq[i]
            trans_var_i = transition_var_seq[i]
            dt_ratios = input_dts[:, i + 1] / reference_dt

            cur_lstm = [[lstm_flat[2 * k], lstm_flat[2 * k + 1]]
                        for k in range(n_lstm // 2)]

            (Next_coefs, Next_biases, Next_LP, Next_segment_len,
             Next_gamma_mean, Next_gamma_var, Next_states, new_lstm) = RNN_cell_LSTM(
                input_i, Prev_coefs, Prev_biases, LP, segment_len,
                lstm, cur_lstm, oh_row, oh_col, transition_mask,
                sequence_phase_1, sequence_phase_2, transition_sequence,
                trans_mean_i, trans_var_i,
                gamma_dist_mean, gamma_dist_var, states, dt_ratios)

            mask_coef = mask_i[None, :, None, None]
            mask_scalar = mask_i[:, None]
            mask_state = mask_i[:, None, None, None]
            mask_lstm = mask_i[:, None]

            Prev_coefs = Next_coefs * mask_coef + Prev_coefs * (1 - mask_coef)
            Prev_biases = Next_biases * mask_coef + Prev_biases * (1 - mask_coef)
            LP = Next_LP * mask_scalar + LP * (1 - mask_scalar)
            segment_len = Next_segment_len * mask_scalar + segment_len * (1 - mask_scalar)
            gamma_dist_mean = Next_gamma_mean * mask_scalar + gamma_dist_mean * (1 - mask_scalar)
            gamma_dist_var = Next_gamma_var * mask_scalar + gamma_dist_var * (1 - mask_scalar)
            states = Next_states * mask_state + states * (1 - mask_state)

            new_lstm_flat = []
            for k in range(n_lstm // 2):
                h = new_lstm[k][0] * mask_lstm + cur_lstm[k][0] * (1 - mask_lstm)
                c = new_lstm[k][1] * mask_lstm + cur_lstm[k][1] * (1 - mask_lstm)
                new_lstm_flat += [h, c]

            return (i + 1, Prev_coefs, Prev_biases, LP, segment_len,
                    gamma_dist_mean, gamma_dist_var, states,
                    All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta,
                    *new_lstm_flat)

        cond = lambda i, *_: i < num_steps

        loop_out = tf.while_loop(
            cond, body,
            loop_vars=[tf.constant(0), Prev_coefs, Prev_biases, LP, segment_len,
                       gamma_dist_mean, gamma_dist_var, states,
                       All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta,
                       *lstm_flat],
            parallel_iterations=1, swap_memory=True)

        (i, Prev_coefs, Prev_biases, LP, segment_len,
         gamma_dist_mean, gamma_dist_var, states,
         All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta) = loop_out[:12]
        final_lstm_flat = loop_out[12:]
        final_lstm = [[final_lstm_flat[2 * k], final_lstm_flat[2 * k + 1]]
                      for k in range(n_lstm // 2)]

        All_states = tf.transpose(All_states_ta.stack(), perm=[1, 0, 2, 3])[:, :, 0, :]
        All_coefs = tf.transpose(All_coefs_ta.stack(), perm=[2, 0, 3, 1, 4])
        All_biases = tf.transpose(All_biases_ta.stack(), perm=[2, 0, 3, 1, 4])
        All_LPs = tf.transpose(All_LP_ta.stack(), perm=[1, 0, 2])
        All_states = All_states[:, sequence_length - 1:]

        return (Prev_coefs, Prev_biases, LP, segment_len,
                gamma_dist_mean, gamma_dist_var,
                All_states, All_coefs, All_biases, All_LPs, states, final_lstm)


# ============================================================================
#  Model builder
# ============================================================================
def build_segment_model_LSTM(track_len,
                             nb_states,
                             initial_fractions,
                             transition_rates,
                             transition_shapes,
                             batch_size,
                             reference_dt,
                             nb_dims=2,
                             sequence_length=3,
                             nb_hidden_vars=2,
                             nb_obs_vars=1,
                             integration_variable_index=1,
                             lstm_units=128,
                             nb_lstm_layers=2,
                             feature_hidden=128,
                             vary_initial_fractions=True,
                             vary_transition_shapes=False,
                             vary_transition_rates=True,
                             nb_LocErr_dims=1,
                             stop_gradient_features=False):
    """
    Same wiring as `build_segment_model`, but the constraint function is the deep
    LSTM.  `params` / `initial_params` are no longer needed (the LSTM learns the
    dynamics); a *template* constraint function is used solely to discover the
    integration schedules via `get_sequences`.

    Note: input_LocErrs / input_dts are still accepted so the I/O signature
    matches your training pipeline (TrackSegmentSequence).  LocErrs are not fed
    to the LSTM by default; if you want them as inputs, append them in
    `featurize` (they are available as a per-step input — see TODO below).
    """
    H = nb_hidden_vars
    O = nb_obs_vars
    nb_gaussians = O + H

    # ---- template constraint function: schedules only -----------------------
    template_cf = constraint_function_arbitrary_KF(H, O, integration_variable_index)
    template_params = np.random.rand(
        nb_states, H ** 2 + (H + 1) * H // 2 + (O + 1) * O // 2 + (H + 1) * (O + 1)) * 4 - 2
    template_initial_params = np.random.rand(
        nb_states, H * (H + 1) // 2 + H) * 4 - 2
    
    (init_seq1, init_seq2, rec_seq1, rec_seq2, final_seq1, trans_seq) = get_sequences(
        template_params, template_initial_params, template_cf,
        nb_gaussians, H, dtype)
    
    # ---- shared LSTM dynamics ----------------------------------------------
    with tf.device('/GPU:0'):
        lstm_module = LSTM_constraint_function(
            nb_hidden_vars=H, nb_obs_vars=O,
            integration_variable_index=integration_variable_index,
            nb_states=nb_states, nb_dims=nb_dims, batch_size=batch_size,
            sequence_length=sequence_length, lstm_units=lstm_units,
            nb_lstm_layers=nb_lstm_layers, feature_hidden=feature_hidden,
            carryover=True, stop_gradient_features=stop_gradient_features, dtype=dtype)
        lstm_module.build(None)   # force-build sub-layers / carry vars
    
    # ---- inputs -------------------------------------------------------------
    inputs = tf.keras.Input(batch_shape=(batch_size, track_len, nb_obs_vars, nb_dims),
                            name='tracks', dtype=dtype)
    if nb_LocErr_dims > 0:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len, nb_obs_vars, nb_LocErr_dims),
                                       name='Localization errors', dtype=dtype)
    else:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len),
                                       name='Localization errors', dtype=dtype)
    input_dts = tf.keras.Input(batch_shape=(batch_size, track_len + 1),
                               name='frame durations', dtype=dtype)
    input_mask = tf.keras.Input(batch_shape=(batch_size, track_len),
                                name='masks', dtype=dtype)
    input_isfirst = tf.keras.Input(batch_shape=(batch_size,), name='isfirsts', dtype=dtype)

    '''
    all_inputs, outputs = seq[0]
    inputs = all_inputs[0]
    inputs.shape
    input_LocErrs = all_inputs[1]
    input_LocErrs.shape
    input_dts = all_inputs[2]
    input_mask = tf.constant(all_inputs[3], dtype = dtype)
    input_isfirst = tf.constant(all_inputs[4], dtype = dtype)
    '''

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

    reshaped_inputs = tf.keras.layers.Lambda(
        lambda x: x[:, None, :, None], dtype=dtype)(inputs)
    transposed_inputs = transpose_layer(dtype=dtype)(reshaped_inputs, perm=[2, 1, 0, 3, 4, 5])
    
    Init_layer = Initial_LSTM_layer(
        lstm_module, initial_fractions, init_seq1, init_seq2,
        nb_states, sequence_length, reference_dt,
        vary_initial_fractions = vary_initial_fractions,
        carryover=True, dtype=dtype)
    
    _, initial_states, lstm_states1 = Init_layer(transposed_inputs, input_dts, input_isfirst)
    Prev_coefs, Prev_biases, LP = initial_states
    
    first_mask_layer = IsfirstMaskLayer(dtype=dtype)
    Prev_coefs = first_mask_layer(Prev_coefs, Init_layer.carryout_coefs,
                                  input_isfirst[None, :, None, None])
    Prev_biases = first_mask_layer(Prev_biases, Init_layer.carryout_biases,
                                   input_isfirst[None, :, None, None])
    LP = first_mask_layer(LP, Init_layer.carryout_LP, input_isfirst[:, None])

    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype=dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype=dtype)(input_mask)
    
    rnn_layer = Custom_RNN_LSTM_layer(
        batch_size, lstm_module, transition_shapes, transition_rates, nb_states,
        rec_seq1, rec_seq2, trans_seq, transition_param_function,
        sequence_length = sequence_length,
        vary_transition_shapes = vary_transition_shapes,
        vary_transition_rates = vary_transition_rates,
        carryover=True, dtype=dtype)
    
    (Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var,
     All_motion_states, All_coefs, All_biases, All_LPs, motion_states,
     final_lstm) = rnn_layer(sliced_inputs, input_dts, reference_dt, sliced_mask,
                             Prev_coefs, Prev_biases, LP, lstm_states1,
                             isfirst=input_isfirst)

    states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]

    # carry-over: analytic state + dwell-time bookkeeping + LSTM (h,c)
    carry_vars = [Init_layer.carryout_coefs, Init_layer.carryout_biases,
                  Init_layer.carryout_LP, rnn_layer.carryout_segment_len,
                  rnn_layer.carryout_gamma_dist_mean, rnn_layer.carryout_gamma_dist_var]
    carry_states = [Prev_coefs, Prev_biases, LP, segment_len,
                    gamma_dist_mean, gamma_dist_var]
    for i in range(lstm_module.nb_lstm_layers):
        carry_vars += [lstm_module.carry_h[i], lstm_module.carry_c[i]]
        carry_states += [final_lstm[i][0], final_lstm[i][1]]

    carryover_layer = CarryoverAssignLayer(carryout_variables=carry_vars, dtype=dtype)

    F_layer = Final_layer(final_seq1, nb_dims=nb_dims,
                          sequence_length=sequence_length, dtype=dtype)
    outputs, All_states = F_layer(states)
    outputs = carryover_layer(outputs, carry_states)

    model = tf.keras.Model(
        inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
        outputs=outputs, name='LSTM_diffusion_model')
    pred_model = tf.keras.Model(
        inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
        outputs=(outputs, All_states, All_coefs, All_biases, All_LPs),
        name='LSTM_diffusion_model')

    return model, pred_model


def extract_hidden_variables_general(All_coefs, All_biases, All_LPs,
                                     nb_dims, sequence_length,
                                     ridge=1e-12, eps=1e-30):
    """
    Filtered (online) estimate of every hidden variable for an arbitrary
    ``constraint_function_arbitrary_KF`` model.

    Parameters
    ----------
    All_coefs : array, shape (nb_tracks, track_len, nb_sequences, H, H)
        Coefficient matrices C.  ``All_coefs[..., g, h]`` is the coefficient of
        hidden variable ``h`` in Gaussian ``g``. (3rd output of ``pred_model``.)
    All_biases : array, shape (nb_tracks, track_len, nb_sequences, H, nb_dims)
        Bias vectors b. (4th output of ``pred_model``.)
    All_LPs : array, shape (nb_tracks, track_len, nb_sequences)
        Per-sequence log-likelihoods, hidden variables NOT yet integrated out.
        (5th output of ``pred_model``.)
    nb_dims : int
        Number of independent spatial dimensions.
    sequence_length : int
        Number of parallel histories per state
        (``nb_sequences = sequence_length * nb_states``).
    ridge : float
        Tikhonov term added to the diagonal of C before inversion / slogdet,
        for numerical stability on near-singular (e.g. padded) steps. The
        recurrence coefficients are precision square-roots, typically O(1) or
        larger, so this is negligible for real data.
    eps : float
        Floor inside logarithms.

    Returns
    -------
    dict with keys
        'per_state_mean' : (nb_tracks, track_len, nb_states, H, nb_dims)
        'per_state_std'  : (nb_tracks, track_len, nb_states, H, nb_dims)
        'collapsed_mean' : (nb_tracks, track_len, H, nb_dims)
        'collapsed_std'  : (nb_tracks, track_len, H, nb_dims)
        'seq_mean'       : (nb_tracks, track_len, nb_sequences, H, nb_dims)
                           raw per-sequence MAP (pre-recombination)
        'seq_std'        : (nb_tracks, track_len, nb_sequences, H)
        'seq_LP'         : (nb_tracks, track_len, nb_sequences, H)
                           per-variable weighting log-likelihoods
    """
    All_coefs  = np.asarray(All_coefs,  dtype=np.float64)
    All_biases = np.asarray(All_biases, dtype=np.float64)
    All_LPs    = np.asarray(All_LPs,    dtype=np.float64)

    nb_tracks, track_len, nb_sequences = All_LPs.shape
    H = All_coefs.shape[-1]
    if nb_sequences % sequence_length != 0:
        raise ValueError("nb_sequences (%d) is not a multiple of "
                         "sequence_length (%d)" % (nb_sequences, sequence_length))
    nb_states = nb_sequences // sequence_length

    # ---- per-sequence multivariate-Gaussian moments -----------------------
    C = All_coefs + ridge * np.eye(H, dtype=np.float64)        # (..., H, H)
    Cinv = np.linalg.inv(C)                                    # (..., H, H)

    # mean[..., h, d] = -(C^{-1} b)[..., h, d]
    seq_mean = -np.einsum('...hg,...gd->...hd', Cinv, All_biases)   # (..., H, nb_dims)

    # marginal variance of variable h = sum_g (C^{-1})[h, g]^2 = Sigma[h, h]
    seq_var = np.sum(Cinv ** 2, axis=-1)                      # (..., H)
    seq_std = np.sqrt(np.maximum(seq_var, 0.0))

    # ---- per-variable weighting log-likelihoods ---------------------------
    # LP_h = All_LPs - nb_dims * (log|det C| + log sigma_h)
    _sign, logdet = np.linalg.slogdet(C)                      # (...,)
    log_sigma = 0.5 * np.log(seq_var + eps)                   # (..., H)
    seq_LP = (All_LPs[..., None]
              - nb_dims * logdet[..., None]
              - nb_dims * log_sigma)                          # (..., nb_sequences, H)

    # ---- recombination helper (softmax weights + mixture moments) ---------
    def _mix(mean, var, lp, axis):
        w = scipy_softmax(lp, axis=axis)                     # normalise over `axis`
        wb = w[..., None]                                    # add nb_dims axis
        vb = var[..., None]
        mix_mean = np.sum(mean * wb, axis=axis)
        mix_e2   = np.sum((vb + mean ** 2) * wb, axis=axis)
        mix_var  = mix_e2 - mix_mean ** 2
        return mix_mean, np.sqrt(np.maximum(mix_var, 0.0))

    # ---- per-state estimate (mix the sequence_length histories per state) --
    shp_m = (nb_tracks, track_len, sequence_length, nb_states, H, nb_dims)
    shp_v = (nb_tracks, track_len, sequence_length, nb_states, H)
    per_state_mean, per_state_std = _mix(
        seq_mean.reshape(shp_m), seq_var.reshape(shp_v),
        seq_LP.reshape(shp_v), axis=2)                       # drop sequence_length axis

    # ---- collapsed estimate (mix every sequence: state-marginalised) -------
    collapsed_mean, collapsed_std = _mix(seq_mean, seq_var, seq_LP, axis=2)

    return {
        'per_state_mean': per_state_mean,    # (N, T, nb_states, H, nb_dims)
        'per_state_std':  per_state_std,     # (N, T, nb_states, H, nb_dims)
        'collapsed_mean': collapsed_mean,    # (N, T, H, nb_dims)
        'collapsed_std':  collapsed_std,     # (N, T, H, nb_dims)
        'seq_mean':       seq_mean,          # (N, T, nb_sequences, H, nb_dims)
        'seq_std':        seq_std,           # (N, T, nb_sequences, H)
        'seq_LP':         seq_LP,            # (N, T, nb_sequences, H)
    }


class HiddenStateExtractor:
    """
    Convenience wrapper: runs ``pred_model`` on a dataset and returns the
    inferred hidden variables for an arbitrary ``constraint_function_arbitrary_KF``
    model.

    Parameters
    ----------
    pred_model : tf.keras.Model
        The ``pred_model`` returned by ``build_segment_model`` (outputs
        ``(outputs, All_states, All_coefs, All_biases, All_LPs)``).
    sequence_length : int
        Same ``sequence_length`` the model was built with.
    nb_dims : int, optional
        Spatial dimensions; inferred from the track array if omitted.
    """

    def __init__(self, pred_model, sequence_length, nb_dims=None):
        self.pred_model = pred_model
        self.sequence_length = sequence_length
        self.nb_dims = nb_dims

    # ---- raw forward pass -------------------------------------------------
    def predict_raw(self, tracks, LocErrs, dts, masks, batch_size, is_first=None):
        if is_first is None:
            is_first = np.ones(np.asarray(tracks).shape[0])
        return self.pred_model.predict(
            (tracks, LocErrs, dts, masks, is_first), batch_size=batch_size)

    # ---- filtered (online) hidden-variable inference ----------------------
    def extract(self, tracks, LocErrs, dts, masks, batch_size, is_first=None,
                ridge=1e-12):
        nb_dims = self.nb_dims if self.nb_dims is not None else np.asarray(tracks).shape[-1]
        outputs, All_states, All_coefs, All_biases, All_LPs = self.predict_raw(
            tracks, LocErrs, dts, masks, batch_size, is_first)
        result = extract_hidden_variables_general(
            All_coefs, All_biases, All_LPs, nb_dims, self.sequence_length, ridge=ridge)
        result['state_preds'] = All_states     # (N, T, nb_states) online state posterior
        result['LP'] = outputs                 # final per-track log-likelihood
        return result

    # ---- smoothed inference (forward + time-reversed fusion) --------------
    def extract_smooth(self, tracks, LocErrs, dts, masks, batch_size,
                       shape_weight_index=8, rate_weight_index=7,
                       reverse_parity=None, ridge=1e-12):
        """
        Forward+reverse precision-weighted fusion, generalising
        ``extract_smooth_hidden_variables``.

        Notes
        -----
        * The transition Gamma shape/rate matrices are transposed for the reverse
          pass (``s -> s'`` becomes ``s' -> s``), exactly as in the original.
          ``shape_weight_index`` / ``rate_weight_index`` are the indices of those
          weights in ``pred_model.weights`` (8 / 7 for the segmented model).
        * ``reverse_parity`` is an array of length H giving, per hidden variable,
          the sign acquired under time reversal (-1 for odd/velocity-like
          variables whose sign flips, +1 for even/position-like variables).
          The arbitrary-KF parameterisation has no built-in notion of which
          variable is a drift, so this MUST be supplied by the caller when any
          variable is direction-dependent; defaults to all +1 (no flip).
          Any dt-dependent rescaling (e.g. displacement -> velocity) is left to
          the caller, since it is model-specific.

        Returns the precision-weighted fused ``per_state_mean`` / ``per_state_std``
        and ``collapsed_mean`` / ``collapsed_std`` aligned on the forward time axis.
        """
        import tensorflow as tf
        nb_dims = self.nb_dims if self.nb_dims is not None else np.asarray(tracks).shape[-1]
        tracks = np.asarray(tracks)
        H = None

        # forward pass
        fwd = self.extract(tracks, LocErrs, dts, masks, batch_size, ridge=ridge)

        # reverse pass: transpose the Gamma matrices, reverse time
        w = self.pred_model.weights
        w[rate_weight_index].assign(tf.transpose(w[rate_weight_index]))
        w[shape_weight_index].assign(tf.transpose(w[shape_weight_index]))
        try:
            inverse_dts = np.concatenate((dts[:, -1:], dts[:, :-1]), axis=1)[:, ::-1]
            rev = self.extract(tracks[:, ::-1], np.asarray(LocErrs)[:, ::-1],
                               inverse_dts, np.asarray(masks)[:, ::-1],
                               batch_size, ridge=ridge)
        finally:
            # always restore the model
            w[rate_weight_index].assign(tf.transpose(w[rate_weight_index]))
            w[shape_weight_index].assign(tf.transpose(w[shape_weight_index]))

        H = fwd['per_state_mean'].shape[-2]
        if reverse_parity is None:
            reverse_parity = np.ones(H)
        reverse_parity = np.asarray(reverse_parity, dtype=np.float64)[None, None, None, :, None]

        def _fuse(key_mean, key_std, slicer_fwd, slicer_rev, parity):
            m1 = fwd[key_mean][slicer_fwd]
            s1 = fwd[key_std][slicer_fwd]
            m2 = parity * rev[key_mean][slicer_rev]
            s2 = rev[key_std][slicer_rev]
            v1 = s1 ** 2 + 1e-30
            v2 = s2 ** 2 + 1e-30
            wfused = 1.0 / v1 + 1.0 / v2
            mean = (m1 / v1 + m2 / v2) / wfused
            std = (1.0 / wfused) ** 0.5
            return mean, std

        # align forward [1:] with reverse [:0:-1] (drop the boundary step on each)
        ps_mean, ps_std = _fuse(
            'per_state_mean', 'per_state_std',
            (slice(None), slice(1, None)),
            (slice(None), slice(None, 0, -1)),
            reverse_parity)
        col_parity = reverse_parity[:, :, 0]   # (1,1,H,1) for collapsed (no state axis)
        col_mean, col_std = _fuse(
            'collapsed_mean', 'collapsed_std',
            (slice(None), slice(1, None)),
            (slice(None), slice(None, 0, -1)),
            col_parity)

        return {'per_state_mean': ps_mean,
            'per_state_std':  ps_std,
            'collapsed_mean': col_mean,
            'collapsed_std':  col_std}



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
            shape_idx = 7
            rate_idx = 6
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
        
        weights[7]
        
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
        shape_IDs = 7
        rates_IDs = 6
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
    params = weights[0]
    initial_params = weights[1]

    param_dict = {'params': params, 'initial params': initial_params, 'transition rates': transition_rates, 'transition shapes': transition_shapes, 'Fractions': tf.math.softmax(weights[2][0]).numpy()}
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

#%% Track segmentation
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


@njit(cache=True)
def _segment_tracks_core(
    tracks_flat,        # (total_len, nb_obs_vars, nb_dims)        float64
    locerrs_flat,       # (total_len, nb_obs_vars, locerr_dims)    float64
    dts_flat,           # (total_dt_len,)                          float64
    track_offsets,      # (nb_tracks + 1,)                         int64  -- cumsum of track lengths
    dt_offsets,         # (nb_tracks + 1,)                         int64  -- cumsum of dt lengths
    batch_size,
    segment_length,
    min_segment_length,
    max_nb_batches,
    mean_dt):

    nb_obs_vars = tracks_flat.shape[1]
    nb_dims     = tracks_flat.shape[2]
    locerr_dims = locerrs_flat.shape[2]

    track_batches   = np.zeros((max_nb_batches, batch_size, segment_length, nb_obs_vars, nb_dims))
    locerr_batches  = np.zeros((max_nb_batches, batch_size, segment_length, nb_obs_vars, locerr_dims))
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
            # LHS (seg_len, nb_obs_vars, nb_dims) == RHS (seg_len, nb_obs_vars, nb_dims)
            track_batches[row, index_ID, :seg_len] = \
                tracks_flat[track_start + seg_start : track_start + seg_end]
            for o in range(nb_obs_vars):
                for k in range(nb_dims):
                    pad_val = tracks_flat[last_track_idx, o, k]
                    for j in range(seg_len, segment_length):
                        track_batches[row, index_ID, j, o, k] = pad_val

            # --- localization errors: same pattern ---
            locerr_batches[row, index_ID, :seg_len] = \
                locerrs_flat[track_start + seg_start : track_start + seg_end]
            for o in range(nb_obs_vars):
                for k in range(locerr_dims):
                    pad_val = locerrs_flat[last_track_idx, o, k]
                    for j in range(seg_len, segment_length):
                        locerr_batches[row, index_ID, j, o, k] = pad_val

            # --- dt: one extra time step, uses the *unclipped* end + 1 ---
            # dt has no obs-vars axis: it is a single duration per time-step.
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
    Numba-accelerated segment_tracks for the generalized ExaTrack algorithm.

    Compared with the previous version, each localization now carries an extra
    `nb_obs_vars` axis (the number of observed variables per time point), so:

      * track_list[i]  has shape (track_len, nb_obs_vars, nb_dims)
      * LocErr_list[i] has shape (track_len, nb_obs_vars, nb_LocErr_dims)
                       (or (track_len, nb_obs_vars), implying nb_LocErr_dims == 1)
      * dt_list[i]     has shape (track_len,)                  -- unchanged

    Outputs:
      * track_batches  : (nb_batches, batch_size, segment_length, nb_obs_vars, nb_dims)
      * locerr_batches : (nb_batches, batch_size, segment_length, nb_obs_vars, nb_LocErr_dims)
                         (last axis squeezed if the input LocErr was 2D per track)
      * dt_batches     : (nb_batches, batch_size, segment_length + 1)
      * mask_batches   : (nb_batches, batch_size, segment_length)
      * isfirst_batches: (nb_batches, batch_size)

    Key changes vs. the original:
      * Variable-length lists are concatenated into flat float64 arrays + offsets
        before entering the jitted core. tracks_flat / locerrs_flat now keep the
        obs-vars axis: (total_len, nb_obs_vars, nb_dims/locerr_dims).
      * The per-track np.where placement is replaced by an O(batch_size)
        "leftmost column with minimum next-empty-row" lookup, reproducing the
        original C-order placement.
      * The hot inner loop (copy segment, pad with last value over both the
        obs-vars and dims axes, write mask) is compiled with @njit(cache=True).
    """

    # ---- clip cutoff (preserve original behavior) ----
    if cutoff_batch_treshhold > 1:
        cutoff_batch_treshhold = 1
    elif cutoff_batch_treshhold < 1 / batch_size:
        cutoff_batch_treshhold = 0.5 / batch_size

    nb_tracks   = len(track_list)
    nb_obs_vars = track_list[0].shape[1]
    nb_dims     = track_list[0].shape[2]

    if LocErr_list is None:
        # default: one error value per (time point, obs var)
        LocErr_list = [np.ones((len(track), nb_obs_vars)) for track in track_list]

    # A 2D per-track LocErr means (track_len, nb_obs_vars) -> nb_LocErr_dims == 1.
    # A 3D per-track LocErr means (track_len, nb_obs_vars, nb_LocErr_dims).
    locerr_is_2d = (LocErr_list[0].ndim == 2)
    locerr_dims  = 1 if locerr_is_2d else LocErr_list[0].shape[2]

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

    tracks_flat  = np.zeros((total_track_len, nb_obs_vars, nb_dims),     dtype=np.float64) - 1
    locerrs_flat = np.zeros((total_track_len, nb_obs_vars, locerr_dims), dtype=np.float64)
    dts_flat     = np.zeros(total_dt_len,                                dtype=np.float64)

    for i in range(nb_tracks):
        s, e = int(track_offsets[i]), int(track_offsets[i + 1])
        tracks_flat[s:e] = track_list[i]
        if locerr_is_2d:
            locerrs_flat[s:e, :, 0] = LocErr_list[i]
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

    # ---- restore squeezed LocErr shape if input was 2D per track ----
    if locerr_is_2d:
        locerr_batches = locerr_batches[..., 0]

    return track_batches, locerr_batches, dt_batches, mask_batches, isfirst_batches


'''
lines to visually test if the segments are built well.
track_batches.shape

# pick one obs var (here index 0) and one dim (here index 1) to visualise
for i in range(len(track_batches)):
    track_batches[i, :, 0, 0] = track_batches[i, :, 0, 0] * (1 - isfirst_batches[i, :, None])

img = track_batches[..., 0, 1].transpose([0, 2, 1])   # (nb_batches, seg_len, batch_size)
img = np.concatenate(img)

plt.figure()
plt.imshow(img[-200:, :])
'''


class TrackSegmentSequence(tf.keras.utils.Sequence):
    """Keras Sequence that pre-computes all batches from segment_tracks."""

    def __init__(self, track_list, LocErr_list, dt_list, batch_size, segment_length=20,
                 min_segment_length=4, cutoff_batch_treshhold=0.01, shuffle=False,
                 dtype='float32'):
        """
        Parameters
        ----------
        track_list:  List of numpy arrays of shape (number of time points, nb_obs_vars, nb_dims).
                     The number of time points can vary across tracks, but nb_obs_vars and nb_dims
                     must be constant.
        LocErr_list: List of localization errors corresponding to the localizations in track_list.
                     Each element must have shape (number of time points, nb_obs_vars, nb_LocErr_dims)
                     or (number of time points, nb_obs_vars) for a single error dim.
                     Can also be set to None to assume unit errors (shape (track_len, nb_obs_vars)).
        dt_list:     List of durations between frames corresponding to the localizations in track_list.
                     For a given track, dt_list[i] must equal time[i+1] - time[i]. A last element with
                     value reference_dt must be provided. Can also be set to None to assume a dt shared
                     across tracks and time points. Each element must have shape (number of time points).

        batch_size:             (int) Batch size of the model.
        segment_length:         (int) length of the track segments (time-steps per batch of the model).
        min_segment_length:     (int) minimal segment length that we consider.
        cutoff_batch_treshhold: Fraction of the non-full batches we keep. 0 removes all non-full
                                batches (which might be problematic); 1 keeps all of them.
        """
        self.track_list = track_list
        if LocErr_list is None:
            LocErr_list = [np.ones((len(track), track.shape[1])) for track in track_list]
        if dt_list is None:
            dt_list = [np.ones(len(track)) for track in track_list]

        self.LocErr_list = LocErr_list
        self.dt_list = dt_list
        self.segment_length = segment_length
        self.min_segment_length = min_segment_length
        self.cutoff_batch_treshhold = cutoff_batch_treshhold
        self.shuffle = shuffle
        self.dtype = dtype

        self.tracks, self.LocErrs, self.dts, self.masks, self.isfirsts = segment_tracks(
            track_list, LocErr_list, dt_list, batch_size, segment_length,
            min_segment_length, cutoff_batch_treshhold)
        self.batch_size = batch_size

        # Pre-build dummy labels once (same array every batch)
        self.dummy_labels = np.zeros((self.batch_size,), dtype=dtype)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        inputs = (self.tracks[idx],     # (batch_size, seg_len, nb_obs_vars, nb_dims)
                  self.LocErrs[idx],     # (batch_size, seg_len, nb_obs_vars, nb_LocErr_dims)
                  self.dts[idx],         # (batch_size, seg_len + 1)
                  self.masks[idx],       # (batch_size, seg_len)
                  self.isfirsts[idx])    # (batch_size,)
        return inputs, self.dummy_labels

    def on_epoch_end(self):
        """Shuffle batch order between epochs."""
        if self.shuffle:
            self.tracks, self.LocErrs, self.dts, self.masks, self.isfirsts = segment_tracks(
                self.track_list, self.LocErr_list, self.dt_list, self.batch_size,
                self.segment_length, self.min_segment_length,
                self.cutoff_batch_treshhold, self.shuffle)


# -*- coding: utf-8 -*-
"""
Monte-Carlo multi-step forecasting for the LSTM-driven segmented state-space
model (`build_segment_model_LSTM` / `LSTM_constraint_function`).

What this does
--------------
Given a *trained* `pred_model` and an observed prefix of each track, it produces
a forecast of the next `horizon` observations (and hidden variables and motion
states) by **ancestral forward simulation of the model's own predictive
distribution**:

    for each rollout step:
        1. step the LSTM once -> emitted Gaussian factors g  (the "emit" half of
           the recurrence cell)
        2. sample an observation y_{t+1} from the model's one-step predictive
           mixture  p(y_{t+1} | history), i.e.
              - sample a hypothesis k and next state s' from
                    softmax( LP_k + log P(s'|k) )            (Gamma hazard)
              - sample the hidden x_{t+1} from that component
                    (carried predictive if no transition; carry vars 0..idx-1 and
                     reset vars idx..H-1 from state s' prior if a transition)
              - sample y_{t+1} from state s' measurement Gaussian given x_{t+1}
        3. feed that *sampled* y back through the real recurrence (`_cell_core`,
           the "update" half) to advance belief + dwell-time + LSTM (h,c).

Running many independent particles per track and reading the per-step empirical
mean / std / quantiles of the sampled y (and x, and state posterior) gives the
forecast and its uncertainty.

Why this is in-distribution
---------------------------
The belief carriers the LSTM consumes are always genuine filtered mixtures: a
sample drawn from the model is distributed identically to a real observation, so
conditioning on it yields a belief of exactly the kind seen during training.
Hence no masked retraining is required to handle the forecast regime.

Validation hook
---------------
`one_step_predictive_y` enumerates the *same* mixture analytically (no sampling).
At horizon 1 the MC mean/std of y must match it; this is also a fast exact
next-observation predictor on its own.

Usage
-----
    from mc_rollout_forecaster import MCRolloutForecaster
    fc = MCRolloutForecaster(pred_model)
    out = fc.rollout(tracks, LocErrs, dts, masks,
                     horizon=20, nb_particles=200)
    # out['y_mean'] : (nb_tracks, horizon, O, nb_dims)
    # out['y_std']  : (nb_tracks, horizon, O, nb_dims)
    # out['y_quantiles'] : (nb_tracks, horizon, n_q, O, nb_dims)
    # out['x_mean'], out['x_std'] : (nb_tracks, horizon, H, nb_dims)
    # out['state_probs'] : (nb_tracks, horizon, nb_states)
    # out['y_samples'] (optional) : (nb_tracks, nb_particles, horizon, O, nb_dims)

Adjust the import below to the module where your model classes live.
"""



# ===========================================================================
#  The "update" half of RNN_cell_LSTM: identical body, but the LSTM has
#  already been stepped outside (so we never advance (h,c) twice). It takes the
#  emitted factors `g` and the new lstm states, and the sampled observation.
# ===========================================================================
def _cell_core(g, new_lstm_states, input_i,
               Prev_coefs, Prev_biases, LP, segment_len,
               lstm_module, oh_row, oh_col, transition_mask,
               sequence_phase_1, sequence_phase_2, transition_sequence,
               transition_mean, transition_var,
               gamma_dist_mean, gamma_dist_var, states, dt_ratios):
    """Verbatim body of RNN_cell_LSTM after the `lstm.step`/`assemble` lines."""
    nb_dims = input_i.shape[-1]
    nb_tracks = LP.shape[0]
    nb_hidden_vars = Prev_coefs.shape[3]
    nb_states = lstm_module.nb_states
    sequence_length = LP.shape[1] // nb_states

    reccurent_obs_var_coefs         = g['rec_obs']
    reccurent_hidden_var_coefs      = g['rec_hidden_cur']
    reccurent_next_hidden_var_coefs = g['rec_hidden_next']
    reccurent_biases                = g['rec_biases']
    transition_hidden_var_coefs     = g['trans_hidden']
    transition_biases               = g['trans_biases']
    Log_factors_NS                  = g['Log_factors']
    transition_Log_factors_NS       = g['transition_Log_factors']

    transition_hidden_var_coefs = tf.concat(
        [transition_hidden_var_coefs] * (sequence_length * nb_states), axis=2)
    transition_biases = tf.concat(
        [transition_biases] * (sequence_length * nb_states), axis=2)

    flat_Log = tf.einsum('ns,ps->np', Log_factors_NS, oh_row)
    flat_trans = tf.einsum('ns,ps->np', transition_Log_factors_NS, oh_col)
    reshaped_Log_factors = flat_trans * transition_mask + flat_Log * (1 - transition_mask)

    current_states = states[:, :, -1:]

    Prev_coefs2 = tf.repeat(Prev_coefs, nb_states, axis=2)
    Prev_biases2 = tf.repeat(Prev_biases, nb_states, axis=2)
    LP2 = tf.repeat(LP, nb_states, axis=1)
    segment_len = tf.repeat(segment_len, nb_states, axis=1)

    alternative_Prev_coefs = tf.concat(
        (Prev_coefs2, tf.identity(transition_hidden_var_coefs)), axis=0)
    alternative_Prev_biases = tf.concat(
        (Prev_biases2, tf.identity(transition_biases)), axis=0)

    transition_Prev_coefs, transition_Prev_biases, LC = transition_RNN_reccurence_formula(
        current_hidden_var_coefs=alternative_Prev_coefs,
        next_hidden_var_coefs=tf.constant(0, dtype=dtype, shape=alternative_Prev_coefs.shape),
        biases=alternative_Prev_biases,
        transition_sequence=transition_sequence,
        nb_dims=nb_dims,
        dtype=dtype)

    LP2 += LC * transition_mask + reshaped_Log_factors

    current_shapes = gamma_dist_mean ** 2 / gamma_dist_var
    current_rates = gamma_dist_mean / gamma_dist_var

    all_Prev_coefs = (transition_Prev_coefs * transition_mask[None, :, :, None]
                      + Prev_coefs2 * (1 - transition_mask[None, :, :, None]))
    all_prev_biases = (transition_Prev_biases * transition_mask[None, :, :, None]
                       + Prev_biases2 * (1 - transition_mask[None, :, :, None]))

    gamma = tf.compat.v1.distributions.Gamma(current_shapes, current_rates)
    S_old = 1 - gamma.cdf(segment_len + 1e-12) + 1e-12
    S_new = 1 - gamma.cdf(segment_len + 1e-12 + dt_ratios[:, None]) + 1e-12
    transition_probas = tf.clip_by_value(1 - S_new / S_old,
                                         clip_value_min=1e-20, clip_value_max=1 - 1e-10)

    non_transition_probas = tf.repeat(
        1 - tf.clip_by_value(
            tf.reduce_sum(tf.reshape(transition_probas * transition_mask,
                                     shape=(nb_tracks, nb_states * sequence_length, nb_states)),
                          axis=2),
            clip_value_min=1 - 20, clip_value_max=1 - 1e-10),
        nb_states, axis=1)

    transition_probas = (transition_probas * transition_mask
                         + non_transition_probas * (1 - transition_mask))
    all_LP = LP2 + tf.math.log(transition_probas)

    current_reccurent_obs_var_coefs = tf.concat(
        [reccurent_obs_var_coefs] * (sequence_length * nb_states), axis=2)
    current_reccurent_hidden_var_coefs = tf.concat(
        [reccurent_hidden_var_coefs] * (sequence_length * nb_states), axis=2)
    current_reccurent_next_hidden_var_coefs = tf.concat(
        [reccurent_next_hidden_var_coefs] * (sequence_length * nb_states), axis=2)
    current_reccurent_biases = tf.concat(
        [reccurent_biases] * (sequence_length * nb_states), axis=2)

    current_hidden_var_coefs = tf.concat(
        (all_Prev_coefs, tf.identity(current_reccurent_hidden_var_coefs)), axis=0)
    zero_tensor = tf.constant(0, dtype=dtype, shape=all_Prev_coefs.shape)
    next_hidden_var_coefs = tf.concat(
        (zero_tensor, tf.identity(current_reccurent_next_hidden_var_coefs)), axis=0)
    current_biases = tf.identity(current_reccurent_biases)
    current_biases += tf.reduce_sum(
        current_reccurent_obs_var_coefs[:, :, :, :, None] * input_i, (-2))
    biases = tf.concat((all_prev_biases, current_biases), axis=0)

    Next_coefs, Next_biases, LC = RNN_reccurence_formula(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        sequence_phase_1, sequence_phase_2, nb_dims=nb_dims, dtype=dtype)

    all_LP += LC

    reshaped_Next_coefs = tf.reshape(
        Next_coefs, Next_coefs.shape[:2] + [sequence_length * nb_states, nb_states, nb_hidden_vars])
    transition_LPs = tf.reshape(all_LP - 200 * (1 - transition_mask),
                                (nb_tracks, sequence_length * nb_states, nb_states)) \
        - nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(reshaped_Next_coefs, [1, 2, 3, 0, 4])),
            axis=-1)) + 1e-20)

    max_transition_LPs = tf.reduce_max(transition_LPs, axis=1, keepdims=True)
    transition_Ps = tf.math.exp(transition_LPs - max_transition_LPs)
    transition_weights = transition_Ps / tf.reduce_sum(transition_Ps, 1, keepdims=True)

    transition_states = tf.reduce_sum(
        states[:, :, None] * transition_weights[:, :, :, None, None], 1)

    transition_Next_coefs = tf.reshape(
        Next_coefs, Next_coefs.shape[:2] + [sequence_length * nb_states, nb_states, nb_hidden_vars])
    transition_Next_coefs = tf.reduce_sum(
        transition_Next_coefs * transition_weights[None, :, :, :, None], axis=2)

    transition_Next_biases = tf.reshape(
        Next_biases, Next_biases.shape[:2] + [sequence_length * nb_states, nb_states, nb_dims])
    transition_Next_biases = tf.reduce_sum(
        transition_Next_biases * transition_weights[None, :, :, :, None], axis=2)

    transition_LPs = tf.math.log(tf.reduce_sum(transition_Ps, axis=1)) \
        + max_transition_LPs[:, 0] \
        + nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(transition_Next_coefs, [1, 2, 0, 3])),
            axis=-1)) + 1e-20)

    stable_LPs = tf.reshape(all_LP, (nb_tracks, sequence_length * nb_states, nb_states))
    stable_weights = tf.reshape((1 - transition_mask),
                                (sequence_length * nb_states, nb_states))[None]
    stable_LPs = tf.reduce_sum(stable_LPs * stable_weights, 2)
    stable_states = tf.reduce_sum(states[:, :, None] * stable_weights[:, :, :, None, None], 2)

    stable_Next_coefs = tf.reduce_sum(
        tf.reshape(Next_coefs, Next_coefs.shape[:2] +
                   [sequence_length * nb_states, nb_states, nb_hidden_vars])
        * stable_weights[None, :, :, :, None], axis=3)
    stable_Next_biases = tf.reduce_sum(
        tf.reshape(Next_biases, Next_biases.shape[:2] +
                   [sequence_length * nb_states, nb_states, nb_dims])
        * stable_weights[None, :, :, :, None], axis=3)
    stable_segment_len = tf.reduce_sum(
        tf.reshape(segment_len, (nb_tracks, sequence_length * nb_states, nb_states))
        * stable_weights, axis=2)

    current_gamma_dist_mean = tf.concat([transition_mean, gamma_dist_mean], axis=1)
    current_gamma_dist_var = tf.concat([transition_var, gamma_dist_var], axis=1)

    Next_coefs = tf.concat([transition_Next_coefs, stable_Next_coefs], axis=2)
    Next_biases = tf.concat([transition_Next_biases, stable_Next_biases], axis=2)
    new_LP = tf.concat([transition_LPs, stable_LPs], axis=1)
    additional_stable_segment_len = dt_ratios[:, None]
    current_segment_len = tf.concat(
        [tf.zeros((nb_tracks, nb_states), dtype=dtype), stable_segment_len], axis=1)
    current_segment_len = current_segment_len + additional_stable_segment_len
    Next_states = tf.concat([transition_states, stable_states], axis=1)

    saved_Next_coefs = Next_coefs[:, :, :-nb_states * 2]
    saved_Next_biases = Next_biases[:, :, :-nb_states * 2]
    saved_LP = new_LP[:, :-nb_states * 2]
    saved_segment_len = current_segment_len[:, :-nb_states * 2]
    saved_gamma_dist_mean = current_gamma_dist_mean[:, :-nb_states ** 2 * 2]
    saved_gamma_dist_var = current_gamma_dist_var[:, :-nb_states ** 2 * 2]
    saved_states = Next_states[:, :-nb_states * 2]

    nb_prev_gaussians = Next_coefs.shape[0]
    last_Next_coefs = tf.reshape(Next_coefs[:, :, -nb_states * 2:],
                                 (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_hidden_vars))
    last_Next_biases = tf.reshape(Next_biases[:, :, -nb_states * 2:],
                                  (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_dims))
    last_LP = tf.reshape(new_LP[:, -nb_states * 2:], (nb_tracks, 2, nb_states)) \
        - nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(last_Next_coefs, [1, 2, 3, 0, 4])),
            axis=-1)) + 1e-20)
    last_segment_len = tf.reshape(current_segment_len[:, -nb_states * 2:],
                                  (nb_tracks, 2, nb_states))
    last_gamma_dist_mean = tf.reshape(current_gamma_dist_mean[:, -nb_states ** 2 * 2:],
                                      (nb_tracks, 2, nb_states, nb_states))
    last_gamma_dist_var = tf.reshape(current_gamma_dist_var[:, -nb_states ** 2 * 2:],
                                     (nb_tracks, 2, nb_states, nb_states))
    last_states = tf.reshape(Next_states[:, -nb_states * 2:],
                             (nb_tracks, 2, nb_states, sequence_length, nb_states))

    last_LP_max = tf.reduce_max(last_LP, axis=1, keepdims=True)
    last_P = tf.math.exp(last_LP - last_LP_max)
    sum_last_P = tf.reduce_sum(last_P, 1, keepdims=True)

    weight_last_P = tf.math.exp(last_LP - tf.reduce_max(last_LP, axis=1, keepdims=True))
    last_weights = weight_last_P / tf.reduce_sum(weight_last_P, 1, keepdims=True)

    reduced_last_Next_coefs = tf.reduce_sum(
        last_Next_coefs * last_weights[None, :, :, :, None], axis=2)
    reduced_last_Next_biases = tf.reduce_sum(
        last_Next_biases * last_weights[None, :, :, :, None], axis=2)
    reduced_last_LPs = (tf.math.log(sum_last_P + 1e-100) + last_LP_max)[:, 0] \
        + nb_dims * tf.math.log(tf.math.abs(tf.reduce_prod(
            tf.linalg.diag_part(tf.transpose(reduced_last_Next_coefs, [1, 2, 0, 3])),
            axis=-1)) + 1e-20)
    reduced_last_segment_len = tf.reduce_sum(last_segment_len * last_weights, axis=1)
    reduced_last_gamma_dist_mean = tf.reduce_sum(
        last_gamma_dist_mean * last_weights[:, :, :, None], axis=1)
    reduced_last_gamma_dist_var = tf.reduce_sum(
        (last_gamma_dist_var + (last_gamma_dist_mean - reduced_last_gamma_dist_mean[:, None]) ** 2)
        * last_weights[:, :, :, None], axis=1)
    reduced_last_gamma_dist_mean = tf.reshape(reduced_last_gamma_dist_mean,
                                              (nb_tracks, nb_states ** 2))
    reduced_last_gamma_dist_var = tf.reshape(reduced_last_gamma_dist_var,
                                             (nb_tracks, nb_states ** 2))
    reduced_last_states = tf.reduce_sum(last_states * last_weights[:, :, :, None, None], axis=1)

    new_Next_coefs = tf.concat((saved_Next_coefs, reduced_last_Next_coefs), axis=2)
    new_Next_biases = tf.concat((saved_Next_biases, reduced_last_Next_biases), axis=2)
    new_LPs = tf.concat((saved_LP, reduced_last_LPs), axis=1)
    new_segment_len = tf.concat((saved_segment_len, reduced_last_segment_len), axis=1)
    new_gamma_dist_mean = tf.concat((saved_gamma_dist_mean, reduced_last_gamma_dist_mean), axis=1)
    new_gamma_dist_var = tf.concat((saved_gamma_dist_var, reduced_last_gamma_dist_var), axis=1)
    new_states = tf.concat((saved_states, reduced_last_states), axis=1)
    new_states = tf.concat((new_states, current_states), axis=2)[:, :, 1:]

    return (new_Next_coefs, new_Next_biases, new_LPs, new_segment_len,
            new_gamma_dist_mean, new_gamma_dist_var, new_states, new_lstm_states)


# ===========================================================================
#  Forecaster
# ===========================================================================
class MCRolloutForecaster:
    """
    Monte-Carlo multi-step forecaster for a trained LSTM segmented state-space
    model. Construct from the `pred_model` returned by `build_segment_model_LSTM`.
    """

    def __init__(self, pred_model, init_layer=None, rnn_layer=None):
        self.pred_model = pred_model
        # ---- fetch the custom layers (and the shared LSTM module) -----------
        if init_layer is None or rnn_layer is None:
            for l in pred_model.layers:
                if isinstance(l, Custom_RNN_LSTM_layer):
                    rnn_layer = l
                elif isinstance(l, Initial_LSTM_layer):
                    init_layer = l
        if init_layer is None or rnn_layer is None:
            raise ValueError("Could not locate Initial_LSTM_layer / "
                             "Custom_RNN_LSTM_layer inside pred_model. Pass them "
                             "explicitly via init_layer=, rnn_layer=.")
        self.init_layer = init_layer
        self.rnn_layer = rnn_layer
        self.lstm = rnn_layer.lstm_module

        L = self.lstm
        self.H = L.nb_hidden_vars
        self.O = L.nb_obs_vars
        self.idx = L.integration_variable_index
        self.S = L.nb_states
        self.seq = L.sequence_length
        self.nb_dims = L.nb_dims
        self.batch_size = L.batch_size
        self.reference_dt = float(init_layer.reference_dt)
        self.dtype = L.dtype

        # std bounds used inside assemble (must match to recover raw sigmas)
        self.LOG_STD_MIN = np.log(1e-10)
        self.LOG_STD_MAX = np.log(1e10)

    # ------------------------------------------------------------------ #
    #  input prep (mirrors build_segment_model_LSTM wiring)
    # ------------------------------------------------------------------ #
    def _prep_inputs(self, tracks):
        """tracks: (B, T, O, dims) tf -> (T, 1, B, 1, O, dims)."""
        x = tracks[:, None, :, None]
        return tf.transpose(x, [2, 1, 0, 3, 4, 5])

    # ------------------------------------------------------------------ #
    #  prefix filtering -> carriers at the forecast origin
    # ------------------------------------------------------------------ #
    def _filter_prefix(self, tracks, dts, masks):
        """
        tracks (B,Tp,O,dims), dts (B,Tp+1), masks (B,Tp). B must equal batch_size.
        Returns carriers (Prev_coefs, Prev_biases, LP, segment_len,
        gamma_dist_mean, gamma_dist_var, states) and lstm_states.
        """
        d = self.dtype
        tracks = tf.constant(tracks, dtype=d)
        dts = tf.constant(dts, dtype=d)
        masks = tf.constant(masks, dtype=d)
        isfirst = tf.ones((tracks.shape[0],), dtype=d)

        transposed = self._prep_inputs(tracks)
        _, init_states, lstm_states1 = self.init_layer(transposed, dts, isfirst)
        Prev_coefs, Prev_biases, LP = init_states

        sliced_inputs = transposed[1:]
        sliced_mask = masks[:, 1:]

        out = self.rnn_layer(sliced_inputs, dts, self.reference_dt, sliced_mask,
                             Prev_coefs, Prev_biases, LP, lstm_states1,
                             isfirst=isfirst)
        (Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var,
         _All_states, _All_coefs, _All_biases, _All_LPs, states, final_lstm) = out
        carriers = [Prev_coefs, Prev_biases, LP, segment_len,
                    gamma_dist_mean, gamma_dist_var, states]
        return carriers, final_lstm

    # ------------------------------------------------------------------ #
    #  Gamma-moment schedule for future steps (mirrors the layer)
    # ------------------------------------------------------------------ #
    def _future_gamma_schedule(self, future_dts_TN):
        """future_dts_TN: (horizon, B). Returns trans_mean (horizon,B,S^2),
        trans_var (horizon,B,S^2) for *new* segments at each future step."""
        d = self.dtype
        rl = self.rnn_layer
        ts = rl.transition_shapes
        tr = rl.transition_rates
        dts = tf.constant(future_dts_TN, dtype=d)
        shapes_full, rates_full = rl.transition_param_function(
            ts, tr, dts, self.reference_dt, d)
        oh_row = rl.oh_row
        oh_col = rl.oh_col
        rates_flat = tf.einsum('tnij,pi,pj->tnp', rates_full, oh_row, oh_col)
        shapes_flat = tf.einsum('ij,pi,pj->p', shapes_full, oh_row, oh_col)
        mean_full = shapes_flat[None, None] / rates_flat
        var_full = shapes_flat[None, None] / (rates_flat ** 2)
        # the cell consumes only the first S^2 candidates as the new-segment moments
        return mean_full[:, :, :self.S ** 2], var_full[:, :, :self.S ** 2]

    # ------------------------------------------------------------------ #
    #  emit: one LSTM step -> assembled factors g + raw KF blocks (numpy)
    # ------------------------------------------------------------------ #
    def _emit(self, carriers, lstm_states):
        L = self.lstm
        Prev_coefs, Prev_biases, LP, segment_len, gmean, gvar, _states = carriers
        features = L.featurize(Prev_coefs, Prev_biases, LP, segment_len, gmean, gvar)
        raw, new_lstm = L.step(features, lstm_states)
        g = L.assemble(raw, self.nb_dims)
        blocks = self._raw_blocks(raw)            # numpy generative blocks
        return g, new_lstm, blocks

    def _raw_blocks(self, raw):
        """Re-slice `raw` exactly like LSTM_constraint_function.assemble to recover
        the *generative* measurement and prior blocks (as numpy arrays).

        Returns dict with per-(N,S) numpy arrays:
            M       (N,S,O,H)   measurement-over-hidden coefficients
            P       (N,S,O,O)   unit-lower-tri measurement-over-obs coefficients
            b_meas  (N,S,O)     measurement bias
            sig_meas(N,S,O)     measurement std
            Lprior  (N,S,H,H)   unit-lower-tri prior
            init_std(N,S,H)
            init_bias(N,S,H)
        """
        L = self.lstm
        H, O = self.H, self.O
        n_sl_H, n_sl_O = L.n_sl_H, L.n_sl_O
        N = self.batch_size
        S = self.S

        rec = raw[..., :L.L_rec]
        ini = raw[..., L.L_rec:]

        o = 0
        o += H * H                                   # A_flat (skip)
        o += n_sl_H                                  # dyn_next_off (skip)
        meas_hidden = rec[..., o:o + O * H];  o += O * H
        meas_obs_off = rec[..., o:o + n_sl_O]; o += n_sl_O
        log_stds = rec[..., o:o + H + O];     o += H + O
        biases_p = rec[..., o:o + H + O];     o += H + O

        log_stds = self.LOG_STD_MIN + (self.LOG_STD_MAX - self.LOG_STD_MIN) * tf.math.sigmoid(log_stds)
        stds = tf.math.exp(log_stds)
        M = tf.reshape(meas_hidden, (N, S, O, H))
        P = L._unit_lower(meas_obs_off, O, L.SLO_idx, L.SLO_mask)
        sig_meas = stds[..., H:]
        b_meas = biases_p[..., H:]

        i0 = 0
        L_lower = ini[..., i0:i0 + n_sl_H]; i0 += n_sl_H
        log_init = ini[..., i0:i0 + H];     i0 += H
        init_bias = ini[..., i0:i0 + H];    i0 += H
        log_init = self.LOG_STD_MIN + (self.LOG_STD_MAX - self.LOG_STD_MIN) * tf.math.sigmoid(log_init)
        init_std = tf.math.exp(log_init)
        Lprior = L._unit_lower(L_lower, H, L.SL_idx, L.SL_mask)

        return dict(M=M.numpy(), P=P.numpy(), b_meas=b_meas.numpy(),
                    sig_meas=sig_meas.numpy(), Lprior=Lprior.numpy(),
                    init_std=init_std.numpy(), init_bias=init_bias.numpy())

    # ------------------------------------------------------------------ #
    #  carried predictive over the current hidden x (per hypothesis), numpy
    # ------------------------------------------------------------------ #
    def _carried_xpred(self, carriers, ridge=1e-12):
        Prev_coefs = carriers[0].numpy()      # (H, N, K, H)
        Prev_biases = carriers[1].numpy()     # (H, N, K, dims)
        H = self.H
        C = np.moveaxis(Prev_coefs, 0, -2)    # (N, K, H[gauss], H[var])
        b = np.moveaxis(Prev_biases, 0, -2)   # (N, K, H[gauss], dims)
        C = C + ridge * np.eye(H)[None, None]
        Cinv = np.linalg.inv(C)               # (N,K,H,H)
        mu = -np.einsum('nkgh,nkgd->nkhd', Cinv, b)            # (N,K,H,dims)
        Sigma = np.einsum('nkhg,nklg->nkhl', Cinv, Cinv)       # (N,K,H,H)
        return mu, Sigma

    # ------------------------------------------------------------------ #
    #  per-hypothesis next-state transition probabilities P(s'|k), numpy
    #  (mirrors the hazard computation inside the cell)
    # ------------------------------------------------------------------ #
    def _transition_probs(self, carriers, dt_ratio):
        S = self.S
        seq = self.seq
        gmean = carriers[4].numpy()           # (N, P) P = K*S
        gvar = carriers[5].numpy()
        seg = carriers[3].numpy()             # (N, K)
        N, P = gmean.shape
        shapes = gmean ** 2 / gvar
        rates = gmean / gvar
        seg_rep = np.repeat(seg, S, axis=1)   # (N,P)
        dr = np.asarray(dt_ratio).reshape(N, 1)
        S_old = gamma.sf(seg_rep + 1e-12, a=shapes, scale=1.0 / rates) + 1e-12
        S_new = gamma.sf(seg_rep + 1e-12 + dr, a=shapes, scale=1.0 / rates) + 1e-12
        hazard = np.clip(1.0 - S_new / S_old, 1e-20, 1 - 1e-10)   # (N,P)
        # candidate p -> (k=p//S, s'=p%S), current state c(k)=k%S
        k_of_p = np.arange(P) // S
        sp_of_p = np.arange(P) % S
        c_of_p = k_of_p % S
        mask_trans = (c_of_p != sp_of_p).astype(np.float64)[None]   # (1,P)
        # stay prob per k = 1 - sum_{s'!=c(k)} hazard
        hz_resh = (hazard * mask_trans).reshape(N, seq * S, S)
        stay = 1.0 - np.clip(np.sum(hz_resh, axis=2), 1e-20, 1 - 1e-10)  # (N, K)
        stay_full = np.repeat(stay, S, axis=1)                          # (N,P)
        Tprob = hazard * mask_trans + stay_full * (1 - mask_trans)      # (N,P)
        return Tprob, k_of_p, sp_of_p, c_of_p

    # ------------------------------------------------------------------ #
    #  sample one observation per particle from the one-step predictive
    # ------------------------------------------------------------------ #
    def _sample_y(self, carriers, blocks, dt_ratio, rng):
        """Returns y (N,O,dims), x (N,H,dims), s' (N,) sampled from the predictive."""
        N = self.batch_size
        H, O = self.H, self.O
        idx = self.idx
        D = self.nb_dims

        LP = carriers[2].numpy()                       # (N, K)
        mu, Sigma = self._carried_xpred(carriers)      # (N,K,H,dims),(N,K,H,H)
        Tprob, k_of_p, sp_of_p, c_of_p = self._transition_probs(carriers, dt_ratio)

        logw = LP[:, k_of_p] + np.log(Tprob)           # (N, P)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)

        M, P, b_meas, sig_meas = blocks['M'], blocks['P'], blocks['b_meas'], blocks['sig_meas']
        Lp, isd, ib = blocks['Lprior'], blocks['init_std'], blocks['init_bias']

        y_out = np.zeros((N, O, D))
        x_out = np.zeros((N, H, D))
        s_out = np.zeros((N,), dtype=np.int64)

        for n in range(N):
            p = rng.choice(w.shape[1], p=w[n])
            k = k_of_p[p]; sp = sp_of_p[p]; c = c_of_p[p]
            s_out[n] = sp

            # ---- sample hidden x (per dimension; Sigma shared across dims) ----
            x = np.zeros((H, D))
            if sp == c:
                Sig = Sigma[n, k] + 1e-12 * np.eye(H)
                Lc = np.linalg.cholesky(Sig)
                for dd in range(D):
                    x[:, dd] = mu[n, k, :, dd] + Lc @ rng.standard_normal(H)
            else:
                # carry 0..idx-1 from carried predictive marginal
                if idx > 0:
                    Sig_AA = Sigma[n, k, :idx, :idx] + 1e-12 * np.eye(idx)
                    Lc = np.linalg.cholesky(Sig_AA)
                    for dd in range(D):
                        x[:idx, dd] = mu[n, k, :idx, dd] + Lc @ rng.standard_normal(idx)
                # reset idx..H-1 from state sp prior, conditional on x[:idx]
                for h in range(idx, H):
                    coupling = Lp[n, sp, h, :h] @ x[:h, :]            # (D,)
                    base = -(coupling + ib[n, sp, h])
                    x[h, :] = base + isd[n, sp, h] * rng.standard_normal(D)

            # ---- sample observation y from state sp measurement given x ----
            y = np.zeros((O, D))
            for oo in range(O):
                coupling_y = P[n, sp, oo, :oo] @ y[:oo, :] if oo > 0 else 0.0
                coupling_x = M[n, sp, oo, :] @ x                      # (D,)
                base = -(coupling_y + coupling_x + b_meas[n, sp, oo])
                y[oo, :] = base + sig_meas[n, sp, oo] * rng.standard_normal(D)

            x_out[n] = x
            y_out[n] = y

        return y_out, x_out, s_out

    # ------------------------------------------------------------------ #
    #  EXACT enumerated one-step predictive over y (validation / fast head)
    # ------------------------------------------------------------------ #
    def one_step_predictive_y(self, tracks, dts, masks, dt_ratio=None):
        """Analytic mean/cov of y_{t+1} given the observed prefix, by enumerating
        the same predictive mixture the sampler draws from. Returns
        (y_mean (N,O,dims), y_cov (N,O,O), state_probs (N,S))."""
        tracks = self._as_BTOD(tracks)
        B = tracks.shape[0]
        dts = self._pad_dts(dts, tracks.shape[1])
        if masks is None:
            masks = np.ones((B, tracks.shape[1]))
        if dt_ratio is None:
            dt_ratio = np.ones((B,))
        carriers, lstm_states = self._filter_prefix(tracks, dts, masks)
        _g, _new_lstm, blocks = self._emit(carriers, lstm_states)

        H, O, idx, D, S = self.H, self.O, self.idx, self.nb_dims, self.S
        LP = carriers[2].numpy()
        mu, Sigma = self._carried_xpred(carriers)
        Tprob, k_of_p, sp_of_p, c_of_p = self._transition_probs(carriers, dt_ratio)
        logw = LP[:, k_of_p] + np.log(Tprob)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)         # (B,P)

        M, P, b_meas, sig_meas = blocks['M'], blocks['P'], blocks['b_meas'], blocks['sig_meas']
        Lp, isd, ib = blocks['Lprior'], blocks['init_std'], blocks['init_bias']

        Pn = w.shape[1]
        y_mean = np.zeros((B, O, D))
        y_sec = np.zeros((B, O, O, D))     # E[y y^T] accumulation per dim
        state_probs = np.zeros((B, S))
        for n in range(B):
            for p in range(Pn):
                wp = w[n, p]
                if wp <= 0:
                    continue
                k = k_of_p[p]; sp = sp_of_p[p]; c = c_of_p[p]
                state_probs[n, sp] += wp
                # component hidden mean/cov
                if sp == c:
                    mx = mu[n, k]                    # (H,D)
                    Sx = Sigma[n, k]                 # (H,H)
                else:
                    mx = np.zeros((H, D)); Sx = np.zeros((H, H))
                    if idx > 0:
                        mx[:idx] = mu[n, k, :idx]
                        Sx[:idx, :idx] = Sigma[n, k, :idx, :idx]
                    # x_B = F x_A + d + G eps   (rows idx..H-1 of prior)
                    L_BB = Lp[n, sp, idx:H, idx:H]
                    L_BB_inv = np.linalg.inv(L_BB)
                    L_BA = Lp[n, sp, idx:H, :idx] if idx > 0 else np.zeros((H - idx, 0))
                    F = -L_BB_inv @ L_BA if idx > 0 else np.zeros((H - idx, 0))
                    dvec = -(L_BB_inv @ ib[n, sp, idx:H])            # (H-idx,)
                    G = L_BB_inv @ np.diag(isd[n, sp, idx:H])
                    GGt = G @ G.T
                    for dd in range(D):
                        xA = mx[:idx, dd] if idx > 0 else np.zeros((0,))
                        mx[idx:, dd] = (F @ xA if idx > 0 else 0.0) + dvec
                    if idx > 0:
                        S_AA = Sx[:idx, :idx]
                        S_AB = S_AA @ F.T
                        Sx[:idx, idx:] = S_AB
                        Sx[idx:, :idx] = S_AB.T
                        Sx[idx:, idx:] = F @ S_AA @ F.T + GGt
                    else:
                        Sx[idx:, idx:] = GGt
                # push hidden -> obs:  y = A x + cvec + noise
                Pinv = np.linalg.inv(P[n, sp])
                A = -Pinv @ M[n, sp]                  # (O,H)
                cvec = -(Pinv @ b_meas[n, sp])        # (O,)
                Rn = Pinv @ np.diag(sig_meas[n, sp] ** 2) @ Pinv.T   # (O,O)
                my = np.zeros((O, D))
                for dd in range(D):
                    my[:, dd] = A @ mx[:, dd] + cvec
                Cy = A @ Sx @ A.T + Rn                # (O,O), shared across dims
                y_mean[n] += wp * my
                for dd in range(D):
                    y_sec[n, :, :, dd] += wp * (Cy + np.outer(my[:, dd], my[:, dd]))
        # mixture covariance: E[yy^T] - E[y]E[y]^T  (return diagonal-friendly)
        y_cov = np.zeros((B, O, O, D))
        for dd in range(D):
            for n in range(B):
                y_cov[n, :, :, dd] = y_sec[n, :, :, dd] - np.outer(y_mean[n, :, dd], y_mean[n, :, dd])
        return y_mean, y_cov, state_probs

    # ------------------------------------------------------------------ #
    #  helpers for shaping user inputs
    # ------------------------------------------------------------------ #
    def _as_BTOD(self, tracks):
        tracks = np.asarray(tracks, dtype=np.float64)
        if tracks.ndim == 3:                 # (B,T,dims) -> add O axis
            tracks = tracks[:, :, None, :]
        return tracks                        # (B,T,O,dims)

    def _pad_dts(self, dts, T):
        if dts is None:
            dts = np.full((1, T), self.reference_dt)
        dts = np.asarray(dts, dtype=np.float64)
        if dts.shape[1] == T:                # need T+1 for the model
            dts = np.concatenate([dts, dts[:, -1:]], axis=1)
        return dts

    # ------------------------------------------------------------------ #
    #  MAIN: MC rollout
    # ------------------------------------------------------------------ #
    def rollout(self, tracks, LocErrs, dts, masks,
                horizon, nb_particles=200, future_dts=None,
                quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
                return_samples=False, seed=0, verbose=True):
        """
        Parameters
        ----------
        tracks  : (nb_tracks, Tp, dims) or (nb_tracks, Tp, O, dims)
            observed prefix per track.
        LocErrs : same leading shape as tracks (unused by the LSTM front-end but
            kept for signature symmetry; pass zeros if you have none).
        dts     : (nb_tracks, Tp) or (nb_tracks, Tp+1) frame durations.
        masks   : (nb_tracks, Tp) or None (defaults to all-ones / fully observed).
        horizon : number of future steps to forecast.
        nb_particles : MC rollouts per track.
        future_dts : (nb_tracks, horizon) future frame durations, or scalar, or
            None (=> reference_dt).

        Returns dict of forecasts (see module docstring).
        """
        tracks = self._as_BTOD(tracks)
        nb_tracks, Tp = tracks.shape[0], tracks.shape[1]
        O, D = self.O, self.nb_dims
        H, S = self.H, self.S
        bs = self.batch_size

        dts = self._pad_dts(dts, Tp)
        if dts.shape[0] == 1 and nb_tracks > 1:
            dts = np.repeat(dts, nb_tracks, axis=0)
        if masks is None:
            masks = np.ones((nb_tracks, Tp))
        masks = np.asarray(masks, dtype=np.float64)

        # future dt ratios per track
        if future_dts is None:
            fdt = np.full((nb_tracks, horizon), self.reference_dt)
        elif np.isscalar(future_dts):
            fdt = np.full((nb_tracks, horizon), float(future_dts))
        else:
            fdt = np.asarray(future_dts, dtype=np.float64)
        fdt_ratio = fdt / self.reference_dt                       # (nb_tracks, horizon)

        # outputs
        y_mean = np.zeros((nb_tracks, horizon, O, D))
        y_std = np.zeros((nb_tracks, horizon, O, D))
        y_q = np.zeros((nb_tracks, horizon, len(quantiles), O, D))
        x_mean = np.zeros((nb_tracks, horizon, H, D))
        x_std = np.zeros((nb_tracks, horizon, H, D))
        state_probs = np.zeros((nb_tracks, horizon, S))
        all_samples = None
        if return_samples:
            all_samples = np.zeros((nb_tracks, nb_particles, horizon, O, D))

        n_chunks = int(np.ceil(nb_particles / bs))
        for t in range(nb_tracks):
            if verbose:
                print(f"[track {t+1}/{nb_tracks}] {n_chunks} chunk(s) "
                      f"x {bs} particles, horizon {horizon}")
            ys_track = np.zeros((n_chunks * bs, horizon, O, D))
            xs_track = np.zeros((n_chunks * bs, horizon, H, D))
            ss_track = np.zeros((n_chunks * bs, horizon), dtype=np.int64)

            # replicate this track across the batch (particles share the prefix)
            tr = np.repeat(tracks[t:t + 1], bs, axis=0)            # (bs,Tp,O,D)
            dt_pref = np.repeat(dts[t:t + 1], bs, axis=0)          # (bs,Tp+1)
            mk = np.repeat(masks[t:t + 1], bs, axis=0)             # (bs,Tp)
            future_ratio_TN = np.repeat(fdt_ratio[t:t + 1].T, bs, axis=1)  # (horizon,bs)
            trans_mean_full, trans_var_full = self._future_gamma_schedule(future_ratio_TN)
            trans_mean_full = trans_mean_full.numpy()              # (horizon,bs,S^2)
            trans_var_full = trans_var_full.numpy()

            for ch in range(n_chunks):
                rng = np.random.default_rng(seed + 100003 * t + ch)
                carriers, lstm_states = self._filter_prefix(tr, dt_pref, mk)

                for h in range(horizon):
                    g, new_lstm, blocks = self._emit(carriers, lstm_states)
                    dr = future_ratio_TN[h]                         # (bs,)
                    y_s, x_s, s_s = self._sample_y(carriers, blocks, dr, rng)

                    row0 = ch * bs
                    ys_track[row0:row0 + bs, h] = y_s
                    xs_track[row0:row0 + bs, h] = x_s
                    ss_track[row0:row0 + bs, h] = s_s

                    # feed the sampled observation back through the real cell
                    input_i = tf.constant(
                        y_s.reshape(1, bs, 1, O, D), dtype=self.dtype)  # (1,N,1,O,dims)
                    tmean = tf.constant(trans_mean_full[h], dtype=self.dtype)  # (bs,S^2)
                    tvar = tf.constant(trans_var_full[h], dtype=self.dtype)
                    drt = tf.constant(dr, dtype=self.dtype)

                    new = _cell_core(
                        g, new_lstm, input_i,
                        carriers[0], carriers[1], carriers[2], carriers[3],
                        self.lstm, self.rnn_layer.oh_row, self.rnn_layer.oh_col,
                        self.rnn_layer.transition_mask,
                        self.rnn_layer.sequence_phase_1, self.rnn_layer.sequence_phase_2,
                        self.rnn_layer.transition_sequence,
                        tmean, tvar, carriers[4], carriers[5], carriers[6], drt)
                    (nc, nb, nlp, nseg, ngm, ngv, nst, nlstm) = new
                    carriers = [nc, nb, nlp, nseg, ngm, ngv, nst]
                    lstm_states = nlstm

            ys = ys_track[:nb_particles]      # (nb_particles,horizon,O,D)
            xs = xs_track[:nb_particles]
            ss = ss_track[:nb_particles]

            y_mean[t] = ys.mean(axis=0)
            y_std[t] = ys.std(axis=0)
            y_q[t] = np.quantile(ys, quantiles, axis=0)            # (n_q,horizon,O,D) -> reorder
            y_q[t] = np.moveaxis(np.quantile(ys, quantiles, axis=0), 0, 1)
            x_mean[t] = xs.mean(axis=0)
            x_std[t] = xs.std(axis=0)
            for h in range(horizon):
                counts = np.bincount(ss[:, h], minlength=S).astype(np.float64)
                state_probs[t, h] = counts / counts.sum()
            if return_samples:
                all_samples[t] = ys

        out = dict(y_mean=y_mean, y_std=y_std, y_quantiles=y_q,
                   x_mean=x_mean, x_std=x_std, state_probs=state_probs,
                   quantiles=np.asarray(quantiles))
        if return_samples:
            out['y_samples'] = all_samples
        return out


# ===========================================================================
#  Example usage + horizon-1 sanity check
# ===========================================================================
if __name__ == '__main__':
    """
    Minimal smoke test. Replace the model construction with loading YOUR trained
    pred_model, and `tracks`/`dts`/`masks` with a real observed prefix.

    The sanity check confirms the MC one-step y mean/std matches the analytic
    enumerated one-step predictive (they should agree to MC error ~ 1/sqrt(N)).
    """
    # ----- you provide these -------------------------------------------------
    # from your_training_script import pred_model
    # tracks: (nb_tracks, Tp, nb_dims) observed prefix
    # dts:    (nb_tracks, Tp) frame durations (or None for reference_dt)
    # masks:  (nb_tracks, Tp) or None
    #
    # fc = MCRolloutForecaster(pred_model)
    #
    # # ---- horizon-1 validation ----
    # ym, yc, sp = fc.one_step_predictive_y(tracks, dts, masks)
    # mc = fc.rollout(tracks, np.zeros_like(tracks), dts, masks,
    #                 horizon=1, nb_particles=4000, return_samples=True, verbose=False)
    # print("analytic y_mean :", ym[:, 0])
    # print("MC       y_mean :", mc['y_mean'][:, 0])
    # print("analytic y_std  :", np.sqrt(yc[:, range(fc.O), range(fc.O)]))
    # print("MC       y_std  :", mc['y_std'][:, 0])
    #
    # # ---- multi-step forecast ----
    # out = fc.rollout(tracks, np.zeros_like(tracks), dts, masks,
    #                  horizon=20, nb_particles=300)
    # print("forecast mean shape:", out['y_mean'].shape)
    # print("forecast std  shape:", out['y_std'].shape)
    # print("state probs   shape:", out['state_probs'].shape)
    print("Module loaded. See the commented example in __main__ to run a forecast.")
#%% Additional algorithms

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


