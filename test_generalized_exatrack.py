# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:45:59 2026

@author: Franc
"""

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
#import exatrack_var_shape as exatrack
#import exatrack_var_shape as exatrack
import generalized_exatrack as exatrack
from glob import glob

track_len = 60
nb_tracks = 1000
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 1])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.05

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.5],   # State 0 -> State 1
                                [0.5, 0.00]]),
    shape_matrix=np.array([[0, 5],
                           [5, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.002,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 50

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(1), np.log(0.01), np.log(0.01), np.log(0.0002), 1],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.001), 0]])
nb_states = len(params)

initial_params = np.array([[np.log(60)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 3 * np.eye(nb_states, dtype='float64')
transition_rates[0,0] = 2
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = True

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 20

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.10
nb_batches
epochs = 50
epoch_decay = 30
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
verbose = 1

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
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates,
                LocErr_type = 'Linear')

#device = '/GPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

'''
Results
loss: -72.5056{'Model types': array(['Directed motion', 'Confined motion'], dtype='<U15'), 'anomalous factors': [0.0504, 0.23], 'Localization errors': [1.002, 1.18], 'd': [0.005, 0.193], 'anomalous variation': [8e-05, 0.00047], 'transition rates': [[0.898, 0.483], [0.564, 0.907]], 'transition shapes': [[1.0, 4.742], [6.054, 1.0]], 'Fractions': [0.38, 0.62, 0.0]}
60/60 [==============================] - 60s 998ms/step - loss: -72.5056
'''

'''
Then, we modify the constraint function
'''

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
#import exatrack_var_shape as exatrack
#import exatrack_var_shape as exatrack
import generalized_exatrack as exatrack
from glob import glob

track_len = 60
nb_tracks = 200
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 1               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 1])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.05

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.5],   # State 0 -> State 1
                                [0.5, 0.00]]),
    shape_matrix=np.array([[0, 5],
                           [5, 0]]),
    LocErr_std = 0.0001,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.0001,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

nb_obs_vars = 1
nb_hidden_vars = 2
nb_dims = 1
integration_variable_index = 1

nb_states = 2
initial_params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2) + 1
params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + 
                            nb_hidden_vars * nb_obs_vars + 2 * (nb_obs_vars + nb_hidden_vars)) + 1

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 3 * np.eye(nb_states, dtype='float64')
tf.math.softmax(transition_rates)
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0

vary_params = np.ones(params.shape)
vary_initial_params = np.ones(initial_params.shape)
vary_initial_fractions = np.ones(initial_fractions.shape)

vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = True
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

nb_dims = 1
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

all_inputs, outputs = seq[0]
inputs = all_inputs[0]
LocErrs = all_inputs[1]
dts = all_inputs[2]
dtype = 'float64'
input_mask = tf.constant(all_inputs[3], dtype = dtype)
input_isfirst = tf.constant(all_inputs[4], dtype = dtype)


class constraint_function:
    """
    Generic constraint function for a recurrent conditional-Gaussian process
    with an arbitrary number of hidden (H = nb_hidden_vars) and observed
    (O = nb_obs_vars) variables.
 
    It produces, for every (time, track, state) triple, the coefficients,
    biases and log-normalisers of the univariate Gaussian factors
 
        f_i( sum_j A[i, j] * var_j  +  bias_i )           (std_i == 1)
 
    that, taken together, encode the joint distribution
        p(observation_t, hidden_t, hidden_{t+1} | state_t).
 
    Three families of Gaussians are built (see the schematic the coefficients
    are taken from):
 
      * INITIAL gaussians (H of them, only at t = 0) -- the prior on the
        hidden state x_0.  Lower-triangular: initial gaussian k touches the
        current hidden variables x_0[0 .. k].
 
      * INTERNAL-STATE / dynamics gaussians (H of them per recurrent step) --
        each defines ONE hidden variable at t+1 from the current hidden
        variables.  Dynamics gaussian j has
            - a diagonal coefficient on the next variable x_{t+1}[j]
            - current-variable coefficients on x_t[H-1-j .. H-1]
              (the "anti-triangular" pattern of the schematic: predicting the
              first next variable depends on the last current variable, the
              last next variable depends on every current variable).
 
      * OBSERVATION / measurement gaussians (O of them per recurrent step) --
        each relates the observed variables to ALL current hidden variables
        (dense over x_t) and to the observed variables (O x O block), with no
        dependence on the next variables.
 
    Per recurrent step the gaussians are ordered  [dynamics_0 .. dynamics_{H-1},
    measurement_0 .. measurement_{O-1}], i.e. dynamics first, matching the
    schematic (gaussian 1..H then the measurement gaussian last).
 
    Coefficients AND biases are read from the parameter vectors; the output
    standard deviations are all 1 (the scale lives in the coefficients, so it
    is not duplicated in the std).
 
    Parameter packing
    -----------------
    all_initial_params[s] : length H*(H+1)/2
        lower-triangular prior coefficients, filled row by row
        (k = 0..H-1, m = 0..k).
 
    all_params[s] : length  H*(H+1)/2 + H*(O+2) + O**2, in order
        [ dyn_cur     : H*(H+1)/2   anti-triangular dynamics current coefs
          dyn_next    : H           diagonal next-variable coefs
          dyn_bias    : H           dynamics drift biases
          meas_hidden : O*H         measurement coefs over current hidden vars
          meas_obs    : O*O         measurement coefs over observed vars ]
 
    The only non-zero biases are the H dynamics drifts; initial and
    measurement biases are zero (no parameters are allocated for them).
    
    initial_hidden_vars
    <tf.Tensor: shape=(2, 50, 2, 2)
    
    hidden_vars
    <tf.Tensor: shape=(20, 3, 50, 2, 4)
    
    """
    
    def __init__(self, nb_hidden_vars=2, nb_obs_vars=1, integration_variable_index=1):
        self.nb_hidden_vars = nb_hidden_vars
        self.nb_obs_vars = nb_obs_vars
        self.integration_variable_index = integration_variable_index
  
    def __call__(self, *args, **kwargs):
        # allow calling the instance directly:  cf(params, ...)  -> cf.call(params, ...)
        return self.call(*args, **kwargs)
    
    # ------------------------------------------------------------------
    # static index maps (depend only on H and O, built once at trace time)
    # ------------------------------------------------------------------
    def _build_index_maps(self):
        H = self.nb_hidden_vars
 
        # initial / prior  L : lower triangular, row k touches columns 0..k
        L_idx = np.zeros((H, H), dtype=np.int32)
        L_mask = np.zeros((H, H), dtype=np.float64)
        p = 0
        for k in range(H):
            for m in range(k + 1):
                L_idx[k, m] = p
                L_mask[k, m] = 1.0
                p += 1
        n_L = p                                            # = H*(H+1)/2
 
        # dynamics current  Acur : row j touches columns H-1-j .. H-1
        A_idx = np.zeros((H, H), dtype=np.int32)
        A_mask = np.zeros((H, H), dtype=np.float64)
        q = 0
        for j in range(H):
            for kcol in range(H - 1 - j, H):
                A_idx[j, kcol] = q
                A_mask[j, kcol] = 1.0
                q += 1
        n_A = q                                            # = H*(H+1)/2
 
        return L_idx, L_mask, n_L, A_idx, A_mask, n_A
 
    #@tf.function(jit_compile=False)
    def call(self, all_params, all_initial_params, LocErrs, dts,
             nb_dims, reference_dt, LocErr_function, dtype):
        H = self.nb_hidden_vars
        O = self.nb_obs_vars
        idx = self.integration_variable_index
        n_g = O + H                                        # recurrent gaussians
        n_trans = H - idx                                  # transition gaussians
        eps = tf.constant(1e-20, dtype=dtype)
        
        all_params = tf.cast(all_params, dtype)
        all_initial_params = tf.cast(all_initial_params, dtype)
        S = all_params.shape[0]                            # nb_states (static)
        
        L_idx, L_mask, n_L, A_idx, A_mask, n_A = self._build_index_maps()
        L_mask = tf.constant(L_mask, dtype=dtype)
        A_mask = tf.constant(A_mask, dtype=dtype)
        
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
        # slice the parameter vector
        # --------------------------------------------------------------
        o0 = 0
        dyn_cur = all_params[:, o0:o0 + n_A];               o0 += n_A
        dyn_next = all_params[:, o0:o0 + H];                o0 += H
        dyn_bias = all_params[:, o0:o0 + H];                o0 += H
        meas_hidden = all_params[:, o0:o0 + O * H];         o0 += O * H
        meas_obs = all_params[:, o0:o0 + O * O];            o0 += O * O
        
        meas_hidden = tf.reshape(meas_hidden, (S, O, H))
        meas_obs = tf.reshape(meas_obs, (S, O, O))
        
        # coefficient matrices  (S, H, H) etc.
        Acur = tf.reshape(tf.gather(dyn_cur, A_idx.reshape(-1), axis=1),
                          (S, H, H)) * A_mask               # anti-triangular
        L = tf.reshape(tf.gather(all_initial_params, L_idx.reshape(-1), axis=1),
                       (S, H, H)) * L_mask                  # lower-triangular prior
        
        # --------------------------------------------------------------
        # recurrent hidden-variable coefficients, last axis = 2H
        #   [ current_0..H-1 , next_0..H-1 ]
        # --------------------------------------------------------------
        Dnext = tf.linalg.diag(dyn_next)                    # (S, H, H), row j = e_j * dyn_next[j]
        dyn_rows = tf.concat([Acur, Dnext], axis=-1)        # (S, H, 2H)
        
        meas_next = tf.zeros((S, O, H), dtype=dtype)
        meas_rows = tf.concat([meas_hidden, meas_next], axis=-1)   # (S, O, 2H)
 
        C = tf.concat([dyn_rows, meas_rows], axis=1)        # (S, n_g, 2H) dynamics first
        C = tf.transpose(C, [1, 0, 2])                      # (n_g, S, 2H)
        hidden_vars = tf.broadcast_to(
            C[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, 2 * H]))
        
        # recurrent observation coefficients, last axis = O
        obs_dyn = tf.zeros((H, S, O), dtype=dtype)
        obs_meas = tf.transpose(meas_obs, [1, 0, 2])        # (O, S, O)
        OBS = tf.concat([obs_dyn, obs_meas], axis=0)        # (n_g, S, O)
        obs_vars = tf.broadcast_to(
            OBS[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, O]))
        
        # recurrent biases (dynamics drift only), last axis = nb_dims
        Bdyn = tf.broadcast_to(dyn_bias[:, :, None], (S, H, nb_dims))
        Bmeas = tf.zeros((S, O, nb_dims), dtype=dtype)
        B = tf.concat([Bdyn, Bmeas], axis=1)                # (S, n_g, nb_dims)
        B = tf.transpose(B, [1, 0, 2])                      # (n_g, S, nb_dims)
        biases = tf.broadcast_to(
            B[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, nb_dims]))
        
        Gaussian_stds = tf.ones(
            tf.stack([track_len, n_g, nb_tracks, S, 1]), dtype=dtype)
        
        # --------------------------------------------------------------
        # initial gaussians (no time axis), last axis = H
        # --------------------------------------------------------------
        IL = tf.transpose(L, [1, 0, 2])                     # (H, S, H)
        initial_hidden_vars = tf.broadcast_to(
            IL[:, None, :, :], tf.stack([H, nb_tracks, S, H]))
        initial_obs_vars = tf.zeros(tf.stack([H, nb_tracks, S, O]), dtype=dtype)
        initial_Gaussian_stds = tf.ones(tf.stack([H, nb_tracks, S, 1]), dtype=dtype)
        initial_biases = tf.zeros(tf.stack([H, nb_tracks, S, nb_dims]), dtype=dtype)
                
        # --------------------------------------------------------------
        # transition gaussians (fresh prior on vars idx..H-1, re-injected at a
        # state change). Same lower-triangular prior rows as the initial ones.
        # --------------------------------------------------------------
        TL = tf.transpose(L[:, idx:H, :], [1, 0, 2])        # (n_trans, S, H)
        transition_hidden_vars = tf.broadcast_to(
            TL[None, :, None, :, :], tf.stack([track_len, n_trans, nb_tracks, S, H]))
        transition_Gaussian_stds = tf.ones(
            tf.stack([track_len, n_trans, nb_tracks, S, 1]), dtype=dtype)
        transition_biases = tf.zeros(
            tf.stack([track_len, n_trans, nb_tracks, S, nb_dims]), dtype=dtype)
        
        integration_variable_index = tf.constant(idx)
        
        # --------------------------------------------------------------
        # log-normalising factors (std == 1, so the normaliser is the log of
        # the diagonal/pivot coefficient of every gaussian)
        # --------------------------------------------------------------
        log_dyn = tf.reduce_sum(tf.math.log(tf.abs(dyn_next) + eps), axis=-1)        # (S,)
        obs_diag = tf.linalg.diag_part(meas_obs)                                     # (S, O)
        log_obs = tf.reduce_sum(tf.math.log(tf.abs(obs_diag) + eps), axis=-1)        # (S,)
        rec_state = log_dyn + log_obs                                                # (S,)
        Log_factors = tf.broadcast_to(rec_state[None, None, :],
                                      tf.stack([track_len, nb_tracks, S]))
        
        L_diag = tf.linalg.diag_part(L)                                              # (S, H)
        log_L = tf.math.log(tf.abs(L_diag) + eps)
        init_norm_all = tf.reduce_sum(log_L, axis=-1)                                # (S,)
        init_norm_reset = tf.reduce_sum(log_L[:, idx:H], axis=-1)                    # (S,)
        
        initial_Log_factors = Log_factors[0] + init_norm_all[None, :]                # (N, S)
        transition_Log_factors = Log_factors + init_norm_reset[None, None, :]        # (T, N, S)
        
        return (hidden_vars, obs_vars, Gaussian_stds, biases,
                initial_hidden_vars, initial_obs_vars,
                initial_Gaussian_stds, initial_biases,
                transition_hidden_vars, transition_Gaussian_stds,
                transition_biases, integration_variable_index,
                Log_factors, initial_Log_factors, transition_Log_factors)

self = constraint_function(2,1,1)
class constraint_function:
    """
    Generic constraint function for a recurrent conditional-Gaussian process
    with an arbitrary number of hidden (H = nb_hidden_vars) and observed
    (O = nb_obs_vars) variables.
 
    It produces, for every (time, track, state) triple, the coefficients,
    biases, STANDARD DEVIATIONS and log-normalisers of the univariate Gaussian
    factors
 
        f_i( sum_j A[i, j] * var_j  +  bias_i )           (std_i learnable)
 
    that, taken together, encode the joint distribution
        p(observation_t, hidden_t, hidden_{t+1} | state_t).
 
    Difference from the original ("scale-in-the-coefficient") version
    -----------------------------------------------------------------
    The pivot (diagonal) coefficients of the dynamics and measurement
    gaussians are now FIXED to 1 (identity blocks) and the scale of those
    gaussians is carried by explicit, learnable standard deviations instead:
 
        * dyn_next   ->  identity  (next-variable block = I_H),   scale -> std_dyn
        * meas_obs   ->  identity  (observed-variable block = I_O), scale -> std_meas
 
    A factor with pivot 1 and std sigma is equivalent to one with pivot
    c = 1/sigma and std 1, so the only knock-on effect is the sign of the
    per-gaussian log-normaliser:  log|c|  ->  -log(sigma).
 
    Three families of Gaussians are built (see the schematic the coefficients
    are taken from):
 
      * INITIAL gaussians (H of them, only at t = 0) -- the prior on the
        hidden state x_0.  Lower-triangular: initial gaussian k touches the
        current hidden variables x_0[0 .. k].  (Unchanged: the prior scale
        still lives in diag(L), std == 1.)
 
      * INTERNAL-STATE / dynamics gaussians (H of them per recurrent step) --
        each defines ONE hidden variable at t+1 from the current hidden
        variables.  Dynamics gaussian j has
            - a UNIT (identity) coefficient on the next variable x_{t+1}[j]
            - current-variable coefficients on x_t[H-1-j .. H-1]
              (the "anti-triangular" pattern of the schematic)
            - a learnable standard deviation std_dyn[j] (the process noise).
 
      * OBSERVATION / measurement gaussians (O of them per recurrent step) --
        each relates the observed variables to ALL current hidden variables
        (dense over x_t) and to the observed variables through the IDENTITY
        (O x O identity block), with no dependence on the next variables and a
        learnable standard deviation std_meas[i] (the measurement noise).
 
    Per recurrent step the gaussians are ordered  [dynamics_0 .. dynamics_{H-1},
    measurement_0 .. measurement_{O-1}], i.e. dynamics first, matching the
    schematic (gaussian 1..H then the measurement gaussian last).
 
    Parameter packing
    -----------------
    all_initial_params[s] : length H*(H+1)/2
        lower-triangular prior coefficients, filled row by row
        (k = 0..H-1, m = 0..k).
 
    all_params[s] : length  H*(H+1)/2 + H*(O+2) + O, in order
        [ dyn_cur     : H*(H+1)/2   anti-triangular dynamics current coefs
          dyn_bias    : H           dynamics drift biases
          meas_hidden : O*H         measurement coefs over current hidden vars
          std_dyn     : H           dynamics gaussian standard deviations
          std_meas    : O           measurement gaussian standard deviations ]
 
    (Relative to the original packing, dyn_next[H] and meas_obs[O*O] are gone
    and std_dyn[H] and std_meas[O] take their place, so the total length only
    changes by O**2 -> O.)
 
    The only non-zero biases are the H dynamics drifts; initial and
    measurement biases are zero (no parameters are allocated for them).
    Standard deviations are read directly from the parameter vector and made
    positive with abs(.) + eps; swap in tf.math.softplus / tf.exp if you
    prefer an unconstrained (log-std) parameterisation.
 
    initial_hidden_vars
    <tf.Tensor: shape=(2, 50, 2, 2)
 
    hidden_vars
    <tf.Tensor: shape=(20, 3, 50, 2, 4)
    
    all_params = params
    all_initial_params = initial_params
    """
 
    def __init__(self, nb_hidden_vars=2, nb_obs_vars=1, integration_variable_index=1):
        self.nb_hidden_vars = nb_hidden_vars
        self.nb_obs_vars = nb_obs_vars
        self.integration_variable_index = integration_variable_index
 
    def __call__(self, *args, **kwargs):
        # allow calling the instance directly:  cf(params, ...)  -> cf.call(params, ...)
        return self.call(*args, **kwargs)
 
    # ------------------------------------------------------------------
    # static index maps (depend only on H and O, built once at trace time)
    # ------------------------------------------------------------------
    def _build_index_maps(self):
        H = self.nb_hidden_vars
 
        # initial / prior  L : lower triangular, row k touches columns 0..k
        L_idx = np.zeros((H, H), dtype=np.int32)
        L_mask = np.zeros((H, H), dtype=np.float64)
        p = 0
        for k in range(H):
            for m in range(k + 1):
                L_idx[k, m] = p
                L_mask[k, m] = 1.0
                p += 1
        n_L = p                                            # = H*(H+1)/2
 
        # dynamics current  Acur : row j touches columns H-1-j .. H-1
        A_idx = np.zeros((H, H), dtype=np.int32)
        A_mask = np.zeros((H, H), dtype=np.float64)
        q = 0
        for j in range(H):
            for kcol in range(H - 1 - j, H):
                A_idx[j, kcol] = q
                A_mask[j, kcol] = 1.0
                q += 1
        n_A = q                                            # = H*(H+1)/2
 
        return L_idx, L_mask, n_L, A_idx, A_mask, n_A
 
    #@tf.function(jit_compile=False)
    def call(self, all_params, all_initial_params, LocErrs, dts,
             nb_dims, reference_dt, LocErr_function, dtype):
        
        H = self.nb_hidden_vars
        O = self.nb_obs_vars
        idx = self.integration_variable_index
        n_g = O + H                                        # recurrent gaussians
        n_trans = H - idx                                  # transition gaussians
        eps = tf.constant(1e-20, dtype=dtype)
 
        all_params = tf.cast(all_params, dtype)
        all_initial_params = tf.cast(all_initial_params, dtype)
        S = all_params.shape[0]                            # nb_states (static)
 
        L_idx, L_mask, n_L, A_idx, A_mask, n_A = self._build_index_maps()
        L_mask = tf.constant(L_mask, dtype=dtype)
        A_mask = tf.constant(A_mask, dtype=dtype)
 
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
        # slice the parameter vector
        #   dyn_next  (pivot of dynamics)   -> fixed identity, see Dnext below
        #   meas_obs  (pivot of measurement)-> fixed identity, see meas_obs below
        #   std_dyn / std_meas now carry the scale of those gaussians
        # --------------------------------------------------------------
        o0 = 0
        dyn_cur = all_params[:, o0:o0 + n_A];               o0 += n_A
        meas_hidden = all_params[:, o0:o0 + O * H];         o0 += O * H
        stds = all_params[:, o0:o0 + H + O];                 o0 += H + O
        biases = all_params[:, o0:o0 + H + O];                o0 += H + O
        
        meas_hidden = tf.reshape(meas_hidden, (S, O, H))
        
        # positive standard deviations (scale of the dynamics / measurement gaussians)
        stds = tf.math.exp(stds) + eps                     # (S, H + O)
        
        # coefficient matrices  (S, H, H) etc.
        Acur = tf.reshape(tf.gather(dyn_cur, A_idx.reshape(-1), axis=1),
                          (S, H, H)) * A_mask               # anti-triangular
        L = tf.reshape(tf.gather(all_initial_params, L_idx.reshape(-1), axis=1),
                       (S, H, H)) * L_mask                  # lower-triangular prior
        
        # --------------------------------------------------------------
        # recurrent hidden-variable coefficients, last axis = 2H
        #   [ current_0..H-1 , next_0..H-1 ]
        # next-variable block is now the identity (scale moved to std_dyn)
        # --------------------------------------------------------------
        dyn_next_coefs = - tf.eye(H, batch_shape=[S], dtype=dtype)     # (S, H, H) identity next block
        dyn_rows = tf.concat([Acur, dyn_next_coefs], axis=-1)        # (S, H, 2H)
        
        meas_next = tf.zeros((S, O, H), dtype=dtype)
        meas_rows = tf.concat([meas_hidden, meas_next], axis=-1)   # (S, O, 2H)
 
        C = tf.concat([dyn_rows, meas_rows], axis=1)        # (S, n_g, 2H) dynamics first
        C = tf.transpose(C, [1, 0, 2])                      # (n_g, S, 2H)
        hidden_vars = tf.broadcast_to(
            C[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, 2 * H]))
        
        # recurrent observation coefficients, last axis = O
        # observed-variable block is now the identity (scale moved to std_meas)
        meas_obs = - tf.eye(O, batch_shape=[S], dtype=dtype)  # (S, O, O) identity obs block
        obs_dyn = tf.zeros((H, S, O), dtype=dtype)
        obs_meas = tf.transpose(meas_obs, [1, 0, 2])        # (O, S, O)
        OBS = tf.concat([obs_dyn, obs_meas], axis=0)        # (n_g, S, O)
        obs_vars = tf.broadcast_to(
            OBS[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, O]))
 
        # recurrent biases (dynamics drift only), last axis = nb_dims
        biases = tf.broadcast_to(biases[:, :, None], (S, H + O, nb_dims))
        biases = tf.transpose(biases, [1, 0, 2])                      # (n_g, S, nb_dims)
        biases = tf.broadcast_to(biases[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, nb_dims]))
        
        # --------------------------------------------------------------
        # recurrent standard deviations (variable now), last axis = 1
        #   [ std_dyn_0..H-1 , std_meas_0..O-1 ]  (dynamics first, matches C)
        # --------------------------------------------------------------
        rec_std = tf.transpose(stds, [1, 0])             # (n_g, S)
        Gaussian_stds = tf.broadcast_to(
            rec_std[None, :, None, :, None],
            tf.stack([track_len, n_g, nb_tracks, S, 1]))
        
        # --------------------------------------------------------------
        # initial gaussians (no time axis), last axis = H
        # (unchanged: prior scale stays in diag(L), std == 1)
        # --------------------------------------------------------------
        IL = tf.transpose(L, [1, 0, 2])                     # (H, S, H)
        initial_hidden_vars = tf.broadcast_to(
            IL[:, None, :, :], tf.stack([H, nb_tracks, S, H]))
        initial_obs_vars = tf.zeros(tf.stack([H, nb_tracks, S, O]), dtype=dtype)
        initial_Gaussian_stds = tf.ones(tf.stack([H, nb_tracks, S, 1]), dtype=dtype)
        initial_biases = tf.zeros(tf.stack([H, nb_tracks, S, nb_dims]), dtype=dtype)
        
        # --------------------------------------------------------------
        # transition gaussians (fresh prior on vars idx..H-1, re-injected at a
        # state change). Same lower-triangular prior rows as the initial ones.
        # (unchanged: std == 1)
        # --------------------------------------------------------------
        TL = tf.transpose(L[:, idx:H, :], [1, 0, 2])        # (n_trans, S, H)
        transition_hidden_vars = tf.broadcast_to(
            TL[None, :, None, :, :], tf.stack([track_len, n_trans, nb_tracks, S, H]))
        transition_Gaussian_stds = tf.ones(
            tf.stack([track_len, n_trans, nb_tracks, S, 1]), dtype=dtype)
        transition_biases = tf.zeros(
            tf.stack([track_len, n_trans, nb_tracks, S, nb_dims]), dtype=dtype)
        
        integration_variable_index = tf.constant(idx)
 
        # --------------------------------------------------------------
        # log-normalising factors
        #   recurrent gaussians: pivot coefficients are fixed to 1, so the
        #   per-gaussian normaliser is  -log(std)  (the scale now lives in std).
        #   initial / transition gaussians: pivot still lives in diag(L)
        #   (std == 1), so their normaliser is  log|L_diag|  as before.
        # --------------------------------------------------------------
        #log_dyn = -tf.reduce_sum(tf.math.log(std_dyn), axis=-1)        # (S,)
        #log_obs = -tf.reduce_sum(tf.math.log(std_meas), axis=-1)       # (S,)
        rec_logs = -tf.reduce_sum(tf.math.log(rec_std), axis=0) #log_dyn + log_obs                                  # (S,)
        Log_factors = tf.broadcast_to(rec_logs[None, None, :],
                                      tf.stack([track_len, nb_tracks, S]))
        
        L_diag = tf.linalg.diag_part(L)                                              # (S, H)
        log_L = tf.math.log(tf.abs(L_diag) + eps)
        init_norm_all = tf.reduce_sum(log_L, axis=-1)                                # (S,)
        init_norm_reset = tf.reduce_sum(log_L[:, idx:H], axis=-1)                    # (S,)
        
        initial_Log_factors = Log_factors[0] + init_norm_all[None, :]                # (N, S)
        transition_Log_factors = Log_factors + init_norm_reset[None, None, :]        # (T, N, S)
        
        return (hidden_vars, obs_vars, Gaussian_stds, biases,
                initial_hidden_vars, initial_obs_vars,
                initial_Gaussian_stds, initial_biases,
                transition_hidden_vars, transition_Gaussian_stds,
                transition_biases, integration_variable_index,
                Log_factors, initial_Log_factors, transition_Log_factors)

current_constraint_function = constraint_function(nb_hidden_vars, nb_obs_vars, integration_variable_index)
#self = current_constraint_function
current_constraint_function(params, initial_params, LocErrs, dts,
         nb_dims, reference_dt, 0, dtype)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.005
nb_batches
epochs = 150
epoch_decay = 80
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
verbose = 1

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
                current_constraint_function = current_constraint_function,
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates,
                nb_LocErr_dims = 1,
                LocErr_type = 'Linear')

#device = '/GPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.995, clipvalue=0.1) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])



weights = model.get_weights()

max_track_len = np.max([len(track) for track in track_list])
seq_full = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=max_track_len,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

_, pred_model = exatrack.build_segment_model(max_track_len, # maximum number of time points in the input tracks
                nb_states, # Number of states of their model
                params = weights[0], # recurrent parameters of the model
                initial_params = weights[1], # initial parameters of the model
                transition_rates = weights[6], # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes = weights[7], # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions = weights[2],
                batch_size = batch_size, # number of tracks analysed at the same time
                reference_dt = reference_dt,
                nb_dims = nb_dims, # Number of dimensions of the tracks
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                current_constraint_function = current_constraint_function,
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates,
                nb_LocErr_dims = 1,
                LocErr_type = 'Linear')

exatrack.get_model_params(model, track_segmentation=True)

LPs, preds, All_coefs, All_biases, All_LPs = pred_model.predict(seq_full)

tracks = np.concatenate([seq_full[i][0][0] for i in range(len(seq_full))], 0)
LocErrs = np.concatenate([seq_full[i][0][1] for i in range(len(seq_full))], 0)
time_steps = np.concatenate([seq_full[i][0][2] for i in range(len(seq_full))], 0)
masks = np.concatenate([seq_full[i][0][3] for i in range(len(seq_full))], 0)


'''
Plot tracks with state predictions
'''

colors = np.array([[1,0,0],
                   [0,0,1]])

plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = p@colors, s = 7)
        plt.scatter(track[0,0], track[0,1] , c = 'k', s = 3, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')


plt.figure(figsize = (15, 15))
plt.title('ExaTrack state predictions')
lim = 2.2 # MreB
nb_rows = 6
#IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        mask = masks[ID]
        track = tracks[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i]]
        p = preds[ID, mask.astype(bool)]
        p = np.clip(p, 0, 1)
        plt.plot(np.arange(len(track))*0.01 + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*0.01 + lim*j, track[:,0] , c = p@colors, s = 7)
plt.gca().set_aspect('equal', adjustable='box')

'''
Comparison to exatrack fitting
'''

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
#import exatrack_var_shape as exatrack
#import exatrack_var_shape as exatrack
import exatrack
from glob import glob

track_len = 60
nb_tracks = 200
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 1               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 1])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.05

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.5],   # State 0 -> State 1
                                [0.5, 0.00]]),
    shape_matrix=np.array([[0, 5],
                           [5, 0]]),
    LocErr_std = 0.0001,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.0001,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(1), np.log(0.01), np.log(0.01), np.log(0.0002), 1],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.001), 0]])
nb_states = len(params)

initial_params = np.array([[np.log(60)]]*nb_states, dtype='float64')

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
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = True

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 1
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.05
nb_batches
epochs = 50
epoch_decay = 30
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
verbose = 1

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
                LocErr_type = 'Linear')

device = '/GPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters(track_segmentation = True)], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])




'''
Inferring the hidden real positions and velocity vector
'''

motion_types = list(params[:, 4]) + [0]
position_mean, position_std, anomalous_mean, anomalous_std, mean_preds = exatrack.extract_smooth_hidden_variables(tracks, LocErrs, time_steps, masks, pred_model, batch_size, sequence_length, motion_types, reference_dt)

for track_ID in range(3):
    plt.figure()
    mask = masks[track_ID, 1:-1].astype(bool)
    plt.title('Example of refined positions, track %s'%track_ID)
    plt.errorbar(np.arange(len(position_mean[track_ID,mask,0])), position_mean[track_ID,mask,0] - np.mean(position_mean[track_ID,mask,0]), yerr = position_std[track_ID,mask,0])
    plt.errorbar(np.arange(len(position_mean[track_ID,mask,1])), position_mean[track_ID,mask,1] - np.mean(position_mean[track_ID,mask,1]), yerr = position_std[track_ID,mask,1])
    plt.xlim([-1, np.sum(mask)])
    plt.ylabel('Position')
    plt.xlabel('Time point')
    plt.legend(['x', 'y'])

track_ID = 0
directed_state_ID = 0
mask = masks[track_ID, 1:-1].astype(bool)
plt.figure()
plt.title('Velocity assuming directed state, track %s'%track_ID)
plt.plot((anomalous_mean[track_ID, mask,directed_state_ID, 0]**2 + anomalous_mean[track_ID,mask,directed_state_ID,1]**2)**0.5)
plt.xlabel('Time step')
plt.ylabel('Estimated velocity (um/time step)')

track_ID = 0
mask = masks[track_ID,].astype(bool)
plt.figure()
plt.title('state probabilities, track %s'%track_ID)
plt.plot(preds[track_ID,mask])
plt.xlabel('Time step')
plt.ylabel('Estimated velocity (um/time step)')
plt.legend(['State 0', 'State 1', 'Misslinking'])

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

# Refined positions
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





'''
Determining the number of states
'''

track_len = 50
nb_tracks = 50
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.000025, 0.25])

dt = 0.02
ds = (2*Ds*dt)**0.5 
velocity = 0.005

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition( 
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.3, 0.3, 0.4]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0, 0.2]),
    conf_Ds=np.array([0.0, 0, 0.0]),         # No diffusion of confinement center 
    conf_dists=np.array([0.0,0, 0.0]),
    transition_matrix=np.array([[0.00, 0, 0.02],   # State 0 -> State 2
                                [0.06, 0, 0.03],
                                [0.0, 0.05, 0]]),
    shape_matrix=np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]), 
    dt=dt, 
    dt_std = 0.001,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0, 
    bleaching_rate = 0.02) 

# Plot tracks
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
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 20

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(1), np.log(0.001), np.log(0.0001), np.log(0.0002), 1],
                   [np.log(1), np.log(0.001), np.log(0.1), np.log(0.0002), 0],
                   [np.log(1), np.log(0.001), np.log(0.005), np.log(0.0002), 1],
                   [np.log(1), np.log(0.003), np.log(0.1), np.log(0.001), 0],                   [np.log(0.025), np.log(0.01), np.log(0.01), np.log(0.002), 1],
                   [np.log(1), np.log(0.01), np.log(0.1), np.log(0.002), 0],
                   [np.log(1), np.log(0.05), np.log(0.05), np.log(0.002), 1],
                   [np.log(1), np.log(0.05), np.log(0.05), np.log(0.002), 0],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.0002), 1],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.0002), 0]])

nb_states = len(params)

initial_params = np.array([[np.log(60)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 3 * np.eye(nb_states, dtype='float64')
'''
transition_rates[0,1] = -5
transition_rates[1,0] = -7
'''
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')
tf.math.softmax(transition_rates, 1)

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 10

seq = exatrack.TrackSegmentSequence(track_list,
                                    LocErr_list=LocErr_list, 
                                    dt_list=dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.01
nb_batches
epochs = 50
epoch_decay = 13
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
verbose = 1
LocErr_type = 'Linear'

model_results = exatrack.get_number_of_states(track_list,
                         params,
                         initial_params,
                         transition_shapes,
                         transition_rates,
                         initial_fractions,
                         reference_dt,
                         dt_list = dt_list,
                         LocErr_list = LocErr_list,
                         nb_dims = 2,
                         sequence_length = 10,
                         max_linking_distance = 0.4,
                         estimated_density = 0.001,
                         epochs = epochs,
                         epoch_decay = epoch_decay,
                         learning_rate = 0.02,
                         decay_rate = 0.005,
                         batch_size = batch_size,
                         vary_params = True,
                         vary_initial_params = True,
                         vary_initial_fractions = True,
                         vary_transition_shapes = False,
                         vary_transition_rates = True,
                         device = device,
                         track_segmentation = True,
                         segment_length = 10,
                         LocErr_type = 'Linear')


log_likelihoods = [model_results[i]['log_likelihood'].numpy() for i in range(1, 11)]

model_results[3]

plt.figure()
plt.plot(np.arange(1, 11), log_likelihoods)
exatrack.equilibrium_distribution(np.array([[0.98, 0, 0.02],   # State 0 -> State 2
                            [0.06, 0.91, 0.03],
                            [0.0, 0.05, 0.95]]))


'''
Using Model_finder to determine if each state is rather confined or directed for a given number of states 
'''

track_len = 60
nb_tracks = 200
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 1])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.005

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.25],   # State 0 -> State 1
                                [0.25, 0.00]]),
    shape_matrix=np.array([[0, 5],
                           [5, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.002,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.0001)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 2 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')

track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]

batch_size = 50

# Prepare parameters for a 4 states model

# Initialize with generic guesses (Localization error, diffusion length, anomalous parameter, change of anomalous parameter)
# Here, we are going to run the model with LocErr_type = 'Linear' so the localization error parameter should be initialized at 1

params = np.array([[np.log(1), np.log(0.01), np.log(0.01), np.log(0.0002), 0],
                   [np.log(1), np.log(0.1), np.log(0.1), np.log(0.001), 0]])
nb_states = len(params)

initial_params = np.array([[np.log(60)]]*nb_states, dtype='float64')

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
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')#-7

# we fix the localization error of the two bound states otherwise the short lived state does not appear to the profit of two long lived states with different localization errors
vary_params = np.ones(params.shape)
# If we are confident we our estimations of the localization error we can fix it:
# vary_params[:, 0] = 0
vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = True

# We prevent transitions between the two bound states to improve readability
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
device = '/CPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 2
sequence_length = 5
max_linking_distance = 1
segment_length = 20
epochs = 60

model, pred_model = exatrack.Model_finder(track_list,
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
                 LocErr_list = LocErr_list,
                 dt_list = dt_list,
                 segment_length = 10,
                 learning_rate = 1/20,
                 decay_fraction = 0.2,
                 decay_rate = 0.002,
                 device = '/GPU:0', 
                 verbose = 1,       
                 shuffle = False,
                 vary_params = vary_params,
                 vary_initial_params = vary_initial_params,
                 vary_initial_fractions = vary_initial_fractions,
                 vary_transition_shapes = vary_transition_shapes,
                 vary_transition_rates = vary_transition_rates,
                 LocErr_type = 'Linear')