# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:45:59 2026

@author: Franc

We removed the misslinking state from ExaTrack and modified the constraint function
so it can match any arbitrary state space model
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
nb_tracks = 200
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 1               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.16])
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
nb_hidden_vars = 3
nb_dims = 1
integration_variable_index = 1

nb_states = 2
initial_params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + nb_hidden_vars)*4 - 2
params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + 
                            nb_hidden_vars * nb_obs_vars + 2 * (nb_obs_vars + nb_hidden_vars) + 20)*4 - 2

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states], dtype='float64')

# Transition matrices

# Transition matrices
transition_rates = 2 * np.eye(nb_states, dtype='float64')
tf.math.softmax(transition_rates)/3
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
vary_transition_rates = False
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)

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
    params = np.random.rand(nb_states, nb_hidden_vars**2 + nb_hidden_vars * (nb_hidden_vars+1)//2 + 
                                nb_hidden_vars * nb_obs_vars + 2 * (nb_obs_vars + nb_hidden_vars)) + 1

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
        
        nb_hidden_vars**2 + (nb_hidden_vars + 1) * nb_hidden_vars //2 + (nb_obs_vars + 1) * nb_obs_vars //2 +  nb_hidden_vars * nb_obs_vars
        
        
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


class constraint_function_KF_diag_Q_R:
    """
    Generic constraint function for a recurrent conditional-Gaussian process
    with an arbitrary number of hidden (H = nb_hidden_vars) and observed
    (O = nb_obs_vars) variables. This implementation is aimed to be strictly
    equivalent to a state space model with diagonal process and measurement noises.
 
    It produces, for every (time, track, state) triple, the coefficients,
    biases, STANDARD DEVIATIONS and log-normalisers of the univariate Gaussian
    factors
 
        f_i( sum_j A[i, j] * var_j  +  bias_i )           (std_i learnable)
 
    that, taken together, encode the joint distribution
        p(observation_t, hidden_t, hidden_{t+1} | state_t).
 
    Scale convention ("identity pivot + explicit std")
    ---------------------------------------------------
    The pivot (diagonal) coefficient of EVERY gaussian -- dynamics,
    measurement, AND the initial / transition priors -- is FIXED to a constant
    (+-1) and the scale of the gaussian is carried by an explicit, learnable
    standard deviation.  A factor with unit pivot and std sigma is equivalent
    to one with pivot 1/sigma and std 1, so the per-gaussian log-normaliser is
    simply  -log(sigma).
 
        * dyn_next   -> -I_H  (next-variable block),       scale -> std_dyn
        * meas_obs   -> -I_O  (observed-variable block),   scale -> std_meas
        * prior diag -> +I_H  (diagonal of L, see prior_diag), scale -> init_std
 
    Three families of Gaussians are built (see the schematic the coefficients
    are taken from):
 
      * INITIAL gaussians (H of them, only at t = 0) -- the prior on the
        hidden state x_0.  Lower-triangular: initial gaussian k touches the
        current hidden variables x_0[0 .. k].  The diagonal coefficient is
        fixed to prior_diag (== 1); only the strictly-lower off-diagonals are
        learned.  Each has a learnable std (init_std[k]) and bias
        (init_bias[k]).
 
      * INTERNAL-STATE / dynamics gaussians (H of them per recurrent step) --
        each defines ONE hidden variable at t+1 from the current hidden
        variables.  Dynamics gaussian j has
            - a UNIT (negated identity) coefficient on the next variable
              x_{t+1}[j]
            - current-variable coefficients on x_t[H-1-j .. H-1]
              (the "anti-triangular" pattern of the schematic)
            - a learnable standard deviation std_dyn[j] (process noise).
 
      * OBSERVATION / measurement gaussians (O of them per recurrent step) --
        each relates the observed variables to ALL current hidden variables
        (dense over x_t) and to the observed variables through the (negated)
        IDENTITY (O x O block), with no dependence on the next variables and a
        learnable standard deviation std_meas[i] (measurement noise).
 
    Per recurrent step the gaussians are ordered  [dynamics_0 .. dynamics_{H-1},
    measurement_0 .. measurement_{O-1}], i.e. dynamics first.
 
    Transition gaussians (re-injected at a state change) reuse the SAME prior
    rows idx..H-1 of L, and therefore the SAME std / bias slice init_std[idx:H]
    / init_bias[idx:H] as the corresponding initial gaussians.
 
    Parameter packing
    -----------------
    all_initial_params[s] : length  H*(H-1)/2 + 2*H, in order
        [ L_lower   : H*(H-1)/2   STRICTLY lower-triangular prior off-diagonals
                                   (row k, cols 0..k-1; diagonal fixed to 1)
          init_std  : H           log-std of each initial / prior gaussian
          init_bias : H           bias  of each initial / prior gaussian ]
        (stds are stored as log-std: sigma = exp(param) + eps.)
 
    all_params[s] : length  H*(H+1)/2 + O*H + 2*(H+O), in order
        [ dyn_cur     : H*(H+1)/2   anti-triangular dynamics current coefs
          meas_hidden : O*H         measurement coefs over current hidden vars
          stds        : H+O         log-std of [dynamics_0..H-1, meas_0..O-1]
          biases      : H+O         bias    of [dynamics_0..H-1, meas_0..O-1] ]
 
    Standard deviations everywhere use the log-std parameterisation
    sigma = exp(param) + eps; swap in tf.abs / tf.math.softplus if preferred.
 
    initial_hidden_vars
    <tf.Tensor: shape=(2, 50, 2, 2)
 
    hidden_vars
    <tf.Tensor: shape=(20, 3, 50, 2, 4)
        
    # parameter to initialize to populate the constraint function
    initial_params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + nb_hidden_vars)*4 - 2
    params = np.random.rand(nb_states, nb_hidden_vars * (nb_hidden_vars+1)//2 + 
                                nb_hidden_vars * nb_obs_vars + 2 * (nb_obs_vars + nb_hidden_vars)) + 1

    all_params = params
    all_initial_params = initial_params
    self = constraint_function(3,1,1)
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
 
        # initial / prior  L : STRICTLY lower triangular now (diagonal fixed
        # to 1 and added separately), row k touches columns 0..k-1.
        L_idx = np.zeros((H, H), dtype=np.int32)
        L_mask = np.zeros((H, H), dtype=np.float64)
        p = 0
        for k in range(H):
            for m in range(k):                 # m = 0 .. k-1  (excludes diagonal)
                L_idx[k, m] = p
                L_mask[k, m] = 1.0
                p += 1
        n_L = p                                # = H*(H-1)/2
 
        # dynamics current  Acur : row j touches columns H-1-j .. H-1
        A_idx = np.zeros((H, H), dtype=np.int32)
        A_mask = np.zeros((H, H), dtype=np.float64)
        q = 0
        for j in range(H):
            for kcol in range(0, j + 1):
                A_idx[j, kcol] = q
                A_mask[j, kcol] = 1.0
                q += 1
        n_A = q                                # = H*(H+1)/2
 
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
        
        # pivot of the initial / transition prior gaussians (diagonal of L).
        # Fixed to +1 as requested; set to -1.0 to match the -I convention of
        # the dynamics / measurement pivots (mathematically equivalent).
        prior_diag = tf.constant(1.0, dtype=dtype)
        
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
        # slice the recurrent parameter vector
        # --------------------------------------------------------------
        o0 = 0
        dyn_cur = all_params[:, o0:o0 + n_A];               o0 += n_A
        meas_hidden = all_params[:, o0:o0 + O * H];         o0 += O * H
        stds = all_params[:, o0:o0 + H + O];                o0 += H + O
        biases = all_params[:, o0:o0 + H + O];              o0 += H + O
 
        meas_hidden = tf.reshape(meas_hidden, (S, O, H))
        stds = tf.math.exp(stds) + eps                      # (S, H + O)  recurrent stds
        
        # --------------------------------------------------------------
        # slice the initial / prior parameter vector
        #   [ L_lower : n_L , init_std : H , init_bias : H ]
        # --------------------------------------------------------------
        i0 = 0
        L_lower = all_initial_params[:, i0:i0 + n_L];       i0 += n_L
        init_std = all_initial_params[:, i0:i0 + H];        i0 += H
        init_bias = all_initial_params[:, i0:i0 + H];       i0 += H
        init_std = tf.math.exp(init_std) + eps              # (S, H)
 
        # --------------------------------------------------------------
        # coefficient matrices  (S, H, H)
        # --------------------------------------------------------------
        Acur = tf.reshape(tf.gather(dyn_cur, A_idx.reshape(-1), axis=1),
                          (S, H, H)) * A_mask               # anti-triangular
        
        # prior L: strictly-lower off-diagonals from params + fixed diagonal.
        # (guard n_L == 0, which happens for H == 1: no off-diagonals exist.)
        if n_L > 0:
            L_off = tf.reshape(tf.gather(L_lower, L_idx.reshape(-1), axis=1),
                               (S, H, H)) * L_mask
        else:
            L_off = tf.zeros((S, H, H), dtype=dtype)
        L = L_off + prior_diag * tf.eye(H, batch_shape=[S], dtype=dtype)
 
        # --------------------------------------------------------------
        # recurrent hidden-variable coefficients, last axis = 2H
        #   [ current_0..H-1 , next_0..H-1 ]   (next block = -I, scale in std)
        # --------------------------------------------------------------
        dyn_next_coefs = -tf.eye(H, batch_shape=[S], dtype=dtype)   # (S, H, H)
        dyn_rows = tf.concat([Acur, dyn_next_coefs], axis=-1)       # (S, H, 2H)
 
        meas_next = tf.zeros((S, O, H), dtype=dtype)
        meas_rows = tf.concat([meas_hidden, meas_next], axis=-1)    # (S, O, 2H)
 
        C = tf.concat([dyn_rows, meas_rows], axis=1)        # (S, n_g, 2H) dynamics first
        C = tf.transpose(C, [1, 0, 2])                      # (n_g, S, 2H)
        hidden_vars = tf.broadcast_to(
            C[None, :, None, :, :],
            tf.stack([track_len, n_g, nb_tracks, S, 2 * H]))
 
        # recurrent observation coefficients, last axis = O  (obs block = -I)
        meas_obs = -tf.eye(O, batch_shape=[S], dtype=dtype)  # (S, O, O)
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
 
        # --------------------------------------------------------------
        # recurrent standard deviations, last axis = 1
        #   [ std_dyn_0..H-1 , std_meas_0..O-1 ]  (dynamics first, matches C)
        # --------------------------------------------------------------
        rec_std = tf.transpose(stds, [1, 0])                # (n_g, S)
        Gaussian_stds = tf.broadcast_to(
            rec_std[None, :, None, :, None],
            tf.stack([track_len, n_g, nb_tracks, S, 1]))
 
        # --------------------------------------------------------------
        # initial gaussians (no time axis), last axis = H
        #   diagonal of L fixed to prior_diag; std and bias now learnable
        # --------------------------------------------------------------
        IL = tf.transpose(L, [1, 0, 2])                     # (H, S, H)
        initial_hidden_vars = tf.broadcast_to(
            IL[:, None, :, :], tf.stack([H, nb_tracks, S, H]))
        initial_obs_vars = tf.zeros(tf.stack([H, nb_tracks, S, O]), dtype=dtype)
        
        init_std_g = tf.transpose(init_std, [1, 0])         # (H, S)
        initial_Gaussian_stds = tf.broadcast_to(
            init_std_g[:, None, :, None],
            tf.stack([H, nb_tracks, S, 1]))
        
        init_bias_g = tf.transpose(init_bias, [1, 0])       # (H, S)
        initial_biases = tf.broadcast_to(
            init_bias_g[:, None, :, None],
            tf.stack([H, nb_tracks, S, nb_dims]))
        
        # --------------------------------------------------------------
        # transition gaussians (fresh prior on vars idx..H-1 at a state change).
        #   Same prior rows idx..H-1 of L  ->  same std / bias slice idx..H-1.
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
        
        # diagonal of L is now fixed to +-1, so log|diag| = 0; the prior
        # scale lives entirely in init_std -> normaliser is -log(init_std).
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



current_constraint_function = constraint_function_KF_diag_Q_R(nb_hidden_vars, nb_obs_vars, integration_variable_index)
current_constraint_function = constraint_function_arbitrary_KF(nb_hidden_vars, nb_obs_vars, integration_variable_index)
#self = current_constraint_function
current_constraint_function(params, initial_params, LocErrs, dts,
         nb_dims, reference_dt, 0, dtype)

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.01
nb_batches
epochs = 150
epoch_decay = 50
decay_threshold = epoch_decay*nb_batches
verbose = 1

# Compute the required decay rate to decay the learning rate by a factor decay_ratio
decay_ratio = 0.001 
decay_rate = - np.log(decay_ratio)/((epochs - epoch_decay) * nb_batches)


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

device = '/CPU:0'
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=0.1) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

#preds = model.predict(seq)
#log_likelihood = exatrack.MLE_loss(preds, preds)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

plt.figure()
plt.plot(history.history['loss'])

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
        plt.plot(np.arange(len(track))*0.02 + lim*j, track[:,0], ':k', alpha = 0.5)
        plt.scatter(np.arange(len(track))*0.02 + lim*j, track[:,0] , c = p@colors, s = 7)
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