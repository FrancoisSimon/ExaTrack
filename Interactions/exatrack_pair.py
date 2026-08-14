# -*- coding: utf-8 -*-
"""
exatrack_pair.py
================
 
Two-particle ("pair") extension of ExaTrack: given the *simultaneous* tracks of
two particles, decide -- time point by time point -- whether they are
co-diffusing (interacting) or moving independently.
 
Idea
----
The generic ExaTrack engine represents the joint law of observations and hidden
variables as a product of unit-variance Gaussians whose arguments are linear
combinations of the hidden variables, and integrates the hidden variables out
analytically.  Nothing in that engine is specific to one particle: it only needs
 
    * the number of observed variables      (nb_obs_vars)
    * the number of hidden variables        (nb_hidden_vars)
    * the coefficient / bias tensors        (constraint_function)
 
So the pair model is obtained by doubling the state vector and adding one
coupling term:
 
    hidden variables per time step and per spatial dimension
        h = [ x1 , x2 , a1 , a2 ]
            x1, x2 : true positions of particle 1 and 2
            a1, a2 : "anomalous" variables (well anchor if confined,
                      drift displacement if directed), redrawn at transitions
    observed variables
        y = [ y1 , y2 ]
 
    dynamics (per state s, per spatial dimension, per time step)
 
        x1' = (1 - li/2) [ (1-l1) x1 + l1 a1 ]  +  (li/2) x2  + N(0, gx1**2)
        x2' = (1 - li/2) [ (1-l2) x2 + l2 a2 ]  +  (li/2) x1  + N(0, gx2**2)
        a1' = r1 a1 + N(0, q1**2)          (r1 = dt ratio if directed, 1 else)
        a2' = r2 a2 + N(0, q2**2)
        y1  = (1 - b/2) x1 + (b/2) x1' + N(0, LocErr1**2)     (b = blur_ratio)
        y2  = (1 - b/2) x2 + (b/2) x2' + N(0, LocErr2**2)
 
    li = state-dependent *interaction* (co-diffusion) factor.  Splitting it
    evenly between the two particles makes the centre of mass c = (x1+x2)/2
    diffuse freely while the relative coordinate r = x1 - x2 follows an exact
    Ornstein-Uhlenbeck process,
 
        r' = (1 - li) r + noise ,      1 - li = exp(-li_c * dt/reference_dt)
 
    so that li ~ 0  <=>  independent particles and li -> 1  <=>  the two
    particles are locked together within one frame.  The stationary width of the
    relative coordinate is  pair_distance = sqrt(d1**2 + d2**2) / sqrt(1-exp(-2 li_c)),
    which is used as the prior on x1 - x2 at the first time point of a segment.
 
Consequences for the engine
---------------------------
    nb_obs_vars             = 2
    nb_hidden_vars          = 4
    nb_gaussians            = 6   (2 localisation + 2 position + 2 anomalous)
    integration_variable_index = 2        (a1, a2 are redrawn at transitions)
    nb_transition_gaussians = 2           (their priors)
and the engine invariant  integration_variable_index + nb_transition_gaussians
== nb_hidden_vars  is satisfied (2 + 2 == 4), exactly as 1 + 1 == 2 for the
single particle model.
 
Parameter layout  (one row per state, 8 + nb_LocErr_cols + 1 columns)
---------------------------------------------------------------------
    col  0        log d1        diffusion length of particle 1 per reference_dt
    col  1        ano1          logit(confinement) if confined, log(speed*dt) if directed
    col  2        log q1        noise of the anomalous variable of particle 1
    col  3        log d2
    col  4        ano2
    col  5        log q2
    col  6        ano_int       logit of the interaction factor li  <-- the co-diffusion parameter
    col  7        is_dir2       1 -> particle 2 directed, 0 -> confined
    col  8:-1     log LocErr    1, 2 (one per particle) or 2*nb_dims columns
    col -1        is_dir1       1 -> particle 1 directed, 0 -> confined
 
Columns 0, 1, 2 and -1 keep the meaning they have in the single-particle model,
which is required because `Initial_layer_constraints.call` builds the extra
"mislinking" state as
    [log(max_linking_distance), -15, log(1e-5)] + params[-1, 3:-1] + [0].
The mislinking row is then patched inside the constraint function so that *both*
particles get the large linking distance and no interaction.
 
Usage
-----
    import exatrack_pair as ep
    params = ep.make_pair_params(d1=[...], d2=[...], l_int=[...], ...)
    model, pred_model = ep.build_pair_segment_model(...)
Tracks are fed as arrays of shape (track_len, 2*nb_dims) with the columns
ordered particle-major:  [x1, y1, (z1), x2, y2, (z2)], so that
`exatrack.segment_tracks` / `TrackSegmentSequence` can be reused unchanged.
"""
 
import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
 
import sys
import os
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except NameError:
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)
import exatrack
 
dtype = 'float64'
jit_compile = False

# =====================================================================
# 0.  Patch of RNN_cell
# ---------------------------------------------------------------------
# `RNN_cell` reads the number of hidden variables as `Prev_coefs.shape[3]`
# whereas `Prev_coefs` has the layout (nb_gaussians, nb_tracks, nb_sequences,
# nb_dims, nb_hidden_vars): axis 3 is nb_dims.  The two happen to coincide in
# the single particle 2D case (nb_dims == nb_hidden_vars == 2), which is why the
# bug is invisible there, but the pair model has nb_hidden_vars = 4 != nb_dims.
# The function below is the original one with that single line fixed
# (`shape[-1]`), and it also fixes 3D single-particle tracking.
# =====================================================================
@tf.function(jit_compile=False)
def RNN_cell(input_i, Prev_coefs, Prev_biases, LP, segment_len,
             reshaped_Log_factors, reshaped_transition_Log_factors,
             reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
             reccurent_next_hidden_var_coefs, reccurent_biases,
             transition_hidden_var_coefs, transition_biases,
             sequence_phase_1, sequence_phase_2, transition_mask,
             transition_sequence, transition_mean, transition_var,
             gamma_dist_mean, gamma_dist_var, states, dt_ratios):
    """Identical to `exatrack.RNN_cell` except for the `nb_hidden_vars` line."""
 
    current_states = states[:, :, -1:]
    nb_dims = input_i.shape[-2]
    nb_tracks = LP.shape[0]
    nb_hidden_vars = Prev_coefs.shape[-1]          # <-- was Prev_coefs.shape[3]
    nb_states = reccurent_hidden_var_coefs.shape[2]
    sequence_length = LP.shape[1] // nb_states
 
    Prev_coefs2 = tf.repeat(Prev_coefs, nb_states, axis=2)
    Prev_biases2 = tf.repeat(Prev_biases, nb_states, axis=2)
    LP2 = tf.repeat(LP, nb_states, axis=1)
    segment_len = tf.repeat(segment_len, nb_states, axis=1)
 
    alternative_Prev_coefs = tf.concat(
        (Prev_coefs2, tf.identity(transition_hidden_var_coefs)), axis=0)
    alternative_Prev_biases = tf.concat(
        (Prev_biases2, tf.identity(transition_biases)), axis=0)
 
    transition_Prev_coefs, transition_Prev_biases, LC = \
        exatrack.transition_RNN_reccurence_formula(
            current_hidden_var_coefs=alternative_Prev_coefs,
            next_hidden_var_coefs=tf.constant(
                0, dtype=dtype, shape=alternative_Prev_coefs.shape),
            biases=alternative_Prev_biases,
            transition_sequence=transition_sequence,
            dtype=dtype)
 
    LP2 += LC * transition_mask + reshaped_Log_factors
 
    current_shapes = gamma_dist_mean ** 2 / gamma_dist_var
    current_rates = gamma_dist_mean / gamma_dist_var
 
    all_Prev_coefs = (transition_Prev_coefs * transition_mask[None, :, :, None, None]
                      + Prev_coefs2 * (1 - transition_mask[None, :, :, None, None]))
    all_prev_biases = (transition_Prev_biases * transition_mask[None, :, :, None]
                       + Prev_biases2 * (1 - transition_mask[None, :, :, None]))
 
    gamma = tf.compat.v1.distributions.Gamma(current_shapes, current_rates)
    S_old = 1 - gamma.cdf(segment_len + 1e-12) + 1e-12
    S_new = 1 - gamma.cdf(segment_len + 1e-12 + dt_ratios[:, None]) + 1e-12
    transition_probas = tf.clip_by_value(1 - S_new / S_old,
                                         clip_value_min=1e-20,
                                         clip_value_max=1 - 1e-10)
 
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
    current_biases += tf.reduce_sum(current_reccurent_obs_var_coefs * input_i, (-1))
    biases = tf.concat((all_prev_biases, current_biases), axis=0)
 
    Next_coefs, Next_biases, LC = exatrack.RNN_reccurence_formula(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        sequence_phase_1, sequence_phase_2, dtype=dtype)
 
    all_LP += LC
 
    reshaped_Next_coefs = tf.reshape(
        Next_coefs, Next_coefs.shape[:2] + [sequence_length * nb_states, nb_states,
                                            nb_dims, nb_hidden_vars])
    transition_LPs = tf.reshape(all_LP - 200 * (1 - transition_mask),
                                (nb_tracks, sequence_length * nb_states, nb_states)) \
        - tf.math.reduce_sum(
            tf.math.log(tf.math.abs(tf.linalg.diag_part(
                tf.transpose(reshaped_Next_coefs, [1, 2, 3, 4, 0, 5]))) + 1e-20),
            axis=(-2, -1))
 
    max_transition_LPs = tf.reduce_max(transition_LPs, axis=1, keepdims=True)
    transition_Ps = tf.math.exp(transition_LPs - max_transition_LPs)
    transition_weights = transition_Ps
    transition_weights = transition_weights / tf.reduce_sum(transition_weights, 1, keepdims=True)
 
    transition_states = tf.reduce_sum(states[:, :, None] * transition_weights[:, :, :, None, None], 1)
 
    transition_Next_coefs = tf.reshape(
        Next_coefs, Next_coefs.shape[:2] + [sequence_length * nb_states, nb_states,
                                            nb_dims, nb_hidden_vars])
    transition_Next_coefs = tf.reduce_sum(
        transition_Next_coefs * transition_weights[None, :, :, :, None, None], axis=2)
 
    transition_Next_biases = tf.reshape(
        Next_biases, Next_biases.shape[:2] + [sequence_length * nb_states, nb_states, nb_dims])
    transition_Next_biases = tf.reduce_sum(
        transition_Next_biases * transition_weights[None, :, :, :, None], axis=2)
 
    transition_LPs = tf.math.log(tf.reduce_sum(transition_Ps, axis=1)) \
        + max_transition_LPs[:, 0] \
        + tf.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(
            tf.transpose(transition_Next_coefs, [1, 2, 3, 0, 4]))) + 1e-20), axis=(-2, -1))
 
    stable_LPs = tf.reshape(all_LP, (nb_tracks, sequence_length * nb_states, nb_states))
    stable_weights = tf.reshape((1 - transition_mask), (sequence_length * nb_states, nb_states))[None]
    stable_LPs = tf.reduce_sum(stable_LPs * stable_weights, 2)
 
    stable_states = tf.reduce_sum(states[:, :, None] * stable_weights[:, :, :, None, None], 2)
 
    stable_Next_coefs = tf.reduce_sum(
        tf.reshape(Next_coefs, Next_coefs.shape[:2] + [sequence_length * nb_states, nb_states,
                                                       nb_dims, nb_hidden_vars])
        * stable_weights[None, :, :, :, None, None], axis=3)
    stable_Next_biases = tf.reduce_sum(
        tf.reshape(Next_biases, Next_biases.shape[:2] + [sequence_length * nb_states,
                                                         nb_states, nb_dims])
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
                                 (nb_prev_gaussians, nb_tracks, 2, nb_states,
                                  nb_dims, nb_hidden_vars))
    last_Next_biases = tf.reshape(Next_biases[:, :, -nb_states * 2:],
                                  (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_dims))
    last_LP = tf.reshape(new_LP[:, -nb_states * 2:], (nb_tracks, 2, nb_states)) \
        - tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(
            tf.transpose(last_Next_coefs, [1, 2, 3, 4, 0, 5]))) + 1e-20), axis=(-2, -1))
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
 
    weight_last_LP = last_LP
    weight_last_P = tf.math.exp(weight_last_LP - tf.reduce_max(weight_last_LP, axis=1, keepdims=True))
    last_weights = weight_last_P / tf.reduce_sum(weight_last_P, 1, keepdims=True)
 
    reduced_last_Next_coefs = tf.reduce_sum(
        last_Next_coefs * last_weights[None, :, :, :, None, None], axis=2)
    reduced_last_Next_biases = tf.reduce_sum(
        last_Next_biases * last_weights[None, :, :, :, None], axis=2)
    reduced_last_LPs = (tf.math.log(sum_last_P + 1e-100) + last_LP_max)[:, 0] \
        + tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.diag_part(
            tf.transpose(reduced_last_Next_coefs, [1, 2, 3, 0, 4]))) + 1e-20), axis=(-2, -1))
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
            new_gamma_dist_mean, new_gamma_dist_var, new_states)
 
 
# `Custom_RNN_layer.call` resolves `RNN_cell` in the exatrack module namespace,
# so replacing the attribute is enough to make the pair model use the fixed cell.
exatrack.RNN_cell = RNN_cell
 
 
# =====================================================================
# 1.  Parameter helpers
# =====================================================================
def logit(x):
    x = np.asarray(x, dtype='float64')
    return np.log(x / (1 - x))
 
 
def make_pair_params(d1, d2, l_int,
                     q1=2e-4, q2=2e-4,
                     l_1=1e-6, l_2=1e-6,
                     v_1=None, v_2=None,
                     is_dir1=0.0, is_dir2=0.0,
                     LocErr=1.0, nb_LocErr_cols=2):
    """
    Build the (nb_states, 8 + 1 + nb_LocErr_cols) parameter matrix of the pair
    model.  Every argument is a scalar or a per-state sequence.
 
    Column order:
        [log d1, ano1, log q1, log d2, ano2, log q2,
         ano_int, is_dir1, is_dir2, log LocErr...]
    d1, d2  : diffusion length of each particle per reference_dt, i.e.
              sqrt(2 * D * reference_dt).
    l_int   : interaction factor in (0, 1).  0 -> independent particles,
              close to 1 -> the two particles are locked together within a frame.
              This is *the* co-diffusion parameter.
    l_1,l_2 : per-particle confinement factor (used when is_dir == 0).
    v_1,v_2 : per-particle drift displacement per reference_dt (used when
              is_dir == 1); overrides l_1 / l_2 for the states where is_dir == 1.
    q1, q2  : diffusion of the anomalous variable (well anchor / drift) per
              reference_dt.
    LocErr  : initial value of the localisation-error parameter.  With
              LocErr_type='Linear' the input localisation errors are multiplied
              by it, so 1.0 is the natural starting point.  nb_LocErr_cols can
              be 1 (shared), 2 (one per particle) or 2*nb_dims (per particle and
              per spatial dimension, particle-major ordering).
    """
    def arr(v):
        v = np.atleast_1d(np.asarray(v, dtype='float64'))
        return v
 
    d1, d2, l_int = arr(d1), arr(d2), arr(l_int)
    nb_states = max(len(d1), len(d2), len(l_int))
 
    def full(v, default=None):
        if v is None:
            v = default
        v = arr(v)
        if len(v) == 1:
            v = np.repeat(v, nb_states)
        return v
 
    d1, d2, l_int = full(d1), full(d2), full(l_int)
    q1, q2 = full(q1), full(q2)
    l_1, l_2 = full(l_1), full(l_2)
    is_dir1, is_dir2 = full(is_dir1), full(is_dir2)
    v_1 = full(v_1, d1) if v_1 is not None else d1.copy()
    v_2 = full(v_2, d2) if v_2 is not None else d2.copy()
 
    ano1 = np.where(is_dir1 >= 0.5, np.log(v_1), logit(np.clip(l_1, 1e-12, 1 - 1e-12)))
    ano2 = np.where(is_dir2 >= 0.5, np.log(v_2), logit(np.clip(l_2, 1e-12, 1 - 1e-12)))
    ano_int = logit(np.clip(l_int, 1e-12, 1 - 1e-12))
 
    LocErr = np.asarray(LocErr, dtype='float64')
    if LocErr.ndim == 0:
        LocErr = np.full((nb_states, nb_LocErr_cols), float(LocErr))
    elif LocErr.ndim == 1:
        LocErr = np.tile(LocErr[None], (nb_states, 1))
 
    params = np.concatenate([
        np.log(d1)[:, None], ano1[:, None], np.log(q1)[:, None],
        np.log(d2)[:, None], ano2[:, None], np.log(q2)[:, None],
        ano_int[:, None], np.asarray(is_dir1, dtype='float64')[:, None],
        np.asarray(is_dir2, dtype='float64')[:, None],
        np.log(LocErr)], axis=1)
    return params
 
 
def pair_params_to_dict(params, reference_dt=1.0):
    """Human readable version of the raw (log-space) pair parameters."""
    params = np.asarray(params)
    l_int = 1 / (1 + np.exp(-params[:, 6]))
    out = {
        'd1': np.exp(params[:, 0]),
        'd2': np.exp(params[:, 3]),
        'D1': np.exp(params[:, 0]) ** 2 / (2 * reference_dt),
        'D2': np.exp(params[:, 3]) ** 2 / (2 * reference_dt),
        'interaction factor l_int': l_int,
        'interaction rate (per time unit)': -np.log(np.maximum(1 - l_int, 1e-300)) / reference_dt,
        'anomalous factor 1': np.where(params[:, 7] >= 0.5,
                                       np.exp(params[:, 1]), 1 / (1 + np.exp(-params[:, 1]))),
        'anomalous factor 2': np.where(params[:, 8] >= 0.5,
                                       np.exp(params[:, 4]), 1 / (1 + np.exp(-params[:, 4]))),
        'q1': np.exp(params[:, 2]),
        'q2': np.exp(params[:, 5]),
        'Model type 1': np.array(['Confined', 'Directed'])[params[:, 7].astype(int)],
        'Model type 2': np.array(['Confined', 'Directed'])[params[:, 8].astype(int)],
        'LocErr params': np.exp(params[:, 9:])}
    return out
 
 
def get_pair_model_params(model, reference_dt=1.0, track_segmentation=True):
    """Same role as `exatrack.get_model_params` but for the pair parameter layout."""
    weights = model.weights
    nb_states = weights[0].shape[0]
    shape_id, rate_id = (8, 7) if track_segmentation else (5, 4)
    transition_shapes = tf.math.exp(weights[shape_id]).numpy()
    max_off_rate = 1 / exatrack.min_lifetime
    tr = weights[rate_id]
    tr = (tf.eye(nb_states, dtype=dtype) * (1 - max_off_rate + max_off_rate * tf.math.softmax(tr, axis=1))
          + (1 - tf.eye(nb_states, dtype=dtype)) * (max_off_rate * tf.math.softmax(tr, axis=1)) + 1e-7)
    transition_rates = (tr * transition_shapes).numpy()
 
    out = pair_params_to_dict(weights[0].numpy(), reference_dt)
    out['transition rates'] = transition_rates
    out['transition shapes'] = transition_shapes
    out['Fractions'] = tf.math.softmax(weights[2][0]).numpy()
    return out
 
 
class get_pair_parameters(tf.keras.callbacks.Callback):
    """Drop-in replacement for `exatrack.get_parameters` printing pair parameters."""
 
    def __init__(self, reference_dt=1.0, track_segmentation=True):
        super().__init__()
        self.reference_dt = reference_dt
        self.track_segmentation = track_segmentation
 
    def on_epoch_end(self, epoch, logs=None):
        p = get_pair_model_params(self.model, self.reference_dt, self.track_segmentation)
        printable = {k: np.round(v, 4) if np.issubdtype(np.asarray(v).dtype, np.number) else v
                     for k, v in p.items()}
        print({k: printable[k] for k in ['D1', 'D2', 'interaction factor l_int',
                                         'LocErr params', 'transition rates',
                                         'transition shapes', 'Fractions']})
 
 
# =====================================================================
# 2.  The pair constraint function
# =====================================================================
def make_pair_constraint_function(nb_dims):
    """
    Returns the `constraint_function` of the pair model for `nb_dims` spatial
    dimensions.
 
    The number of spatial dimensions is baked in through the closure because
    `exatrack.get_sequences` traces the constraint function with nb_dims = 1 to
    discover the sparsity pattern, which would clash with per-dimension
    localisation-error parameters.  The traced pattern is identical for every
    nb_dims, so this only makes the (single time point, single track) trace
    marginally more expensive.
    """
    D = int(nb_dims)
 
    @tf.function
    def pair_constraint_function(all_params, all_initial_params, LocErrs, dts,
                                 _nb_dims, reference_dt, LocErr_function,
                                 blur_ratio=1, dtype='float64', Multi_LocErrs=False):
 
        nb_states = all_params.shape[0]
        nb_hidden_vars = 4
        nb_obs_vars = 2
        nb_transition_gaussians = 2
        integration_variable_index = tf.constant(2)
 
        if all_params.shape[1] < 10:
            raise ValueError(
                'The pair model needs at least 10 parameter columns '
                '(8 dynamical + is_dir1 + is_dir2 + >=1 localisation error). '
                'Use exatrack_pair.make_pair_params to build them.')
 
        # -------------------------------------------------- time steps
        dts = tf.cast(dts, dtype)
        dts = tf.transpose(dts, [1, 0])[..., None, None]          # (T+1, N, 1, 1)
        reference_dt = tf.cast(reference_dt, dtype)
        dt_ratio = dts / reference_dt + 0.9e-20
        dt_sqrt_ratio = tf.sqrt(dt_ratio)
 
        # -------------------------------------------------- localisation errors
        LocErrs = tf.cast(LocErrs, dtype)
        if LocErrs.shape.rank == 2:
            LocErrs = LocErrs[..., None]
        L_in = int(LocErrs.shape[-1])
        nb_tracks = tf.shape(LocErrs)[0]
        track_len = tf.shape(LocErrs)[1]
 
        if L_in == 2 * D and D > 1:                 # (N,T,2*D) particle-major
            LocErrs = tf.reshape(LocErrs, tf.stack([nb_tracks, track_len, 2, D]))
            LocErrs = tf.transpose(LocErrs, [0, 1, 3, 2])         # (N,T,D,2)
        elif L_in == 2:                             # one per particle
            LocErrs = LocErrs[:, :, None, :]                       # (N,T,1,2)
        elif L_in == 1:                             # shared
            LocErrs = LocErrs[:, :, :, None]                       # (N,T,1,1)
        else:
            raise ValueError('input localisation errors must have 1, 2 or 2*nb_dims '
                             'columns, got %s' % L_in)
        LocErrs = tf.transpose(LocErrs, [1, 0, 2, 3])[:, :, None]  # (T,N,1,d,p)
 
        LocErr_cols = all_params[:, 9:]
        L_par = int(LocErr_cols.shape[-1])
        LocErr_param = tf.math.exp(LocErr_cols)
        if L_par == 2 * D and D > 1:
            LocErr_param = tf.transpose(tf.reshape(LocErr_param, (nb_states, 2, D)), [0, 2, 1])
        elif L_par == 2:
            LocErr_param = LocErr_param[:, None, :]
        elif L_par == 1:
            LocErr_param = LocErr_param[:, :, None]
        else:
            raise ValueError('params[:, 9:] must have 1, 2 or 2*nb_dims columns, '
                             'got %s' % L_par)
        LocErr_param = LocErr_param[None, None]                    # (1,1,S,d,p)
 
        LocErrs = LocErr_function(LocErrs, LocErr_param)
        shp5 = tf.stack([track_len, nb_tracks, nb_states, D, 2])
        LocErrs = tf.broadcast_to(LocErrs + 1e-20, shp5)
        sig1 = LocErrs[..., 0]                                     # (T,N,S,D)
        sig2 = LocErrs[..., 1]
 
        # -------------------------------------------------- per state parameters
        p = all_params
        log_d1 = p[:, 0][None, None, :, None]
        ano1 = p[:, 1][None, None, :, None]
        log_q1 = p[:, 2][None, None, :, None]
        log_d2 = p[:, 3][None, None, :, None]
        ano2 = p[:, 4][None, None, :, None]
        log_q2 = p[:, 5][None, None, :, None]
        ano_int = p[:, 6][None, None, :, None]
        is_dir1 = p[:, 7][None, None, :, None]
        is_dir2 = p[:, 8][None, None, :, None]
        log_spread = all_initial_params[:, 0][None, None, :, None]
 
        # The mislinking state (last one, appended by Initial_layer_constraints)
        # inherits columns 3:-1 from the last physical state, and the layer's
        # hard-coded trailing [0.] lands on the last LocErr column (a harmless
        # unit multiplier).  Overwrite the inherited columns here so that BOTH
        # particles get the large linking distance (already put in cols 0-2 for
        # particle 1 by the layer), no interaction, and confined Brownian motion
        # (is_dir = 0).  This reproduces the single-particle mislink behaviour.
        mis = tf.one_hot(nb_states - 1, nb_states, dtype=dtype)[None, None, :, None]
        keep = 1.0 - mis
        log_d2 = keep * log_d2 + mis * log_d1
        ano2 = keep * ano2 + mis * ano1
        log_q2 = keep * log_q2 + mis * log_q1
        ano_int = keep * ano_int + mis * tf.constant(-15.0, dtype=dtype)
        is_dir1 = keep * is_dir1        # force confined (is_dir1 = 0) for the mislink
        is_dir2 = keep * is_dir2        # force confined (is_dir2 = 0) for the mislink
 
        isdir1 = tf.cast(is_dir1 >= 0.5, dtype)
        isconf1 = 1.0 - isdir1
        isdir2 = tf.cast(is_dir2 >= 0.5, dtype)
        isconf2 = 1.0 - isdir2
 
        # -------------------------------------------------- reference_dt -> dts
        dtr = dt_ratio[:track_len]
        dsr = dt_sqrt_ratio[:track_len]
 
        d1 = tf.exp(log_d1) * dsr + 1e-20
        d2 = tf.exp(log_d2) * dsr + 1e-20
        q1 = tf.exp(log_q1) * dsr + 1e-20
        q2 = tf.exp(log_q2) * dsr + 1e-20
        v1 = tf.exp(ano1) * dtr + 1e-20
        v2 = tf.exp(ano2) * dtr + 1e-20
 
        # continuous-time confinement rates:  l = 1 - exp(-l_c),
        # l_c(reference_dt) = -log(1 - sigmoid(ano)) = softplus(ano)
        l1_c = tf.math.softplus(ano1) * dtr + 1e-12
        l2_c = tf.math.softplus(ano2) * dtr + 1e-12
        li_c = tf.math.softplus(ano_int) * dtr + 1e-12
        l1 = -tf.math.expm1(-l1_c)
        l2 = -tf.math.expm1(-l2_c)
        li = -tf.math.expm1(-li_c)
 
        # exact OU step noise of the individual well (identical to the single
        # particle model); the interaction does not rescale it, which keeps the
        # centre of mass diffusing at the right rate whatever the interaction.
        corr1 = tf.sqrt(-tf.math.expm1(-2 * l1_c) / (2 * l1_c))
        corr2 = tf.sqrt(-tf.math.expm1(-2 * l2_c) / (2 * l2_c))
        gx1_std = d1 * (isdir1 + isconf1 * corr1)
        gx2_std = d2 * (isdir2 + isconf2 * corr2)
 
        # stationary width of the well and of the pair (used as priors at the
        # first time point and after a state transition).
        # NB: well_distance = d / sqrt(2 l_c) is the width that is *consistent*
        # with the step noise above; the single-particle module uses
        # d / sqrt(2 (1-exp(-2 l_c))) which differs by sqrt(l_c/(1-exp(-2 l_c))).
        well_dist1 = d1 / tf.sqrt(2 * l1_c)
        well_dist2 = d2 / tf.sqrt(2 * l2_c)
        pair_dist = tf.sqrt((d1 ** 2 + d2 ** 2) / (-tf.math.expm1(-2 * li_c)))
 
        # directed motion: ano_t = speed * dts[t], so E[ano_{t+1}|ano_t] scales
        # with the ratio of consecutive frame durations.
        ano_step_ratio = dt_ratio[1:] / dt_ratio[:-1]
        r1 = ano_step_ratio * isdir1 + (1.0 - isdir1)
        r2 = ano_step_ratio * isdir2 + (1.0 - isdir2)
 
        initial_spread = tf.exp(log_spread)
 
        # -------------------------------------------------- broadcasting helpers
        shp = tf.stack([track_len, nb_tracks, nb_states, D])
        z = tf.zeros(shp, dtype=dtype)
        tiny = tf.fill(shp, tf.constant(1e-15, dtype=dtype))
 
        def b(x):
            return tf.broadcast_to(tf.cast(x, dtype) + z, shp)
 
        blur = tf.cast(blur_ratio, dtype)
 
        # -------------------------------------------------- recurrent Gaussians
        # coefficient order: [x1, x2, a1, a2, x1', x2', a1', a2']
        inv_s1 = 1.0 / sig1
        inv_s2 = 1.0 / sig2
        g_y1 = tf.stack([b((1 - 0.5 * blur) * inv_s1), z, z, z,
                         b(0.5 * blur * inv_s1), z, z, z], axis=-1)
        g_y2 = tf.stack([z, b((1 - 0.5 * blur) * inv_s2), z, z,
                         z, b(0.5 * blur * inv_s2), z, z], axis=-1)
 
        half = li / 2.0                       # each particle takes half of the pull
        one_minus_half = 1.0 - half
        
        own1_x = (1.0 - l1) * isconf1 + isdir1
        own1_a = l1 * isconf1 + isdir1
        own2_x = (1.0 - l2) * isconf2 + isdir2
        own2_a = l2 * isconf2 + isdir2
 
        inv_g1 = 1.0 / gx1_std
        inv_g2 = 1.0 / gx2_std
 
        g_x1 = tf.stack([b(one_minus_half * own1_x * inv_g1),
                         b(half * inv_g1),
                         b(one_minus_half * own1_a * inv_g1),
                         z,
                         b(-inv_g1), z, z, z], axis=-1)
        g_x2 = tf.stack([b(half * inv_g2),
                         b(one_minus_half * own2_x * inv_g2),
                         z,
                         b(one_minus_half * own2_a * inv_g2),
                         z, b(-inv_g2), z, z], axis=-1)
 
        inv_q1 = 1.0 / q1
        inv_q2 = 1.0 / q2
        g_a1 = tf.stack([z, z, b(r1 * inv_q1), z, z, z, b(-inv_q1), z], axis=-1)
        g_a2 = tf.stack([z, z, z, b(r2 * inv_q2), z, z, z, b(-inv_q2)], axis=-1)
 
        hidden_vars = tf.stack([g_y1, g_y2, g_x1, g_x2, g_a1, g_a2], axis=1)
 
        # -------------------------------------------------- observation coefs
        obs_y1 = tf.stack([b(-inv_s1), z], axis=-1)
        obs_y2 = tf.stack([z, b(-inv_s2)], axis=-1)
        obs_0 = tf.stack([z, z], axis=-1)
        obs_vars = tf.stack([obs_y1, obs_y2, obs_0, obs_0, obs_0, obs_0], axis=1)
 
        # -------------------------------------------------- initial Gaussians
        # ig0 : broad prior on x1
        # ig1 : prior on the pair separation x1 - x2 (stationary OU width)
        # ig2 : prior linking a1 to x1 (well) or setting the drift scale
        # ig3 : same for particle 2
        ig0 = tf.stack([b(1.0 / initial_spread), z, z, z], axis=-1)
        ig1 = tf.stack([b(1.0 / pair_dist), b(-1.0 / pair_dist), z, z], axis=-1)
 
        ig2_c0 = (1.0 / well_dist1) * isconf1 + tiny * isdir1
        ig2_c1 = (-1.0 / well_dist1) * isconf1 + (1.0 / v1) * isdir1
        ig2 = tf.stack([b(ig2_c0), z, b(ig2_c1), z], axis=-1)
 
        ig3_c0 = (1.0 / well_dist2) * isconf2 + tiny * isdir2
        ig3_c1 = (-1.0 / well_dist2) * isconf2 + (1.0 / v2) * isdir2
        ig3 = tf.stack([z, b(ig3_c0), z, b(ig3_c1)], axis=-1)
 
        initial_hidden_vars = tf.stack([ig0[0], ig1[0], ig2[0], ig3[0]], axis=0)
 
        # -------------------------------------------------- transition Gaussians
        # only the anomalous variables are redrawn at a state transition, the two
        # positions (and therefore the pair separation) are continuous.
        transition_hidden_vars = tf.stack([ig2, ig3], axis=1)      # (T,2,N,S,D,4)
 
        # -------------------------------------------------- scaffolding
        Gaussian_stds = tf.ones(tf.stack([track_len, nb_obs_vars + nb_hidden_vars,
                                          nb_tracks, nb_states, D, 1]), dtype=dtype)
        biases = tf.zeros(tf.stack([track_len, nb_obs_vars + nb_hidden_vars,
                                    nb_tracks, nb_states, D]), dtype=dtype)
        initial_obs_vars = tf.zeros(tf.stack([nb_hidden_vars, nb_tracks, nb_states,
                                              D, nb_obs_vars]), dtype=dtype)
        initial_Gaussian_stds = tf.ones(tf.stack([nb_hidden_vars, nb_tracks, nb_states,
                                                  D, 1]), dtype=dtype)
        initial_biases = tf.zeros(tf.stack([nb_hidden_vars, nb_tracks, nb_states, D]),
                                  dtype=dtype)
        transition_Gaussian_stds = tf.ones(tf.stack([track_len, nb_transition_gaussians,
                                                     nb_tracks, nb_states, D, 1]), dtype=dtype)
        transition_biases = tf.zeros(tf.stack([track_len, nb_transition_gaussians,
                                               nb_tracks, nb_states, D]), dtype=dtype)
 
        # -------------------------------------------------- log normalisations
        Log_factors = tf.reduce_sum(
            - tf.math.log(sig1) - tf.math.log(sig2)
            - tf.math.log(gx1_std) - tf.math.log(gx2_std)
            - tf.math.log(q1) - tf.math.log(q2), axis=-1)          # (T,N,S)
 
        anomalous_factor = (- tf.math.log(well_dist1) * isconf1 - tf.math.log(v1) * isdir1
                            - tf.math.log(well_dist2) * isconf2 - tf.math.log(v2) * isdir2)
        anomalous_factor = tf.reduce_sum(b(anomalous_factor), axis=-1)
 
        pair_factor = tf.reduce_sum(b(- tf.math.log(pair_dist)
                                      - tf.math.log(initial_spread)), axis=-1)
 
        initial_Log_factors = Log_factors[0] + anomalous_factor[0] + pair_factor[0]
        transition_Log_factors = Log_factors + anomalous_factor
 
        return (hidden_vars, obs_vars, Gaussian_stds, biases,
                initial_hidden_vars, initial_obs_vars,
                initial_Gaussian_stds, initial_biases,
                transition_hidden_vars, transition_Gaussian_stds,
                transition_biases, integration_variable_index,
                Log_factors, initial_Log_factors, transition_Log_factors)
 
    return pair_constraint_function
 
 
# =====================================================================
# 3.  Model builder
# =====================================================================
def build_pair_segment_model(track_len,
                             nb_states,
                             params,
                             initial_params,
                             transition_rates,
                             transition_shapes,
                             initial_fractions,
                             batch_size,
                             reference_dt,
                             nb_dims=2,
                             sequence_length=3,
                             max_linking_distance=3,
                             estimated_density=0.001,
                             vary_params=True,
                             vary_initial_params=True,
                             vary_initial_fractions=True,
                             vary_transition_shapes=False,
                             vary_transition_rates=True,
                             nb_LocErr_dims=2,
                             blur_ratio=1,
                             LocErr_type='Linear',
                             transition_param_function=None):
    """
    Pair version of `exatrack.build_segment_model`.
 
    Inputs of the returned models
    -----------------------------
    tracks   : (batch_size, track_len, 2*nb_dims)  columns [x1,y1,(z1),x2,y2,(z2)]
    LocErrs  : (batch_size, track_len, nb_LocErr_dims) with nb_LocErr_dims in
               {0 -> (batch_size, track_len), 1, 2, 2*nb_dims}
    dts      : (batch_size, track_len+1)
    masks    : (batch_size, track_len)
    isfirst  : (batch_size,)
 
    Returns (model, pred_model) exactly like the single-particle builder;
    `pred_model` additionally outputs the per-time-point state probabilities.
    """
    nb_obs_vars = 2
    nb_hidden_vars = 4
    nb_gaussians = nb_obs_vars + nb_hidden_vars
 
    params = np.asarray(params)
    nb_LocErr_cols = params.shape[1] - 9
    if nb_LocErr_cols < 1:
        raise ValueError(
            'the pair model expects params with 8 dynamical columns, the two '
            'is_dir flags and at least one localisation-error column, i.e. at '
            'least 10 columns; got %s. '
            'Use exatrack_pair.make_pair_params to build them.' % params.shape[1])
    if nb_LocErr_cols not in (1, 2, 2 * nb_dims):
        raise ValueError('params[:, 9:] must have 1, 2 or 2*nb_dims (%s) columns, got %s'
                         % (2 * nb_dims, nb_LocErr_cols))
    if nb_LocErr_dims not in (0, 1, 2, 2 * nb_dims):
        raise ValueError('nb_LocErr_dims must be 0, 1, 2 or 2*nb_dims (%s), got %s'
                         % (2 * nb_dims, nb_LocErr_dims))
    if params.shape[0] != nb_states:
        raise ValueError('params has %s rows but nb_states is %s (the mislinking state '
                         'is added internally, do not include it)'
                         % (params.shape[0], nb_states))
 
    if transition_param_function is None:
        transition_param_function = exatrack.transition_param_function
 
    constraint_function = make_pair_constraint_function(nb_dims)
 
    inputs = tf.keras.Input(batch_shape=(batch_size, track_len, 2 * nb_dims),
                            name='tracks', dtype=dtype)
    if nb_LocErr_dims > 0:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len, nb_LocErr_dims),
                                       name='Localization errors', dtype=dtype)
    else:
        input_LocErrs = tf.keras.Input(batch_shape=(batch_size, track_len),
                                       name='Localization errors', dtype=dtype)
    input_dts = tf.keras.Input(batch_shape=(batch_size, track_len + 1),
                               name='frame durations', dtype=dtype)
    input_mask = tf.keras.Input(batch_shape=(batch_size, track_len), name='masks', dtype=dtype)
    input_isfirst = tf.keras.Input(batch_shape=(batch_size,), name='isfirsts', dtype=dtype)
 
    # (N, T, 2*D) -> (N, T, 2, D) -> (N, T, D, 2) -> (N, 1, T, 1, D, 2)
    def _reshape_pair(x):
        x = tf.reshape(x, (batch_size, track_len, 2, nb_dims))
        x = tf.transpose(x, [0, 1, 3, 2])
        return x[:, None, :, None]
 
    reshaped_inputs = tf.keras.layers.Lambda(_reshape_pair, dtype=dtype)(inputs)
    transposed_inputs = exatrack.transpose_layer(dtype=dtype)(
        reshaped_inputs, perm=[2, 1, 0, 3, 4, 5])
 
    Init_layer = exatrack.Initial_layer_constraints(
        nb_states, nb_gaussians, nb_obs_vars, nb_hidden_vars, nb_dims,
        params, initial_params, initial_fractions, max_linking_distance,
        constraint_function,
        reference_dt=reference_dt,
        vary_params=vary_params,
        vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        sequence_length=sequence_length,
        carryover=True,
        LocErr_type=LocErr_type,
        blur_ratio=blur_ratio,
        dtype=dtype)
 
    tensor1, initial_states = Init_layer(transposed_inputs, input_LocErrs, input_dts)
 
    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds = Init_layer.param_vars[:, 0]              # log d1 (particle 1)
    anomalous_factors = Init_layer.param_vars[:, 1]   # ano1
    isdir = Init_layer.param_vars[:, 7]               # is_dir1 (new layout)
 
    (Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors,
     reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
     reccurent_next_hidden_var_coefs, reccurent_biases,
     transition_hidden_var_coefs, transition_biases) = initial_states
 
    first_mask_layer = exatrack.IsfirstMaskLayer(dtype=dtype)
    Prev_coefs = first_mask_layer(Prev_coefs, Init_layer.carryout_coefs,
                                  input_isfirst[None, :, None, None, None])
    Prev_biases = first_mask_layer(Prev_biases, Init_layer.carryout_biases,
                                   input_isfirst[None, :, None, None])
    LP = first_mask_layer(LP, Init_layer.carryout_LP, input_isfirst[:, None])
 
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype=dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype=dtype)(input_mask)
 
    layer = exatrack.Custom_RNN_layer(
        batch_size, transition_shapes, transition_rates, estimated_density, nb_states,
        Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2,
        Init_layer.transition_sequence, transition_param_function,
        sequence_length=sequence_length,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates,
        carryover=True, dtype=dtype)
 
    (Prev_coefs, Prev_biases, LP, segment_len, gamma_dist_mean, gamma_dist_var,
     All_motion_states, All_coefs, All_biases, All_LPs, motion_states) = layer(
        sliced_inputs, input_dts, reference_dt, sliced_mask, Prev_coefs, Prev_biases, LP,
        Log_factors, transition_Log_factors, reccurent_obs_var_coefs,
        reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases,
        transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions,
        anomalous_factors, isdir, isfirst=input_isfirst)
 
    states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]
 
    carryover_layer = exatrack.CarryoverAssignLayer(
        carryout_variables=[Init_layer.carryout_coefs,
                            Init_layer.carryout_biases,
                            Init_layer.carryout_LP,
                            layer.carryout_segment_len,
                            layer.carryout_gamma_dist_mean,
                            layer.carryout_gamma_dist_var],
        dtype=dtype)
 
    F_layer = exatrack.Final_layer(Init_layer.final_sequence_phase_1, nb_dims=nb_dims,
                                   sequence_length=sequence_length, dtype=dtype)
    outputs, All_states = F_layer(states)
 
    outputs = carryover_layer(outputs, [Prev_coefs, Prev_biases, LP, segment_len,
                                        gamma_dist_mean, gamma_dist_var])
 
    model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
                           outputs=outputs, name='Pair_diffusion_model')
    pred_model = tf.keras.Model(inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
                                outputs=(outputs, All_states, All_coefs, All_biases, All_LPs),
                                name='Pair_diffusion_model')
    return model, pred_model
 
 
# =====================================================================
# 4.  Reaction-diffusion simulation of two-colour SPT data
# ---------------------------------------------------------------------
# Two populations of particles diffuse in a field of view:
#   * particles A are imaged in channel 1
#   * particles B are imaged in channel 2
# An unbound A and an unbound B can bind ONLY when their separation is below a
# reaction radius `r_bind`, and while they are within that radius the binding is
# Markovian (constant hazard `k_on` per unit time -> exponential waiting time,
# NOT the position-independent gamma dwell time used previously).  A bound A-B
# complex co-diffuses (shared centre of mass with diffusion `D_bound`, relative
# coordinate held together as a stiff Ornstein-Uhlenbeck bond of width
# `bond_length`) and unbinds with a Markovian rate `k_off`.
#
# The output is NOT a list of pre-formed couples: each A and each B gives an
# independent single-particle track (as a real experiment would, after
# per-channel linking).  The task of the analysis is then to (a) propose
# candidate A-B couples that come close enough for long enough, and (b) let the
# pair model decide, per time point, which candidates are actually co-diffusing.
# =====================================================================
def _reflect(pos, box):
    """Reflecting boundary conditions on [0, box] per dimension.
 
    Positions stay continuous (no periodic wrapping), so the resulting tracks
    have Gaussian displacements everywhere and are directly usable by the pair
    model.  `pos` : (N, nb_dims), `box` : (nb_dims,)."""
    box = np.asarray(box, dtype='float64')
    two = 2.0 * box
    p = np.mod(pos, two)
    return np.where(p > box, two - p, p)
 
 
def simulate_reaction_diffusion(nb_A=200,
                                nb_B=200,
                                field_of_view=(10.0, 10.0),
                                track_len=50,
                                dt=0.02,
                                dt_std=0.0,
                                nb_dims=2,
                                D_A=0.1,
                                D_B=0.1,
                                D_bound=None,
                                r_bind=0.1,
                                k_on=200.0,
                                k_off=3.0,
                                bond_length=0.03,
                                bond_relax_rate=60.0,
                                LocErr=0.02,
                                LocErr_std=0.0,
                                nb_sub_steps=20,
                                nb_burnin_frames=40,
                                track_dropout=0.0,
                                min_track_len=5,
                                seed=None):
    """
    Simulate a two-colour reaction-diffusion SPT experiment.
 
    Parameters
    ----------
    nb_A, nb_B : int
        Numbers of channel-1 (A) and channel-2 (B) particles.
    field_of_view : sequence of length nb_dims
        Box size (um).  Reflecting boundaries keep the density constant.
    track_len : int
        Number of recorded frames.
    dt, dt_std : float
        Mean and standard deviation of the frame duration (s).
    D_A, D_B : float
        Free diffusion coefficients (um^2/s).
    D_bound : float or None
        Centre-of-mass diffusion of a bound complex.  Defaults to
        (D_A + D_B) / 4 (rigid-dimer-like slowdown).
    r_bind : float
        Reaction radius (um).  Binding can occur only below this separation.
    k_on : float
        Markovian binding hazard while within r_bind (per s).
    k_off : float
        Markovian unbinding hazard (per s) -> exponential bound lifetimes.
    bond_length : float
        Stationary width of the relative coordinate of a bound complex (um).
    bond_relax_rate : float
        Relaxation rate of the bond (per s); sets how tightly the two particles
        track each other within a frame.
    LocErr, LocErr_std : float
        Mean and spread of the per-localisation error (um).
    nb_sub_steps : int
        Sub-steps per frame for the continuous-time dynamics and reactions.
    nb_burnin_frames : int
        Number of frames simulated before recording, so the bound fraction
        reaches its steady state.
    track_dropout : float
        Fraction of the movie length over which a particle may randomly start
        late / end early (0 -> every particle is present for the whole movie).
    min_track_len : int
        Tracks shorter than this after dropout are discarded.
    seed : int or None
 
    Returns
    -------
    dict with keys
        tracks_A, tracks_B   : lists of (Ti, nb_dims) arrays (channels 1 and 2)
        frames_A, frames_B   : lists of (Ti,) integer frame indices
        LocErr_A, LocErr_B   : lists of (Ti,) per-localisation errors
        dt_array             : (track_len,) frame durations
        ids_A, ids_B         : original particle indices of each track
        bound_partner        : (track_len, nb_A) int, index of the B bound to
                               each A at each frame (-1 if free)
        true_couples         : dict (a, b) -> boolean (track_len,) co-diffusion
                               mask, restricted to frames where both tracks exist
        params               : echo of the physical parameters
    """
    rng = np.random.default_rng(seed)
    box = np.asarray(field_of_view, dtype='float64')[:nb_dims]
    if D_bound is None:
        D_bound = 0.25 * (D_A + D_B)
 
    posA = rng.random((nb_A, nb_dims)) * box
    posB = rng.random((nb_B, nb_dims)) * box
    partnerA = np.full(nb_A, -1, dtype=np.int64)   # index of bound B, or -1
    partnerB = np.full(nb_B, -1, dtype=np.int64)
 
    if dt_std > 0:
        scale = dt_std ** 2 / dt
        dt_array = np.clip(rng.gamma(dt / scale, scale, track_len), 0.2 * dt, 3 * dt)
    else:
        dt_array = np.full(track_len, dt)
 
    a_bond = np.exp(-bond_relax_rate * (dt_array.mean() / nb_sub_steps))
    bond_noise = bond_length * np.sqrt(max(1 - a_bond ** 2, 1e-12))
 
    def _substep(delta):
        nonlocal posA, posB, partnerA, partnerB
        a_b = np.exp(-bond_relax_rate * delta)
        b_noise = bond_length * np.sqrt(max(1 - a_b ** 2, 1e-12))
 
        # --- bound complexes : COM diffusion + stiff relative bond ---
        ba = np.where(partnerA >= 0)[0]
        if ba.size:
            bb = partnerA[ba]
            com = 0.5 * (posA[ba] + posB[bb])
            r = posA[ba] - posB[bb]
            com = com + rng.normal(0, np.sqrt(2 * D_bound * delta), com.shape)
            r = r * a_b + rng.normal(0, b_noise, r.shape)
            posA[ba] = com + 0.5 * r
            posB[bb] = com - 0.5 * r
 
        # --- free particles : independent Brownian steps ---
        fa = np.where(partnerA < 0)[0]
        posA[fa] += rng.normal(0, np.sqrt(2 * D_A * delta), (fa.size, nb_dims))
        fb = np.where(partnerB < 0)[0]
        posB[fb] += rng.normal(0, np.sqrt(2 * D_B * delta), (fb.size, nb_dims))
 
        posA[:] = _reflect(posA, box)
        posB[:] = _reflect(posB, box)
 
        # --- unbinding : Markovian k_off ---
        ba = np.where(partnerA >= 0)[0]
        if ba.size:
            off = rng.random(ba.size) < (1 - np.exp(-k_off * delta))
            for k in np.where(off)[0]:
                b = partnerA[ba[k]]
                partnerA[ba[k]] = -1
                partnerB[b] = -1
 
        # --- binding : radius-gated Markovian k_on ---
        fa = np.where(partnerA < 0)[0]
        fb = np.where(partnerB < 0)[0]
        if fa.size and fb.size:
            tB = cKDTree(posB[fb])
            near = cKDTree(posA[fa]).query_ball_tree(tB, r_bind)
            cand = [(fa[i], fb[j]) for i, lst in enumerate(near) for j in lst]
            if cand:
                rng.shuffle(cand)
                p_on = 1 - np.exp(-k_on * delta)
                for ai, bj in cand:
                    if partnerA[ai] == -1 and partnerB[bj] == -1 and rng.random() < p_on:
                        partnerA[ai] = bj
                        partnerB[bj] = ai
 
    # ---- burn-in to reach the binding/unbinding steady state ----
    burn_delta = dt / nb_sub_steps
    for _ in range(nb_burnin_frames * nb_sub_steps):
        _substep(burn_delta)
 
    # ---- recorded movie ----
    positions_A = np.zeros((track_len, nb_A, nb_dims))
    positions_B = np.zeros((track_len, nb_B, nb_dims))
    bound_partner = np.full((track_len, nb_A), -1, dtype=np.int64)
    for t in range(track_len):
        delta = dt_array[t] / nb_sub_steps
        for _ in range(nb_sub_steps):
            _substep(delta)
        positions_A[t] = posA
        positions_B[t] = posB
        bound_partner[t] = partnerA
 
    # ---- localisation error ----
    if LocErr_std > 0:
        scale = LocErr_std ** 2 / LocErr
        errA = rng.gamma(LocErr / scale, scale, (track_len, nb_A))
        errB = rng.gamma(LocErr / scale, scale, (track_len, nb_B))
    else:
        errA = np.full((track_len, nb_A), LocErr)
        errB = np.full((track_len, nb_B), LocErr)
    obsA = positions_A + rng.normal(0, 1, positions_A.shape) * errA[:, :, None]
    obsB = positions_B + rng.normal(0, 1, positions_B.shape) * errB[:, :, None]
 
    # ---- optional per-particle start/end dropout -> variable track lengths ----
    def _window(n):
        if track_dropout <= 0:
            return np.zeros(n, dtype=int), np.full(n, track_len, dtype=int)
        span = int(round(track_dropout * track_len))
        start = rng.integers(0, span + 1, n)
        end = track_len - rng.integers(0, span + 1, n)
        return start, np.maximum(end, start + 1)
 
    startA, endA = _window(nb_A)
    startB, endB = _window(nb_B)
 
    tracks_A, frames_A, LocErr_A, ids_A = [], [], [], []
    for i in range(nb_A):
        s, e = startA[i], endA[i]
        if e - s >= min_track_len:
            tracks_A.append(obsA[s:e, i])
            frames_A.append(np.arange(s, e))
            LocErr_A.append(errA[s:e, i])
            ids_A.append(i)
    tracks_B, frames_B, LocErr_B, ids_B = [], [], [], []
    for j in range(nb_B):
        s, e = startB[j], endB[j]
        if e - s >= min_track_len:
            tracks_B.append(obsB[s:e, j])
            frames_B.append(np.arange(s, e))
            LocErr_B.append(errB[s:e, j])
            ids_B.append(j)
 
    # ---- ground-truth co-diffusing couples (in original particle indices) ----
    true_couples = {}
    for t in range(track_len):
        for a in range(nb_A):
            b = bound_partner[t, a]
            if b >= 0:
                key = (a, b)
                if key not in true_couples:
                    true_couples[key] = np.zeros(track_len, dtype=bool)
                true_couples[key][t] = True
 
    return {'tracks_A': tracks_A, 'frames_A': frames_A, 'LocErr_A': LocErr_A, 'ids_A': ids_A,
            'tracks_B': tracks_B, 'frames_B': frames_B, 'LocErr_B': LocErr_B, 'ids_B': ids_B,
            'dt_array': dt_array, 'bound_partner': bound_partner,
            'true_couples': true_couples,
            'params': dict(nb_A=nb_A, nb_B=nb_B, field_of_view=tuple(box), track_len=track_len,
                           dt=dt, D_A=D_A, D_B=D_B, D_bound=D_bound, r_bind=r_bind,
                           k_on=k_on, k_off=k_off, bond_length=bond_length,
                           bond_relax_rate=bond_relax_rate, LocErr=LocErr)}
 
"""
Numba-accelerated two-colour reaction-diffusion SPT simulator.

Drop-in replacement for the original `simulate_reaction_diffusion`. Same
signature, same return dict. The physics/algorithm is unchanged; only the
per-substep integration + binding search (previously scipy.cKDTree + Python
loops, re-run tens of thousands of times) is replaced with a single
JIT-compiled kernel.

Notes on parity with the original
----------------------------------
* Numbers will NOT match the scipy/`default_rng` version bit-for-bit, even
  for the same `seed`: Numba's `np.random` inside `@njit` code uses its own
  internal Mersenne-Twister stream (a different algorithm from NumPy's
  default_rng/PCG64), and the grid-based binding search below visits
  candidate pairs in a different order than a KD-tree would, before they are
  shuffled anyway. The *statistics* (diffusion, bound fraction, lifetimes)
  are the same model and match the original within run-to-run noise.
* Binding candidates are found with a uniform grid / cell-list neighbour
  search (rebuilt every substep) instead of scipy's cKDTree. Cell size is
  chosen adaptively: large enough that the number of cells stays of order
  `nb_A + nb_B` (so rebuilding the grid every substep stays cheap), but never
  smaller than `r_bind` (which is what guarantees correctness -- any pair
  closer than `r_bind` is guaranteed to fall in the same or a directly
  touching cell, so only the 3**nb_dims neighbouring cells are ever checked).
* The two unused `a_bond` / `bond_noise` lines in the original code (computed
  from `dt_array.mean()` but never referenced inside `_substep`, which
  recomputes its own `a_b`/`b_noise` from the actual per-call `delta`) were
  dead code and have been dropped.
"""

import numpy as np
from numba import njit


# --------------------------------------------------------------------------
# Compiled kernel
# --------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _seed_numba_rng(seed):
    """Seed Numba's *internal* random stream (separate from NumPy's)."""
    np.random.seed(seed)


@njit(cache=True, fastmath=True)
def _reflect_inplace(pos, box):
    n, d = pos.shape
    for i in range(n):
        for k in range(d):
            L = box[k]
            period = 2.0 * L
            x = pos[i, k] % period
            if x > L:
                x = period - x
            pos[i, k] = x


# --------------------------------------------------------------------------
# Uniform grid / cell-list neighbour search for the binding step
# --------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _grid_params(box, nb_A, nb_B, r_bind, nb_dims):
    """Grid geometry, computed once and reused every substep (box, r_bind,
    nb_A, nb_B, nb_dims never change during a run)."""
    vol = 1.0
    for d in range(nb_dims):
        vol *= box[d]
    target_n = nb_A + nb_B
    if target_n < 1:
        target_n = 1
    # cell size for ~1 particle/cell on average, but never below r_bind
    # (cell_size >= r_bind is what makes checking only neighbouring cells safe)
    cell_size = (vol / target_n) ** (1.0 / nb_dims)
    if cell_size < r_bind:
        cell_size = r_bind

    ncells = np.empty(nb_dims, dtype=np.int64)
    for d in range(nb_dims):
        nc = int(box[d] / cell_size) + 1
        if nc < 1:
            nc = 1
        ncells[d] = nc

    strides = np.empty(nb_dims, dtype=np.int64)
    strides[nb_dims - 1] = 1
    for d in range(nb_dims - 2, -1, -1):
        strides[d] = strides[d + 1] * ncells[d + 1]
    total_cells = strides[0] * ncells[0]

    pow3 = 3 ** nb_dims
    offsets_table = np.empty((pow3, nb_dims), dtype=np.int64)
    for combo in range(pow3):
        rem = combo
        for d in range(nb_dims):
            digit = rem % 3 - 1
            rem //= 3
            offsets_table[combo, d] = digit

    return cell_size, ncells, strides, total_cells, offsets_table


@njit(cache=True, fastmath=True)
def _binding_search_grid(posA, posB, partnerA, partnerB, r_bind, cell_size,
                          ncells, strides, total_cells, offsets_table,
                          cell_of_B, counts, fill_pos, bucket, cell_coord_buf,
                          cand_ai, cand_bj):
    nb_A, nb_dims = posA.shape
    nb_B = posB.shape[0]
    pow3 = offsets_table.shape[0]
    max_cand = cand_ai.shape[0]

    # --- bucket free B particles into grid cells (counting sort / CSR) ---
    for i in range(total_cells + 1):
        counts[i] = 0

    for b in range(nb_B):
        if partnerB[b] < 0:
            cidx = 0
            for d in range(nb_dims):
                c = int(posB[b, d] / cell_size)
                if c < 0:
                    c = 0
                elif c >= ncells[d]:
                    c = ncells[d] - 1
                cidx += c * strides[d]
            cell_of_B[b] = cidx
            counts[cidx + 1] += 1
        else:
            cell_of_B[b] = -1

    for i in range(1, total_cells + 1):
        counts[i] += counts[i - 1]

    for i in range(total_cells):
        fill_pos[i] = counts[i]
    for b in range(nb_B):
        c = cell_of_B[b]
        if c >= 0:
            p = fill_pos[c]
            bucket[p] = b
            fill_pos[c] += 1

    # --- query: each free A particle checks its cell + neighbouring cells ---
    r2 = r_bind * r_bind
    n_cand = 0
    for a in range(nb_A):
        if partnerA[a] < 0:
            for d in range(nb_dims):
                c = int(posA[a, d] / cell_size)
                if c < 0:
                    c = 0
                elif c >= ncells[d]:
                    c = ncells[d] - 1
                cell_coord_buf[d] = c
            for combo in range(pow3):
                valid = True
                cidx = 0
                for d in range(nb_dims):
                    nc = cell_coord_buf[d] + offsets_table[combo, d]
                    if nc < 0 or nc >= ncells[d]:
                        valid = False
                        break
                    cidx += nc * strides[d]
                if valid:
                    start = counts[cidx]
                    end = counts[cidx + 1]
                    for kk in range(start, end):
                        b = bucket[kk]
                        d2 = 0.0
                        for d in range(nb_dims):
                            diff = posA[a, d] - posB[b, d]
                            d2 += diff * diff
                        if d2 < r2 and n_cand < max_cand:
                            cand_ai[n_cand] = a
                            cand_bj[n_cand] = b
                            n_cand += 1
    return n_cand


@njit(cache=True, fastmath=True)
def _substep(posA, posB, partnerA, partnerB, box,
             D_A, D_B, D_bound, r_bind, k_on, k_off,
             bond_length, bond_relax_rate, delta,
             cell_size, ncells, strides, total_cells, offsets_table,
             cell_of_B, counts, fill_pos, bucket, cell_coord_buf,
             cand_ai, cand_bj):
    nb_A, nb_dims = posA.shape
    nb_B = posB.shape[0]

    a_b = np.exp(-bond_relax_rate * delta)
    b_noise = bond_length * np.sqrt(max(1.0 - a_b * a_b, 1e-12))
    sigma_com = np.sqrt(2.0 * D_bound * delta)
    sigma_A = np.sqrt(2.0 * D_A * delta)
    sigma_B = np.sqrt(2.0 * D_B * delta)

    # --- bound complexes: COM diffusion + relaxing relative bond ---
    for a in range(nb_A):
        b = partnerA[a]
        if b >= 0:
            for k in range(nb_dims):
                com = 0.5 * (posA[a, k] + posB[b, k])
                r = posA[a, k] - posB[b, k]
                com += np.random.normal(0.0, sigma_com)
                r = r * a_b + np.random.normal(0.0, b_noise)
                posA[a, k] = com + 0.5 * r
                posB[b, k] = com - 0.5 * r

    # --- free particles: independent Brownian steps ---
    for a in range(nb_A):
        if partnerA[a] < 0:
            for k in range(nb_dims):
                posA[a, k] += np.random.normal(0.0, sigma_A)
    for b in range(nb_B):
        if partnerB[b] < 0:
            for k in range(nb_dims):
                posB[b, k] += np.random.normal(0.0, sigma_B)

    _reflect_inplace(posA, box)
    _reflect_inplace(posB, box)

    # --- unbinding: Markovian k_off ---
    p_off = 1.0 - np.exp(-k_off * delta)
    for a in range(nb_A):
        b = partnerA[a]
        if b >= 0:
            if np.random.random() < p_off:
                partnerA[a] = -1
                partnerB[b] = -1

    # --- binding: radius-gated Markovian k_on (grid/cell-list neighbour scan) ---
    n_cand = _binding_search_grid(posA, posB, partnerA, partnerB, r_bind, cell_size,
                                   ncells, strides, total_cells, offsets_table,
                                   cell_of_B, counts, fill_pos, bucket, cell_coord_buf,
                                   cand_ai, cand_bj)

    if n_cand > 0:
        perm = np.arange(n_cand)
        np.random.shuffle(perm)
        p_on = 1.0 - np.exp(-k_on * delta)
        for ii in range(n_cand):
            m = perm[ii]
            ai = cand_ai[m]
            bj = cand_bj[m]
            if partnerA[ai] < 0 and partnerB[bj] < 0:
                if np.random.random() < p_on:
                    partnerA[ai] = bj
                    partnerB[bj] = ai


@njit(cache=True, fastmath=True)
def _run_core(posA, posB, partnerA, partnerB, box,
              D_A, D_B, D_bound, r_bind, k_on, k_off,
              bond_length, bond_relax_rate,
              nb_sub_steps, nb_burnin_frames, burn_delta,
              dt_array, positions_A_out, positions_B_out, bound_partner_out):
    nb_A, nb_dims = posA.shape
    nb_B = posB.shape[0]
    track_len = dt_array.shape[0]

    # grid geometry is invariant for the whole run (box, r_bind, nb_A, nb_B,
    # nb_dims never change) -> compute once and reuse every substep
    cell_size, ncells, strides, total_cells, offsets_table = _grid_params(
        box, nb_A, nb_B, r_bind, nb_dims)

    # scratch buffers reused across every substep call (no repeated allocation)
    cell_of_B = np.empty(nb_B, dtype=np.int64)
    counts = np.empty(total_cells + 1, dtype=np.int64)
    fill_pos = np.empty(total_cells, dtype=np.int64)
    bucket = np.empty(nb_B, dtype=np.int64)
    cell_coord_buf = np.empty(nb_dims, dtype=np.int64)
    cand_ai = np.empty(nb_A * nb_B, dtype=np.int64)
    cand_bj = np.empty(nb_A * nb_B, dtype=np.int64)

    # ---- burn-in to reach the binding/unbinding steady state ----
    for _ in range(nb_burnin_frames * nb_sub_steps):
        _substep(posA, posB, partnerA, partnerB, box, D_A, D_B, D_bound,
                  r_bind, k_on, k_off, bond_length, bond_relax_rate,
                  burn_delta,
                  cell_size, ncells, strides, total_cells, offsets_table,
                  cell_of_B, counts, fill_pos, bucket, cell_coord_buf,
                  cand_ai, cand_bj)

    # ---- recorded movie ----
    for t in range(track_len):
        delta = dt_array[t] / nb_sub_steps
        for _ in range(nb_sub_steps):
            _substep(posA, posB, partnerA, partnerB, box, D_A, D_B, D_bound,
                      r_bind, k_on, k_off, bond_length, bond_relax_rate,
                      delta,
                      cell_size, ncells, strides, total_cells, offsets_table,
                      cell_of_B, counts, fill_pos, bucket, cell_coord_buf,
                      cand_ai, cand_bj)
        positions_A_out[t] = posA
        positions_B_out[t] = posB
        bound_partner_out[t] = partnerA


# --------------------------------------------------------------------------
# Public, drop-in function (same signature/return dict as the original)
# --------------------------------------------------------------------------

def simulate_reaction_diffusion(nb_A=200,
                                 nb_B=200,
                                 field_of_view=(10.0, 10.0),
                                 track_len=50,
                                 dt=0.02,
                                 dt_std=0.0,
                                 nb_dims=2,
                                 D_A=0.1,
                                 D_B=0.1,
                                 D_bound=None,
                                 r_bind=0.1,
                                 k_on=200.0,
                                 k_off=3.0,
                                 bond_length=0.03,
                                 bond_relax_rate=60.0,
                                 LocErr=0.02,
                                 LocErr_std=0.0,
                                 nb_sub_steps=20,
                                 nb_burnin_frames=40,
                                 track_dropout=0.0,
                                 min_track_len=5,
                                 seed=None):
    """
    Simulate a two-colour reaction-diffusion SPT experiment (Numba-accelerated).

    Same parameters and return value as the original implementation -- see
    module docstring for the caveats on exact reproducibility of a given
    `seed` relative to the scipy/cKDTree version.

    Parameters
    ----------
    nb_A, nb_B : int
        Numbers of channel-1 (A) and channel-2 (B) particles.
    field_of_view : sequence of length nb_dims
        Box size (um).  Reflecting boundaries keep the density constant.
    track_len : int
        Number of recorded frames.
    dt, dt_std : float
        Mean and standard deviation of the frame duration (s).
    D_A, D_B : float
        Free diffusion coefficients (um^2/s).
    D_bound : float or None
        Centre-of-mass diffusion of a bound complex.  Defaults to
        (D_A + D_B) / 4 (rigid-dimer-like slowdown).
    r_bind : float
        Reaction radius (um).  Binding can occur only below this separation.
    k_on : float
        Markovian binding hazard while within r_bind (per s).
    k_off : float
        Markovian unbinding hazard (per s) -> exponential bound lifetimes.
    bond_length : float
        Stationary width of the relative coordinate of a bound complex (um).
    bond_relax_rate : float
        Relaxation rate of the bond (per s); sets how tightly the two particles
        track each other within a frame.
    LocErr, LocErr_std : float
        Mean and spread of the per-localisation error (um).
    nb_sub_steps : int
        Sub-steps per frame for the continuous-time dynamics and reactions.
    nb_burnin_frames : int
        Number of frames simulated before recording, so the bound fraction
        reaches its steady state.
    track_dropout : float
        Fraction of the movie length over which a particle may randomly start
        late / end early (0 -> every particle is present for the whole movie).
    min_track_len : int
        Tracks shorter than this after dropout are discarded.
    seed : int or None

    Returns
    -------
    dict with keys
        tracks_A, tracks_B   : lists of (Ti, nb_dims) arrays (channels 1 and 2)
        frames_A, frames_B   : lists of (Ti,) integer frame indices
        LocErr_A, LocErr_B   : lists of (Ti,) per-localisation errors
        dt_array             : (track_len,) frame durations
        ids_A, ids_B         : original particle indices of each track
        bound_partner        : (track_len, nb_A) int, index of the B bound to
                               each A at each frame (-1 if free)
        true_couples         : dict (a, b) -> boolean (track_len,) co-diffusion
                               mask, restricted to frames where both tracks exist
        params               : echo of the physical parameters
    """
    rng = np.random.default_rng(seed)
    box = np.asarray(field_of_view, dtype='float64')[:nb_dims]
    if D_bound is None:
        D_bound = 0.25 * (D_A + D_B)

    posA = rng.random((nb_A, nb_dims)) * box
    posB = rng.random((nb_B, nb_dims)) * box
    partnerA = np.full(nb_A, -1, dtype=np.int64)   # index of bound B, or -1
    partnerB = np.full(nb_B, -1, dtype=np.int64)

    if dt_std > 0:
        scale = dt_std ** 2 / dt
        dt_array = np.clip(rng.gamma(dt / scale, scale, track_len), 0.2 * dt, 3 * dt)
    else:
        dt_array = np.full(track_len, dt)

    # seed Numba's own RNG stream (independent of `rng` above) so the whole
    # simulation is reproducible for a fixed `seed`
    numba_seed = int(rng.integers(0, 2 ** 31 - 1))
    _seed_numba_rng(numba_seed)

    burn_delta = dt / nb_sub_steps

    positions_A = np.zeros((track_len, nb_A, nb_dims))
    positions_B = np.zeros((track_len, nb_B, nb_dims))
    bound_partner = np.full((track_len, nb_A), -1, dtype=np.int64)

    _run_core(posA, posB, partnerA, partnerB, box,
              float(D_A), float(D_B), float(D_bound), float(r_bind),
              float(k_on), float(k_off), float(bond_length), float(bond_relax_rate),
              int(nb_sub_steps), int(nb_burnin_frames), float(burn_delta),
              dt_array, positions_A, positions_B, bound_partner)

    # ---- localisation error ----
    if LocErr_std > 0:
        scale = LocErr_std ** 2 / LocErr
        errA = rng.gamma(LocErr / scale, scale, (track_len, nb_A))
        errB = rng.gamma(LocErr / scale, scale, (track_len, nb_B))
    else:
        errA = np.full((track_len, nb_A), LocErr)
        errB = np.full((track_len, nb_B), LocErr)
    obsA = positions_A + rng.normal(0, 1, positions_A.shape) * errA[:, :, None]
    obsB = positions_B + rng.normal(0, 1, positions_B.shape) * errB[:, :, None]

    # ---- optional per-particle start/end dropout -> variable track lengths ----
    def _window(n):
        if track_dropout <= 0:
            return np.zeros(n, dtype=int), np.full(n, track_len, dtype=int)
        span = int(round(track_dropout * track_len))
        start = rng.integers(0, span + 1, n)
        end = track_len - rng.integers(0, span + 1, n)
        return start, np.maximum(end, start + 1)

    startA, endA = _window(nb_A)
    startB, endB = _window(nb_B)

    tracks_A, frames_A, LocErr_A, ids_A = [], [], [], []
    for i in range(nb_A):
        s, e = startA[i], endA[i]
        if e - s >= min_track_len:
            tracks_A.append(obsA[s:e, i])
            frames_A.append(np.arange(s, e))
            LocErr_A.append(errA[s:e, i])
            ids_A.append(i)
    tracks_B, frames_B, LocErr_B, ids_B = [], [], [], []
    for j in range(nb_B):
        s, e = startB[j], endB[j]
        if e - s >= min_track_len:
            tracks_B.append(obsB[s:e, j])
            frames_B.append(np.arange(s, e))
            LocErr_B.append(errB[s:e, j])
            ids_B.append(j)

    # ---- ground-truth co-diffusing couples (in original particle indices) ----
    true_couples = {}
    for t in range(track_len):
        for a in range(nb_A):
            b = bound_partner[t, a]
            if b >= 0:
                key = (a, b)
                if key not in true_couples:
                    true_couples[key] = np.zeros(track_len, dtype=bool)
                true_couples[key][t] = True

    return {'tracks_A': tracks_A, 'frames_A': frames_A, 'LocErr_A': LocErr_A, 'ids_A': ids_A,
            'tracks_B': tracks_B, 'frames_B': frames_B, 'LocErr_B': LocErr_B, 'ids_B': ids_B,
            'dt_array': dt_array, 'bound_partner': bound_partner,
            'true_couples': true_couples,
            'params': dict(nb_A=nb_A, nb_B=nb_B, field_of_view=tuple(box), track_len=track_len,
                            dt=dt, D_A=D_A, D_B=D_B, D_bound=D_bound, r_bind=r_bind,
                            k_on=k_on, k_off=k_off, bond_length=bond_length,
                            bond_relax_rate=bond_relax_rate, LocErr=LocErr)}    



 
def reaction_bound_fraction(res):
    """Fraction of A particles that are bound, averaged over the recorded movie."""
    bp = res['bound_partner']
    return float(np.mean(bp >= 0))
 
 
def tune_reaction_density(target_fraction=0.5,
                          nb_A=150, nb_B=150,
                          field_of_view=(10.0, 10.0),
                          track_len=20,
                          nb_burnin_frames=30,
                          box_range=(3.0, 25.0),
                          n_iter=8,
                          verbose=True,
                          **kwargs):
    """
    Find the (square) field-of-view size that yields roughly `target_fraction`
    of bound A particles, by bisection on the box size (smaller box -> higher
    density -> more binding).  Returns (field_of_view, achieved_fraction).
 
    All other physical parameters are passed through `kwargs` to
    `simulate_reaction_diffusion`.
    """
    lo, hi = box_range
    nb_dims = len(field_of_view)
    best = None
    for it in range(n_iter):
        L = 0.5 * (lo + hi)
        res = simulate_reaction_diffusion(nb_A=nb_A, nb_B=nb_B,
                                          field_of_view=(L,) * nb_dims,
                                          track_len=track_len,
                                          nb_burnin_frames=nb_burnin_frames,
                                          seed=it, **kwargs)
        f = reaction_bound_fraction(res)
        if verbose:
            print('  [tune %d] box = %.2f um -> bound fraction = %.3f' % (it, L, f))
        if best is None or abs(f - target_fraction) < abs(best[1] - target_fraction):
            best = ((L,) * nb_dims, f)
        if f > target_fraction:      # too dense -> enlarge the box
            lo = L
        else:                        # too dilute -> shrink the box
            hi = L
    return best
 
 
# =====================================================================
# 5.  Candidate co-diffusing couples
# ---------------------------------------------------------------------
# From the two channels we build the couples worth testing: an (A, B) pair is a
# candidate when the two tracks overlap in time and come within `search_radius`
# for at least `min_overlap` frames.  Neighbour search is done frame by frame
# with a KD-tree so the cost scales with the number of localisations, not with
# nb_A * nb_B.  Each candidate is returned already aligned on its common frame
# window in the (x1, y1, x2, y2) particle-major layout expected by the pair model.
# =====================================================================
class Candidate(dict):
    """A candidate co-diffusing couple, with attribute access for convenience."""
    __getattr__ = dict.get
 
 
def find_codiffusion_candidates(res,
                                search_radius=None,
                                min_overlap=5):
    """
    Build candidate couples from a `simulate_reaction_diffusion` result (or any
    dict exposing tracks_A/frames_A/LocErr_A and the B counterparts).
 
    Parameters
    ----------
    res : dict
        Output of `simulate_reaction_diffusion` (only the track/frame/LocErr
        lists and dt_array are used).
    search_radius : float or None
        Two localisations are neighbours if closer than this.  Defaults to
        5 * r_bind (falls back to a few times the typical step when r_bind is
        unknown), which is generous enough to catch every truly bound couple.
    min_overlap : int
        Minimum number of frames within `search_radius` to keep a couple.
 
    Returns
    -------
    list[Candidate]
        Each candidate has:
            idA, idB       : original particle indices
            trackA_index,
            trackB_index   : indices into res['tracks_A'] / res['tracks_B']
            frames         : (L,) integer frames of the aligned window
            track          : (L, 2*nb_dims) particle-major [x1,y1,x2,y2]
            LocErr         : (L, 2) per-particle localisation error
            dt             : (L,) frame durations
            mask           : (L,) ones (aligned window is contiguous)
            min_dist       : minimum A-B distance over the window
            n_close        : number of frames within search_radius
    """
    tracks_A, frames_A, LocErr_A = res['tracks_A'], res['frames_A'], res['LocErr_A']
    tracks_B, frames_B, LocErr_B = res['tracks_B'], res['frames_B'], res['LocErr_B']
    ids_A = res.get('ids_A', list(range(len(tracks_A))))
    ids_B = res.get('ids_B', list(range(len(tracks_B))))
    dt_array = res.get('dt_array')
    nb_dims = tracks_A[0].shape[1]
 
    if search_radius is None:
        r_bind = res.get('params', {}).get('r_bind', None)
        search_radius = 5 * r_bind if r_bind is not None else 5 * np.median(
            [np.sqrt(np.mean(np.sum(np.diff(t, axis=0) ** 2, 1))) for t in tracks_A])
 
    # frame -> lists of (track_index, position) for each channel
    max_frame = max(int(f[-1]) for f in frames_A + frames_B) + 1
    perframe_A = [[] for _ in range(max_frame)]
    perframe_B = [[] for _ in range(max_frame)]
    for ti, (tr, fr) in enumerate(zip(tracks_A, frames_A)):
        for k, f in enumerate(fr):
            perframe_A[f].append((ti, tr[k]))
    for tj, (tr, fr) in enumerate(zip(tracks_B, frames_B)):
        for k, f in enumerate(fr):
            perframe_B[f].append((tj, tr[k]))
 
    counts = {}          # (ti, tj) -> number of frames within search_radius
    for f in range(max_frame):
        la, lb = perframe_A[f], perframe_B[f]
        if not la or not lb:
            continue
        idxA = np.array([x[0] for x in la])
        idxB = np.array([x[0] for x in lb])
        posA = np.array([x[1] for x in la])
        posB = np.array([x[1] for x in lb])
        near = cKDTree(posA).query_ball_tree(cKDTree(posB), search_radius)
        for i, lst in enumerate(near):
            for j in lst:
                counts[(idxA[i], idxB[j])] = counts.get((idxA[i], idxB[j]), 0) + 1
 
    candidates = []
    for (ti, tj), n_close in counts.items():
        if n_close < min_overlap:
            continue
        fA, fB = frames_A[ti], frames_B[tj]
        f0 = max(fA[0], fB[0])
        f1 = min(fA[-1], fB[-1])
        if f1 - f0 + 1 < min_overlap:
            continue
        frames = np.arange(f0, f1 + 1)
        ia = f0 - fA[0]
        ib = f0 - fB[0]
        L = len(frames)
        segA = tracks_A[ti][ia:ia + L]
        segB = tracks_B[tj][ib:ib + L]
        errA = np.asarray(LocErr_A[ti])[ia:ia + L]
        errB = np.asarray(LocErr_B[tj])[ib:ib + L]
        track = np.concatenate([segA, segB], axis=1)         # [x1,y1,x2,y2]
        LocErr = np.stack([errA, errB], axis=1)              # (L, 2)
        if dt_array is not None:
            dt = dt_array[frames]
        else:
            dt = np.full(L, res.get('params', {}).get('dt', 1.0))
        dist = np.sum((segA - segB) ** 2, axis=1) ** 0.5
        candidates.append(Candidate(
            idA=ids_A[ti], idB=ids_B[tj],
            trackA_index=ti, trackB_index=tj,
            frames=frames, track=track, LocErr=LocErr, dt=dt,
            mask=np.ones(L), min_dist=float(dist.min()), n_close=int(n_close)))
    return candidates
 
 
def candidate_recall(candidates, res):
    """
    Fraction of ground-truth co-diffusing couples that appear among the
    candidates, plus the candidate count and the number of ground-truth couples.
    """
    truth = set(res['true_couples'].keys())
    found = set((c['idA'], c['idB']) for c in candidates)
    recovered = truth & found
    recall = len(recovered) / max(1, len(truth))
    return {'recall': recall, 'nb_true_couples': len(truth),
            'nb_candidates': len(candidates), 'nb_recovered': len(recovered)}
 
 
# =====================================================================
# 6.  Scoring / classification of candidate couples
# ---------------------------------------------------------------------
# `couple_state_loglik` runs the exact pair engine over one candidate for every
# physical state; `score_candidates` turns that into a per-couple log Bayes
# factor in favour of co-diffusion.  These give a model-based score without any
# training and are also what the offline tests use.  At runtime the trained
# two-state model provides the per-time-point co-diffusion probability instead
# (see `predict_codiffusion`).
# =====================================================================
def _identity_LocErr(LocErrs, LocErr_param):
    return LocErrs * LocErr_param
 
 
def couple_state_loglik(params, initial_params, track, LocErr, dt, reference_dt,
                        nb_dims, constraint_function=None, schedules=None,
                        blur_ratio=0.0):
    """
    Total log-likelihood of one aligned couple under every physical state of
    `params`.  `params` must be in the pair layout and its LAST row is treated
    as the mislinking placeholder (as inside the model), so provide
    nb_physical_states + 1 rows and read back only the first ones.
 
    Returns an array of length params.shape[0] (log-likelihood per state).
    """
    dtype = 'float64'
    if constraint_function is None:
        constraint_function = make_pair_constraint_function(nb_dims)
    if schedules is None:
        schedules = exatrack.get_sequences(params, initial_params,
                                           constraint_function, 6, 4, blur_ratio, dtype)
    init1, init2, rec1, rec2, final1 = schedules[0], schedules[1], schedules[2], schedules[3], schedules[4]
 
    T = track.shape[0]
    obs = track.reshape(T, 2, nb_dims).transpose(0, 2, 1)      # (T, nb_dims, 2)
    LocErr_in = np.asarray(LocErr)[None]                        # (1, T, 2)
    dt_in = np.concatenate([dt, dt[-1:]])[None]                # (1, T+1)
 
    out = constraint_function(tf.constant(params), tf.constant(initial_params),
                              tf.constant(LocErr_in), tf.constant(dt_in),
                              nb_dims, reference_dt, _identity_LocErr, blur_ratio, dtype)
    (hidden_vars, obs_vars, _, biases, initial_hidden_vars, initial_obs_vars,
     _, initial_biases, _, _, _, _, Log_factors, initial_Log_factors, _) = out
 
    inp = tf.constant(obs[:, None, None, None], dtype=dtype)    # (T,1,1,1,D,2)
    cur = tf.concat((initial_hidden_vars, hidden_vars[0][..., :4]), 0)
    nxt = tf.concat((tf.zeros_like(initial_hidden_vars), hidden_vars[0][..., 4:]), 0)
    bb = tf.concat((initial_biases + tf.reduce_sum(initial_obs_vars * inp[0], -1),
                    biases[0] + tf.reduce_sum(obs_vars[0] * inp[0], -1)), 0)
    P, B, LC = exatrack.RNN_reccurence_formula(cur, nxt, bb, init1, init2, dtype=dtype)
    LP = LC + initial_Log_factors
    for t in range(1, T):
        cur = tf.concat((P, hidden_vars[t][..., :4]), 0)
        nxt = tf.concat((tf.zeros_like(P), hidden_vars[t][..., 4:]), 0)
        bb = tf.concat((B, biases[t] + tf.reduce_sum(obs_vars[t] * inp[t], -1)), 0)
        P, B, LC = exatrack.RNN_reccurence_formula(cur, nxt, bb, rec1, rec2, dtype=dtype)
        LP = LP + LC + Log_factors[t]
    _, _, LC = exatrack.RNN_reccurence_formula(P, tf.zeros_like(P), B, final1, [[], []], dtype=dtype)
    return np.array(LP + LC)[0]
 
 
def score_candidates(candidates, interacting_params, independent_params,
                     initial_params, reference_dt, nb_dims, blur_ratio=0.0,
                     mislink_params=None):
    """
    Per-couple log Bayes factor log P(couple | co-diffusing) - log P(couple |
    independent), computed with the exact pair engine and untrained (physical)
    parameters.  A positive value favours co-diffusion.
 
    interacting_params / independent_params : single parameter rows (shape (11,)
        for 2D with one LocErr column per particle) describing the two hypotheses.
    """
    interacting_params = np.atleast_2d(interacting_params)
    independent_params = np.atleast_2d(independent_params)
    if mislink_params is None:
        mislink_params = independent_params.copy()
    params = np.concatenate([interacting_params, independent_params, mislink_params], 0)
    initial_params = np.asarray(initial_params)
    if initial_params.shape[0] != 3:
        initial_params = np.repeat(initial_params[:1], 3, axis=0)
 
    cf = make_pair_constraint_function(nb_dims)
    schedules = exatrack.get_sequences(params, initial_params, cf, 6, 4, blur_ratio, 'float64')
 
    scores = np.zeros(len(candidates))
    for k, c in enumerate(candidates):
        ll = couple_state_loglik(params, initial_params, c['track'], c['LocErr'],
                                 c['dt'], reference_dt, nb_dims, cf, schedules, blur_ratio)
        scores[k] = ll[0] - ll[1]
        c['score'] = float(scores[k])
    return scores
 
 
# =====================================================================
# 7.  Running the trained pair model on the candidate couples
# =====================================================================
def build_codiffusion_sequence(candidates, batch_size, segment_length,
                               min_segment_length=2, cutoff_batch_treshhold=None):
    """
    Wrap the candidate couples in a `TrackSegmentSequence` ready for the pair
    model.  Candidates shorter than `min_segment_length` are dropped; the
    surviving ones are returned so predictions can be mapped back.
 
    Returns (sequence, kept_candidates).
    """
    kept = [c for c in candidates if len(c['frames']) >= min_segment_length]
    track_list = [c['track'] for c in kept]
    LocErr_list = [c['LocErr'] for c in kept]
    dt_list = [c['dt'] for c in kept]
    if cutoff_batch_treshhold is None:
        cutoff_batch_treshhold = 0.5 / batch_size      # keep every couple
    seq = exatrack.TrackSegmentSequence(
        track_list, LocErr_list=LocErr_list, dt_list=dt_list,
        batch_size=batch_size, segment_length=segment_length,
        min_segment_length=min_segment_length,
        cutoff_batch_treshhold=cutoff_batch_treshhold)
    return seq, kept
 
 
def predict_codiffusion(pred_model, candidates, batch_size, segment_length=None,
                        min_segment_length=2):
    """
    Run the trained pair `pred_model` on the candidate couples and attach a
    per-frame co-diffusion probability to each kept candidate.
 
    `segment_length` should be >= the longest candidate (defaults to it) so that
    every couple is processed as a single segment and predictions map back in
    candidate order.
 
    Returns the list of kept candidates, each with an added field
    `p_codiffusion` : (L,) array = P(co-diffusing) at every frame of its window.
    """
    if segment_length is None:
        segment_length = max(len(c['frames']) for c in candidates)
    seq, kept = build_codiffusion_sequence(candidates, batch_size, segment_length,
                                           min_segment_length)
    _, preds, _, _, _ = pred_model.predict(seq)
 
    masks = np.concatenate([seq[i][0][3] for i in range(len(seq))], axis=0)
    # preds order matches candidate order (see run script reconstruction)
    p_bound = preds[..., 0]                          # state 0 = co-diffusing
    row = 0
    for c in kept:
        L = len(c['frames'])
        m = masks[row].astype(bool)
        c['p_codiffusion'] = p_bound[row][m][:L]
        row += 1
    return kept
 
 
def codiffusion_report(candidates, min_fraction=0.3, min_run=3,
                       use='p_codiffusion'):
    """
    Turn scored/predicted candidates into the final list of couples that are
    potentially co-diffusing.
 
    A couple is reported when either
      * `use == 'p_codiffusion'` and it spends at least `min_fraction` of its
        window co-diffusing (P > 0.5) with a contiguous co-diffusing run of at
        least `min_run` frames, or
      * `use == 'score'` and its log Bayes factor is positive.
 
    Returns a list of dicts sorted by decreasing evidence:
        idA, idB, frames (first, last), overlap, min_dist,
        codiff_fraction / score, and the per-frame probability when available.
    """
    report = []
    for c in candidates:
        entry = {'idA': c['idA'], 'idB': c['idB'],
                 'first_frame': int(c['frames'][0]), 'last_frame': int(c['frames'][-1]),
                 'overlap': len(c['frames']), 'min_dist': c['min_dist']}
        if use == 'p_codiffusion' and 'p_codiffusion' in c:
            p = np.asarray(c['p_codiffusion'])
            bound = p > 0.5
            frac = float(bound.mean())
            longest = _longest_true_run(bound)
            entry['codiff_fraction'] = frac
            entry['longest_run'] = longest
            entry['p_codiffusion'] = p
            keep = frac >= min_fraction and longest >= min_run
            entry['evidence'] = frac
        else:
            entry['score'] = c.get('score', np.nan)
            keep = entry['score'] > 0
            entry['evidence'] = entry['score']
        if keep:
            report.append(entry)
    report.sort(key=lambda e: e['evidence'], reverse=True)
    return report
 
 
def _longest_true_run(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best
 
 
# =====================================================================
# 8.  Two-body helper: radius-gated Markovian binding (quick pair generator)
# ---------------------------------------------------------------------
# Kept for convenience / for exercising the pair model directly on pre-formed
# couples.  Unlike the previous version, entering and leaving the co-diffusing
# state now follows the same physics as the full simulation: binding is possible
# only while the two particles are within `r_bind` and is Markovian there.
# =====================================================================
def simulate_codiffusion(nb_pairs=500,
                         track_len=50,
                         nb_dims=2,
                         dt=0.02,
                         dt_std=0.0,
                         D_A=0.1,
                         D_B=0.05,
                         D_bound=None,
                         r_bind=0.1,
                         k_on=200.0,
                         k_off=3.0,
                         bond_length=0.03,
                         bond_relax_rate=60.0,
                         box_size=0.6,
                         LocErr=0.02,
                         LocErr_std=0.0,
                         nb_sub_steps=20,
                         seed=None):
    """
    Simulate `nb_pairs` A-B couples confined together in a small reflecting box
    (so they encounter each other repeatedly) with radius-gated Markovian
    binding.  Returns pre-aligned couples, matching the pair-model layout, for
    quick tests of the model itself.
 
    Returns
    -------
    track_list  : list of (track_len, 2*nb_dims) arrays, columns [x1,y1,x2,y2]
    LocErr_list : list of (track_len, 2) arrays
    dt_list     : list of (track_len,) arrays
    state_list  : list of (track_len,) arrays, 0 = co-diffusing, 1 = free
    """
    rng = np.random.default_rng(seed)
    box = np.full(nb_dims, box_size, dtype='float64')
    if D_bound is None:
        D_bound = 0.25 * (D_A + D_B)
 
    track_list, LocErr_list, dt_list, state_list = [], [], [], []
    for _ in range(nb_pairs):
        if dt_std > 0:
            scale = dt_std ** 2 / dt
            dts = np.clip(rng.gamma(dt / scale, scale, track_len), 0.2 * dt, 3 * dt)
        else:
            dts = np.full(track_len, dt)
 
        x1 = rng.random(nb_dims) * box
        x2 = rng.random(nb_dims) * box
        bound = False
        positions = np.zeros((track_len, 2, nb_dims))
        states = np.zeros(track_len, dtype=int)
 
        for t in range(track_len):
            delta = dts[t] / nb_sub_steps
            a_b = np.exp(-bond_relax_rate * delta)
            b_noise = bond_length * np.sqrt(max(1 - a_b ** 2, 1e-12))
            for _s in range(nb_sub_steps):
                if bound:
                    com = 0.5 * (x1 + x2) + rng.normal(0, np.sqrt(2 * D_bound * delta), nb_dims)
                    r = (x1 - x2) * a_b + rng.normal(0, b_noise, nb_dims)
                    x1 = com + 0.5 * r
                    x2 = com - 0.5 * r
                    if rng.random() < 1 - np.exp(-k_off * delta):
                        bound = False
                else:
                    x1 = x1 + rng.normal(0, np.sqrt(2 * D_A * delta), nb_dims)
                    x2 = x2 + rng.normal(0, np.sqrt(2 * D_B * delta), nb_dims)
                    if np.sum((x1 - x2) ** 2) ** 0.5 < r_bind:
                        if rng.random() < 1 - np.exp(-k_on * delta):
                            bound = True
                x1 = _reflect(x1[None], box)[0]
                x2 = _reflect(x2[None], box)[0]
            positions[t, 0] = x1
            positions[t, 1] = x2
            states[t] = 0 if bound else 1
 
        if LocErr_std > 0:
            scale = LocErr_std ** 2 / LocErr
            errs = rng.gamma(LocErr / scale, scale, (track_len, 2))
        else:
            errs = np.full((track_len, 2), LocErr)
        noisy = positions + rng.normal(0, 1, positions.shape) * errs[:, :, None]
        track_list.append(noisy.reshape(track_len, 2 * nb_dims))
        LocErr_list.append(errs)
        dt_list.append(dts)
        state_list.append(states)
 
    return track_list, LocErr_list, dt_list, state_list
 
 
def interaction_rate_to_l_int(rate, reference_dt):
    """Continuous interaction / bond relaxation rate (per time unit) -> per-frame factor."""
    return 1 - np.exp(-np.asarray(rate, dtype='float64') * reference_dt)
 
 
def pair_separation(track):
    """Distance between the two particles of a (track_len, 2*nb_dims) array."""
    track = np.asarray(track)
    nb_dims = track.shape[1] // 2
    return np.sum((track[:, :nb_dims] - track[:, nb_dims:]) ** 2, axis=1) ** 0.5

