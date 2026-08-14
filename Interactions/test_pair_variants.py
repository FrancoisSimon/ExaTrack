# -*- coding: utf-8 -*-
"""
Second validation pass: the pair coefficients must reproduce the exact Kalman
likelihood for every supported configuration
    - 1, 2 and 3 spatial dimensions
    - localisation error inputs / parameters with 1, 2 or 2*nb_dims columns
    - constant and variable frame durations
    - confined / directed motion for each of the two particles
    - interacting and independent states
"""

import itertools
import numpy as np
import tensorflow as tf


import sys
import os
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except NameError:
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)

import exatrack_pair as ep
dtype = 'float64'
np.random.seed(7)
 
T = 12
reference_dt = 0.02
 
 
def LocErr_function(LocErrs, LocErr_param):
    return LocErrs * LocErr_param
 
 
def kalman_ll(hidden_vars, initial_hidden_vars, observations, state, D):
    """Exact log-likelihood, summed over the (independent) spatial dimensions."""
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    total = 0.0
    for d in range(D):
        A = np.array(initial_hidden_vars[:, 0, state, d, :])
        Sigma = np.linalg.inv(A.T @ A)
        hv = np.array(hidden_vars[:, :, 0, state, d, :])
        mu = np.zeros(4)
        for t in range(T):
            R = np.diag([(1.0 / hv[t, 0, 0]) ** 2, (1.0 / hv[t, 1, 1]) ** 2])
            innov = observations[t, d] - H @ mu
            S = H @ Sigma @ H.T + R
            Sinv = np.linalg.inv(S)
            total += -0.5 * (innov @ Sinv @ innov + np.log(np.linalg.det(2 * np.pi * S)))
            K = Sigma @ H.T @ Sinv
            mu = mu + K @ innov
            Sigma = Sigma - K @ H @ Sigma
            if t < T - 1:
                F = np.zeros((4, 4))
                Q = np.zeros((4, 4))
                for g in range(2, 6):
                    j = g - 2
                    std = -1.0 / hv[t, g, 4 + j]
                    F[j] = hv[t, g, :4] * std
                    Q[j, j] = std ** 2
                mu = F @ mu
                Sigma = F @ Sigma @ F.T + Q
    return total
 
 
def run_chain(coefs, schedules, observations, T):
    (hidden_vars, obs_vars, biases, initial_hidden_vars, initial_obs_vars,
     initial_biases, Log_factors, initial_Log_factors) = coefs
    init1, init2, rec1, rec2, final1 = schedules
    inp = tf.constant(observations[:, None, None, None], dtype=dtype)
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
    _, _, LC = exatrack.RNN_reccurence_formula(
        P, tf.zeros_like(P), B, final1, [[], []], dtype=dtype)
    return np.array(LP + LC)[0]
 
 
configs = []
for D in [1, 2, 3]:
    for L in sorted({1, 2, 2 * D}):
        for variable_dt in [False, True]:
            for is_dir1, is_dir2 in [(0, 0), (1, 0), (1, 1)]:
                configs.append((D, L, variable_dt, is_dir1, is_dir2))
 
print('%-4s %-4s %-9s %-8s %-8s %-14s %-14s %s' %
      ('dims', 'LEc', 'var_dt', 'is_dir1', 'is_dir2', 'engine', 'kalman', 'diff'))
worst = 0.0
for (D, L, variable_dt, is_dir1, is_dir2) in configs:
    params = ep.make_pair_params(d1=[0.09, 0.09, 1.0], d2=[0.06, 0.06, 1.0],
                                 l_int=[0.4, 1e-6, 1e-6],
                                 q1=[3e-3, 3e-3, 1e-5], q2=[2e-3, 2e-3, 1e-5],
                                 l_1=[0.05, 0.05, 1e-6], l_2=[0.02, 0.02, 1e-6],
                                 v_1=[0.03, 0.03, 1.0], v_2=[0.02, 0.02, 1.0],
                                 is_dir1=[is_dir1, is_dir1, 0],
                                 is_dir2=[is_dir2, is_dir2, 0],
                                 LocErr=1.0, nb_LocErr_cols=L)
    initial_params = np.array([[np.log(30.0)]] * 3)
    cf = ep.make_pair_constraint_function(D)
 
    seqs = exatrack.get_sequences(params, initial_params, cf, 6, 4, 0.0, dtype)
    schedules = (seqs[0], seqs[1], seqs[2], seqs[3], seqs[4])
 
    LocErr_in = 0.02 * (1 + 0.3 * np.random.rand(1, T, L))
    if variable_dt:
        dts = reference_dt * (0.5 + np.random.rand(1, T + 1))
    else:
        dts = np.full((1, T + 1), reference_dt)
 
    out = cf(tf.constant(params), tf.constant(initial_params), tf.constant(LocErr_in),
             tf.constant(dts), D, reference_dt, LocErr_function, 0.0, dtype)
    coefs = (out[0], out[1], out[3], out[4], out[5], out[7], out[12], out[13])
 
    obs = np.random.normal(0, 0.05, (T, D, 2)).cumsum(0)
    engine = run_chain(coefs, schedules, obs, T)
    for s in [0, 1]:
        ref = kalman_ll(out[0], out[4], obs, s, D)
        diff = abs(engine[s] - ref)
        worst = max(worst, diff / max(1.0, abs(ref)))
        if s == 0:
            print('%-4d %-4d %-9s %-8d %-8d %-14.6f %-14.6f %.2e'
                  % (D, L, variable_dt, is_dir1, is_dir2, engine[s], ref, diff))
        assert diff < 1e-5, 'MISMATCH for config %s state %s (%.3e)' % (
            (D, L, variable_dt, is_dir1, is_dir2), s, diff)
 
print('\nall %d configurations x 2 states matched the Kalman reference '
      '(worst relative error %.2e)' % (len(configs), worst))
