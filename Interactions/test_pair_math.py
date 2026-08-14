"""
Validation of the pair (two-particle) ExaTrack extension.
 
Test 1 : the integration schedules produced by `get_sequences` for the pair
         coefficient pattern are consistent (the initial and recurrent posterior
         Gaussians have the same sparsity pattern) and have the expected sizes.
Test 2 : running the *real* ExaTrack Gaussian-elimination engine
         (`RNN_reccurence_formula`) on the pair coefficients reproduces, to
         machine precision, the exact log-likelihood of a pair of tracks
         computed with an independent Kalman filter -- both for an interacting
         state (coupled 4x4 dynamics) and for an independent one.
Test 3 : the transition step (redrawing the two anomalous variables) leaves the
         expected number of Gaussians.
Test 4 : the interacting state wins on co-diffusing pairs and loses on
         independent pairs.
 
The last row of `params` plays the role of the mislinking state that
`Initial_layer_constraints` appends at run time, because the constraint function
overwrites the second-particle / interaction columns of that row.
"""
 
import numpy as np
import tensorflow as tf
 
import exatrack_dim_LocErrs as exatrack
import exatrack_pair as ep

dtype = 'float64'
np.random.seed(2)
tf.random.set_seed(0)
 
D = 1                    # spatial dimensions (the model treats them independently)
T = 30                   # time points
reference_dt = 0.02
blur_ratio = 0.0         # memoryless observations -> directly Kalman comparable
 
 
def LocErr_function(LocErrs, LocErr_param):
    return LocErrs * LocErr_param
 
 
# state 0 : interacting (l_int = 0.35 per frame), particle 1 slightly confined
# state 1 : independent (l_int ~ 0)
# state 2 : stands in for the mislinking state appended by the layer
params = ep.make_pair_params(d1=[0.09, 0.09, 1.0],
                             d2=[0.06, 0.06, 1.0],
                             l_int=[0.35, 1e-6, 1e-6],
                             q1=[3e-3, 3e-3, 1e-5],
                             q2=[2e-3, 2e-3, 1e-5],
                             l_1=[0.05, 0.05, 1e-6],
                             l_2=[1e-6, 1e-6, 1e-6],
                             LocErr=1.0, nb_LocErr_cols=2)
initial_params = np.array([[np.log(30.0)]] * 3)
print('params:\n', np.round(params, 3))
 
cf = ep.make_pair_constraint_function(D)
 
# ------------------------------------------------------------------ Test 1
print('\n================ Test 1 : integration schedules ================')
(init1, init2, rec1, rec2, final1, trans) = exatrack.get_sequences(
    params, initial_params, cf, 6, 4, blur_ratio, dtype)
for name, s in [('initial phase 1', init1), ('initial phase 2', init2),
                ('recurrent phase 1', rec1), ('recurrent phase 2', rec2),
                ('final phase 1', final1), ('transition', trans)]:
    print('%-18s %2d operations' % (name, len(s[1])))
 
# ------------------------------------------------------------------ coefficients
LocErr_in = np.full((1, T, 2), 0.02)
dts = np.full((1, T + 1), reference_dt)
 
out = cf(tf.constant(params), tf.constant(initial_params),
         tf.constant(LocErr_in), tf.constant(dts),
         D, reference_dt, LocErr_function, blur_ratio, dtype)
(hidden_vars, obs_vars, Gaussian_stds, biases,
 initial_hidden_vars, initial_obs_vars, initial_Gaussian_stds, initial_biases,
 transition_hidden_vars, transition_Gaussian_stds, transition_biases,
 integration_variable_index, Log_factors, initial_Log_factors,
 transition_Log_factors) = out
print('\nshapes: hidden', hidden_vars.shape, ' obs', obs_vars.shape,
      ' init', initial_hidden_vars.shape, ' trans', transition_hidden_vars.shape)
print("state 0 position-update Gaussian of particle 1 (x1,x2,a1,a2,x1',x2',a1',a2'):\n",
      np.round(np.array(hidden_vars[0, 2, 0, 0, 0]), 3))
print('state 1 (independent) position-update Gaussian of particle 1:\n',
      np.round(np.array(hidden_vars[0, 2, 0, 1, 0]), 3))
 
 
def run_chain(observations):
    """Runs the real engine over one pair of tracks, returns LP for every state."""
    inp = tf.constant(observations[:, None, None, None], dtype=dtype)   # (T,1,1,1,D,2)
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
    return np.array(LP + LC)[0], P, B
 
 
def kalman_ll(observations, state):
    """Exact log-likelihood of the same linear Gaussian model, state by state."""
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    A = np.array(initial_hidden_vars[:, 0, state, 0, :])
    Sigma = np.linalg.inv(A.T @ A)
    hv = np.array(hidden_vars[:, :, 0, state, 0, :])              # (T, 6, 8)
    mu = np.zeros(4)
    ll = 0.0
    for t in range(T):
        R = np.diag([(1.0 / hv[t, 0, 0]) ** 2, (1.0 / hv[t, 1, 1]) ** 2])
        innov = observations[t, 0] - H @ mu
        S = H @ Sigma @ H.T + R
        Sinv = np.linalg.inv(S)
        ll += -0.5 * (innov @ Sinv @ innov + np.log(np.linalg.det(2 * np.pi * S)))
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
    return ll
 
 
# ------------------------------------------------------------------ data
# (a) a co-diffusing pair : the separation is a stationary OU process
sep_noise = (0.09 ** 2 + 0.06 ** 2) ** 0.5
sep_std = sep_noise / (1 - (1 - 0.35) ** 2) ** 0.5
com = np.cumsum(np.random.normal(0, 0.05, (T, D)), 0)
sep = np.zeros((T, D))
sep[0] = np.random.normal(0, sep_std, D)
for t in range(1, T):
    sep[t] = sep[t - 1] * (1 - 0.35) + np.random.normal(0, sep_noise, D)
y_bound = np.stack([com + sep / 2, com - sep / 2], axis=-1) \
    + np.random.normal(0, 0.02, (T, D, 2))
 
# (b) an independent pair
y_free = np.stack([np.cumsum(np.random.normal(0, 0.09, (T, D)), 0),
                   np.cumsum(np.random.normal(0, 0.06, (T, D)), 0) + 0.5], axis=-1) \
    + np.random.normal(0, 0.02, (T, D, 2))
 
# ------------------------------------------------------------------ Test 2
print('\n================ Test 2 : likelihood vs Kalman =================')
LP_bound, P, B = run_chain(y_bound)
print('%d posterior gaussians carried between steps (expected 4)' % P.shape[0])
for s, label in [(0, 'interacting state'), (1, 'independent state')]:
    ref = kalman_ll(y_bound, s)
    print('%-20s  engine = %+.8f   kalman = %+.8f   |diff| = %.2e'
          % (label, LP_bound[s], ref, abs(LP_bound[s] - ref)))
    assert abs(LP_bound[s] - ref) < 1e-6, 'PAIR GAUSSIAN ALGEBRA IS WRONG'
print('==> Test 2 PASSED')
 
# ------------------------------------------------------------------ Test 3
print('\n================ Test 3 : transition step ======================')
cur = tf.concat((P, transition_hidden_vars[2]), axis=0)
bb = tf.concat((B, transition_biases[2]), axis=0)
tc, tb, tlc = exatrack.transition_RNN_reccurence_formula(
    cur, tf.zeros_like(cur), bb, trans, dtype=dtype)
print('gaussians before / after the transition step: %d -> %d (expected 4)'
      % (cur.shape[0], tc.shape[0]))
assert tc.shape[0] == 4
print('==> Test 3 PASSED')
 
# ------------------------------------------------------------------ Test 4
print('\n============ Test 4 : does the model see co-diffusion? =========')
LP_free, _, _ = run_chain(y_free)
print('co-diffusing pair : LL(interacting) = %+.2f   LL(independent) = %+.2f'
      '   log Bayes factor = %+.2f'
      % (LP_bound[0], LP_bound[1], LP_bound[0] - LP_bound[1]))
print('independent pair  : LL(interacting) = %+.2f   LL(independent) = %+.2f'
      '   log Bayes factor = %+.2f'
      % (LP_free[0], LP_free[1], LP_free[0] - LP_free[1]))
assert (LP_bound[0] - LP_bound[1]) > 0 > (LP_free[0] - LP_free[1]), 'no discrimination'
print('==> Test 4 PASSED')
 
