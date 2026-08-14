# -*- coding: utf-8 -*-
"""
Two-colour co-diffusion demo with the pair ExaTrack extension.
 
Pipeline
--------
1. Reaction-diffusion simulation: many A particles (channel 1) and B particles
   (channel 2) diffuse in a field of view.  An A and a B bind ONLY when closer
   than a reaction radius r_bind, Markovian while within it, and unbind with a
   constant rate.  The density (field-of-view size) is tuned so ~50% of the
   particles are bound.  Each particle yields an independent single-particle
   track -- the simulation returns two lists of tracks, NOT pre-formed couples.
2. Candidate pairing: propose (A, B) couples that overlap in time and come close
   for long enough.
3. Pair model: fit the two-state (co-diffusing / independent) pair model on the
   candidate couples, then classify each candidate per time point.
4. Report: the couples that are potentially co-diffusing, with their frame
   windows and per-frame co-diffusion probability.
 
The final deliverable is `report`, a list of couples of tracks (not a list of
tracks).
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
import os
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except NameError:
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)

import exatrack
import exatrack_pair as ep



# =====================================================================
# 0. Physical parameters
# =====================================================================
reference_dt = 0.02          # s per frame
nb_dims = 2
track_len = 200
D_A = 0.25                    # um^2/s, free A
D_B = 0.25                    # free B
D_bound = 0.25               # bound complex centre of mass
LocErr = 0.005
r_bind = 0.050                # um, reaction radius
k_on = 1000.0                 # per s, binding hazard within r_bind
k_off = 2.0                  # per s, unbinding hazard (mean bound time 0.25 s)
bond_length = 0.03           # um, size of the bound complex
bond_relax_rate = 60.0       # per s, how tightly the pair tracks within a frame
 
1/(reference_dt*k_off)
1/(reference_dt*k_on)

rd_kwargs = dict(dt=reference_dt, nb_dims=nb_dims, D_A=D_A, D_B=D_B, D_bound=D_bound,
                 r_bind=r_bind, k_on=k_on, k_off=k_off, bond_length=bond_length,
                 bond_relax_rate=bond_relax_rate, LocErr=LocErr, nb_sub_steps=50, nb_burnin_frames=500)
 
# =====================================================================
# 1. Tune the density to ~50% bound, then simulate the movie
# =====================================================================
nb_A = nb_B = 10000
'''
print('Tuning the field of view for ~50% bound particles ...')
fov, frac = ep.tune_reaction_density(target_fraction=0.01, nb_A=nb_A, nb_B=nb_B,
                                     track_len=track_len,
                                     box_range=(1.0, 800), n_iter=3, **rd_kwargs)
print('chosen field of view = %s um -> bound fraction = %.3f\n' % (np.round(fov, 2), frac))
'''
fov = (1000, 1000)
movie = ep.simulate_reaction_diffusion(nb_A=nb_A, nb_B=nb_B, field_of_view=fov,
                                       track_len=track_len,
                                       track_dropout=0.2, seed=0, **rd_kwargs)

print('recorded %d A tracks (channel 1) and %d B tracks (channel 2)'
      % (len(movie['tracks_A']), len(movie['tracks_B'])))
print('actual bound fraction over the movie: %.3f' % ep.reaction_bound_fraction(movie))
print('number of ground-truth co-diffusing couples: %d\n' % len(movie['true_couples']))
 
# quick look at the two channels
plt.figure(figsize=(7, 7))
for tr in movie['tracks_A'][:80]:
    plt.plot(tr[:, 0], tr[:, 1], '-', color='tab:red', lw=0.6, alpha=0.5)
    plt.scatter(tr[:1, 0], tr[:1, 1], c = 'k', alpha=0.5)
for tr in movie['tracks_B'][:80]:
    plt.plot(tr[:, 0], tr[:, 1], '-', color='tab:blue', lw=0.6, alpha=0.5)
plt.gca().set_aspect('equal')
plt.title('channel 1 (A, red) and channel 2 (B, blue)')

# quick look at the two channels
plt.figure(figsize=(7, 7))
for tr in movie['tracks_A'][:80]:
    plt.plot(np.arange(len(tr[:, 0])), tr[:, 1], '-', color='tab:red', lw=1.5, alpha=0.5)
for tr in movie['tracks_B'][:80]:
    plt.plot(np.arange(len(tr[:, 0])), tr[:, 1], '-', color='tab:blue', lw=1.5, alpha=0.5)
plt.title('channel 1 (A, red) and channel 2 (B, blue)')

# =====================================================================
# 2. Candidate co-diffusing couples
# =====================================================================
candidates = ep.find_codiffusion_candidates(movie, search_radius=20 * r_bind, min_overlap=5)
rec = ep.candidate_recall(candidates, movie)
print('candidate pairing: %d candidates cover %d/%d true couples (recall %.2f)'
      % (rec['nb_candidates'], rec['nb_recovered'], rec['nb_true_couples'], rec['recall']))
print('  (instead of the %d exhaustive A x B pairs)\n'
      % (len(movie['tracks_A']) * len(movie['tracks_B'])))
 
# assemble the candidate couples as a training set for the pair model:
# each couple is a "track" of shape (len, 2*nb_dims) = [x1, y1, x2, y2]
track_list = [c['track'] for c in candidates]
LocErr_list = [c['LocErr'] for c in candidates]
dt_list = [c['dt'] for c in candidates]
 

plt.figure(figsize =(10,10))
nb_rows = 5
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows +j
        plt.subplot(nb_rows,nb_rows,ID + 1)
        plt.plot(np.arange(len( track_list[ID])), track_list[ID][:,0])
        plt.plot(np.arange(len( track_list[ID])), track_list[ID][:,2])


# =====================================================================
# 3. Pair model: two states (co-diffusing / independent)
# =====================================================================
batch_size = 100
segment_length = 10
sequence_length = 10
nb_states = 2
 
l_int0 = ep.interaction_rate_to_l_int(bond_relax_rate, reference_dt)   # bond stiffness
params = ep.make_pair_params(
    d1=[(2 * D_bound * reference_dt) ** 0.5, (2 * D_A * reference_dt) ** 0.5],
    d2=[(2 * D_bound * reference_dt) ** 0.5, (2 * D_B * reference_dt) ** 0.5],
    l_int=[l_int0, 1e-6],              # state 0 co-diffusing, state 1 independent
    q1=[2e-4, 2e-4], q2=[2e-4, 2e-4],
    l_1=[1e-6, 1e-6], l_2=[1e-6, 1e-6],
    is_dir1=[0, 0], is_dir2=[0, 0],
    LocErr=1.0, nb_LocErr_cols=2)
print('initial pair parameters:\n', np.round(params, 3), '\n')
 
initial_params = np.array([[np.log(60.0)]] * nb_states)
initial_fractions = np.array([[0.0] * nb_states + [-5.0]])
transition_rates = 3 * np.eye(nb_states, dtype='float64')
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')
 
vary_params = np.ones(params.shape)
vary_params[:, 6] = 0        # is_dir1 flags fixed
vary_params[:, 7] = 0        # is_dir2 flags fixed
vary_params[1, 8] = 0        # the independent state must stay independent

seq = exatrack.TrackSegmentSequence(track_list, LocErr_list=LocErr_list, dt_list=dt_list,
                                    batch_size=batch_size, segment_length=segment_length,
                                    min_segment_length=4, cutoff_batch_treshhold=0.5)
nb_batches = len(seq)

model, pred_model = ep.build_pair_segment_model(
    segment_length, nb_states, params, initial_params, transition_rates,
    transition_shapes, initial_fractions, batch_size, reference_dt,
    nb_dims=nb_dims, sequence_length=sequence_length,
    max_linking_distance=1.0, estimated_density=1e-5,
    vary_params=vary_params, vary_initial_params=True, vary_initial_fractions=True,
    vary_transition_shapes=False, vary_transition_rates=np.ones(transition_rates.shape),
    nb_LocErr_dims=2, blur_ratio=0.0, LocErr_type='Linear')

epochs = 40
lr = exatrack.WarmupLearningRateSchedule(10, 0.05, 0.005, 25 * nb_batches)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0)
model.compile(loss=exatrack.MLE_loss, optimizer=optimizer, jit_compile=False)

with tf.device('/GPU:0'):
    model.fit(seq, epochs=epochs, callbacks=[ep.get_pair_parameters(reference_dt)],
              shuffle=False, verbose=1)

print('\nfitted parameters:')
for k, v in ep.get_pair_model_params(model, reference_dt).items():
    if type(v[0])==np.float64:
        print('  %-30s %s' % (k, np.round(v, 5) if np.asarray(v).dtype != object else v))

ep.get_pair_model_params(model, reference_dt)['transition rates']

# =====================================================================
# 4. Classify the candidates per time point, then report the couples
# =====================================================================
weights = model.get_weights()
max_len = max(len(c['frames']) for c in candidates)
_, pred_model = ep.build_pair_segment_model(
    max_len, nb_states, params=weights[0], initial_params=weights[1],
    transition_rates=weights[7], transition_shapes=weights[8],
    initial_fractions=weights[2], batch_size=batch_size, reference_dt=reference_dt,
    nb_dims=nb_dims, sequence_length=sequence_length, max_linking_distance=1.0,
    estimated_density=1e-5, nb_LocErr_dims=2, blur_ratio=0.0, LocErr_type='Linear')
 
kept = ep.predict_codiffusion(pred_model, candidates, batch_size,
                              segment_length=max_len, min_segment_length=2)

report = kept
ep.codiffusion_report(kept, min_fraction=0.1, min_run=3, use='p_codiffusion')
kept[1]
report[1]


print('\n%d couples reported as potentially co-diffusing:' % len(report))
truth = set(movie['true_couples'].keys())
reported = set((r['idA'], r['idB']) for r in report)
prec = len(reported & truth) / max(1, len(reported))
recl = len(reported & truth) / max(1, len(truth))
print('precision = %.2f   recall = %.2f' % (prec, recl))
for r in report[:15]:
    tag = 'TRUE ' if (r['idA'], r['idB']) in truth else 'false'
    print('  A%-4d B%-4d frames %3d-%3d  co-diff fraction %.2f  longest run %2d  [%s]'
          % (r['idA'], r['idB'], r['first_frame'], r['last_frame'],
             r['codiff_fraction'], r['longest_run'], tag))
 
# ---- plot a few reported couples coloured by the co-diffusion probability ----
by_index = {(c['idA'], c['idB']): c for c in kept}
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, r in zip(axes.ravel(), report[:6]):
    c = by_index[(r['idA'], r['idB'])]
    tr, p = c['track'], np.asarray(c['p_codiffusion'])
    ax.scatter(tr[:, 0], tr[:, 1], c=p, cmap='coolwarm', vmin=0, vmax=1, s=14)
    ax.scatter(tr[:, 2], tr[:, 3], c=p, cmap='coolwarm', vmin=0, vmax=1, s=14, marker='s')
    ax.plot(tr[:, 0], tr[:, 1], '-', color=[0, 0.5, 0], lw=0.8)
    ax.plot(tr[:, 2], tr[:, 3], '--', color=[0.2, 0., 0.2], lw=0.8)
    ax.set_aspect('equal',  adjustable = 'datalim')
    ax.set_title('A%d - B%d' % (r['idA'], r['idB']), fontsize=9)
fig.suptitle('reported couples: red = co-diffusing, blue = apart '
             '(circles A, squares B)')
plt.show()



