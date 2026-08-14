# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:50:57 2026

@author: Franc
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# Import the ExaTrack module (ensure exatrack.py is in your path)
import os
import sys
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, rootdir)
except:
    sys.path.append(r"C:\Users\Franc\Data\ExaTrack_Orientation") 
import exatrack_rotational as exatrack
#import exatrack as exatrack
from glob import glob

from mpl_toolkits.mplot3d import Axes3D

dt = 0.02

'''
In this first test, we imagine a plausible scenario where a particle is either in free diffusion 
or in directed motion according to a processive process. In such cases, the rotational diffusion coefficient
of the free diffusive state is typically much higher than what can be acquired. In case of the 
processive state, we can quite easily imagine that the angles are confined within a small region 
of values. The rotational diffusion and the confinement factor that produce this confinement
remain unclear. Here, we will assume visible time correlations of the angles with a low rotational diffusion
coefficient and a low confinement factor to see if our model is able to assess these quantities in a
regime where it is theoretically feasible.
'''


0.2/(2*np.pi) # rotational diffusion length (in percentage of the circle)
rot_D0 = 0.2**2/(2*0.02)
0.2/(2*0.2)**0.5/(2*np.pi) # confinement radius (in percentage of the circle)


(2*rot_D0*0.02)**0.5

# Typical rotational diffusion of a free diffusing particle
rot_D1 = 1e6 
rot_D1 = 4
(2*rot_D1*0.02)**0.5
blur_ratio = 1

all_tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(max_track_len=100,                          # Maximal track length
                              nb_tracks=200,                              # Number of tracks
                              LocErr=0.01,                                # Localization error (position)
                              Fs=np.array([0.5, 0.5]),                    # Initial fractions of each state
                              # ---- translational motion ----
                              Ds=np.array([0.0, 0.25]),                   # Diffusion coefficients
                              velocities=np.array([0.05, 0.0]),           # Directed-motion speed
                              angular_Ds=np.array([0.0, 0.0]),            # Angular diffusion of the directed velocity vector
                              conf_forces=np.array([0.0, 0.01]),           # Attraction toward the potential well
                              conf_Ds=np.array([0.0, 0.0]),               # Diffusion of the potential well
                              conf_dists=np.array([0.0, 0.0]),            # Std of the potential well
                              # ---- rotational motion (particle orientation) ----
                              rot_Ds=np.array([rot_D0, rot_D1]),                # Rotational diffusion coefficients
                              rot_velocities=np.array([0., 0.0]),         # Persistent angular velocity (directed rotation)
                              rot_conf_forces=np.array([0.2, 0.]),        # Attraction toward the preferred orientation
                              rot_conf_Ds=np.array([0., 0.0]),            # Diffusion of the preferred orientation
                              rot_conf_dists=np.array([0.0, 0.0]),        # Std of the preferred-orientation well
                              OrientErr=0.05,                             # Orientation measurement error
                              OrientErr_std=0.0,                          # Std of the orientation error (0 = constant)
                              # ---- kinetics / acquisition ----
                              transition_matrix=np.array([[0.00, 0.05],    # Transition rates
                                                          [0.05, 0.00]]),
                              shape_matrix=np.array([[0, 1],              # Transition shapes (gamma dwell times)
                                                     [1, 0]]),
                              bleaching_rate=0.0001,                      # Bleaching rate per time step
                              LocErr_std=0.002,                           # Std of the localization error (0 = constant)
                              dt=dt,                            # Time step
                              dt_std=0.00,                                # Std of the time steps (0 = constant)
                              field_of_view=np.array([10, 10, 10]),       # Size of the (3-D) field of view
                              nb_burning_steps=100,                       # Burn-in steps for equilibration
                              nb_sub_steps=10,                            # Sub-steps per step (continuous transitions)
                              blur_ratio=blur_ratio,                      # Motion-blur exposure fraction
                              gap_probability=0)


plt.figure(figsize = (15, 15))
nb_plotted_tracks = 3
for i in range(nb_plotted_tracks):
    probs = (all_states[i, :,None]==np.arange(3)[None]).astype(float)
    for k in range(5):
        plt.subplot(nb_plotted_tracks, 5, i*5+k+1)
        track = all_tracks[i, :, k]
        plt.plot(np.arange(len(track)), track, color="gray")
        plt.scatter(np.arange(len(track)), track, c = probs)

i = 0
tracks = [all_tracks[i, all_masks[i].astype(bool)] for i in range(len(all_tracks))]

for i in range(len(tracks)):
    tracks[i] = tracks[i] - tracks[i][:1] + np.random.normal(0,1,(1,5))

dt_list = [all_dts[i, all_masks[i].astype(bool)] for i in range(len(all_tracks))]
#dt_list = [np.concatenate(((frame[1:] - frame[:-1])*0+reference_dt, [reference_dt])) for frame in frames]

# Prepare parameters for a 4 states model
nb_states = 2
'''
 col  name         meaning
  0   log_d        translational diffusion length (log)
  1   ano          translational anomalous parameter
                   (log drift speed if directed, logit confinement if confined)
  2   log_q        translational anomalous variation speed (log)
  3   is_dir       translational motion type   (0 = confined, 1 = directed)
  4   log_LE_x     localization error, x        (log)
  5   log_LE_y     localization error, y        (log)
  6   log_LE_z     localization error, z        (log)
  7   rot_log_d    rotational diffusion length  (log)
  8   rot_ano      rotational anomalous parameter
  9   rot_log_q    rotational anomalous variation speed (log)
 10   rot_is_dir   rotational motion type       (0 = confined, 1 = directed)
 11   log_OE_1     orientation error, angle 1   (log)
 12   log_OE_2     orientation error, angle 2   (log)
 
 -309.7104d=[0.0002 0.0159]  ano=[0.0415 0.0128]  q=[0.     0.0254]   Loc Error=[[0.0103 0.0103 0.0102]
  [0.0303 0.0313 0.0311]] rot_d=[0.1137 0.2014]  Rot Error=[[0.0763 0.077 ]
  [0.075  0.0779]] rot_ano=[0.2077 0.0048] rot_q=[0.0013 0.0006] rates=[[0.9354 0.0646]
  [0.0607 0.9393]]
 40/40 [==============================] - 25s 619ms/step - loss: -309.7104
'''

# Initialize with generic guesses
params = np.array([[np.log(0.001), np.log(0.05)                  , np.log(0.0001), 1, np.log(0.02), np.log(0.02), np.log(0.02), np.log(0.2), np.log(0.10), np.log(0.001), 0, np.log(0.06), np.log(0.06)],
                   [np.log(0.100), np.log(0.20) - np.log(1 -0.20), np.log(0.0001), 0, np.log(0.02), np.log(0.02), np.log(0.02), np.log(0.4), np.log(0.01), np.log(0.001), 1, np.log(0.06), np.log(0.06)]], dtype='float64')

initial_params = np.array([[np.log(1.0), np.log(1.0)]]*nb_states, dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0]*nb_states+[-5.0]], dtype='float64')

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
#vary_params[1, 2] = 0
#vary_params[2, 2] = 0

vary_initial_params = True
vary_initial_fractions = True
vary_transition_shapes = False

# We prevent transitions between the two bound states to improve readability 
vary_transition_rates = np.ones(transition_rates.shape)
#vary_transition_rates[:2, :2] = 0
tf.math.softmax(transition_rates)
batch_size = 50
nb_batches = len(tracks)//batch_size
device = '/GPU:0'

estimated_density = 0.00001 # Negligible density
nb_dims = 5
sequence_length = 10
max_linking_distance = 1
segment_length = 10
reference_dt = dt

0.06**2/(2*0.02)

model, pred_model = exatrack.build_segment_model(segment_length, # maximum number of time points in the input tracks 
                nb_states, # Number of states of their model
                params, # recurrent parameters of the model
                initial_params, # initial parameters of the model
                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                initial_fractions, 
                batch_size, # number of tracks analysed at the same time
                reference_dt = reference_dt,
                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                estimated_density = estimated_density, # Estimated density of the sample.
                vary_params = vary_params,
                vary_initial_params = vary_initial_params,
                vary_initial_fractions = vary_initial_fractions,
                vary_transition_shapes = vary_transition_shapes,
                vary_transition_rates = vary_transition_rates,
                blur_ratio=blur_ratio)

seq = exatrack.TrackSegmentSequence(track_list = tracks,
                                    LocErr_list=None,
                                    dt_list = dt_list,
                                    batch_size=batch_size,
                                    segment_length=segment_length,
                                    min_segment_length=4,
                                    cutoff_batch_treshhold=0.5)

model.weights[0]

seq[0][0][0]

nb_batches = len(seq)

#all_masks = masks
learning_rate = 0.01
nb_batches
epochs = 70
epoch_decay = 60
decay_threshold = epoch_decay*nb_batches
decay_rate = 0.005
np.exp(-20*64*0.001)

device = '/GPU:0'
shuffle = True
verbose = 1
print('Final learning rate:', learning_rate*np.exp(-max(0, epochs-epoch_decay)*decay_rate*nb_batches))

lr = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold) # learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.99, beta_2=0.999, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
MLE_loss = exatrack.MLE_loss
model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile = False)

LPs, preds, All_coefs, All_biases, All_LPs = pred_model.predict(seq)
likelihood = MLE_loss(LPs, LPs).numpy()
print(likelihood)

with tf.device(device):
    history = model.fit(seq, epochs = epochs, callbacks=[exatrack.get_parameters()], shuffle=False, verbose = verbose) #, callbacks  = [l_callback])

exatrack.get_model_params(model)

def plot_3d_track(track, probs, dot_size=25, line_width=1.5):
    """
    Plot a 3D particle track colored by state probabilities.

    Parameters
    ----------
    track : np.ndarray
        Shape (N, 3), xyz coordinates of the trajectory.

    probs : np.ndarray
        Shape (N, 3), probabilities of the three states.
        Each row should sum approximately to 1.

    dot_size : float
        Size of the localization dots.

    line_width : float
        Width of the gray trajectory line.
    """

    track = np.asarray(track)
    probs = np.asarray(probs)

    assert track.shape[0] == probs.shape[0], \
        "track and probs must have the same length"

    assert track.shape[1] == 3, \
        "track must have dimensions (N,3)"

    assert probs.shape[1] == 3, \
        "probs must have dimensions (N,3)"

    # Normalize probabilities in case they are not exactly normalized
    rgb = probs / probs.sum(axis=1, keepdims=True)

    x, y, z = track.T

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Gray connecting trajectory
    ax.plot(
        x, y, z,
        color="gray",
        linewidth=line_width,
        alpha=0.7,
        zorder=1
    )

    # RGB colored localizations
    ax.scatter(
        x, y, z,
        c=rgb,
        s=dot_size,
        edgecolors="none",
        zorder=2
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    ax.set_title("3D track colored by state probability")
    
    # Equal aspect ratio
    max_range = np.ptp(track, axis=0).max()
    center = track.mean(axis=0)
    
    ax.set_xlim(center[0]-max_range/2, center[0]+max_range/2)
    ax.set_ylim(center[1]-max_range/2, center[1]+max_range/2)
    ax.set_zlim(center[2]-max_range/2, center[2]+max_range/2)
    
    plt.tight_layout()
    plt.show()

