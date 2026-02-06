# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:46:56 2025

@author: Franc
"""


import numpy as np
import tensorflow as tf
from exatrack import anomalous_diff_transition, build_model
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import pandas as pd
dtype = 'float64'


track_len=20
nb_tracks = 1000

all_tracks, all_states, all_masks = anomalous_diff_transition(max_track_len=track_len,
                                                   nb_tracks = nb_tracks,
                                                   LocErr=0.02, # localization error in x, y and z (even if not used)
                                                   Fs = np.array([0.5, 0.5]),
                                                   Ds = np.array([0, 0.16]),
                                                   nb_dims = 2,
                                                   velocities = np.array([0.01, 0.0]),
                                                   angular_Ds = np.array([0.0, 0.0]),
                                                   conf_forces = np.array([0.0, 0.1]),
                                                   conf_Ds = np.array([0.0, 0.0]),
                                                   conf_dists = np.array([0.0, 0.0]),
                                                   transition_matrix = np.array([[0.00, 0.1],
                                                                                 [0.1, 0.00]]),
                                                   shape_matrix = np.array([[1, 1],
                                                                            [1, 1]]),
                                                   bleaching_rate= 0.001,
                                                   LocErr_std = 0,
                                                   dt = 0.02,
                                                   field_of_view = [10, 10])

k=0
plt.figure()
plt.plot(all_tracks[k, all_masks[k].astype(bool), 0], all_tracks[k, all_masks[k].astype(bool), 1], 'k:', alpha = 0.5)
plt.scatter(all_tracks[k, all_masks[k].astype(bool), 0], all_tracks[k, all_masks[k].astype(bool), 1], c=plt.cm.jet(np.linspace(0,1,np.sum(all_masks[k]).astype(int))))
plt.gca().set_aspect('equal', adjustable='box')
print(all_states[k, :np.sum(all_masks[k]).astype(int)])

'''
building the model
'''

nb_states = 2 # Here the user can adapt the number of states of their model
sequence_length = 3 # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
nb_dimensions = 2 # Number of spatial dimensions
max_linking_distance = 3 # Maximum linking distance or standard deviation for the expected misslinking distance.
estimated_density = 0.001 # Estimated density of the sample. 

# Initial guesses on the model parameters, each row represents one state so len(params) and len(initial_params) must equal nb_states
# The 5th column is the model type indicator: 0 = confined motion, 1 = directed motion
params = tf.constant([[np.log(0.01), np.log(0.001), -0.03, np.log(0.001), 1],
                      [np.log(0.01), np.log(0.4), 0.2, np.log(0.0001), 0]], dtype = dtype)
initial_params = tf.constant([[np.log(8), np.log(0.05)],
                              [np.log(8), np.log(0.05)]], dtype = dtype) # col 0: spread of the position in the field of view, col 1: relation between the true position and the new anomalous variable

transition_shapes = tf.ones((nb_states, nb_states), dtype = dtype)
transition_rates = tf.ones((nb_states, nb_states), dtype = dtype)*0.08

# initial_fractions: fraction of each state in logit scale (softmax to get fractions).
# Length is nb_states+1 because the last entry represents the mislinking state.
initial_fractions = np.ones((1, nb_states+1))
initial_fractions[0,-1] = -1

batch_size = nb_tracks

model, pred_model = build_model(track_len, # maximum number of time points in the input tracks
                                nb_states, # Number of states of their model
                                params, # recurrent parameters of the model
                                initial_params, # initial parameters of the model
                                transition_rates, # transition rates for each pair of states (gamma distributed transition lifetimes)
                                transition_shapes, # transition shapes for each pair of states (gamma distributed transition lifetimes)
                                initial_fractions,
                                batch_size = batch_size, # number of tracks analysed at the same time
                                nb_dims = nb_dimensions,
                                sequence_length = sequence_length, # sequence of the previous states that are considered without alterations (computation time and memory usage proportional to sequence_length)
                                max_linking_distance = max_linking_distance, # Maximum linking distance or standard deviation for the expected misslinking distance.
                                estimated_density = estimated_density, # Estimated density of the sample.
                                )

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

class get_parameters(tf.keras.callbacks.Callback):
    def __init__(self, layer_name='params'):
        super(get_parameters, self).__init__()
        self.layer_name = layer_name  # Name of the layer whose weights you want to monitor
    
    def on_epoch_end(self, epoch, logs=None):
        # Retrieve the weights of the model
        weights = model.weights
        #params = tf.concat((model.weights[0][:, 2], np.exp(weights[0][:, 0]), np.exp(weights[0][:, 1]), [weights[3][0, 1]], [weights[3][1, 0]], [weights[4][0, 1]], [weights[4][1, 0]]), 0)
        params = {'anomalous factors': list(np.round(model.weights[0][:, 2],3)), 'Localization errors': list(np.round(np.exp(weights[0][:, 0]),3)), 'd': list(np.round(np.exp(weights[0][:, 1]), 3)), 'transition rates': list(np.round([weights[4][0, 1], weights[4][1, 0]], 3))}
        # For Dense layers, the first item is the kernel, and the second is the bias (if any)
        
        # Print the weights (you can customize this part to show what you need)
        print(params)
        
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
    
tracks = tf.constant(all_tracks[:,None, :, None, None, :nb_dimensions], dtype)

lr = WarmupLearningRateSchedule(15, 1/100, 0.007, 200)
adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/GPU:0'):
    history1 = model.fit((tracks, all_masks), tracks, epochs = 500, batch_size = batch_size, callbacks=[get_parameters()], shuffle=False, verbose = 1) #, callbacks  = [l_callback])

'''
Extracting and saving the model parameters (for now we keep the shape parameters fixed to 1)
'''
params = {'anomalous factors': model.weights[0][:, 2], 'Localization errors': np.exp(model.weights[0][:, 0]), 'd': np.exp(model.weights[0][:, 1]), 'transition rates': model.weights[4], 'q': np.exp(model.weights[0][:, 3]), 'transition shapes': model.weights[5]}

parameter_values = np.concatenate([params['Localization errors'][:, None], params['d'][:, None], params['anomalous factors'][:, None], params['q'][:, None],  params['transition rates']*(1-np.identity(nb_states)), params['transition shapes']], axis = 1)

rate_cols = []
shape_cols = []
for state in range(nb_states):
    rate_cols = rate_cols + ['transition rate (per step) to state %s'%state]
    shape_cols = shape_cols + ['transition shape (per step) to state %s'%state]

parameters = pd.DataFrame(parameter_values, columns = ['Localization errors', 'diffusion length', 'anomalous parameter', 'anomalous speed variation'] + rate_cols + shape_cols)

parameters.to_csv('Saved_parameters.csv')

'''
Computing the probabilities of each state
'''

state_preds = pred_model.predict((tracks, all_masks), batch_size = batch_size)

plt.figure(figsize = (10,10))

nb_rows = 5
lim = 0.7

for i in range(nb_rows):
    for j in range(nb_rows):
        track = all_tracks[i*nb_rows+j][:np.sum(all_masks[i*nb_rows+j]).astype(int)]
        track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
        
        preds = state_preds[i*nb_rows+j, :track.shape[0]]
        plt.plot(track[:, 0], track[:, 1], ':k')
        plt.scatter(track[:, 0], track[:, 1], c = preds[:,:3])
plt.gca().set_aspect('equal', adjustable='box')

