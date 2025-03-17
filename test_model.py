# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:46:56 2025

@author: Franc
"""

track_len=20
all_tracks, all_states, all_masks = anomalous_diff_transition(max_track_len=track_len,
                                                   nb_tracks = 1000,
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

# Defining the hyperparameters of the model
dtype = 'float64'
nb_states = 2
nb_obs_vars = 1
nb_independent_vars = 2 # This accounts for variables that are independ and which follow the same relationships (e.g. the spatial dimensions in tracking). 
nb_hidden_vars = 2
nb_gaussians = nb_obs_vars + nb_hidden_vars
nb_states = 2

# Initial guesses on the model parameters
params = tf.constant([[np.log(0.01), np.log(0.001), -0.03, np.log(0.001)],
                      [np.log(0.01), np.log(0.4), 0.2, np.log(0.0001)]], dtype = dtype)
initial_params = tf.constant([[np.log(8), np.log(0.05)],
                              [np.log(8), np.log(0.05)]], dtype = dtype) # col 0: spread of the position in the field of view, col 1: relation between the true position and the new anomalous variable

transition_shapes = tf.ones((nb_states, nb_states), dtype = dtype)
transition_rates = tf.ones((nb_states, nb_states), dtype = dtype)*0.08

tracks = tf.constant(all_tracks[:,None, :, None, None, :nb_independent_vars], dtype)

k=0
plt.figure()
plt.plot(all_tracks[k, all_masks[k].astype(bool), 0], all_tracks[k, all_masks[k].astype(bool), 1], 'k:', alpha = 0.5)
plt.scatter(all_tracks[k, all_masks[k].astype(bool), 0], all_tracks[k, all_masks[k].astype(bool), 1], c=plt.cm.jet(np.linspace(0,1,np.sum(all_masks[k]).astype(int))))
plt.gca().set_aspect('equal', adjustable='box')
print(all_states[k, :np.sum(all_masks[k]).astype(int)])
#print(segment_len[k]+0.5)



nb_tracks = len(tracks)
batch_size = nb_tracks

sequence_length = 3
max_linking_distance = 3
estimated_density = 0.001

inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1, 1, nb_independent_vars), dtype = dtype)
input_mask = tf.keras.Input(batch_shape = (batch_size, track_len), dtype = dtype)

#inputs = tracks
#input_mask = all_masks

transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4, 5])

Init_layer = Initial_layer_constraints(nb_states,
                                       nb_gaussians,
                                       nb_obs_vars,
                                       nb_hidden_vars,
                                       params,
                                       initial_params,
                                       max_linking_distance,
                                       constraint_function,
                                       sequence_length,
                                       dtype = dtype)

tensor1, initial_states = Init_layer(transposed_inputs)

softmax_inv_Fractions = Init_layer.initial_fractions
log_ds = Init_layer.param_vars[:, 1]
anomalous_factors = Init_layer.param_vars[:, 2]

Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases = initial_states

sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)
sliced_mask = tf.keras.layers.Lambda(lambda x: x[:, 1:], dtype = dtype)(input_mask)

layer = Custom_RNN_layer(batch_size, transition_shapes, transition_rates, estimated_density, nb_states, Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, Init_layer.transition_sequence, transition_param_function, sequence_length, dtype = dtype)
states = layer(sliced_inputs, sliced_mask, Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases, transition_hidden_var_coefs, transition_biases, log_ds, softmax_inv_Fractions, anomalous_factors)

F_layer = Final_layer(Init_layer.final_sequence_phase_1, nb_dims = nb_independent_vars, sequence_length = sequence_length, dtype = dtype)
outputs, All_states = F_layer(states)

model = tf.keras.Model(inputs=(inputs, input_mask), outputs=outputs, name="Diffusion_model")

pred_model = tf.keras.Model(inputs=(inputs, input_mask), outputs=All_states, name="Diffusion_model")

pred_model
model.summary()
model.weights

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

preds = model.predict((tracks, all_masks), batch_size = batch_size)
MLE_loss(preds, preds)

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
    
lr = WarmupLearningRateSchedule(15, 1/100, 0.007, 200)
adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/GPU:0'):
    history1 = model.fit((tracks, all_masks), tracks, epochs = 500, batch_size = batch_size, callbacks=[get_parameters()], shuffle=False, verbose = 1) #, callbacks  = [l_callback])

state_preds = pred_model.predict((tracks, all_masks), batch_size = batch_size)

plt.figure(figsize = (10,10))

nb_rows = 5
lim = 2.5

for i in range(nb_rows):
    for j in range(nb_rows):
        track = all_tracks[i*nb_rows+j][:np.sum(all_masks[i*nb_rows+j]).astype(int)]
        track = track - np.mean(track, 0, keepdims = True) + [[lim*i, lim*j]]
        
        preds = state_preds[i*nb_rows+j, :track.shape[0]]
        plt.plot(track[:, 0], track[:, 1], ':k')
        plt.scatter(track[:, 0], track[:, 1], c = preds[:,:3])
plt.gca().set_aspect('equal', adjustable='box')





















