"""Checks for the two small fixes in get_number_of_states.

1. nb_data_points must count observed positions, i.e. sum the mask
   (index 3 of the batch inputs), not the localization errors
   (index 1).
2. Pruned fractions are fed back through tf.math.softmax
   (Init_layer, exatrack.py:1184), so the inverse is log(p), which
   softmax maps back to p exactly; log(p/(1-p)) does not.

Run: python3.11 test_state_count_inputs.py
"""
import numpy as np
import tensorflow as tf
from exatrack import TrackSegmentSequence

rng = np.random.default_rng(0)
nb_tracks, track_len = 40, 30
tracks = [np.cumsum(rng.normal(0, 0.1, (track_len, 2)), axis=0)
          for _ in range(nb_tracks)]
LocErrs = [np.full((track_len, 2), 0.02) for _ in range(nb_tracks)]
dts = [np.full(track_len, 0.02) for _ in range(nb_tracks)]

seq = TrackSegmentSequence(tracks, LocErrs, dts, batch_size=20,
                           segment_length=10, min_segment_length=4,
                           cutoff_batch_treshhold=0.5)
nb = len(seq)
idx1 = np.concatenate([seq[i][0][1] for i in range(nb)], axis=0)
idx3 = np.concatenate([seq[i][0][3] for i in range(nb)], axis=0)
print("index 1 shape", idx1.shape, "sum[:,1:]", float(np.sum(idx1[:, 1:])))
print("index 3 shape", idx3.shape, "sum[:,1:]", float(np.sum(idx3[:, 1:])))
assert set(np.unique(idx3)) <= {0.0, 1.0}, "index 3 is not a 0/1 mask"
assert np.allclose(idx1[idx1 != 0], 0.02), "index 1 is not the LocErr array"
print("fix 1 ok: index 3 is the mask, index 1 is LocErr")

p = np.array([0.6, 0.25, 0.1, 0.05])
old = tf.math.softmax(np.log(p / (1 - p))).numpy()
new = tf.math.softmax(np.log(p)).numpy()
print("target   ", p)
print("old logit", np.round(old, 4))
print("new log p", np.round(new, 4))
assert np.allclose(new, p) and not np.allclose(old, p)
print("fix 2 ok: softmax(log p) recovers p; softmax(logit p) does not")
