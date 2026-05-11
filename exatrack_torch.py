# -*- coding: utf-8 -*-
"""
PyTorch port of the ExaTrack diffusion model.

Mirrors the structure of the original TensorFlow implementation as closely as
possible so that side-by-side review is straightforward. Section headers below
match the original module's grouping.

Authoring conventions
---------------------
- Default dtype is float64 throughout (matches the TF model).
- Trainable model parameters are nn.Parameters; constraints (log-floor at
  log(minval)) are applied via torch.nn.utils.parametrize so they survive
  optimizer.step() and keep gradients clean.
- Carry-over buffers are non-persistent buffers and are mutated in-place
  inside the forward pass (no equivalent of CarryoverAssignLayer is needed).
- vary_* stop-gradient masks are implemented as `m*x + (1-m)*x.detach()`,
  identical in effect to `m*x + (1-m)*tf.stop_gradient(x)`.
- The pre-built integration "sequences" are still plain Python lists of
  (callable, [coef_index, ID_1, ID_2]) tuples, exactly as in the TF code.
  Each callable operates on Python lists of tensors via list ops, then stacks
  back -- the TF tf.unstack/tf.stack pattern translates to native Python.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

# ---------------------------------------------------------------------------
# Module-level constants (mirrors the top of the TF file)
# ---------------------------------------------------------------------------

DTYPE = torch.float64
PI = math.pi
MINVAL = 1e-14
LOG_MINVAL = math.log(MINVAL)


def _t(x, dtype=DTYPE, device=None) -> torch.Tensor:
    """tf.cast-style helper. Accepts numpy arrays, scalars or tensors."""
    if isinstance(x, torch.Tensor):
        out = x.to(dtype=dtype)
    else:
        out = torch.as_tensor(np.asarray(x), dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """tf.math.divide_no_nan analogue: 0/0 -> 0.

    The TF code regularised C1, C2 by adding tiny normal noise; here we use
    the cleaner `where` form. eps>0 lets callers add a small jitter on the
    denominator if they want the original numerical behaviour.
    """
    if eps > 0:
        b = b + eps
    return torch.where(b != 0, a / b, torch.zeros_like(a))


# ===========================================================================
# Gaussian primitives
# ===========================================================================

def log_gaussian(top: torch.Tensor,
                 variance: torch.Tensor | float = 1.0) -> torch.Tensor:
    if not torch.is_tensor(variance):
        variance = torch.tensor(variance, dtype=top.dtype, device=top.device)
    return -0.5 * torch.log(2 * PI * variance) - top ** 2 / (2 * variance)


def norm_log_gaussian(top: torch.Tensor) -> torch.Tensor:
    return -0.5 * (math.log(2 * PI) + top ** 2)


def RNN_gaussian_product(current_hidden_var_coefs_1: torch.Tensor,
                         current_hidden_var_coefs_2: torch.Tensor,
                         next_hidden_var_coefs_1: torch.Tensor,
                         next_hidden_var_coefs_2: torch.Tensor,
                         biases_1: torch.Tensor,
                         biases_2: torch.Tensor,
                         coef_index: int,
                         nb_dims: int = 1
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor]:
    """Simplify a product of two Gaussians that both depend on the hidden
    variable of index `coef_index` into one Gaussian that depends on this
    variable and another Gaussian that does not. See the TF docstring for
    the detailed shape contract.
    """
    dtype = current_hidden_var_coefs_1.dtype
    device = current_hidden_var_coefs_1.device

    # The TF code added 1e-20-scale noise; here we just use the values directly
    # and rely on _safe_div for true zeros. A small denominator floor matches
    # the original numerical behaviour at log-floor parameters.
    C1 = current_hidden_var_coefs_1[:, :, coef_index:coef_index + 1]
    C2 = current_hidden_var_coefs_2[:, :, coef_index:coef_index + 1]

    current_coefs1 = _safe_div(current_hidden_var_coefs_1, C1)
    current_coefs2 = _safe_div(current_hidden_var_coefs_2, C2)
    next_coefs1 = _safe_div(next_hidden_var_coefs_1, C1)
    next_coefs2 = _safe_div(next_hidden_var_coefs_2, C2)
    biases1 = _safe_div(biases_1, C1[:, :])
    biases2 = _safe_div(biases_2, C2[:, :])

    # 1/C^2 with a tiny floor to match the original TF behaviour where
    # C1/C2 had 1e-20 jitter added to them
    var1 = 1.0 / (C1 ** 2 + 1e-100)
    var2 = 1.0 / (C2 ** 2 + 1e-100)

    var3 = var1 + var2
    std3 = var3 ** 0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3
    biases3 = (biases1 - biases2) / std3[:, :]

    var4 = var1 * var2 / var3
    std4 = var4 ** 0.5
    current_coefs4 = (current_coefs1 * var2 + current_coefs2 * var1) / (var3 * std4)
    next_coefs4 = (next_coefs1 * var2 + next_coefs2 * var1) / (var3 * std4)
    biases4 = (biases1 * var2[:, :] + biases2 * var1[:, :]) / (var3 * std4)[:, :]

    LogConstant = -nb_dims * torch.log(torch.abs(C1 * C2 * std4 * std3))[:, :, 0]
    return (LogConstant, current_coefs3, current_coefs4,
            next_coefs3, next_coefs4, biases3, biases4)


def simple_RNN_gaussian_product(C1, C2,
                                current_hidden_var_coefs_1,
                                current_hidden_var_coefs_2,
                                next_hidden_var_coefs_1,
                                next_hidden_var_coefs_2):
    """1D, scalar-coefficient version used by get_sequences (numpy world)."""
    current_coefs1 = current_hidden_var_coefs_1 / C1
    current_coefs2 = current_hidden_var_coefs_2 / C2
    next_coefs1 = next_hidden_var_coefs_1 / C1
    next_coefs2 = next_hidden_var_coefs_2 / C2

    var1 = 1.0 / C1 ** 2
    var2 = 1.0 / C2 ** 2

    var3 = var1 + var2
    std3 = var3 ** 0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3

    var4 = var1 * var2 / var3
    std4 = var4 ** 0.5
    current_coefs4 = (current_coefs1 * var2 + current_coefs2 * var1) / (var3 * std4)
    next_coefs4 = (next_coefs1 * var2 + next_coefs2 * var1) / (var3 * std4)

    return current_coefs3, current_coefs4, next_coefs3, next_coefs4


# ===========================================================================
# Integration step operators
#
# These take Python lists of tensors (one per Gaussian) and modify them.
# The TF version did the same via tf.unstack -> mutate -> tf.stack; in
# PyTorch we keep them as lists throughout and only stack at the very end of
# the recurrence formula.
# ===========================================================================

def _to_list(t: torch.Tensor) -> List[torch.Tensor]:
    """tf.unstack along axis 0."""
    return list(torch.unbind(t, dim=0))


def _from_list(lst: List[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """tf.stack along axis 0, with explicit shape recovery for empty lists.

    The TF code needed `tf.cast(tf.reshape(tf.stack(...), shape), dtype=...)`
    in the no_RNN cases to get the right empty-bias shape; we reproduce that
    here so downstream `norm_log_gaussian(biases)` reductions don't trip on
    rank-0 tensors.
    """
    if len(lst) == 0:
        new_shape = (0,) + tuple(like.shape[1:])
        return torch.empty(new_shape, dtype=like.dtype, device=like.device)
    return torch.stack(lst, dim=0)


def intermediate_RNN_function(current, next_, biases,
                              coef_index, ID_1, ID_2,
                              nb_hidden_variables, LC, nb_gaussians,
                              kept_next, kept_biases, nb_dims):
    cur_l = _to_list(current)
    nxt_l = _to_list(next_)
    bia_l = _to_list(biases)

    LogConstant, c3, c4, n3, n4, b3, b4 = RNN_gaussian_product(
        cur_l[ID_1], cur_l[ID_2], nxt_l[ID_1], nxt_l[ID_2],
        bia_l[ID_1], bia_l[ID_2], coef_index, nb_dims)

    cur_l[ID_1] = c3
    cur_l[ID_2] = c4
    nxt_l[ID_1] = n3
    nxt_l[ID_2] = n4
    bia_l[ID_1] = b3
    bia_l[ID_2] = b4
    LC = LC + LogConstant

    return (_from_list(cur_l, current), _from_list(nxt_l, next_),
            _from_list(bia_l, biases),
            LC, nb_gaussians, kept_next, kept_biases)


def final_RNN_function_phase_1(current, next_, biases,
                               coef_index, ID_1, ID_2,
                               nb_hidden_variables, LC, nb_gaussians,
                               kept_next, kept_biases, nb_dims):
    current, next_, biases, LC, nb_gaussians, kept_next, kept_biases = \
        intermediate_RNN_function(current, next_, biases,
                                  coef_index, ID_1, ID_2,
                                  nb_hidden_variables, LC, nb_gaussians,
                                  kept_next, kept_biases, nb_dims)

    cur_l = _to_list(current)
    nxt_l = _to_list(next_)
    bia_l = _to_list(biases)

    # Normalise the integrated variable: log_gaussian(xs*c, 1) ==
    #     log_gaussian(xs*c/a, 1/a^2) - log(a)
    LC = LC - nb_dims * torch.log(torch.abs(cur_l[ID_2][:, :, coef_index]))

    cur_l.pop(ID_2)
    nxt_l.pop(ID_2)
    bia_l.pop(ID_2)
    nb_gaussians = nb_gaussians - 1

    return (_from_list(cur_l, current), _from_list(nxt_l, next_),
            _from_list(bia_l, biases),
            LC, nb_gaussians, kept_next, kept_biases)


def no_RNN_function_phase_1(current, next_, biases,
                            coef_index, ID_1, ID_2,
                            nb_hidden_variables, LC, nb_gaussians,
                            kept_next, kept_biases, nb_dims):
    cur_l = _to_list(current)
    nxt_l = _to_list(next_)
    bia_l = _to_list(biases)

    LC = LC - nb_dims * torch.log(torch.abs(cur_l[ID_2][:, :, coef_index]))

    cur_l.pop(ID_2)
    nxt_l.pop(ID_2)
    bia_l.pop(ID_2)
    nb_gaussians = nb_gaussians - 1

    return (_from_list(cur_l, current), _from_list(nxt_l, next_),
            _from_list(bia_l, biases),
            LC, nb_gaussians, kept_next, kept_biases)


def final_RNN_function_phase_2(next_, current, biases,
                               coef_index, ID_1, ID_2,
                               nb_hidden_variables, LC, nb_gaussians,
                               kept_next, kept_biases, nb_dims):
    # Note the (next_, current) argument order, mirroring the TF version
    next_, current, biases, LC, nb_gaussians, kept_next, kept_biases = \
        intermediate_RNN_function(next_, current, biases,
                                  coef_index, ID_1, ID_2,
                                  nb_hidden_variables, LC, nb_gaussians,
                                  kept_next, kept_biases, nb_dims)

    cur_l = _to_list(current)
    nxt_l = _to_list(next_)
    bia_l = _to_list(biases)

    new_next = nxt_l.pop(ID_2)
    new_bias = bia_l.pop(ID_2)

    kept_next_l = _to_list(kept_next)
    kept_biases_l = _to_list(kept_biases)
    kept_next_l.append(new_next)
    kept_biases_l.append(new_bias)

    nb_gaussians = nb_gaussians - 1

    return (_from_list(nxt_l, next_), _from_list(cur_l, current),
            _from_list(bia_l, biases),
            LC, nb_gaussians,
            _from_list(kept_next_l, kept_next),
            _from_list(kept_biases_l, kept_biases))


def no_RNN_function_phase_2(next_, current, biases,
                            coef_index, ID_1, ID_2,
                            nb_hidden_variables, LC, nb_gaussians,
                            kept_next, kept_biases, nb_dims):
    nxt_l = _to_list(next_)
    bia_l = _to_list(biases)

    new_next = nxt_l.pop(ID_2)
    new_bias = bia_l.pop(ID_2)

    kept_next_l = _to_list(kept_next)
    kept_biases_l = _to_list(kept_biases)
    kept_next_l.append(new_next)
    kept_biases_l.append(new_bias)

    nb_gaussians = nb_gaussians - 1

    return (_from_list(nxt_l, next_), current,
            _from_list(bia_l, biases),
            LC, nb_gaussians,
            _from_list(kept_next_l, kept_next),
            _from_list(kept_biases_l, kept_biases))


# ===========================================================================
# Recurrence drivers
# ===========================================================================

def RNN_reccurence_formula(current_hidden_var_coefs: torch.Tensor,
                           next_hidden_var_coefs: torch.Tensor,
                           biases: torch.Tensor,
                           sequence_phase_1: Tuple[List[Callable], List[List[int]]],
                           sequence_phase_2: Tuple[List[Callable], List[List[int]]],
                           nb_dims: int,
                           dtype: torch.dtype = DTYPE
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One recurrence step of the analytic Gaussian integration.

    See the TF docstring for the algorithmic explanation -- this routine is a
    direct line-by-line port. Tensors flow through Python lists between
    integration ops, mirroring the tf.unstack/tf.stack pattern.
    """
    cur = current_hidden_var_coefs.clone()
    nxt = next_hidden_var_coefs.clone()
    bia = biases.clone()

    kept_next = torch.empty((0,) + tuple(nxt.shape[1:]),
                            dtype=nxt.dtype, device=nxt.device)
    kept_biases = torch.empty((0,) + tuple(bia.shape[1:]),
                              dtype=bia.dtype, device=bia.device)

    nb_gaussians = bia.shape[0]
    nb_hidden_variables = cur.shape[-1]

    LC = torch.zeros(cur.shape[1:3], dtype=dtype, device=cur.device)

    fns, slots = sequence_phase_1
    for f, s in zip(fns, slots):
        coef_index, ID_1, ID_2 = s
        cur, nxt, bia, LC, nb_gaussians, kept_next, kept_biases = f(
            cur, nxt, bia, coef_index, ID_1, ID_2,
            nb_hidden_variables, LC, nb_gaussians,
            kept_next, kept_biases, nb_dims)

    fns, slots = sequence_phase_2
    for f, s in zip(fns, slots):
        coef_index, ID_1, ID_2 = s
        nxt, cur, bia, LC, nb_gaussians, kept_next, kept_biases = f(
            nxt, cur, bia, coef_index, ID_1, ID_2,
            nb_hidden_variables, LC, nb_gaussians,
            kept_next, kept_biases, nb_dims)

    new_LCs = norm_log_gaussian(bia).sum(dim=3)
    LC = LC + new_LCs.sum(dim=0)

    # Reverse the kept lists (TF: stack(kept_*[::-1]))
    Next_coefs = torch.flip(kept_next, dims=[0])
    Next_biases = torch.flip(kept_biases, dims=[0])

    return Next_coefs, Next_biases, LC


def transition_RNN_reccurence_formula(current_hidden_var_coefs: torch.Tensor,
                                      next_hidden_var_coefs: torch.Tensor,
                                      biases: torch.Tensor,
                                      transition_sequence,
                                      nb_dims: int,
                                      dtype: torch.dtype = DTYPE):
    """Transition variant: a single phase-1 pass; current/biases survive,
    next is left as-is.
    """
    cur = current_hidden_var_coefs.clone()
    nxt = next_hidden_var_coefs.clone()
    bia = biases.clone()

    kept_next = torch.empty((0,) + tuple(nxt.shape[1:]),
                            dtype=nxt.dtype, device=nxt.device)
    kept_biases = torch.empty((0,) + tuple(bia.shape[1:]),
                              dtype=bia.dtype, device=bia.device)

    nb_gaussians = bia.shape[0]
    nb_hidden_variables = cur.shape[-1]

    LC = torch.zeros(cur.shape[1:3], dtype=dtype, device=cur.device)

    fns, slots = transition_sequence
    for f, s in zip(fns, slots):
        coef_index, ID_1, ID_2 = s
        cur, nxt, bia, LC, nb_gaussians, kept_next, kept_biases = f(
            cur, nxt, bia, coef_index, ID_1, ID_2,
            nb_hidden_variables, LC, nb_gaussians,
            kept_next, kept_biases, nb_dims)

    return cur, bia, LC


# ===========================================================================
# Constraint functions
# ===========================================================================

def constraint_function(all_params: torch.Tensor,
                        all_initial_params: torch.Tensor,
                        LocErr: torch.Tensor,
                        dts: torch.Tensor,
                        nb_dims: int,
                        reference_dt: torch.Tensor | float,
                        dtype: torch.dtype = DTYPE):
    """Vectorised, time-varying constraint function.

    Direct port of the TF version. See that docstring for the full
    parameter / shape contract.
    """
    device = all_params.device

    nb_states = all_params.shape[0]
    integration_variable_index = torch.tensor(1, device=device)
    nb_hidden_vars = 2
    nb_obs_vars = 1
    nb_transition_gaussians = 1     # = nb_hidden_vars - integration_variable_index

    # --- normalise LocErr / dts to (track_len, nb_tracks, 1) ---------------
    LocErr = _t(LocErr, dtype=dtype, device=device)
    if LocErr.dim() == 2:
        LocErr = LocErr.unsqueeze(-1)
    LocErr = LocErr.mean(dim=-1, keepdim=True)
    LocErr = LocErr.permute(1, 0, 2).contiguous()

    dts = _t(dts, dtype=dtype, device=device)
    if dts.dim() == 2:
        dts = dts.unsqueeze(-1)
    dts = dts.mean(dim=-1, keepdim=True)
    dts = dts.permute(1, 0, 2).contiguous()

    reference_dt = _t(reference_dt, dtype=dtype, device=device)

    track_len = LocErr.shape[0]
    nb_tracks = LocErr.shape[1]

    # --- per-state parameters ---------------------------------------------
    log_d = all_params[:, 1].view(1, 1, nb_states)
    ano = all_params[:, 2].view(1, 1, nb_states)
    log_q = all_params[:, 3].view(1, 1, nb_states)
    is_dir = all_params[:, 4].view(1, 1, nb_states)
    log_init_spread = all_initial_params[:, 0].view(1, nb_states)

    isdir_mask = (is_dir >= 0.5).to(dtype)
    isconf_mask = 1.0 - isdir_mask

    # --- per-step rescaling -----------------------------------------------
    dt_ratio = dts / reference_dt + 0.9e-20
    dt_sqrt_ratio = torch.sqrt(dt_ratio)

    d_ref = torch.exp(log_d)
    q_ref = torch.exp(log_q)
    l_ref = torch.sigmoid(ano)
    v_ref = torch.exp(ano)

    d = d_ref * dt_sqrt_ratio[:track_len] + 1e-20
    q = q_ref * dt_sqrt_ratio[:track_len] + 1e-20
    l_ref_c = -torch.log(1.0 - l_ref)
    l_c = l_ref_c * dt_ratio[:track_len]
    l = -torch.expm1(-l_c) + 1e-20
    one_minus_l = torch.exp(-l_c) + 1e-20
    v = v_ref * dt_ratio[:track_len] + 1e-20

    # ano_t scaling for directed regime
    dt_ratio_next = dt_ratio[1:]
    ano_step_ratio = dt_ratio_next / dt_ratio[:-1]
    ano_rescale_per_state = ano_step_ratio * isdir_mask + (1.0 - isdir_mask)

    well_distance = d / torch.sqrt(2 * (1 - torch.exp(-2 * l_c)))

    initial_position_spread = torch.exp(log_init_spread).expand_as(d[0])

    LocErr_b = LocErr.expand(track_len, nb_tracks, nb_states) + 1e-20
    zeros = torch.zeros_like(LocErr_b)
    tiny = torch.full((track_len, nb_tracks, nb_states), 1e-15,
                      dtype=dtype, device=device)

    # --- recurrent hidden-variable coefficients ---------------------------
    # Final shape after stacking: (track_len, 3, nb_tracks, nb_states, 4)
    g0 = torch.stack([1.0 / LocErr_b, zeros, zeros, zeros], dim=-1)

    g1_std = (d * isdir_mask
              + d / (2 * l_c) ** 0.5 * (1 - torch.exp(-2 * l_c)) ** 0.5 * isconf_mask)

    inv_d = 1.0 / g1_std
    g1_c0 = (one_minus_l * isconf_mask + isdir_mask) * inv_d
    g1_c1 = (l * isconf_mask + isdir_mask) * inv_d + 1.1e-20
    g1 = torch.stack([g1_c0, g1_c1, -inv_d, zeros], dim=-1)

    inv_q = 1.0 / q
    g2_c1 = ano_rescale_per_state * inv_q
    g2 = torch.stack([zeros, g2_c1, zeros, -inv_q], dim=-1)

    hidden_vars = torch.stack([g0, g1, g2], dim=1)   # (T, 3, N, S, 4)

    # --- recurrent observation coefficients -------------------------------
    obs_g0 = (-1.0 / LocErr_b).unsqueeze(-1)
    obs_zero = zeros.unsqueeze(-1)
    obs_vars = torch.stack([obs_g0, obs_zero, obs_zero], dim=1)

    # --- initial hidden-variable coefficients -----------------------------
    init_g0 = torch.stack([1.0 / initial_position_spread, zeros[0]], dim=-1)
    init_g1_c0 = (1.0 / well_distance) * isconf_mask + tiny * isdir_mask
    init_g1_c1 = (-1.0 / well_distance) * isconf_mask + (1.0 / v) * isdir_mask
    init_g1 = torch.stack([init_g1_c0, init_g1_c1], dim=-1)

    initial_hidden_vars = torch.stack([init_g0, init_g1[0]], dim=0)

    # --- transition hidden-variable coefficients --------------------------
    transition_hidden_vars = init_g1.unsqueeze(1)    # (T, 1, N, S, 2)

    # --- scaffolding tensors ----------------------------------------------
    Gaussian_stds = torch.ones((track_len, nb_obs_vars + nb_hidden_vars,
                                nb_tracks, nb_states, 1),
                               dtype=dtype, device=device)
    biases_t = torch.zeros((track_len, nb_obs_vars + nb_hidden_vars,
                            nb_tracks, nb_states, nb_dims),
                           dtype=dtype, device=device)

    initial_obs_vars = torch.zeros((nb_hidden_vars, nb_tracks, nb_states, nb_obs_vars),
                                   dtype=dtype, device=device)
    initial_Gaussian_stds = torch.ones((nb_hidden_vars, nb_tracks, nb_states, 1),
                                       dtype=dtype, device=device)
    initial_biases = torch.zeros((nb_transition_gaussians, nb_tracks, nb_states, nb_dims),
                                 dtype=dtype, device=device)
    transition_Gaussian_stds = torch.ones(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, 1),
        dtype=dtype, device=device)
    transition_biases = torch.zeros(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, nb_dims),
        dtype=dtype, device=device)

    # --- log-normalising factors ------------------------------------------
    Log_factors = -torch.log(LocErr + 1e-20) - torch.log(g1_std) - torch.log(q)

    initial_anomalous_factor = (
        (-torch.log(d) + 0.5 * torch.log(2 * (1 - torch.exp(-2 * l_c)) + 1e-20))
        * isconf_mask
        - torch.log(v) * isdir_mask
    )
    initial_Log_factors = Log_factors[0] - log_init_spread + initial_anomalous_factor[0]
    transition_Log_factors = Log_factors + initial_anomalous_factor

    return (hidden_vars, obs_vars, Gaussian_stds, biases_t,
            initial_hidden_vars, initial_obs_vars,
            initial_Gaussian_stds, initial_biases,
            transition_hidden_vars, transition_Gaussian_stds,
            transition_biases, integration_variable_index,
            Log_factors, initial_Log_factors, transition_Log_factors)


def transition_param_function(transition_shapes: torch.Tensor,
                              transition_rates: torch.Tensor,
                              density: float,
                              Fs: torch.Tensor,
                              effective_ds: torch.Tensor,
                              dts: torch.Tensor,
                              reference_dt: torch.Tensor | float,
                              dtype: torch.dtype = DTYPE):
    """Direct port of the TF transition_param_function.

    transition_shapes : (S, S)
    transition_rates  : (S, S) -- raw, pre-softmax
    dts               : (T, N) -- already transposed by the caller
    """
    device = transition_shapes.device
    nb_states = transition_shapes.shape[0]
    nb_time_points, nb_tracks = dts.shape

    transition_shapes = torch.exp(transition_shapes)
    transition_rates = (F.softmax(transition_rates, dim=1)
                        * transition_shapes / reference_dt)
    transition_rates = transition_rates.unsqueeze(0).unsqueeze(0) * dts[..., None, None] + 1e-20

    new_transition_shapes = torch.cat(
        (transition_shapes, torch.ones((1, nb_states), dtype=dtype, device=device)),
        dim=0)
    new_transition_shapes = torch.cat(
        (new_transition_shapes,
         torch.ones((nb_states + 1, 1), dtype=dtype, device=device)),
        dim=1)

    mislinking_dwell_time = torch.full((nb_states,), 0.9 / nb_states,
                                       dtype=dtype, device=device)
    mislinking_dwell_time = torch.cat(
        (mislinking_dwell_time, torch.tensor([0.1], dtype=dtype, device=device)),
        dim=0)
    mislinking_dwell_time = mislinking_dwell_time.view(1, 1, 1, nb_states + 1)
    mislinking_dwell_time = mislinking_dwell_time.expand(
        nb_time_points, nb_tracks, 1, nb_states + 1)

    pair_term = (effective_ds.unsqueeze(1) ** 2 + effective_ds.unsqueeze(0) ** 2) ** 0.5
    mislinking_rates = 1 - torch.exp(
        -0.5 * density * (Fs.unsqueeze(0) * pair_term).sum(dim=0)
    ).unsqueeze(-1)
    mislinking_rates = mislinking_rates.view(1, 1, nb_states, 1).expand(
        nb_time_points, nb_tracks, nb_states, 1)

    new_transition_rates = torch.cat((transition_rates, mislinking_rates), dim=3)
    new_transition_rates = torch.cat((new_transition_rates, mislinking_dwell_time), dim=2)

    return new_transition_shapes, new_transition_rates


# ===========================================================================
# Sequence builder (mostly numpy, but uses a small wrapper around the
# constraint_function to introspect the coefficient sparsity pattern).
# ===========================================================================

def get_sequences(params: np.ndarray,
                  initial_params: np.ndarray,
                  constraint_fn,
                  nb_gaussians: int,
                  nb_hidden_vars: int,
                  dtype: torch.dtype = DTYPE):
    """Build the integration schedules used by RNN_reccurence_formula.

    Returns six (functions, slots) pairs in the same order as the TF version:
        (initial_phase_1, initial_phase_2,
         recurrent_phase_1, recurrent_phase_2,
         final_phase_1,    transition_sequence)
    """
    nb_dims = 1
    LocErrs = np.ones((1, 1))
    dts = np.ones((1, 2))

    params_t = torch.as_tensor(params, dtype=dtype)
    initial_params_t = torch.as_tensor(initial_params, dtype=dtype)

    out = constraint_fn(params_t, initial_params_t, LocErrs, dts,
                        nb_dims, 1.0, dtype)
    (hidden_var_coefs, _, _, _,
     initial_hidden_var_coefs, _, _, _,
     transition_hidden_var_coefs, _, _,
     integration_variable_index,
     _, _, _) = out

    # Move to numpy for the integer-only schedule construction
    hidden_var_coefs = hidden_var_coefs[0].numpy()
    transition_hidden_var_coefs = transition_hidden_var_coefs[0].numpy()
    initial_hidden_var_coefs = initial_hidden_var_coefs.numpy()
    integration_variable_index = int(integration_variable_index.item())

    recurrent_current = np.copy(hidden_var_coefs[:, 0, 0, :nb_hidden_vars])
    recurrent_next = np.copy(hidden_var_coefs[:, 0, 0, nb_hidden_vars:])

    current = hidden_var_coefs[:, 0, 0, :nb_hidden_vars]
    nxt = hidden_var_coefs[:, 0, 0, nb_hidden_vars:]

    cur_init = initial_hidden_var_coefs[:, 0, 0, :nb_hidden_vars]
    nxt_init = np.zeros((nb_hidden_vars, nb_hidden_vars), dtype=cur_init.dtype)

    current = np.concatenate((cur_init, current), axis=0)
    nxt = np.concatenate((nxt_init, nxt), axis=0)

    nb_gauss = len(current)

    def run_phase(coef_iter, current, nxt, *, phase, transition=False):
        """Helper used by phase 1 (eliminating gaussians) and phase 2
        (rearranging next coefs). Returns (functions, slots, current, nxt,
        saved_gaussians_or_None).
        """
        sequence = []
        functions = []
        nonlocal nb_gauss

        if phase == 2:
            saved = np.zeros((nb_hidden_vars, nb_hidden_vars))
        else:
            saved = None

        target_coefs = nxt if phase == 2 else current

        for coef_index in coef_iter:
            non_zero = [g for g in range(nb_gauss)
                        if target_coefs[g, coef_index] != 0]

            for i in range(len(non_zero) - 1):
                ID_1 = non_zero[i]
                ID_2 = non_zero[i + 1]
                sequence.append([coef_index, ID_1, ID_2])
                functions.append(intermediate_RNN_function)

                if phase == 1:
                    C1 = current[ID_1, coef_index]
                    C2 = current[ID_2, coef_index]
                    c3, c4, n3, n4 = simple_RNN_gaussian_product(
                        C1, C2,
                        current[ID_1], current[ID_2],
                        nxt[ID_1], nxt[ID_2])
                    current[ID_1] = c3
                    current[ID_2] = c4
                    nxt[ID_1] = n3
                    nxt[ID_2] = n4
                else:  # phase 2
                    C1 = nxt[ID_1, coef_index]
                    C2 = nxt[ID_2, coef_index]
                    c3, c4, _, _ = simple_RNN_gaussian_product(
                        C1, C2,
                        nxt[ID_1], nxt[ID_2],
                        nxt[ID_1] * 0, nxt[ID_2] * 0)
                    nxt[ID_1] = c3
                    nxt[ID_2] = c4

            if len(non_zero) > 1:
                # promote the last "intermediate" to a "final"
                if phase == 1:
                    functions[-1] = final_RNN_function_phase_1
                else:
                    functions[-1] = final_RNN_function_phase_2
            elif len(non_zero) == 1:
                ID_1 = 0
                ID_2 = non_zero[0]
                sequence.append([coef_index, ID_1, ID_2])
                if phase == 1:
                    functions.append(no_RNN_function_phase_1)
                else:
                    functions.append(no_RNN_function_phase_2)

            if len(non_zero) >= 1:
                if phase == 2:
                    saved[coef_index] = nxt[ID_2]
                    nxt = np.delete(nxt, ID_2, 0)
                else:
                    current = np.delete(current, non_zero[-1], 0)
                    nxt = np.delete(nxt, non_zero[-1], 0)
                nb_gauss -= 1

        return functions, sequence, current, nxt, saved

    # ----- INITIAL STEP ---------------------------------------------------
    init_fns_1, init_seq_1, current, nxt, _ = run_phase(
        range(nb_hidden_vars - 1, -1, -1), current, nxt, phase=1)
    init_fns_2, init_seq_2, current, nxt, init_saved = run_phase(
        range(nb_hidden_vars - 1, -1, -1), current, nxt, phase=2)

    # ----- RECURRENCE STEP ------------------------------------------------
    current = np.concatenate((init_saved, recurrent_current), axis=0)
    nxt = np.concatenate((init_saved * 0, recurrent_next), axis=0)
    nb_gauss = len(current)

    rec_fns_1, rec_seq_1, current, nxt, _ = run_phase(
        range(nb_hidden_vars - 1, -1, -1), current, nxt, phase=1)
    rec_fns_2, rec_seq_2, current, nxt, rec_saved = run_phase(
        range(nb_hidden_vars - 1, -1, -1), current, nxt, phase=2)

    print('Checking that the recurrent next Gaussians have the same form '
          'than the initial next gaussians:',
          np.all((init_saved == 0) == (rec_saved == 0)))

    # ----- TRANSITION STEP ------------------------------------------------
    current = rec_saved
    nxt = rec_saved * 0
    nb_gauss = len(current)
    trans_fns, trans_seq, current, nxt, _ = run_phase(
        list(np.arange(integration_variable_index, nb_hidden_vars))[::-1],
        current, nxt, phase=1)

    current = np.concatenate(
        (current, transition_hidden_var_coefs[:, 0, 0]), axis=0)
    nxt = np.concatenate(
        (nxt, transition_hidden_var_coefs[:, 0, 0] * 0), axis=0)
    nb_gauss = current.shape[0]
    saved_gaussians = current

    # ----- FINAL STEP -----------------------------------------------------
    current = saved_gaussians
    nxt = np.zeros(current.shape)
    nb_gauss = len(current)
    final_fns, final_seq, _, _, _ = run_phase(
        range(nb_hidden_vars - 1, -1, -1), current, nxt, phase=1)

    return ((init_fns_1, init_seq_1),
            (init_fns_2, init_seq_2),
            (rec_fns_1, rec_seq_1),
            (rec_fns_2, rec_seq_2),
            (final_fns, final_seq),
            (trans_fns, trans_seq))


# ===========================================================================
# nn.Module: log-floor parametrisation (the TF `constraint=` lambda)
# ===========================================================================

class LogFloor(nn.Module):
    """Parametrisation that floors a parameter at log(MINVAL).

    Used via `torch.nn.utils.parametrize.register_parametrization(...)`. This
    keeps the floor active across optimizer steps and inside autograd, just
    like the TF Variable's `constraint=` callback.
    """

    def __init__(self, floor: float = LOG_MINVAL):
        super().__init__()
        self.floor = floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.floor)

    # The `right_inverse` is what register_parametrization calls when the
    # raw value is *assigned* to (model.param.data = ...). We just clamp.
    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.floor)


def _apply_vary_mask(param: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`mask*x + (1-mask)*x.detach()` -- TF stop_gradient analogue."""
    return mask * param + (1.0 - mask) * param.detach()


# ===========================================================================
# nn.Module: Initial layer
# ===========================================================================

class IsfirstMaskLayer(nn.Module):
    """Element-wise   init_val * isfirst + prev_val * (1 - isfirst)."""

    def forward(self, init_val, prev_val, isfirst):
        return init_val * isfirst + prev_val * (1.0 - isfirst)


class InitialLayerConstraints(nn.Module):
    """First layer of the model -- owns trainable parameters and runs t=0.

    Faithful port of `Initial_layer_constraints`. See the TF docstring for
    the full responsibility list. Differences:
      - tf.Variable(constraint=...) -> nn.Parameter + LogFloor parametrisation
      - tf.Variable(trainable=False) -> register_buffer(persistent=False)
      - The carry-over buffers' shape is set lazily on the first forward
        because it depends on the input rank (track tensor's last dim),
        mirroring the TF `build()` callback.
    """

    def __init__(self,
                 nb_states: int,
                 nb_gaussians: int,
                 nb_obs_vars: int,
                 nb_hidden_vars: int,
                 params: np.ndarray,
                 initial_params: np.ndarray,
                 initial_fractions: np.ndarray,
                 max_linking_distance: float,
                 constraint_fn,
                 reference_dt: float,
                 vary_params: np.ndarray | None = None,
                 vary_initial_params: np.ndarray | None = None,
                 vary_initial_fractions: np.ndarray | None = None,
                 sequence_length: int = 3,
                 carryover: bool = True,
                 dtype: torch.dtype = DTYPE):
        super().__init__()
        self.dtype_ = dtype
        self.nb_states = nb_states
        self.nb_gaussians = nb_gaussians
        self.nb_obs_vars = nb_obs_vars
        self.nb_hidden_vars = nb_hidden_vars
        self.constraint_fn = constraint_fn
        self.sequence_length = sequence_length
        self.carryover = carryover
        self.reference_dt = reference_dt

        if vary_params is None:
            vary_params = np.ones(np.asarray(params).shape)
        if vary_initial_params is None:
            vary_initial_params = np.ones(np.asarray(initial_params).shape)
        if vary_initial_fractions is None:
            vary_initial_fractions = np.ones(np.asarray(initial_fractions).shape)

        # ---- trainable parameters with log-floor constraint ---------------
        self.param_vars = nn.Parameter(_t(params, dtype=dtype))
        self.initial_param_vars = nn.Parameter(_t(initial_params, dtype=dtype))
        self.initial_fractions = nn.Parameter(_t(initial_fractions, dtype=dtype))

        torch.nn.utils.parametrize.register_parametrization(
            self, 'param_vars', LogFloor())
        torch.nn.utils.parametrize.register_parametrization(
            self, 'initial_param_vars', LogFloor())

        # ---- non-trainable scalars / masks --------------------------------
        self.register_buffer('max_linking_distance_param',
                             _t(max_linking_distance, dtype=dtype),
                             persistent=False)
        self.register_buffer('vary_params', _t(vary_params, dtype=dtype),
                             persistent=False)
        self.register_buffer('vary_initial_params',
                             _t(vary_initial_params, dtype=dtype),
                             persistent=False)
        self.register_buffer('vary_initial_fractions',
                             _t(vary_initial_fractions, dtype=dtype),
                             persistent=False)

        # ---- pre-built integration schedules ------------------------------
        (self.initial_sequence_phase_1,
         self.initial_sequence_phase_2,
         self.recurrent_sequence_phase_1,
         self.recurrent_sequence_phase_2,
         self.final_sequence_phase_1,
         self.transition_sequence) = get_sequences(
            np.asarray(params), np.asarray(initial_params),
            constraint_fn, nb_gaussians, nb_hidden_vars, dtype)

        # ---- carry-over buffers (shape determined on first forward) -------
        self._built = False
        self.carryout_coefs = None
        self.carryout_biases = None
        self.carryout_LP = None

    def _build_carryover(self, nb_tracks: int, nb_dims: int, device, dtype):
        nb_sequences = self.sequence_length * (self.nb_states + 1)
        if self.carryover:
            self.register_buffer(
                'carryout_coefs',
                torch.zeros((self.nb_hidden_vars, nb_tracks, nb_sequences, nb_dims),
                            dtype=dtype, device=device),
                persistent=False)
            self.register_buffer(
                'carryout_biases',
                torch.zeros_like(self.carryout_coefs),
                persistent=False)
            self.register_buffer(
                'carryout_LP',
                torch.zeros((nb_tracks, nb_sequences), dtype=dtype, device=device),
                persistent=False)
        self._built = True

    def duplicate_states(self, param_vars, initial_param_vars, initial_fractions):
        """Identity hook -- subclasses override to tie parameter rows."""
        return param_vars, initial_param_vars, initial_fractions

    def forward(self, inputs: torch.Tensor,
                input_LocErrs: torch.Tensor,
                input_dts: torch.Tensor):
        """
        inputs : (track_len, nb_gaussians, nb_tracks, nb_states, nb_obs_vars, nb_dims)
                 -- already transposed by the caller (transposed_inputs).
        Returns (inputs, initial_states) tuple.
        """
        nb_tracks = inputs.shape[2]
        nb_dims = inputs.shape[-1]
        dtype = self.dtype_
        device = inputs.device

        if not self._built:
            self._build_carryover(nb_tracks, nb_dims, device, dtype)

        # --- vary_* stop-gradient masks -----------------------------------
        param_vars = _apply_vary_mask(self.param_vars, self.vary_params)
        initial_param_vars = _apply_vary_mask(
            self.initial_param_vars, self.vary_initial_params)
        initial_fractions_sm = F.softmax(self.initial_fractions, dim=-1)
        initial_fractions = _apply_vary_mask(
            initial_fractions_sm, self.vary_initial_fractions)

        param_vars, initial_param_vars, initial_fractions = self.duplicate_states(
            param_vars, initial_param_vars, initial_fractions)

        # --- append mislinking state --------------------------------------
        last_lp = param_vars[-1][0:1]
        mislinking_row = torch.stack([
            param_vars[-1][0],
            torch.log(self.max_linking_distance_param.to(dtype=dtype)),
            torch.tensor(-15.0, dtype=dtype, device=device),
            torch.log(torch.tensor(1e-5, dtype=dtype, device=device)),
            torch.tensor(0.0, dtype=dtype, device=device),
        ])
        param_vars = torch.cat((param_vars, mislinking_row.unsqueeze(0)), dim=0)
        initial_param_vars = torch.cat(
            (initial_param_vars, initial_param_vars[-1:]), dim=0)
        nb_states = self.nb_states + 1

        nb_hidden_vars = self.nb_hidden_vars
        sequence_length = self.sequence_length

        # --- constraint function ------------------------------------------
        out = self.constraint_fn(param_vars, initial_param_vars,
                                 input_LocErrs, input_dts, nb_dims,
                                 self.reference_dt, dtype)
        (hidden_var_coefs, obs_var_coefs, Gaussian_stds, biases,
         initial_hidden_var_coefs, initial_obs_var_coefs,
         initial_Gaussian_stds, initial_biases,
         transition_hidden_var_coefs, transition_Gaussian_stds,
         transition_biases, integration_variable_index,
         Log_factors, initial_Log_factors, transition_Log_factors) = out

        hidden_var_coefs = hidden_var_coefs / Gaussian_stds
        obs_var_coefs = obs_var_coefs / Gaussian_stds
        biases = biases / Gaussian_stds

        current_hidden_var_coefs = hidden_var_coefs[..., :nb_hidden_vars]
        next_hidden_var_coefs = hidden_var_coefs[..., nb_hidden_vars:]

        # carry to subsequent layers (full time series)
        reccurent_obs_var_coefs = obs_var_coefs.clone()
        reccurent_hidden_var_coefs = current_hidden_var_coefs.clone()
        reccurent_next_hidden_var_coefs = next_hidden_var_coefs.clone()
        reccurent_biases = biases.clone()

        initial_hidden_var_coefs = initial_hidden_var_coefs / initial_Gaussian_stds
        initial_obs_var_coefs = initial_obs_var_coefs / initial_Gaussian_stds
        initial_biases = initial_biases / initial_Gaussian_stds

        current_initial_hidden_var_coefs = initial_hidden_var_coefs[..., :nb_hidden_vars]
        next_initial_hidden_var_coefs = torch.zeros(
            (nb_hidden_vars, nb_tracks, nb_states, nb_hidden_vars),
            dtype=dtype, device=device)

        transition_hidden_var_coefs = transition_hidden_var_coefs / transition_Gaussian_stds
        transition_biases = transition_biases / transition_Gaussian_stds
        transition_hidden_var_coefs = torch.cat(
            [transition_hidden_var_coefs] * sequence_length * nb_states, dim=3)
        transition_biases = torch.cat(
            [transition_biases] * sequence_length * nb_states, dim=3)

        # --- t = 0 step ---------------------------------------------------
        biases_0 = reccurent_biases[0]
        obs_var_coefs_0 = reccurent_obs_var_coefs[0]
        current_hidden_var_coefs_0 = reccurent_hidden_var_coefs[0]
        next_hidden_var_coefs_0 = reccurent_next_hidden_var_coefs[0]

        biases_0 = biases_0 + (obs_var_coefs_0.unsqueeze(-1) * inputs[0]).sum(dim=-2)
        initial_biases = initial_biases + (
            initial_obs_var_coefs.unsqueeze(-1) * inputs[0]).sum(dim=-2)

        current_hidden_var_coefs_0 = torch.cat(
            (current_initial_hidden_var_coefs, current_hidden_var_coefs_0), dim=0)
        next_hidden_var_coefs_0 = torch.cat(
            (next_initial_hidden_var_coefs, next_hidden_var_coefs_0), dim=0)
        biases_0 = torch.cat((initial_biases, biases_0), dim=0)

        current_hidden_var_coefs_0 = torch.cat(
            [current_hidden_var_coefs_0] * sequence_length, dim=2)
        next_hidden_var_coefs_0 = torch.cat(
            [next_hidden_var_coefs_0] * sequence_length, dim=2)
        biases_0 = torch.cat([biases_0] * sequence_length, dim=2)

        Next_coefs, Next_biases, LC = RNN_reccurence_formula(
            current_hidden_var_coefs_0,
            next_hidden_var_coefs_0,
            biases_0,
            self.initial_sequence_phase_1,
            self.initial_sequence_phase_2,
            nb_dims,
            dtype=dtype)

        init_log_fractions = torch.cat(
            [torch.log(initial_fractions)] * sequence_length, dim=1)
        init_log_factors = torch.cat(
            [nb_dims * initial_Log_factors] * sequence_length, dim=1)

        LP = (LC + init_log_factors + init_log_fractions
              + math.log(1.0 / sequence_length))

        Log_factors = nb_dims * Log_factors
        transition_Log_factors = nb_dims * transition_Log_factors

        initial_states = [Next_coefs, Next_biases, LP,
                          Log_factors, transition_Log_factors,
                          reccurent_obs_var_coefs,
                          reccurent_hidden_var_coefs,
                          reccurent_next_hidden_var_coefs,
                          reccurent_biases,
                          transition_hidden_var_coefs,
                          transition_biases]
        return inputs, initial_states


# ===========================================================================
# RNN cell -- one recurrent step
# ===========================================================================

def RNN_cell(input_i, Prev_coefs, Prev_biases, LP, segment_len,
             reshaped_Log_factors, reshaped_transition_Log_factors,
             reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
             reccurent_next_hidden_var_coefs, reccurent_biases,
             transition_hidden_var_coefs, transition_biases,
             sequence_phase_1, sequence_phase_2,
             transition_mask, transition_sequence,
             transition_mean, transition_var,
             gamma_dist_mean, gamma_dist_var,
             states, dt_ratios):
    """One recurrent step. Direct port of the TF version. See the TF
    docstring for the full pipeline (transition branches, Gamma hazard,
    weighted reduction, sequence-buffer roll).
    """
    dtype = Prev_coefs.dtype
    device = Prev_coefs.device
    nb_dims = input_i.shape[-1]
    nb_tracks = LP.shape[0]
    nb_hidden_vars = Prev_coefs.shape[3]
    nb_states = reccurent_hidden_var_coefs.shape[2]
    sequence_length = LP.shape[1] // nb_states

    current_states = states[:, :, -1:]

    Prev_coefs2 = Prev_coefs.repeat_interleave(nb_states, dim=2)
    Prev_biases2 = Prev_biases.repeat_interleave(nb_states, dim=2)
    LP2 = LP.repeat_interleave(nb_states, dim=1)
    segment_len = segment_len.repeat_interleave(nb_states, dim=1)

    alternative_Prev_coefs = torch.cat(
        (Prev_coefs2, transition_hidden_var_coefs.clone()), dim=0)
    alternative_Prev_biases = torch.cat(
        (Prev_biases2, transition_biases.clone()), dim=0)

    transition_Prev_coefs, transition_Prev_biases, LC = transition_RNN_reccurence_formula(
        current_hidden_var_coefs=alternative_Prev_coefs,
        next_hidden_var_coefs=torch.zeros_like(alternative_Prev_coefs),
        biases=alternative_Prev_biases,
        transition_sequence=transition_sequence,
        nb_dims=nb_dims,
        dtype=dtype)

    LP2 = LP2 + LC * transition_mask + reshaped_Log_factors

    # --- Gamma transition probability ---------------------------------------
    current_shapes = gamma_dist_mean ** 2 / gamma_dist_var
    current_rates = gamma_dist_mean / gamma_dist_var

    # Gradient note: PyTorch has gaps in igamma's autograd. The derivative of
    # gammainc(a, x) w.r.t. its first arg `a` (the concentration) is not
    # implemented, so we detach `current_shapes` when computing the survival.
    # The pdf path stays fully differentiable (lgamma's gradient is
    # supported), and gradient through the rate still flows into the
    # survival via the chain rule on the second arg (which IS implemented).
    # segment_len is detached as well -- it is a "time since last
    # transition" counter, not a quantity we want to optimise against.
    gamma_input = (segment_len + 0.5).detach()
    gamma_pdf = Gamma(concentration=current_shapes, rate=current_rates)
    gamma_cdf = Gamma(concentration=current_shapes.detach(), rate=current_rates)
    pdf = torch.exp(gamma_pdf.log_prob(gamma_input)) + 1e-14
    surv = 1 - gamma_cdf.cdf(gamma_input) + 1e-12
    transition_probas = torch.clamp(pdf / surv,
                                    min=1.0 - 20.0,    # mirrors TF's (1-20)
                                    max=1.0 - 1e-10)

    all_Prev_coefs = (transition_Prev_coefs * transition_mask[None, :, :, None]
                      + Prev_coefs2 * (1 - transition_mask[None, :, :, None]))
    all_prev_biases = (transition_Prev_biases * transition_mask[None, :, :, None]
                       + Prev_biases2 * (1 - transition_mask[None, :, :, None]))

    flat_trans = (transition_probas * transition_mask).reshape(
        nb_tracks, nb_states * sequence_length, nb_states).sum(dim=2)
    non_transition_probas = torch.clamp(flat_trans, min=1.0 - 20.0,
                                        max=1.0 - 1e-10)
    non_transition_probas = (1 - non_transition_probas).repeat_interleave(
        nb_states, dim=1)

    transition_probas = (transition_probas * transition_mask
                         + non_transition_probas * (1 - transition_mask))
    all_LP = LP2 + torch.log(transition_probas)

    # --- recurrent prior gaussians ------------------------------------------
    P = sequence_length * nb_states
    cur_rec_obs = torch.cat([reccurent_obs_var_coefs] * P, dim=2)
    cur_rec_hid = torch.cat([reccurent_hidden_var_coefs] * P, dim=2)
    cur_rec_next_hid = torch.cat([reccurent_next_hidden_var_coefs] * P, dim=2)
    cur_rec_biases = torch.cat([reccurent_biases] * P, dim=2)

    current_hidden_var_coefs = torch.cat(
        (all_Prev_coefs, cur_rec_hid.clone()), dim=0)
    zero_tensor = torch.zeros_like(all_Prev_coefs)
    next_hidden_var_coefs = torch.cat(
        (zero_tensor, cur_rec_next_hid.clone()), dim=0)
    current_biases = cur_rec_biases.clone() + (
        cur_rec_obs.unsqueeze(-1) * input_i).sum(dim=-2)
    biases = torch.cat((all_prev_biases, current_biases), dim=0)

    Next_coefs, Next_biases, LC = RNN_reccurence_formula(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        sequence_phase_1, sequence_phase_2, nb_dims=nb_dims, dtype=dtype)

    all_LP = all_LP + LC

    reshaped_Next_coefs = Next_coefs.reshape(
        Next_coefs.shape[0], Next_coefs.shape[1],
        sequence_length * nb_states, nb_states, nb_hidden_vars)

    log_det = nb_dims * torch.log(
        torch.abs(reshaped_Next_coefs[0, :, :, :, 0]
                  * reshaped_Next_coefs[1, :, :, :, 1]) + 1e-20)

    transition_LPs = (all_LP - 200 * (1 - transition_mask)).reshape(
        nb_tracks, sequence_length * nb_states, nb_states) - log_det

    max_transition_LPs = transition_LPs.max(dim=1, keepdim=True).values
    transition_Ps = torch.exp(transition_LPs - max_transition_LPs)
    transition_weights = transition_Ps / transition_Ps.sum(dim=1, keepdim=True)

    # --- aggregate transition branches --------------------------------------
    transition_states = (states.unsqueeze(2) *
                         transition_weights[:, :, :, None, None]).sum(dim=1)

    transition_Next_coefs = Next_coefs.reshape(
        Next_coefs.shape[0], Next_coefs.shape[1],
        sequence_length * nb_states, nb_states, nb_hidden_vars)
    transition_Next_coefs = (transition_Next_coefs *
                             transition_weights[None, :, :, :, None]).sum(dim=2)

    transition_Next_biases = Next_biases.reshape(
        Next_biases.shape[0], Next_biases.shape[1],
        sequence_length * nb_states, nb_states, nb_dims)
    transition_Next_biases = (transition_Next_biases *
                              transition_weights[None, :, :, :, None]).sum(dim=2)

    transition_LPs = (torch.log(transition_Ps.sum(dim=1))
                      + max_transition_LPs[:, 0]
                      + nb_dims * torch.log(
                          torch.abs(transition_Next_coefs[0, :, :, 0]
                                    * transition_Next_coefs[1, :, :, 1]) + 1e-20))

    # --- stable (non-transition) branches -----------------------------------
    stable_LPs = all_LP.reshape(nb_tracks, sequence_length * nb_states, nb_states)
    stable_weights = (1 - transition_mask).reshape(
        sequence_length * nb_states, nb_states).unsqueeze(0)
    stable_LPs = (stable_LPs * stable_weights).sum(dim=2)

    stable_states = (states.unsqueeze(2) *
                     stable_weights[:, :, :, None, None]).sum(dim=2)

    stable_Next_coefs = (Next_coefs.reshape(
        Next_coefs.shape[0], Next_coefs.shape[1],
        sequence_length * nb_states, nb_states, nb_hidden_vars)
        * stable_weights[None, :, :, :, None]).sum(dim=3)
    stable_Next_biases = (Next_biases.reshape(
        Next_biases.shape[0], Next_biases.shape[1],
        sequence_length * nb_states, nb_states, nb_dims)
        * stable_weights[None, :, :, :, None]).sum(dim=3)
    stable_segment_len = (segment_len.reshape(
        nb_tracks, sequence_length * nb_states, nb_states)
        * stable_weights).sum(dim=2)

    current_gamma_dist_mean = torch.cat([transition_mean, gamma_dist_mean], dim=1)
    current_gamma_dist_var = torch.cat([transition_var, gamma_dist_var], dim=1)

    Next_coefs = torch.cat([transition_Next_coefs, stable_Next_coefs], dim=2)
    Next_biases = torch.cat([transition_Next_biases, stable_Next_biases], dim=2)
    new_LP = torch.cat([transition_LPs, stable_LPs], dim=1)
    additional_stable_segment_len = dt_ratios.unsqueeze(-1)
    current_segment_len = torch.cat(
        [torch.ones((nb_tracks, nb_states), dtype=dtype, device=device),
         stable_segment_len + additional_stable_segment_len], dim=1)
    Next_states = torch.cat([transition_states, stable_states], dim=1)

    # --- merge the last 2*nb_states sequences back into the buffer ----------
    saved_Next_coefs = Next_coefs[:, :, :-nb_states * 2]
    saved_Next_biases = Next_biases[:, :, :-nb_states * 2]
    saved_LP = new_LP[:, :-nb_states * 2]
    saved_segment_len = current_segment_len[:, :-nb_states * 2]
    saved_gamma_dist_mean = current_gamma_dist_mean[:, :-nb_states ** 2 * 2]
    saved_gamma_dist_var = current_gamma_dist_var[:, :-nb_states ** 2 * 2]
    saved_states = Next_states[:, :-nb_states * 2]

    nb_prev_gaussians = Next_coefs.shape[0]
    last_Next_coefs = Next_coefs[:, :, -nb_states * 2:].reshape(
        nb_prev_gaussians, nb_tracks, 2, nb_states, nb_hidden_vars)
    last_Next_biases = Next_biases[:, :, -nb_states * 2:].reshape(
        nb_prev_gaussians, nb_tracks, 2, nb_states, nb_dims)
    last_LP = new_LP[:, -nb_states * 2:].reshape(
        nb_tracks, 2, nb_states) - nb_dims * torch.log(
        torch.abs(last_Next_coefs[0, :, :, :, 0]
                  * last_Next_coefs[1, :, :, :, 1]) + 1e-20)
    last_segment_len = current_segment_len[:, -nb_states * 2:].reshape(
        nb_tracks, 2, nb_states)
    last_gamma_dist_mean = current_gamma_dist_mean[:, -nb_states ** 2 * 2:].reshape(
        nb_tracks, 2, nb_states, nb_states)
    last_gamma_dist_var = current_gamma_dist_var[:, -nb_states ** 2 * 2:].reshape(
        nb_tracks, 2, nb_states, nb_states)
    last_states = Next_states[:, -nb_states * 2:].reshape(
        nb_tracks, 2, nb_states, sequence_length, nb_states)

    last_LP_max = last_LP.max(dim=1, keepdim=True).values
    last_P = torch.exp(last_LP - last_LP_max)
    sum_last_P = last_P.sum(dim=1, keepdim=True)

    weight_last_LP = last_LP
    weight_last_P = torch.exp(weight_last_LP
                              - weight_last_LP.max(dim=1, keepdim=True).values)
    last_weights = weight_last_P / weight_last_P.sum(dim=1, keepdim=True)

    reduced_last_Next_coefs = (last_Next_coefs
                               * last_weights[None, :, :, :, None]).sum(dim=2)
    reduced_last_Next_biases = (last_Next_biases
                                * last_weights[None, :, :, :, None]).sum(dim=2)
    reduced_last_LPs = ((torch.log(sum_last_P + 1e-100) + last_LP_max)[:, 0]
                        + nb_dims * torch.log(
                            torch.abs(reduced_last_Next_coefs[0, :, :, 0]
                                      * reduced_last_Next_coefs[1, :, :, 1]) + 1e-20))
    reduced_last_segment_len = (last_segment_len * last_weights).sum(dim=1)
    reduced_last_gamma_dist_mean = (last_gamma_dist_mean
                                    * last_weights[:, :, :, None]).sum(dim=1)
    reduced_last_gamma_dist_var = ((
        last_gamma_dist_var
        + (last_gamma_dist_mean - reduced_last_gamma_dist_mean.unsqueeze(1)) ** 2)
        * last_weights[:, :, :, None]).sum(dim=1)
    reduced_last_gamma_dist_mean = reduced_last_gamma_dist_mean.reshape(
        nb_tracks, nb_states ** 2)
    reduced_last_gamma_dist_var = reduced_last_gamma_dist_var.reshape(
        nb_tracks, nb_states ** 2)
    reduced_last_states = (last_states * last_weights[:, :, :, None, None]).sum(dim=1)

    new_Next_coefs = torch.cat((saved_Next_coefs, reduced_last_Next_coefs), dim=2)
    new_Next_biases = torch.cat((saved_Next_biases, reduced_last_Next_biases), dim=2)
    new_LPs = torch.cat((saved_LP, reduced_last_LPs), dim=1)
    new_segment_len = torch.cat((saved_segment_len, reduced_last_segment_len), dim=1)
    new_gamma_dist_mean = torch.cat(
        (saved_gamma_dist_mean, reduced_last_gamma_dist_mean), dim=1)
    new_gamma_dist_var = torch.cat(
        (saved_gamma_dist_var, reduced_last_gamma_dist_var), dim=1)
    new_states = torch.cat((saved_states, reduced_last_states), dim=1)
    new_states = torch.cat((new_states, current_states), dim=2)[:, :, 1:]

    return (new_Next_coefs, new_Next_biases, new_LPs, new_segment_len,
            new_gamma_dist_mean, new_gamma_dist_var, new_states)


# ===========================================================================
# nn.Module: Custom_RNN_layer
# ===========================================================================

class CustomRNNLayer(nn.Module):
    """Time-varying recurrent layer driving RNN_cell across a track segment.

    Faithful port of `Custom_RNN_layer`. nb_states here is the *physical*
    state count -- the layer adds one (mislinking) internally, exactly as
    in the TF version.
    """

    def __init__(self,
                 nb_tracks: int,
                 transition_shapes: np.ndarray,
                 transition_rates: np.ndarray,
                 density: float,
                 nb_states: int,
                 sequence_phase_1,
                 sequence_phase_2,
                 transition_sequence,
                 transition_param_fn,
                 sequence_length: int = 3,
                 vary_transition_shapes: np.ndarray | None = None,
                 vary_transition_rates: np.ndarray | None = None,
                 carryover: bool = False,
                 dtype: torch.dtype = DTYPE):
        super().__init__()
        self.dtype_ = dtype
        self.nb_tracks = nb_tracks
        self.density = density
        self.sequence_length = sequence_length
        self.nb_states = nb_states + 1   # physical + mislinking
        self.carryover = carryover
        self.sequence_phase_1 = sequence_phase_1
        self.sequence_phase_2 = sequence_phase_2
        self.transition_sequence = transition_sequence
        self.transition_param_fn = transition_param_fn

        if vary_transition_shapes is None:
            vary_transition_shapes = np.ones(np.asarray(transition_shapes).shape)
        if vary_transition_rates is None:
            vary_transition_rates = np.ones(np.asarray(transition_rates).shape)

        # --- trainable Gamma params (rates have a log-floor) --------------
        self.transition_rates = nn.Parameter(_t(transition_rates, dtype=dtype))
        self.transition_shapes = nn.Parameter(_t(transition_shapes, dtype=dtype))
        torch.nn.utils.parametrize.register_parametrization(
            self, 'transition_rates', LogFloor())

        self.register_buffer('vary_transition_shapes',
                             _t(vary_transition_shapes, dtype=dtype),
                             persistent=False)
        self.register_buffer('vary_transition_rates',
                             _t(vary_transition_rates, dtype=dtype),
                             persistent=False)

        # --- pair-indexing helpers ---------------------------------------
        S = self.nb_states
        rep_part = torch.tensor(list(np.arange(S)) * sequence_length,
                                dtype=torch.long).repeat_interleave(S)
        cat_part = torch.cat([torch.arange(S)] * S * sequence_length, dim=0).long()
        indices = torch.stack([rep_part, cat_part], dim=1)
        self.register_buffer('indices', indices, persistent=False)
        transition_mask = ((indices[:, 0] - indices[:, 1]) != 0).to(dtype).unsqueeze(0)
        self.register_buffer('transition_mask', transition_mask, persistent=False)

        # --- carry-over buffers ------------------------------------------
        if self.carryover:
            self.register_buffer(
                'carryout_segment_len',
                torch.zeros((nb_tracks, sequence_length * S), dtype=dtype),
                persistent=False)
            self.register_buffer(
                'carryout_gamma_dist_mean',
                torch.zeros((nb_tracks, sequence_length * S ** 2), dtype=dtype),
                persistent=False)
            self.register_buffer(
                'carryout_gamma_dist_var',
                torch.zeros((nb_tracks, sequence_length * S ** 2), dtype=dtype),
                persistent=False)

    def forward(self, inputs, input_dts, reference_dt, mask,
                Prev_coefs, Prev_biases, LP,
                Log_factors, transition_Log_factors,
                reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
                reccurent_next_hidden_var_coefs, reccurent_biases,
                transition_hidden_var_coefs, transition_biases,
                log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
                isfirst=None):
        dtype = self.dtype_
        device = inputs.device
        nb_tracks = self.nb_tracks
        nb_states = self.nb_states
        sequence_length = self.sequence_length
        density = self.density

        # --- vary_* stop-gradient masks -----------------------------------
        transition_shapes = _apply_vary_mask(
            self.transition_shapes, self.vary_transition_shapes)
        transition_rates = _apply_vary_mask(
            self.transition_rates, self.vary_transition_rates)

        ds = torch.exp(log_ds)
        Fs = F.softmax(softmax_inv_Fractions[0, :-1], dim=-1)
        effective_ds = ds + 2 * torch.exp(anomalous_factors) * isdir

        dts_TN = input_dts.permute(1, 0).contiguous()
        transition_shapes_full, transition_rates_full = self.transition_param_fn(
            transition_shapes, transition_rates, density, Fs, effective_ds,
            dts_TN, reference_dt, dtype)

        # --- one-hot row/col operators for (S, S) -> P flatten ------------
        oh_row = F.one_hot(self.indices[:, 0], nb_states).to(dtype)
        oh_col = F.one_hot(self.indices[:, 1], nb_states).to(dtype)
        oh_src = oh_col

        flat_Log_full = torch.einsum('tns,ps->tnp', Log_factors, oh_row)
        flat_trans_Log_full = torch.einsum(
            'tns,ps->tnp', transition_Log_factors, oh_src)
        flat_Log_full = (flat_trans_Log_full * self.transition_mask
                         + flat_Log_full * (1 - self.transition_mask))

        transition_rates_flat_full = torch.einsum(
            'tnij,pi,pj->tnp', transition_rates_full, oh_row, oh_col)
        transition_shapes_flat = torch.einsum(
            'ij,pi,pj->p', transition_shapes_full, oh_row, oh_col)

        transition_mean_full = (transition_shapes_flat[None, None]
                                / transition_rates_flat_full)
        transition_var_full = (transition_shapes_flat[None, None]
                               / transition_rates_flat_full ** 2)

        # --- slice time axis so [0] aligns with inputs[0] -----------------
        rec_obs_seq = reccurent_obs_var_coefs[1:]
        rec_hid_seq = reccurent_hidden_var_coefs[1:]
        rec_next_hid_seq = reccurent_next_hidden_var_coefs[1:]
        rec_biases_seq = reccurent_biases[1:]
        trans_hid_seq = transition_hidden_var_coefs[1:]
        trans_biases_seq = transition_biases[1:]

        flat_Log_seq = flat_Log_full[1:]
        flat_trans_Log_seq = flat_trans_Log_full[1:]
        transition_mean_seq = transition_mean_full[1:, :, :nb_states ** 2]
        transition_var_seq = transition_var_full[1:, :, :nb_states ** 2]

        # --- initial loop carriers ----------------------------------------
        segment_len = torch.ones((nb_tracks, sequence_length * nb_states),
                                 dtype=dtype, device=device)
        gamma_dist_mean = transition_mean_full[0]
        gamma_dist_var = transition_var_full[0]

        if self.carryover and isfirst is not None:
            isfirst_b = isfirst.unsqueeze(-1)
            segment_len = (isfirst_b * segment_len
                           + (1 - isfirst_b) * self.carryout_segment_len)
            gamma_dist_mean = (isfirst_b * gamma_dist_mean
                               + (1 - isfirst_b) * self.carryout_gamma_dist_mean)
            gamma_dist_var = (isfirst_b * gamma_dist_var
                              + (1 - isfirst_b) * self.carryout_gamma_dist_var)

        # --- one-hot initial state-history --------------------------------
        states_indices = torch.arange(0, nb_states * sequence_length,
                                      device=device, dtype=torch.long) % nb_states
        states_indices = states_indices.unsqueeze(-1).repeat(1, sequence_length)
        states = F.one_hot(states_indices, nb_states).to(dtype).unsqueeze(0).repeat(
            nb_tracks, 1, 1, 1)

        nb_dims = reccurent_biases.shape[4]
        num_steps = inputs.shape[0]

        # --- diagnostics --------------------------------------------------
        All_states_list = []
        All_coefs_list = []
        All_biases_list = []
        All_LP_list = []

        for i in range(num_steps):
            log_w = LP - nb_dims * torch.log(
                torch.abs(Prev_coefs[0, :, :, 0]
                          * Prev_coefs[1, :, :, 1]) + 1e-20)
            max_log_w = log_w.max(dim=1, keepdim=True).values
            w = torch.exp(log_w - max_log_w)
            w = w / w.sum(dim=1, keepdim=True)
            pred_states = (w.unsqueeze(-1) * states[:, :, 0]).sum(
                dim=1, keepdim=True)

            All_states_list.append(pred_states)
            All_coefs_list.append(Prev_coefs)
            All_biases_list.append(Prev_biases)
            All_LP_list.append(LP)

            input_i = inputs[i]
            mask_i = mask[:, i]

            dt_ratios = input_dts[:, i] / reference_dt

            (Next_coefs, Next_biases, Next_LP, Next_segment_len,
             Next_gamma_mean, Next_gamma_var, Next_states) = RNN_cell(
                input_i, Prev_coefs, Prev_biases, LP, segment_len,
                flat_Log_seq[i], flat_trans_Log_seq[i],
                rec_obs_seq[i], rec_hid_seq[i], rec_next_hid_seq[i],
                rec_biases_seq[i], trans_hid_seq[i], trans_biases_seq[i],
                self.sequence_phase_1, self.sequence_phase_2,
                self.transition_mask, self.transition_sequence,
                transition_mean_seq[i], transition_var_seq[i],
                gamma_dist_mean, gamma_dist_var, states, dt_ratios)

            mc = mask_i[None, :, None, None]
            ms = mask_i[:, None]
            mst = mask_i[:, None, None, None]

            Prev_coefs = Next_coefs * mc + Prev_coefs * (1 - mc)
            Prev_biases = Next_biases * mc + Prev_biases * (1 - mc)
            LP = Next_LP * ms + LP * (1 - ms)
            segment_len = Next_segment_len * ms + segment_len * (1 - ms)
            gamma_dist_mean = Next_gamma_mean * ms + gamma_dist_mean * (1 - ms)
            gamma_dist_var = Next_gamma_var * ms + gamma_dist_var * (1 - ms)
            states = Next_states * mst + states * (1 - mst)

        All_states = torch.stack(All_states_list, dim=0).permute(1, 0, 2, 3)[:, :, 0, :]
        All_coefs = torch.stack(All_coefs_list, dim=0).permute(2, 0, 3, 1, 4)
        All_biases = torch.stack(All_biases_list, dim=0).permute(2, 0, 3, 1, 4)
        All_LPs = torch.stack(All_LP_list, dim=0).permute(1, 0, 2)
        All_states = All_states[:, sequence_length - 1:]

        return (Prev_coefs, Prev_biases, LP, segment_len,
                gamma_dist_mean, gamma_dist_var,
                All_states, All_coefs, All_biases, All_LPs, states)


# ===========================================================================
# nn.Module: Final layer
# ===========================================================================

class FinalLayer(nn.Module):
    """Integrates over the remaining hidden variables and weights the
    sequence buffer to produce the final per-track log-likelihood and
    smoothed state predictions.
    """

    def __init__(self, sequence_phase_1, nb_dims: int, sequence_length: int):
        super().__init__()
        self.sequence_phase_1 = sequence_phase_1
        self.nb_dims = nb_dims
        self.sequence_length = sequence_length

    def forward(self, inputs):
        Prev_coefs, Prev_biases, LP, All_states, last_states = inputs
        nb_dims = self.nb_dims

        if Prev_coefs.shape[0] > 0:
            zero_tensor = torch.zeros_like(Prev_coefs)
            Next_coefs, Next_biases, LC = RNN_reccurence_formula(
                Prev_coefs, zero_tensor, Prev_biases,
                self.sequence_phase_1, ([], []),
                nb_dims=nb_dims, dtype=Prev_coefs.dtype)
            LP = LP + LC

        log_w = LP
        max_log_w = log_w.max(dim=1, keepdim=True).values
        weights = torch.exp(log_w - max_log_w)
        weights = weights / weights.sum(dim=1, keepdim=True)
        pred_states = (weights[:, :, None, None] * last_states).sum(dim=1)
        All_states = torch.cat((All_states, pred_states), dim=1)
        return LP, All_states


# ===========================================================================
# Loss
# ===========================================================================

def MLE_loss(y_pred: torch.Tensor) -> torch.Tensor:
    """Negative log-marginal-likelihood. y_pred is per-sequence LP, shape
    (batch, nb_sequences). y_true is unused (mirrors the TF signature)."""
    max_LP = y_pred.max(dim=1, keepdim=True).values
    reduced = y_pred - max_LP
    log_marginal = torch.log(torch.exp(reduced).sum(dim=1, keepdim=True)) + max_LP
    return -log_marginal.mean()


# ===========================================================================
# Top-level model assembly
# ===========================================================================

class DiffusionModel(nn.Module):
    """End-to-end module equivalent to TF `build_model`'s training graph.

    forward(tracks, LocErrs, dts, mask)  -> per-sequence LP   (training)
    predict(tracks, LocErrs, dts, mask)  -> (LP, All_states, All_coefs,
                                             All_biases, All_LPs)   (inference)
    """

    def __init__(self,
                 track_len: int,
                 nb_states: int,
                 params: np.ndarray,
                 initial_params: np.ndarray,
                 transition_rates: np.ndarray,
                 transition_shapes: np.ndarray,
                 initial_fractions: np.ndarray,
                 batch_size: int,
                 reference_dt: float,
                 nb_dims: int = 2,
                 sequence_length: int = 3,
                 max_linking_distance: float = 3.0,
                 estimated_density: float = 1e-3,
                 vary_params=None,
                 vary_initial_params=None,
                 vary_initial_fractions=None,
                 vary_transition_shapes=None,
                 vary_transition_rates=None,
                 carryover: bool = False,
                 dtype: torch.dtype = DTYPE):
        super().__init__()
        self.dtype_ = dtype
        self.nb_dims = nb_dims
        self.batch_size = batch_size
        self.reference_dt = reference_dt
        self.carryover = carryover

        nb_obs_vars = 1
        nb_hidden_vars = 2
        nb_gaussians = nb_obs_vars + nb_hidden_vars

        self.init_layer = InitialLayerConstraints(
            nb_states, nb_gaussians, nb_obs_vars, nb_hidden_vars,
            params, initial_params, initial_fractions,
            max_linking_distance, constraint_function,
            reference_dt=reference_dt,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            sequence_length=sequence_length,
            carryover=carryover,
            dtype=dtype)

        self.rnn_layer = CustomRNNLayer(
            batch_size, transition_shapes, transition_rates, estimated_density,
            nb_states,
            self.init_layer.recurrent_sequence_phase_1,
            self.init_layer.recurrent_sequence_phase_2,
            self.init_layer.transition_sequence,
            transition_param_function,
            sequence_length=sequence_length,
            vary_transition_shapes=vary_transition_shapes,
            vary_transition_rates=vary_transition_rates,
            carryover=carryover,
            dtype=dtype)

        self.final_layer = FinalLayer(
            self.init_layer.final_sequence_phase_1,
            nb_dims=nb_dims,
            sequence_length=sequence_length)

        self.first_mask_layer = IsfirstMaskLayer()

    def _run(self, tracks, LocErrs, dts, mask, isfirst=None,
             return_diagnostics: bool = False):
        dtype = self.dtype_

        tracks = _t(tracks, dtype=dtype)
        LocErrs = _t(LocErrs, dtype=dtype)
        dts = _t(dts, dtype=dtype)
        mask = _t(mask, dtype=dtype)

        # (B, T, D) -> (B, 1, T, 1, 1, D) -> (T, 1, B, 1, 1, D)
        reshaped = tracks[:, None, :, None, None, :]
        transposed = reshaped.permute(2, 1, 0, 3, 4, 5).contiguous()

        _, init_states = self.init_layer(transposed, LocErrs, dts)

        softmax_inv_Fractions = self.init_layer.initial_fractions
        log_ds = self.init_layer.param_vars[:, 1]
        anomalous_factors = self.init_layer.param_vars[:, 2]
        isdir = self.init_layer.param_vars[:, 4]

        (Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors,
         rec_obs, rec_hid, rec_next_hid, rec_biases,
         trans_hid, trans_biases) = init_states

        if self.carryover and isfirst is not None:
            isfirst = _t(isfirst, dtype=dtype)
            Prev_coefs = self.first_mask_layer(
                Prev_coefs, self.init_layer.carryout_coefs,
                isfirst[None, :, None, None])
            Prev_biases = self.first_mask_layer(
                Prev_biases, self.init_layer.carryout_biases,
                isfirst[None, :, None, None])
            LP = self.first_mask_layer(
                LP, self.init_layer.carryout_LP, isfirst[:, None])

        sliced_inputs = transposed[1:]
        sliced_mask = mask[:, 1:]

        if isfirst is None and self.carryover:
            isfirst = torch.ones(tracks.shape[0], dtype=dtype,
                                 device=tracks.device)

        (Prev_coefs, Prev_biases, LP, segment_len,
         gamma_dist_mean, gamma_dist_var,
         All_states, All_coefs, All_biases, All_LPs,
         states) = self.rnn_layer(
            sliced_inputs, dts, self.reference_dt, sliced_mask,
            Prev_coefs, Prev_biases, LP,
            Log_factors, transition_Log_factors,
            rec_obs, rec_hid, rec_next_hid, rec_biases,
            trans_hid, trans_biases,
            log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
            isfirst=isfirst)

        LP_out, All_states = self.final_layer(
            (Prev_coefs, Prev_biases, LP, All_states, states))

        # Mutable carryover (no CarryoverAssignLayer needed) ----------------
        if self.carryover:
            with torch.no_grad():
                self.init_layer.carryout_coefs.copy_(Prev_coefs.detach())
                self.init_layer.carryout_biases.copy_(Prev_biases.detach())
                self.init_layer.carryout_LP.copy_(LP.detach())
                self.rnn_layer.carryout_segment_len.copy_(segment_len.detach())
                self.rnn_layer.carryout_gamma_dist_mean.copy_(gamma_dist_mean.detach())
                self.rnn_layer.carryout_gamma_dist_var.copy_(gamma_dist_var.detach())

        if return_diagnostics:
            return LP_out, All_states, All_coefs, All_biases, All_LPs
        return LP_out

    def forward(self, tracks, LocErrs, dts, mask, isfirst=None):
        return self._run(tracks, LocErrs, dts, mask, isfirst,
                         return_diagnostics=False)

    @torch.no_grad()
    def predict(self, tracks, LocErrs, dts, mask, isfirst=None):
        return self._run(tracks, LocErrs, dts, mask, isfirst,
                         return_diagnostics=True)
