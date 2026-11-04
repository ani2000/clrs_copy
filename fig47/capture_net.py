"""A byte-for-byte copy of clrs._src.nets.Net / processors.PGN, extended to
also expose the node-update site's two halves as real (stacked, traced)
scan outputs instead of a python side-channel (side effects inside
hk.scan/jax.lax.scan do not see concrete per-step values, only abstract
tracers, so this must be done via the scan's own carry/output mechanism,
the same way nets.py's own `debug=True` exposes `hiddens`).

Per node i at each hint step t, the real PGN/MPNN update (processors.py,
'mpnn' processor kind, msgs_mlp_sizes=[H,H], use_ln=True, no triplets) is:
    z_i        = concat(node_fts_i, hidden_prev_i)                    # (2H,)
    msgs_i     = max_j MLP([H,H])(relu(m1(z_i)+m2(z_j)+m_e(edge_ij)+m_g(g)))
    nxt_hidden_i = LayerNorm(relu(o1(z_i) + o2(msgs_i)))               # (H,)
We expose (z_i, msgs_i) as the node-update INPUT and nxt_hidden_i as its
OUTPUT, for every hint step of every sample -- this is the exact
input/output pair Phase 3's transcoder is trained to reconstruct, and
the exact linear pieces (o1, o2, m1, m2, m_e, m_g, the message MLP, the
ReLU mask, the max-argmax routing, LayerNorm) Phase 5's attribution graph
will need to linearize.

Uses the identical class names/`name=` defaults as the real processors.PGN/
MPNN so haiku parameter paths match the trained checkpoint exactly; only
`__call__`'s *return value* is extended (ret, tri_msgs, z, msgs instead of
ret, tri_msgs), so restore_model()'s params still merge cleanly.
"""
import functools
from typing import Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from clrs._src import decoders, encoders, nets, probing, processors, specs

_Type = specs.Type
BIG_NUMBER = processors.BIG_NUMBER


class PGNCapture(processors.PGN):
  """Same as processors.PGN, also returns (z, msgs)."""

  def __call__(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused_kwargs):
    b, n, _ = node_fts.shape
    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1, m_2, m_e, m_g = (hk.Linear(self.mid_size) for _ in range(4))
    o1, o2 = hk.Linear(self.out_size), hk.Linear(self.out_size)

    msg_1, msg_2, msg_e, msg_g = m_1(z), m_2(z), m_e(edge_fts), m_g(graph_fts)
    tri_msgs = None
    if self.use_triplets:
      triplets = processors.get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)
      o3 = hk.Linear(self.out_size)
      tri_msgs = o3(jnp.max(triplets, axis=1))
      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (jnp.expand_dims(msg_1, 1) + jnp.expand_dims(msg_2, 2) + msg_e +
           jnp.expand_dims(msg_g, (1, 2)))
    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))
    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1), msgs, -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1, h_2 = o1(z), o2(msgs)
    ret = h_1 + h_2
    if self.activation is not None:
      ret = self.activation(ret)
    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)
    if self.gated:
      gate1, gate2 = hk.Linear(self.out_size), hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
      gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1 - gate)
    return ret, tri_msgs, z, msgs


class MPNNCapture(PGNCapture):
  """Same as processors.MPNN (all-ones adjacency), capturing version."""

  def __call__(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused_kwargs):
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


def get_capture_processor_factory(use_ln=True, nb_triplet_fts=8):
  def _factory(out_size):
    return MPNNCapture(out_size=out_size, msgs_mlp_sizes=[out_size, out_size],
                       use_ln=use_ln, use_triplets=False, nb_triplet_fts=0)
  return _factory


@chex.dataclass
class _ScanStateCapture:
  hint_preds: chex.Array
  output_preds: chex.Array
  hiddens: chex.Array
  lstm_state: Optional[hk.LSTMState]
  z: chex.Array
  msgs: chex.Array


class CaptureNet(nets.Net):
  """Copy of nets.Net, threading (z, msgs) through the scan like `hiddens`."""

  def _one_step_pred(self, inputs, hints, hidden, batch_size, nb_nodes,
                     lstm_state, spec, encs, decs, repred):
    node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
    edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
    graph_fts = jnp.zeros((batch_size, self.hidden_dim))
    adj_mat = jnp.repeat(jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

    trajectories = [inputs]
    if self.encode_hints:
      trajectories.append(hints)
    for trajectory in trajectories:
      for dp in trajectory:
        dp = encoders.preprocess(dp, nb_nodes)
        assert dp.type_ != _Type.SOFT_POINTER
        adj_mat = encoders.accum_adj_mat(dp, adj_mat)
        encoder = encs[dp.name]
        edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
        node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
        graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)

    nxt_hidden = hidden
    z = msgs = None
    for _ in range(self.nb_msg_passing_steps):
      nxt_hidden, nxt_edge, z, msgs = self.processor(
          node_fts, edge_fts, graph_fts, adj_mat, nxt_hidden,
          batch_size=batch_size, nb_nodes=nb_nodes)

    if not repred:
      nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)
    nxt_lstm_state = None  # use_lstm=False for this project

    h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)
    e_t = jnp.concatenate([edge_fts, nxt_edge], axis=-1) if nxt_edge is not None else edge_fts
    hint_preds, output_preds = decoders.decode_fts(
        decoders=decs, spec=spec, h_t=h_t, adj_mat=adj_mat, edge_fts=e_t,
        graph_fts=graph_fts, inf_bias=self.processor.inf_bias,
        inf_bias_edge=self.processor.inf_bias_edge, repred=repred)
    return nxt_hidden, output_preds, hint_preds, nxt_lstm_state, z, msgs

  def _msg_passing_step(self, mp_state, i, hints, repred, lengths, batch_size,
                        nb_nodes, inputs, first_step, spec, encs, decs,
                        return_hints, return_all_outputs):
    if self.decode_hints and not first_step:
      hard_postprocess = (self._hint_repred_mode == 'hard' or
                          (self._hint_repred_mode == 'hard_on_eval' and repred))
      decoded_hint = decoders.postprocess(spec, mp_state.hint_preds,
                                          sinkhorn_temperature=0.1, sinkhorn_steps=25,
                                          hard=hard_postprocess)
    if repred and self.decode_hints and not first_step:
      cur_hint = [decoded_hint[hint] for hint in decoded_hint]
    else:
      cur_hint = []
      needs_noise = (self.decode_hints and not first_step and self._hint_teacher_forcing < 1.0)
      force_mask = (jax.random.bernoulli(hk.next_rng_key(), self._hint_teacher_forcing, (batch_size,))
                   if needs_noise else None)
      for hint in hints:
        hint_data = jnp.asarray(hint.data)[i]
        _, loc, typ = spec[hint.name]
        if needs_noise:
          if typ == _Type.POINTER and decoded_hint[hint.name].type_ == _Type.SOFT_POINTER:
            hint_data = hk.one_hot(hint_data, nb_nodes)
            typ = _Type.SOFT_POINTER
          hint_data = jnp.where(nets._expand_to(force_mask, hint_data), hint_data,
                                decoded_hint[hint.name].data)
        cur_hint.append(probing.DataPoint(name=hint.name, location=loc, type_=typ, data=hint_data))

    nxt_hidden, output_preds_cand, hint_preds, lstm_state, z, msgs = self._one_step_pred(
        inputs, cur_hint, mp_state.hiddens, batch_size, nb_nodes,
        mp_state.lstm_state, spec, encs, decs, repred)

    if first_step:
      output_preds = output_preds_cand
    else:
      output_preds = {}
      for outp in mp_state.output_preds:
        is_not_done = nets._is_not_done_broadcast(lengths, i, output_preds_cand[outp])
        output_preds[outp] = (is_not_done * output_preds_cand[outp] +
                              (1.0 - is_not_done) * mp_state.output_preds[outp])

    new_mp_state = _ScanStateCapture(hint_preds=hint_preds, output_preds=output_preds,
                                     hiddens=nxt_hidden, lstm_state=lstm_state, z=z, msgs=msgs)
    accum_mp_state = _ScanStateCapture(
        hint_preds=hint_preds if return_hints else None,
        output_preds=output_preds if return_all_outputs else None,
        hiddens=nxt_hidden, lstm_state=None, z=z, msgs=msgs)
    return new_mp_state, accum_mp_state

  def __call__(self, features_list, repred, algorithm_index, return_hints, return_all_outputs):
    assert algorithm_index != -1
    features = features_list[0]
    self.encoders, self.decoders = self._construct_encoders_decoders()
    self.processor = self.processor_factory(self.hidden_dim)
    self.lstm = None

    inputs, hints, lengths = features.inputs, features.hints, features.lengths
    batch_size, nb_nodes = nets._data_dimensions(features)
    nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
    hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))

    mp_state = _ScanStateCapture(hint_preds=None, output_preds=None, hiddens=hiddens,
                                 lstm_state=None, z=None, msgs=None)
    common_args = dict(hints=hints, repred=repred, inputs=inputs, batch_size=batch_size,
                       nb_nodes=nb_nodes, lengths=lengths, spec=self.spec[algorithm_index],
                       encs=self.encoders[algorithm_index], decs=self.decoders[algorithm_index],
                       return_hints=return_hints, return_all_outputs=return_all_outputs)
    mp_state, lean0 = self._msg_passing_step(mp_state, i=0, first_step=True, **common_args)

    scan_fn = functools.partial(self._msg_passing_step, first_step=False, **common_args)
    output_mp_state, accum_mp_state = hk.scan(scan_fn, mp_state, jnp.arange(nb_mp_steps - 1) + 1,
                                              length=nb_mp_steps - 1)

    accum_mp_state = jax.tree_util.tree_map(
        lambda init, tail: jnp.concatenate([init[None], tail], axis=0), lean0, accum_mp_state)

    hint_preds = None
    if return_hints:
      def invert(d):
        return [dict(zip(d, i)) for i in zip(*d.values())] if d else None
      hint_preds = invert(accum_mp_state.hint_preds)
    output_preds = (output_mp_state.output_preds if not return_all_outputs
                    else {k: jnp.stack(v) for k, v in accum_mp_state.output_preds.items()})
    # z, msgs, hiddens: (nb_mp_steps, batch, nb_nodes, dim) -- every step, incl. step 0.
    return output_preds, hint_preds, accum_mp_state.hiddens, accum_mp_state.z, accum_mp_state.msgs
