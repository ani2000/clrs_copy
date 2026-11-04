"""Pure-numpy port of viewer3.html's client-side attribution engine, so Phase
1/2 can build supernode diagrams and run interventions programmatically
(no browser) across all 12 algorithms. Mirrors buildAttribution/toSupernodes/
runIntervention in viewer3.html exactly -- same linear algebra, same scope
disclosures (leaves = node_fts/msg per step, bias included, top prediction
explained).
"""
import numpy as np


def encode(W, x):
    xn = (x - W['mu_x']) / W['sd_x']
    pre = xn @ W['w_enc'] + W['b_enc']
    act = np.maximum(pre, 0)
    F, K = W['F'], W['K']
    idx = np.argsort(-act)[:K]
    feats = np.zeros(F)
    feats[idx] = act[idx]
    return feats


def reconstruct(W, feats):
    return feats @ W['w_dec'] + W['b_dec']


def decode_all_classes(W, h_t):
    return h_t @ W['W_dec'] + W['b_dec_target']


def build_attribution(W, G, step, node):
    H = W['hidden']
    z, msgs, node_fts = G['z'][step][node], G['msgs'][step][node], G['node_fts'][step][node]
    z_prev = G['z'][step - 1][node] if step > 0 else None
    msgs_prev = G['msgs'][step - 1][node] if step > 0 else None
    feats_cur = encode(W, np.concatenate([z, msgs]))
    feats_prev = encode(W, np.concatenate([z_prev, msgs_prev])) if step > 0 else np.zeros(W['F'])
    hidden_in = reconstruct(W, feats_prev) if step > 0 else np.zeros(H)
    nxt = reconstruct(W, feats_cur)
    h_t = np.concatenate([node_fts, hidden_in, nxt])
    class_logits = decode_all_classes(W, h_t)
    class_idx = int(np.argmax(class_logits))
    decode_vec = W['W_dec'][:, class_idx]
    decode_bias = W['b_dec_target'][class_idx]
    logit = float(h_t @ decode_vec + decode_bias)

    target_name = f"{W['target']}[{step + 1}][{node}]" + (f"={class_idx}" if len(class_logits) > 1 else '')
    edges = []

    def add_edge(s, d, w):
        if abs(w) > 1e-9:
            edges.append((s, d, float(w)))

    for f in range(W['F']):
        if feats_cur[f] > 0:
            g = decode_vec[2 * H:3 * H]
            add_edge(f'f{f}@{step}:{node}', target_name, feats_cur[f] * np.dot(W['w_dec'][f], g))
    add_edge(f'node_fts@{step}:{node}', target_name, np.dot(node_fts, decode_vec[:H]))
    add_edge('bias', target_name, decode_bias)
    if step > 0:
        for f in range(W['F']):
            if feats_prev[f] > 0:
                g = decode_vec[H:2 * H]
                add_edge(f'f{f}@{step - 1}:{node}', target_name, feats_prev[f] * np.dot(W['w_dec'][f], g))

    for t, fa, za, ma in [(step, feats_cur, z, msgs), (step - 1, feats_prev, z_prev, msgs_prev)]:
        if t < 0:
            continue
        for f in range(W['F']):
            if fa[f] > 0:
                gx = W['w_enc'][:, f] / W['sd_x']
                gNf, gHprev, gMsg = gx[:H], gx[H:2 * H], gx[2 * H:]
                add_edge(f'node_fts@{t}:{node}', f'f{f}@{t}:{node}', np.dot(za[:H], gNf))
                add_edge(f'msg@{t}:{node}', f'f{f}@{t}:{node}', np.dot(ma, gMsg))
                if t > 0:
                    z_prev2, msgs_prev2 = G['z'][t - 1][node], G['msgs'][t - 1][node]
                    feats_prev2 = encode(W, np.concatenate([z_prev2, msgs_prev2]))
                    for f2 in range(W['F']):
                        if feats_prev2[f2] > 0:
                            add_edge(f'f{f2}@{t - 1}:{node}', f'f{f}@{t}:{node}',
                                    feats_prev2[f2] * np.dot(W['w_dec'][f2], gHprev))

    return dict(target_name=target_name, step=step, node=node, logit=logit, class_logits=class_logits.tolist(),
               class_idx=class_idx, edges=edges, feats_cur=feats_cur, feats_prev=feats_prev,
               decode_vec=decode_vec, decode_bias=decode_bias, h_t=h_t)


def supernode_key(name, A, label_by_feat):
    if name == A['target_name'] or name == 'bias':
        return name
    import re
    m = re.match(r'^f(\d+)@', name)
    if m:
        lab = label_by_feat.get(int(m.group(1)))
        if lab and lab.get('label') not in (None, '(unlabeled)', '(rare)'):
            return lab['label']
        return '(unlabeled feature)'
    if name.startswith('node_fts'):
        return 'node_fts (all steps)'
    if name.startswith('msg'):
        return 'msg (all steps)'
    return name


def to_supernodes(edges, A, label_by_feat):
    merged = {}
    rep_feature = {}
    for s, d, w in edges:
        ks, kd = supernode_key(s, A, label_by_feat), supernode_key(d, A, label_by_feat)
        if ks == kd:
            continue
        merged[(ks, kd)] = merged.get((ks, kd), 0.0) + w
        import re
        m = re.match(r'^f(\d+)@', s)
        if m and ks not in rep_feature:
            rep_feature[ks] = int(m.group(1))
    return [(s, d, w) for (s, d), w in merged.items()], rep_feature


def run_intervention_group(W, G, A, features, step, node, factor):
    """Clamp ALL of `features` (a supernode's active members at (step,node))
    to factor*their-own-activation simultaneously, recompute the target logit
    through the replacement model, allowing top-k reselection at later steps
    to cascade (real causal test, matching the paper's group-level "inhibit
    feature groups" interventions, not a single-feature toy)."""
    H = W['hidden']
    target_step, target_node = A['step'], A['node']
    z, msgs, node_fts = G['z'][target_step][target_node], G['msgs'][target_step][target_node], G['node_fts'][target_step][target_node]
    base_feats_cur, base_feats_prev = A['feats_cur'].copy(), A['feats_prev'].copy()
    base_logit = A['logit']
    src = base_feats_cur if step == target_step else base_feats_prev
    orig_vals = {f: float(src[f]) for f in features if src[f] > 0}
    if not orig_vals:
        return None  # none of this supernode's features were active at this (step,node) -- nothing to clamp

    def recompute(f_prev, f_cur):
        hidden_in = reconstruct(W, f_prev) if target_step > 0 else np.zeros(H)
        nxt = reconstruct(W, f_cur)
        h_t = np.concatenate([node_fts, hidden_in, nxt])
        return float(h_t @ A['decode_vec'] + A['decode_bias']), h_t

    if step == target_step:
        nf = base_feats_cur.copy()
        for f, v in orig_vals.items(): nf[f] = factor * v
        new_logit, new_h_t = recompute(base_feats_prev, nf)
        mask_changed = not np.array_equal(nf > 0, base_feats_cur > 0)
    else:
        nf1 = base_feats_prev.copy()
        for f, v in orig_vals.items(): nf1[f] = factor * v
        hidden_in_alt = reconstruct(W, nf1)
        z_alt = np.concatenate([node_fts, hidden_in_alt])
        new_feats_cur = encode(W, np.concatenate([z_alt, msgs]))
        new_logit, new_h_t = recompute(nf1, new_feats_cur)
        mask_changed = not np.array_equal(new_feats_cur > 0, base_feats_cur > 0)

    new_class_logits = decode_all_classes(W, new_h_t)  # full distribution, not just the target class
    return dict(features=list(orig_vals.keys()), step=int(step), node=int(node), factor=factor,
               orig_activations={int(f): v for f, v in orig_vals.items()},
               base_logit=base_logit, new_logit=new_logit, delta=new_logit - base_logit, mask_changed=bool(mask_changed),
               new_class_logits=new_class_logits.tolist())
