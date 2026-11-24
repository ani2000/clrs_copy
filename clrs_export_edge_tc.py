"""Convert the edge-site transcoders trained for the 12 CLRS-30 MPNNs (clrs-transcoder-out/<algo>/transcoder.npz)
into the browser format served by the tracer, and write one summary file with their measured quality.

  python clrs_export_edge_tc.py <clrs-transcoder-out folder> edge_tc

Per algorithm it writes
  edge_tc/<algo>.bin   float32, little endian, in this order:
                       s_mu[128] y_mu[128] b_pre[128] b_dec[128] b_enc[D] s_scale y_scale W_enc[128*D] W_dec[D*128]
                       (W_enc is [128, D] and W_dec is [D, 128], both row-major, exactly as in the .npz)
  edge_tc/<algo>.json  {algo, H, D, k, offsets (in floats), sha256 of the .bin}
and one edge_tc/clrs-edge-summary.json copied field by field from each report.json (nothing typed by hand).
Only the edge site (s_ij -> y_ij) is exported. The node-site transcoders are not used by the tracer (see the summary).
"""
import hashlib, json, os, sys
import numpy as np

H = 128
ALGOS = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim', 'mst_kruskal', 'topological_sort',
         'articulation_points', 'bridges', 'strongly_connected_components', 'floyd_warshall']


def export(src, dst):
    os.makedirs(dst, exist_ok=True)
    summary = {}
    for a in ALGOS:
        z = np.load(os.path.join(src, a, 'transcoder.npz'), allow_pickle=False)
        D, k = z['W_enc'].shape[1], int(z['k'])
        assert z['W_enc'].shape == (H, D) and z['W_dec'].shape == (D, H) and z['b_enc'].shape == (D,), a
        parts = [('s_mu', z['s_mu']), ('y_mu', z['y_mu']), ('b_pre', z['b_pre']), ('b_dec', z['b_dec']), ('b_enc', z['b_enc']),
                 ('scales', np.array([z['s_scale'], z['y_scale']])), ('W_enc', z['W_enc']), ('W_dec', z['W_dec'])]
        off, pos, blobs = {}, 0, []
        for name, arr in parts:
            arr = np.ascontiguousarray(arr, dtype='<f4').ravel(); off[name] = [pos, int(arr.size)]; pos += arr.size; blobs.append(arr.tobytes())
        raw = b''.join(blobs)
        open(os.path.join(dst, a + '.bin'), 'wb').write(raw)
        json.dump(dict(algo=a, H=H, D=D, k=k, offsets=off, sha256=hashlib.sha256(raw).hexdigest(), floats=pos),
                  open(os.path.join(dst, a + '.json'), 'w'), separators=(',', ':'))
        r = json.load(open(os.path.join(src, a, 'report.json')))
        e1, e2, dr = r['E1'], r['E2'], r['direct']
        summary[a] = dict(
            tier=r['tier'], baseline_val_f1=r['baseline']['val'], baseline_test_f1=r['baseline']['test'], D=D, k=k,
            nmse_train=e1['train']['nmse'], nmse_val=e1['val']['nmse'], nmse_n64=e1['test_n64']['nmse'] if 'test_n64' in e1 else None,
            dead_frac_train=e1['train']['dead_frac'], l0_val=e1['val']['l0'],
            retention_val=e2['val']['retention_full'], retention_n64=e2['test']['retention_full'] if 'test' in e2 else None,
            f1_full_val=e2['val']['full']['score'], f1_off_val=e2['val']['off']['score'],
            winner_flip_rate=dr['winner_flips']['flip_rate'], winner_flip_violations=dr['winner_flips']['n_violations'],
            hidden_rel_error_direct_mean=float(np.mean(list(dr['hidden_rel_error']['direct'].values()))),
            parity_abs_diff=r['parity']['abs_diff'], minutes=r['minutes'],
            node_site_retention_val=e2['val'].get('retention_node_full'), both_sites_retention_val=e2['val'].get('retention_both_full'))
    json.dump(dict(source='clrs-transcoder-out/*/report.json', site='edge: s_ij -> y_ij', summary=summary),
              open(os.path.join(dst, 'clrs-edge-summary.json'), 'w'), indent=1)
    return summary


if __name__ == '__main__':
    s = export(sys.argv[1], sys.argv[2])
    for a, v in s.items():
        print(f"{a:30s} tier {v['tier']} D={v['D']} k={v['k']} nmse_val={v['nmse_val']:.4f} retention val={v['retention_val']:.3f} n64={v['retention_n64']}")
