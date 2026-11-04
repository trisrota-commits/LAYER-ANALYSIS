"""Aggregate exported dependence cells and actual timing records, not new training."""
import io
import json
import tarfile
import zipfile

import numpy as np
from analysis.thesis_common import csv_rows, require


def build_tables(repo):
    with zipfile.ZipFile(repo / 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip') as packet:
        with tarfile.open(fileobj=io.BytesIO(packet.read('article-dependence-source-20261014-150909.tar.gz'))) as archive:
            files = {m.name.removeprefix('./'): archive.extractfile(m).read() for m in archive if m.isfile() and m.name.endswith('.csv')}
        with tarfile.open(fileobj=io.BytesIO(packet.read('article-timing-baselines-20261014-113919.tar.gz'))) as archive:
            timing_files = {m.name.removeprefix('./'): archive.extractfile(m).read() for m in archive if m.isfile() and m.name.endswith('timings.jsonl')}
    dependence = []
    for stream, name, count in [('early_to_epoch50', 'early_to_epoch50_mic.csv', 147), ('same_epoch', 'same_epoch_mic.csv', 196)]:
        rows = csv_rows(files['files/' + name])
        require(len(rows) == count and all(r['status'] == 'ok' for r in rows), 'Dependence cell coverage/status changed')
        features = sorted({r['feature'] for r in rows})
        require(len(features) == 7, 'Feature count changed')
        for feature in features:
            cells = [r for r in rows if r['feature'] == feature]
            dependence.append(dict(stream=stream, feature=feature, MIC=float(np.mean([float(r['mic']) for r in cells])),
                                   absolute_Spearman=float(np.mean([abs(float(r['spearman_rho'])) for r in cells])),
                                   significant=sum(float(r['mic_bh_fdr_q_value']) <= .05 for r in cells), cells=len(cells)))
    dc = csv_rows(files['files/same_epoch_dependence.csv'])
    require(len(dc) == 196 and len({(r['dataset'], r['source_epoch'], r['feature']) for r in dc}) == 196,
            'Incomplete/duplicate dCor cells')
    require(all(r['status'] == 'ok' for r in dc), 'Invalid dCor cell')
    # Independently recompute BH values, preserving analytical-test provenance.
    p = np.array([float(r['analytical_p_value']) for r in dc]); order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    q = np.empty(len(p)); q[order] = np.minimum(adjusted, 1.)
    require(np.allclose(q, [float(r['global_bh_fdr_q_value']) for r in dc], rtol=0, atol=1e-12), 'Saved dCor BH values disagree')
    layer = [json.loads(line) for line in timing_files['files/results/layer_cost_final_100_60min_v1/timings.jsonl'].decode().splitlines() if line.strip()]
    proxies = [json.loads(line) for line in timing_files['files/zcp-five-cost-v1/results/full/timings.jsonl'].decode().splitlines() if line.strip()]
    complete = [r for r in layer if r['status'] == 'ok']
    proxies = [r for r in proxies if r['status'] == 'ok']
    require(len(complete) == len(proxies) == 96, 'Timing cohort count changed')
    require(len({r['id'] for r in complete}) == len({r['id'] for r in proxies}) == 96, 'Duplicate timing IDs')
    shared = len({r['id'] for r in complete} & {r['id'] for r in proxies})
    require(shared == 93, 'Timing cohort intersection changed')
    timing = []
    for name in ['training', 'evaluation']:
        require(all(r[name]['status'] == 'ok' for r in complete), 'Invalid baseline timing')
        timing.append(dict(operation=name, seconds=float(np.mean([r[name]['seconds'] for r in complete])), n=96,
                           boundary='one epoch', shared_layer_candidates=96))
    for arm in complete[0]['arms']:
        require(all(r['arms'][arm]['status'] == 'ok' for r in complete), 'Incomplete layer timing arm')
        timing.append(dict(operation=arm, seconds=float(np.mean([r['arms'][arm]['seconds'] for r in complete])), n=96,
                           boundary='one checkpoint', shared_layer_candidates=96))
    for name, key in [('zcp_compute', 'five_compute_seconds'), ('zcp_preprocessing', 'end_to_end_seconds')]:
        timing.append(dict(operation=name, seconds=float(np.mean([r[key] for r in proxies])), n=96,
                           boundary='once per candidate', shared_layer_candidates=shared))
    return {'6.1': dependence, '6.13': timing}, dc
