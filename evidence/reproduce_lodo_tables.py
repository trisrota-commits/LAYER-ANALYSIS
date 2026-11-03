#!/usr/bin/env python
"""Reproduce LODO from immutable ZIP inputs, without extraction or model execution.
Requires Python 3.10+, NumPy and SciPy. Outputs are exclusively created; never overwritten.
python reproduce_lodo_tables.py --archive INPUT.zip --output NEW.log
Comparison logs default to the original evidence log directory, independently of output.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
import zipfile
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import rankdata, t

BASE = Path(__file__).resolve().parent
LOGS = BASE / 'overleaf-logs-only-20261015-100424/experiment_logs'
ARCHIVE_SHA = '311fc68d41b988163b12a2af359280eea6433bd98a27ef7a274dbdf77e47cb69'
FOLDS = ['cifar-100', 'svhn', 'imagenette', 'celeba-gender', 'cifar-10', 'mnist', 'places365']
METRICS = ['RMSE', 'MAE', 'Pearson', 'Spearman', 'R2']
SEEDS = list(range(42, 48))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def relative(path):
    require(path.startswith('/a/mm/'), 'Unexpected source prefix: ' + path)
    return path[len('/a/mm/'):]

def correlation(x, y):
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sum(xc * yc, axis=-1) / np.sqrt(np.sum(xc*xc, axis=-1)*np.sum(yc*yc, axis=-1))

def metrics(y, p):
    # p has shape (arms, seeds, records); each metric remains per seed.
    e = p - y
    sse = np.sum(e*e, axis=-1)
    sst = np.sum((y-y.mean())**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1-sse/sst
    # Ranks MUST be recomputed after resampling, including duplicated records.
    return np.stack([np.sqrt(np.mean(e*e, axis=-1)), np.mean(np.abs(e), axis=-1),
                     correlation(y, p), correlation(rankdata(y), rankdata(p, axis=-1)), r2], axis=-1)

def delta(m):
    with np.errstate(invalid='ignore'):
        return (m[1]-m[0])*np.array([-1, -1, 1, 1, 1])

def expected_bootstrap(text):
    result = {}
    aliases = {'CelebA-Gender':'celeba-gender', 'CIFAR-10':'cifar-10', 'MNIST':'mnist', 'Places365':'places365'}
    for line in text.splitlines():
        fields = line.split(',')
        if len(fields) == 8 and fields[0] in aliases:
            result[aliases[fields[0]], fields[3]] = ([float(x) for x in fields[4:7]], int(fields[7]))
        for label, fold in [('CIFAR100','cifar-100'), ('SVHN','svhn'), ('Imagenette','imagenette')]:
            if line.startswith('|'+label+'('):
                cells = line.strip('|').split('|')[1:]
                require(len(cells)==5, 'Handoff table columns')
                for metric, cell in zip(METRICS, cells):
                    nums = re.fullmatch(r'([^\[]+)\[([^,]+),([^\]]+)\]', cell)
                    require(nums is not None, 'Bad handoff cell: '+cell)
                    result[fold, metric] = ([float(x) for x in nums.groups()], 10000)
    require(len(result)==35, 'Expected exactly 35 LODO bootstrap reference rows')
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, default=BASE/'holdout-reproduction-inputs-728444c5.zip')
    ap.add_argument('--output', type=Path, default=LOGS/'lodo_tables_reproduced.log')
    ap.add_argument('--paired-reference', type=Path, default=LOGS/'paired_holdout_tables.log')
    ap.add_argument('--descriptive-reference', type=Path, default=LOGS/'lodo_descriptive_table_evidence.log')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Refusing to overwrite existing output: '+str(args.output))
    require(args.output.parent.is_dir(), 'Output parent must already exist')
    archive_hash = sha(args.archive.read_bytes())
    require(archive_hash==ARCHIVE_SHA, 'Archive SHA256 mismatch')
    paired_bytes = args.paired_reference.read_bytes()
    expected = expected_bootstrap(paired_bytes.decode('utf-8-sig'))
    descriptive_bytes = args.descriptive_reference.read_bytes()
    desc_expected = {}
    condition = None
    for line in descriptive_bytes.decode('utf-8-sig').splitlines():
        if line.startswith('CONDITION '):
            condition = line.split(' ', 1)[1]
        elif line.split('\t')[0] in METRICS:
            cells = line.split('\t')
            desc_expected[condition, cells[0]] = float(cells[1])
    lines = ['LODO TABLES: FRESH SAVED-PREDICTION RECOMPUTATION; NO TRAINING OR INFERENCE',
             'Archive: '+str(args.archive), 'Archive SHA256: '+archive_hash,
             'Script SHA256: '+sha(Path(__file__).read_bytes()),
             'Versions: Python '+platform.python_version()+' NumPy '+np.__version__+' SciPy '+scipy.__version__,
             'Paired reference SHA256: '+sha(paired_bytes), 'Descriptive reference SHA256: '+sha(descriptive_bytes),
             'Six fixed seeds 42-47; descriptive mean +/- t(0.975,df=5)*sample_SD/sqrt(6).',
             'Bootstrap: 10000 draws, numpy.default_rng(0) RESET PER FOLD; first-appearance architecture order;',
             'sample G groups with replacement per draw, retain ALL rows of each sampled group, record-weighted metrics;',
             'metrics per seed then mean; all six differences finite required independently per metric;',
             'Spearman ranks recalculated per draw; positive favors Taylor: ref-candidate errors, candidate-ref correlations/R2.',
             'Percentile CI np.quantile([.025,.975],method=linear); no seed resampling or multiplicity correction.',
             'Model files not required; receipt and seed proofs plus archived prompts/predictions checked.']
    mismatches = []
    with zipfile.ZipFile(args.archive) as z:
        names = [n for n in z.namelist() if not n.endswith('/')]
        require(len(names)==len(set(names)), 'Duplicate ZIP names')
        manifests = [n for n in names if n.endswith('reproduction_input_manifest.json')]
        require(bool(manifests), 'No manifest')
        covered = set()
        for manifest in manifests:
            data = json.loads(z.read(manifest))
            for name, entry in data['files'].items():
                raw = z.read(name)
                require(sha(raw)==entry['sha256'] and len(raw)==entry['bytes'], 'Member integrity failure: '+name)
                covered.add(name)
        require(set(names)-set(manifests)==covered-set(manifests), 'Unmanifested ZIP members')
        lines.append(f'INTEGRITY PASS: {len(covered)} manifest-listed members; manifest itself authenticated by archive SHA256.')
        for name in names:
            lines.append('MEMBER_SHA256 '+sha(z.read(name))+' '+name)
        def js(name):
            return json.loads(z.read(name))
        for fold in FOLDS:
            root = 'lodo-taylor-nolayer-off-v1' if fold in FOLDS[:3] else 'lodo-taylor-rest-off-v1'
            plan = js(root+'/plan.json')
            predictions = []
            canonical_ids = canonical_y = None
            for arm in ['no-layer','taylor']:
                arm_predictions = []
                for seed in SEEDS:
                    key = f'{fold}-{arm}-s{seed}'
                    run = plan['runs'][key]
                    require(run['dataset']==fold and run['arm']==arm and run['seed']==seed, key+' plan identity')
                    require(run['zcp_enabled'] is False and run['accuracy_observations']==3 and run['layer_features']==(['taylor_imp'] if arm=='taylor' else []), key+' conditions')
                    out = relative(run['output'])
                    if key=='mnist-no-layer-s45':
                        out = 'lodo-taylor-rest-off-v1/recovery/mnist-no-layer-s45-eval-v1'
                    if key=='cifar-10-no-layer-s46':
                        out = 'lodo-cifar10-recovery-v1/results/'+key
                    if fold=='places365':
                        out = 'lodo-places-recovery-v1/results/'+key
                    receipt = js(out+'/run_complete.json')
                    proof = js(out+'/seed_verified.json')
                    require(receipt['run']==key and relative(receipt['output'])==out, key+' receipt run/output')
                    for field in ['dataset','arm','seed','base','data','counts','manifest','zcp_enabled','accuracy_observations','layer_features','maximum_epochs','early_stopping_patience']:
                        require(receipt[field]==run[field], key+' receipt '+field)
                    for field in ['seed','maximum_epochs','early_stopping_patience','zcp_enabled']:
                        require(proof[field]==run[field], key+' seed proof '+field)
                    require(proof['layer_enabled']==(arm=='taylor'), key+' seed layer proof')
                    prompt_path = relative(run['data'])+'/test_llm_dataset.jsonl'
                    prompts = [json.loads(line) for line in z.read(prompt_path).decode().splitlines() if line.strip()]
                    ids = [p['architecture_id'] for p in prompts]
                    require(all(isinstance(i,str) and i for i in ids), key+' missing architecture identity')
                    ys = []
                    for p in prompts:
                        answers = [m['content'] for m in p['messages'] if m['role']=='assistant']
                        require(len(answers)==1, key+' assistant answer count')
                        ys.append(float(json.loads(answers[0])['best_accuracy']))
                        users = '\n'.join(m['content'] for m in p['messages'] if m['role']=='user')
                        require(re.search(r'^dataset:\s*'+re.escape(fold)+r'\s*$', users, re.M) is not None, key+' prompt dataset')
                    y = np.array(ys)
                    rows = list(csv.DictReader(io.StringIO(z.read(out+'/test_predictions.csv').decode('utf-8-sig'))))
                    n = run['counts']['test']
                    require(n==len(rows)==len(prompts), key+' row counts')
                    require(run['counts']==plan['folds'][fold]['counts'], key+' fold counts')
                    for field in ['n_samples','n_rows_scored','n_valid']:
                        require(receipt['metrics'][field]==n, key+' metric count '+field)
                    require(receipt['metrics']['parse_failures']==0, key+' parse failures')
                    idx = [int(r['idx']) for r in rows]
                    require(idx==list(range(n)) or idx==list(range(1,n+1)), key+' full ordered idx')
                    require(all(r['dataset']==fold for r in rows), key+' CSV dataset')
                    gt = np.array([float(r['gt_best_accuracy']) for r in rows])
                    pred = np.array([float(r['pred_best_accuracy']) for r in rows])
                    require(np.isfinite(gt).all() and np.isfinite(pred).all() and np.array_equal(gt,y), key+' finite/GT prompt match')
                    if canonical_ids is None:
                        canonical_ids, canonical_y = ids, y
                    require(ids==canonical_ids and np.array_equal(y,canonical_y), key+' cross-arm/seed pairing')
                    arm_predictions.append(pred)
                    lines.append('RUN_PASS '+key+' '+json.dumps({'plan':root+'/plan.json','prompt':prompt_path,'output':out,'records':n,'idx_base':idx[0],'seed_proof':proof}, sort_keys=True))
                predictions.append(arm_predictions)
            p = np.array(predictions)
            y = canonical_y
            point = metrics(y,p)
            require(np.isfinite(point).all(), fold+' nonfinite point metric')
            groups = {}
            for i, group in enumerate(canonical_ids):
                groups.setdefault(group, []).append(i)
            members = [np.array(v,dtype=int) for v in groups.values()]
            lines.append(f'FOLD_PASS {fold} records={len(y)} architecture_groups={len(groups)} fixed_seeds={SEEDS}')
            for a, arm in enumerate(['no-layer','taylor']):
                lines.append('PER_SEED '+fold+' '+arm+' '+json.dumps(point[a].tolist()))
                means = point[a].mean(axis=0)
                half = t.ppf(.975,5)*point[a].std(axis=0,ddof=1)/np.sqrt(6)
                for j, metric in enumerate(METRICS):
                    lines.append(f'DESCRIPTIVE,{fold},{arm},{metric},{means[j]:.10f},{means[j]-half[j]:.10f},{means[j]+half[j]:.10f}')
                    if fold in FOLDS[3:]:
                        old = desc_expected[fold+'-'+arm,metric]
                        ok = f'{means[j]:.10f}'==f'{old:.10f}'
                        lines.append(f'DESCRIPTIVE_COMPARE,{fold},{arm},{metric},{"PASS" if ok else "MISMATCH"},expected={old:.10f},actual={means[j]:.10f}')
                        if not ok:
                            mismatches.append('descriptive '+fold+' '+arm+' '+metric)
            rng = np.random.default_rng(0)
            boot = np.full((10000,5),np.nan)
            for b in range(10000):
                selected = rng.integers(0,len(members),size=len(members))
                ix = np.concatenate([members[g] for g in selected])
                d = delta(metrics(y[ix],p[:,:,ix]))
                valid = np.isfinite(d).all(axis=0)
                boot[b,valid] = d[:,valid].mean(axis=0)
            observed = delta(point).mean(axis=0)
            for j, metric in enumerate(METRICS):
                samples = boot[:,j][np.isfinite(boot[:,j])]
                lo, hi = np.quantile(samples,[.025,.975],method='linear')
                actual = [observed[j],lo,hi]
                old, count = expected[fold,metric]
                ok = all(f'{x:.6f}'==f'{v:.6f}' for x,v in zip(actual,old)) and len(samples)==count
                lines.append(f'BOOTSTRAP,{fold},{len(y)},{len(groups)},{metric},{observed[j]:.6f},{lo:.6f},{hi:.6f},{len(samples)}')
                lines.append('BOOTSTRAP_COMPARE,'+fold+','+metric+','+('PASS' if ok else 'MISMATCH')+',expected='+','.join(f'{x:.6f}' for x in old)+f',valid={count}')
                if not ok:
                    mismatches.append('bootstrap '+fold+' '+metric)
            print('Computed',fold,'records',len(y),'groups',len(groups),flush=True)
    lines.append('SUMMARY '+json.dumps({'integrity':'PASS','pairing':'PASS','runs':84,'descriptive_rows':70,'bootstrap_rows':35,'bootstrap_comparisons':35,'new_descriptive_mean_comparisons':40,'mismatches':mismatches}))
    with args.output.open('x',encoding='utf-8',newline='\n') as f:
        f.write('\n'.join(lines)+'\n')
    print('OUTPUT',args.output,'SHA256',sha(args.output.read_bytes()),'MISMATCHES',len(mismatches),flush=True)
    return 2 if mismatches else 0

if __name__=='__main__':
    sys.exit(main())
